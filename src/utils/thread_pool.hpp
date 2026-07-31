// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <pthread.h>

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class ThreadPool {
   public:
    explicit ThreadPool(size_t threads, int realtime_priority = 0) {
        try {
            workers_.reserve(threads);
            for (size_t i = 0; i < threads; ++i) {
                workers_.emplace_back([this] { worker_loop(); });
                if (realtime_priority > 0) {
                    struct sched_param sp {};
                    sp.sched_priority = realtime_priority;
                    const int error = pthread_setschedparam(
                        workers_.back().native_handle(), SCHED_FIFO, &sp);
                    if (error != 0) {
                        throw std::system_error(
                            error, std::generic_category(),
                            "Failed to set realtime priority for ThreadPool");
                    }
                }
            }
        } catch (...) {
            stop_workers();
            join_workers();
            throw;
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename F>
    void run_parallel(size_t task_count, F&& task) {
        using Task = std::decay_t<F>;
        Task callable(std::forward<F>(task));

        std::unique_lock<std::mutex> lock(state_mutex_);
        task_context_ = &callable;
        task_invoker_ = [](void* context, size_t index) {
            std::invoke(*static_cast<Task*>(context), index);
        };
        task_count_ = task_count;
        next_task_ = 0;
        remaining_tasks_ = task_count;
        first_exception_ = nullptr;
        ++generation_;

        lock.unlock();
        work_cv_.notify_all();
        lock.lock();
        completion_cv_.wait(lock, [this] { return remaining_tasks_ == 0; });

        const std::exception_ptr first_exception = first_exception_;
        task_context_ = nullptr;
        task_invoker_ = nullptr;
        lock.unlock();

        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
    }

    ~ThreadPool() {
        stop_workers();
        join_workers();
    }

   private:
    using TaskInvoker = void (*)(void*, size_t);

    void worker_loop() {
        pthread_setname_np(pthread_self(), "motor_bus");
        size_t observed_generation = 0;

        for (;;) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            work_cv_.wait(lock, [this, &observed_generation] {
                return stop_ || generation_ != observed_generation;
            });
            if (stop_) {
                return;
            }
            observed_generation = generation_;

            while (next_task_ < task_count_) {
                const size_t task_index = next_task_++;
                TaskInvoker invoker = task_invoker_;
                void* context = task_context_;
                lock.unlock();

                std::exception_ptr task_exception;
                try {
                    invoker(context, task_index);
                } catch (...) {
                    task_exception = std::current_exception();
                }

                lock.lock();
                if (task_exception && !first_exception_) {
                    first_exception_ = task_exception;
                }
                if (--remaining_tasks_ == 0) {
                    completion_cv_.notify_one();
                }
            }
        }
    }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stop_ = true;
        }
        work_cv_.notify_all();
    }

    void join_workers() {
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex state_mutex_;
    std::condition_variable work_cv_;
    std::condition_variable completion_cv_;
    bool stop_ = false;
    size_t generation_ = 0;
    size_t task_count_ = 0;
    size_t next_task_ = 0;
    size_t remaining_tasks_ = 0;
    void* task_context_ = nullptr;
    TaskInvoker task_invoker_ = nullptr;
    std::exception_ptr first_exception_;
};
