// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi
//
// Async depth encoder node: preprocess/history/encode run here and publish
// Float32MultiArray on depth_obs_topic so inference_node stays non-blocking.

#include "depth_provider.hpp"

#include <onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <vector>

class DepthNode : public rclcpp::Node {
   public:
    DepthNode() : Node("depth_node") {
        declare_parameter<std::string>("depth_obs_topic", "/depth_obs");
        declare_parameter<int>("depth_encoder_output_dim", 128);
        declare_parameter<int>("intra_threads", 1);
        declare_parameter<double>("depth_process_hz", 50.0);
        // DepthProvider also reads these; declare here so expected size is known first.
        declare_parameter<bool>("use_depth_encoder", true);
        declare_parameter<std::vector<int64_t>>("depth_history_taps",
                                                 std::vector<int64_t>{35, 30, 25, 20, 15, 10, 5, 0});
        declare_parameter<int>("depth_width", 64);
        declare_parameter<int>("depth_height", 36);
        declare_parameter<int>("depth_crop_up", 18);
        declare_parameter<int>("depth_crop_down", 0);
        declare_parameter<int>("depth_crop_left", 16);
        declare_parameter<int>("depth_crop_right", 16);

        get_parameter("depth_obs_topic", depth_obs_topic_);
        get_parameter("depth_encoder_output_dim", depth_encoder_output_dim_);
        get_parameter("intra_threads", intra_threads_);
        get_parameter("use_depth_encoder", use_depth_encoder_);
        double depth_process_hz = 50.0;
        get_parameter("depth_process_hz", depth_process_hz);
        if (depth_process_hz <= 0.0) {
            throw std::runtime_error("depth_process_hz must be positive");
        }
        const auto process_period =
            std::chrono::duration<double>(1.0 / depth_process_hz);

        const std::size_t expected_output_size = static_cast<std::size_t>(output_dim());

        Ort::ThreadingOptions thread_opts;
        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }
        env_ = std::make_unique<Ort::Env>(thread_opts, ORT_LOGGING_LEVEL_WARNING, "DepthEncoder");
        provider_ = std::make_unique<DepthProvider>(*this, *env_, expected_output_size);

        auto data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        depth_obs_publisher_ =
            create_publisher<std_msgs::msg::Float32MultiArray>(depth_obs_topic_, data_qos);
        clear_depth_history_service_ = create_service<std_srvs::srv::Trigger>(
            "clear_depth_history",
            std::bind(&DepthNode::clear_depth_history_srv, this, std::placeholders::_1,
                      std::placeholders::_2));

        process_timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(process_period),
            std::bind(&DepthNode::process_and_publish, this));

        RCLCPP_INFO(get_logger(), "depth_obs_topic: %s", depth_obs_topic_.c_str());
        RCLCPP_INFO(get_logger(), "depth_process_hz: %.3f", depth_process_hz);
        RCLCPP_INFO(get_logger(), "depth output dim: %zu", expected_output_size);
        RCLCPP_INFO(get_logger(), "depth encode runs asynchronously in depth_node");
    }

   private:
    int output_dim() const {
        if (use_depth_encoder_) {
            return depth_encoder_output_dim_;
        }
        int depth_width = 64;
        int depth_height = 36;
        int crop_up = 18;
        int crop_down = 0;
        int crop_left = 16;
        int crop_right = 16;
        std::vector<int64_t> taps;
        get_parameter("depth_width", depth_width);
        get_parameter("depth_height", depth_height);
        get_parameter("depth_crop_up", crop_up);
        get_parameter("depth_crop_down", crop_down);
        get_parameter("depth_crop_left", crop_left);
        get_parameter("depth_crop_right", crop_right);
        get_parameter("depth_history_taps", taps);
        const int obs_num = (depth_width - crop_left - crop_right) * (depth_height - crop_up - crop_down);
        return obs_num * static_cast<int>(taps.size());
    }

    void process_and_publish() {
        std::vector<float> obs;
        if (!provider_->process_latest(obs)) {
            return;
        }
        std_msgs::msg::Float32MultiArray msg;
        msg.data = std::move(obs);
        depth_obs_publisher_->publish(msg);
    }

    void clear_depth_history_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        provider_->reset();
        response->success = true;
        response->message = "depth history cleared";
    }

    std::string depth_obs_topic_;
    int depth_encoder_output_dim_ = 128;
    int intra_threads_ = 1;
    bool use_depth_encoder_ = true;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<DepthProvider> provider_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr depth_obs_publisher_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_depth_history_service_;
    rclcpp::TimerBase::SharedPtr process_timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        RCLCPP_WARN(rclcpp::get_logger("depth_node"), "mlockall failed.");
    }
    try {
        rclcpp::spin(std::make_shared<DepthNode>());
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("depth_node"), "Exception caught: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
