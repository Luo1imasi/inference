#include <onnxruntime_cxx_api.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <vector>

#define PI 3.1415926

class Inference : public rclcpp::Node {
   public:
    Inference() : Node("inference_node") {
        obs_.resize(47);
        act_.resize(12);
        left_obs_.resize(12);
        right_obs_.resize(12);
        imu_obs_.resize(7);
        left_act_.resize(12);
        right_act_.resize(12);
        motor_lower_limit_.resize(12);
        motor_higher_limit_.resize(12);
        step_ = 0;

        this->declare_parameter<std::string>("model_name", "1.onnx");
        this->declare_parameter<int>("intra_threads", -1);
        this->declare_parameter<int>("fps", 50);
        this->declare_parameter<float>("vx", 0.0);
        this->declare_parameter<float>("vy", 0.0);
        this->declare_parameter<float>("dyaw", 0.0);
        this->declare_parameter<float>("cycle_time", 0.0);
        this->declare_parameter<float>("obs_scales_lin_vel", 0.0);
        this->declare_parameter<float>("obs_scales_ang_vel", 0.0);
        this->declare_parameter<float>("obs_scales_dof_pos", 0.0);
        this->declare_parameter<float>("obs_scales_dof_vel", 0.0);
        this->declare_parameter<std::vector<float>>(
            "motor_lower_limit",
            std::vector<float>{-0.15, -1.0, -1.0, -0.4, -1.0, -0.5, -0.7, -1.2, -1.0, -1.5, -1.0, -0.5});
        this->declare_parameter<std::vector<float>>(
            "motor_higher_limit",
            std::vector<float>{0.7, 1.2, 1.0, 1.5, 1.0, 0.5, 0.15, 1.0, 1.0, 0.4, 1.0, 0.5});

        this->get_parameter("model_name", model_name_);
        this->get_parameter("intra_threads", intra_threads_);
        this->get_parameter("fps", fps_);
        this->get_parameter("vx", vx_);
        this->get_parameter("vy", vy_);
        this->get_parameter("dyaw", dyaw_);
        this->get_parameter("cycle_time", cycle_time_);
        this->get_parameter("obs_scales_lin_vel", obs_scales_lin_vel_);
        this->get_parameter("obs_scales_ang_vel", obs_scales_ang_vel_);
        this->get_parameter("obs_scales_dof_pos", obs_scales_dof_pos_);
        this->get_parameter("obs_scales_dof_vel", obs_scales_dof_vel_);
        std::vector<double> tmp;
        this->get_parameter("motor_lower_limit", tmp);
        std::transform(tmp.begin(), tmp.end(), motor_lower_limit_.begin(),
                       [](double val) { return static_cast<float>(val); });
        this->get_parameter("motor_higher_limit", tmp);
        std::transform(tmp.begin(), tmp.end(), motor_higher_limit_.begin(),
                       [](double val) { return static_cast<float>(val); });

        model_path_ = std::string(ROOT_DIR) + "models/" + model_name_;
        RCLCPP_INFO(this->get_logger(), "model_path: %s", model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "intra_threads: %d", intra_threads_);
        RCLCPP_INFO(this->get_logger(), "fps: %d", fps_);
        RCLCPP_INFO(this->get_logger(), "vx: %f", vx_);
        RCLCPP_INFO(this->get_logger(), "vy: %f", vy_);
        RCLCPP_INFO(this->get_logger(), "dyaw: %f", dyaw_);
        RCLCPP_INFO(this->get_logger(), "cycle_time: %f", cycle_time_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_lin_vel: %f", obs_scales_lin_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_ang_vel: %f", obs_scales_ang_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_dof_pos: %f", obs_scales_dof_pos_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_dof_vel: %f", obs_scales_dof_vel_);
        RCLCPP_INFO(this->get_logger(), "motor_lower_limit: %f", motor_lower_limit_);
        RCLCPP_INFO(this->get_logger(), "motor_higher_limit: %f", motor_higher_limit_);

        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");
        Ort::SessionOptions session_options;
        if (intra_threads_ > 0) {
            session_options.SetIntraOpNumThreads(intra_threads_);
        }
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options);
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error: %s", e.what());
        }
        size_t num_inputs = session_->GetInputCount();
        input_names_.resize(num_inputs);
        for (size_t i = 0; i < num_inputs; i++) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator_);
            input_names_[i] = input_name.get();
            auto type_info = session_->GetInputTypeInfo(i);
            input_shape_ = type_info.GetTensorTypeAndShapeInfo().GetShape();
        }
        size_t num_outputs = session_->GetOutputCount();
        output_names_.resize(num_outputs);
        for (size_t i = 0; i < num_outputs; i++) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_[i] = output_name.get();
        }

        left_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_left", 1, std::bind(&Inference::subs_left_callback, this, std::placeholders::_1));
        right_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_right", 1, std::bind(&Inference::subs_right_callback, this, std::placeholders::_1));
        IMU_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/IMU_data", 1, std::bind(&Inference::subs_IMU_callback, this, std::placeholders::_1));
        left_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_left", 1);
        right_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_right", 1);
        timer_ =
            this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&Inference::inference, this));
    }
    ~Inference() {
        for (auto name : input_names_) {
            allocator_.Free(const_cast<void *>(static_cast<const void *>(name)));
        }
        for (auto name : output_names_) {
            allocator_.Free(const_cast<void *>(static_cast<const void *>(name)));
        }
    }

   private:
    std::string model_name_, model_path_;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    std::vector<int64_t> input_shape_;
    int intra_threads_;
    Ort::AllocatorWithDefaultOptions allocator_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_publisher_, right_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_subscription_, right_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr  IMU_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    int step_;
    std::vector<float> obs_, act_;
    std::vector<float> left_obs_, right_obs_, imu_obs_, left_act_, right_act_;
    int fps_;
    float vx_, vy_, dyaw_;
    float cycle_time_, obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_, obs_scales_dof_vel_;
    std::vector<float> motor_lower_limit_, motor_higher_limit_;

    void subs_left_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
        for(int i = 0; i < 6; i++) {
            left_obs_[i] = msg->position[i];
            left_obs_[6 + i] = msg->velocity[i];
        }
    }

    void subs_right_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
        for(int i = 0; i < 6; i++) {
            right_obs_[i] = msg->position[i];
            right_obs_[6 + i] = msg->velocity[i];
        }
    }

    void subs_IMU_callback(const std::shared_ptr<sensor_msgs::msg::Imu> msg) {
        imu_obs_[0] = msg->orientation.w;
        imu_obs_[1] = msg->orientation.x;
        imu_obs_[2] = msg->orientation.y;
        imu_obs_[3] = msg->orientation.z;
        imu_obs_[4] = msg->angular_velocity.x;
        imu_obs_[5] = msg->angular_velocity.y;
        imu_obs_[6] = msg->angular_velocity.z;
    }

    void publish_joint_states() {
        // for (int i = 0; i < 6; i++) {
        //     left_motors[i]->refresh_motor_status();
        //     Timer::ThreadSleepFor(5);
        //     right_motors[i]->refresh_motor_status();
        //     Timer::ThreadSleepFor(5);
        // }
        auto left_message = sensor_msgs::msg::JointState();
        left_message.header.stamp = this->now();
        left_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
        left_message.position = {act_[0], act_[1], act_[2], act_[3], act_[4], act_[5]};
        left_message.velocity = {0, 0, 0, 0, 0, 0};
        left_message.effort = {0, 0, 0, 0, 0, 0};

        left_publisher_->publish(left_message);
        // RCLCPP_INFO(this->get_logger(), "Left Published JointState");

        auto right_message = sensor_msgs::msg::JointState();
        right_message.header.stamp = this->now();
        right_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
        right_message.position = {act_[6], act_[7], act_[8], act_[9], act_[10], act_[11]};
        right_message.velocity = {0, 0, 0, 0, 0, 0};
        right_message.effort = {0, 0, 0, 0, 0, 0};

        right_publisher_->publish(right_message);
        // RCLCPP_INFO(this->get_logger(), "Right Published JointState");
    }

    void quaternion_to_euler(){
        float w, x, y, z;
        w = imu_obs_[0];
        x = imu_obs_[1];
        y = imu_obs_[2];
        z = imu_obs_[3];

        float t0 = 2.0f * (w * x + y * z);
        float t1 = 1.0f - 2.0f * (x * x + y * y);
        obs_[44] = std::atan2(t0, t1);

        float t2 = 2.0 * (w * y - z * x);
        t2 = std::max(-1.0f, std::min(1.0f, t2));
        obs_[45] = std::asin(t2);

        float t3 = 2.0f * (w * z + x * y);
        float t4 = 1.0f - 2.0f * (y * y + z * z);
        obs_[46] = std::atan2(t3, t4);
    }

    void inference() {
        step_ += 1;
        quaternion_to_euler();
        obs_[0] = cos(2.0f * PI * step_ * 1.0f / fps_ / cycle_time_);
        obs_[1] = sin(2.0f * PI * step_ * 1.0f / fps_ / cycle_time_);
        obs_[2] = vx_ * obs_scales_lin_vel_;
        obs_[3] = vy_ * obs_scales_lin_vel_;
        obs_[4] = dyaw_ * obs_scales_ang_vel_;
        for(int i = 0; i < 6; i++) {
            obs_[5 + i] = left_obs_[i] * obs_scales_dof_pos_;
            obs_[17 + i] = left_obs_[6 + i] * obs_scales_ang_vel_;
            obs_[11 + i] = right_obs_[i] * obs_scales_dof_pos_;
            obs_[23 + i] = right_obs_[6 + i] * obs_scales_ang_vel_;
        }
        for(int i = 0; i < 12; i++){
            obs_[29 + i] = act_[i];
        }
        for(int i = 0; i < 12; i++){
            obs_[41 + i] = imu_obs_[4 + i];
        }
        std::vector<float> input = obs_;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(input.data()), input.size(),
                                            input_shape_.data(), input_shape_.size());

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1,
                                            output_names_.data(), output_names_.size());

        std::vector<float> output;
        for (auto &tensor : output_tensors) {
            float *data = tensor.GetTensorMutableData<float>();
            size_t count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
            output.insert(output.end(), data, data + count);
        }
        act_ = output;
        publish_joint_states();
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Inference>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}