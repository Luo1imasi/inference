#include <onnxruntime_cxx_api.h>
#include <string.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <vector>

class Inference : public rclcpp::Node {
   public:
    Inference() : Node("inference_node") {
        obs_.resize(78);
        act_.resize(23);
        last_act_.resize(23);
        left_leg_obs_.resize(12);
        right_leg_obs_.resize(14);
        left_arm_obs_.resize(10);
        right_arm_obs_.resize(10);
        imu_obs_.resize(7);
        imu_obs_[0] = 1.0, imu_obs_[1] = 0.0, imu_obs_[2] = 0.0, imu_obs_[3] = 0.0;
        motor_lower_limit_.resize(23);
        motor_higher_limit_.resize(23);
        motor_default_angle_.resize(23);
        usd2urdf_.resize(23);
        last_output_.resize(23);
        step_ = 0;

        this->declare_parameter<std::string>("model_name", "1.onnx");
        this->declare_parameter<float>("act_alpha", 0.9);
        this->declare_parameter<float>("gyro_alpha", 0.9);
        this->declare_parameter<float>("angle_alpha", 0.9);
        this->declare_parameter<int>("intra_threads", -1);
        this->declare_parameter<int>("frame_stack", 15);
        this->declare_parameter<int>("decimation", 10);
        this->declare_parameter<float>("dt", 0.001);
        this->declare_parameter<float>("vx", 0.0);
        this->declare_parameter<float>("vy", 0.0);
        this->declare_parameter<float>("dyaw", 0.0);
        this->declare_parameter<float>("obs_scales_lin_vel", 1.0);
        this->declare_parameter<float>("obs_scales_ang_vel", 1.0);
        this->declare_parameter<float>("obs_scales_dof_pos", 1.0);
        this->declare_parameter<float>("obs_scales_dof_vel", 1.0);
        this->declare_parameter<float>("obs_scales_gravity_b", 1.0);
        this->declare_parameter<float>("clip_observations", 100.0);
        this->declare_parameter<float>("action_scale", 0.3);
        this->declare_parameter<float>("clip_actions", 18.0);
        this->declare_parameter<std::vector<float>>(
            "motor_lower_limit",
            std::vector<float>{-0.2,  -0.2, -2.5,  0.0,  -0.6,  -0.5,  -1.0, -2.0,  -2.5, 0.0,   -0.6, -0.5,
                               -2.62, -2.0, -0.25, -2.6, -1.00, -1.57, -2.0, -2.25, -2.6, -1.00, -1.57});
        this->declare_parameter<std::vector<float>>(
            "motor_higher_limit",
            std::vector<float>{1.0,  2.0, 0.8,  2.5, 0.6,  0.5,  0.2, 0.2,  0.8, 2.5,  0.6, 0.5,
                               2.62, 2.0, 2.25, 2.6, 1.57, 1.57, 2.0, 0.25, 2.6, 1.57, 1.57});
        this->declare_parameter<std::vector<float>>(
            "motor_default_angle",
            std::vector<float>{0.0, 0.0, -0.24, 0.48, -0.24, 0.0, 0.0, 0.0, -0.24, 0.48, -0.24, 0.0,
                               0.0, 0.2, 0.0,   0.0,  1.0,   0.0, 0.2, 0.0, 0.0,   1.0,  0.0});
        this->declare_parameter<std::vector<float>>(
            "usd2urdf", std::vector<float>{0, 6,  12, 1, 7,  13, 18, 2, 8,  14, 19, 3,
                                           9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22});

        this->get_parameter("model_name", model_name_);
        this->get_parameter("act_alpha", act_alpha_);
        this->get_parameter("gyro_alpha", gyro_alpha_);
        this->get_parameter("angle_alpha", angle_alpha_);
        this->get_parameter("intra_threads", intra_threads_);
        this->get_parameter("frame_stack", frame_stack_);
        this->get_parameter("decimation", decimation_);
        this->get_parameter("dt", dt_);
        this->get_parameter("vx", vx_);
        this->get_parameter("vy", vy_);
        this->get_parameter("dyaw", dyaw_);
        this->get_parameter("obs_scales_lin_vel", obs_scales_lin_vel_);
        this->get_parameter("obs_scales_ang_vel", obs_scales_ang_vel_);
        this->get_parameter("obs_scales_dof_pos", obs_scales_dof_pos_);
        this->get_parameter("obs_scales_dof_vel", obs_scales_dof_vel_);
        this->get_parameter("obs_scales_gravity_b", obs_scales_gravity_b_);
        this->get_parameter("clip_observations", clip_observations_);
        this->get_parameter("action_scale", action_scale_);
        this->get_parameter("clip_actions", clip_actions_);
        std::vector<double> tmp;
        this->get_parameter("motor_lower_limit", tmp);
        std::transform(tmp.begin(), tmp.end(), motor_lower_limit_.begin(),
                       [](double val) { return static_cast<float>(val); });
        this->get_parameter("motor_higher_limit", tmp);
        std::transform(tmp.begin(), tmp.end(), motor_higher_limit_.begin(),
                       [](double val) { return static_cast<float>(val); });
        this->get_parameter("motor_default_angle", tmp);
        std::transform(tmp.begin(), tmp.end(), motor_default_angle_.begin(),
                       [](double val) { return static_cast<float>(val); });
        this->get_parameter("usd2urdf", tmp);
        std::transform(tmp.begin(), tmp.end(), usd2urdf_.begin(),
                       [](double val) { return static_cast<float>(val); });

        model_path_ = std::string(ROOT_DIR) + "models/" + model_name_;
        RCLCPP_INFO(this->get_logger(), "model_path: %s", model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "act_alpha: %f", act_alpha_);
        RCLCPP_INFO(this->get_logger(), "gyro_alpha: %f", gyro_alpha_);
        RCLCPP_INFO(this->get_logger(), "angle_alpha: %f", angle_alpha_);
        RCLCPP_INFO(this->get_logger(), "intra_threads: %d", intra_threads_);
        RCLCPP_INFO(this->get_logger(), "frame_stack: %d", frame_stack_);
        RCLCPP_INFO(this->get_logger(), "decimation: %d", decimation_);
        RCLCPP_INFO(this->get_logger(), "dt: %f", dt_);
        RCLCPP_INFO(this->get_logger(), "vx: %f", vx_);
        RCLCPP_INFO(this->get_logger(), "vy: %f", vy_);
        RCLCPP_INFO(this->get_logger(), "dyaw: %f", dyaw_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_lin_vel: %f", obs_scales_lin_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_ang_vel: %f", obs_scales_ang_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_dof_pos: %f", obs_scales_dof_pos_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_dof_vel: %f", obs_scales_dof_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_gravity_b: %f", obs_scales_gravity_b_);
        RCLCPP_INFO(this->get_logger(), "action_scale: %f", action_scale_);
        RCLCPP_INFO(this->get_logger(), "clip_actions: %f", clip_actions_);
        RCLCPP_INFO(
            this->get_logger(),
            "motor_lower_limit: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
            "%f, %f, %f, %f",
            motor_lower_limit_[0], motor_lower_limit_[1], motor_lower_limit_[2], motor_lower_limit_[3],
            motor_lower_limit_[4], motor_lower_limit_[5], motor_lower_limit_[6], motor_lower_limit_[7],
            motor_lower_limit_[8], motor_lower_limit_[9], motor_lower_limit_[10], motor_lower_limit_[11],
            motor_lower_limit_[12], motor_lower_limit_[13], motor_lower_limit_[14], motor_lower_limit_[15],
            motor_lower_limit_[16], motor_lower_limit_[17], motor_lower_limit_[18], motor_lower_limit_[19],
            motor_lower_limit_[20], motor_lower_limit_[21], motor_lower_limit_[22]);
        RCLCPP_INFO(this->get_logger(),
                    "motor_higher_limit: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                    "%f, %f, %f, %f, %f, %f",
                    motor_higher_limit_[0], motor_higher_limit_[1], motor_higher_limit_[2],
                    motor_higher_limit_[3], motor_higher_limit_[4], motor_higher_limit_[5],
                    motor_higher_limit_[6], motor_higher_limit_[7], motor_higher_limit_[8],
                    motor_higher_limit_[9], motor_higher_limit_[10], motor_higher_limit_[11],
                    motor_higher_limit_[12], motor_higher_limit_[13], motor_higher_limit_[14],
                    motor_higher_limit_[15], motor_higher_limit_[16], motor_higher_limit_[17],
                    motor_higher_limit_[18], motor_higher_limit_[19], motor_higher_limit_[20],
                    motor_higher_limit_[21], motor_higher_limit_[22]);
        RCLCPP_INFO(this->get_logger(),
                    "motor_default_angle: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                    "%f, %f, %f, %f, %f, %f, %f",
                    motor_default_angle_[0], motor_default_angle_[1], motor_default_angle_[2],
                    motor_default_angle_[3], motor_default_angle_[4], motor_default_angle_[5],
                    motor_default_angle_[6], motor_default_angle_[7], motor_default_angle_[8],
                    motor_default_angle_[9], motor_default_angle_[10], motor_default_angle_[11],
                    motor_default_angle_[12], motor_default_angle_[13], motor_default_angle_[14],
                    motor_default_angle_[15], motor_default_angle_[16], motor_default_angle_[17],
                    motor_default_angle_[18], motor_default_angle_[19], motor_default_angle_[20],
                    motor_default_angle_[21], motor_default_angle_[22]);
        RCLCPP_INFO(this->get_logger(),
                    "usd2urdf: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, "
                    "%d, %d, %d, %d",
                    usd2urdf_[0], usd2urdf_[1], usd2urdf_[2], usd2urdf_[3], usd2urdf_[4], usd2urdf_[5],
                    usd2urdf_[6], usd2urdf_[7], usd2urdf_[8], usd2urdf_[9], usd2urdf_[10], usd2urdf_[11],
                    usd2urdf_[12], usd2urdf_[13], usd2urdf_[14], usd2urdf_[15], usd2urdf_[16], usd2urdf_[17],
                    usd2urdf_[18], usd2urdf_[19], usd2urdf_[20], usd2urdf_[21], usd2urdf_[22]);

        for (int i = 0; i < frame_stack_; i++) {
            hist_obs_.push_back(std::vector<float>(78, 0.0f));  // 填充 frame_stack_ 个零向量
        }
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");
        Ort::SessionOptions session_options;
        if (intra_threads_ > 0) {
            session_options.SetIntraOpNumThreads(intra_threads_);
        }
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options);
        num_inputs_ = session_->GetInputCount();
        input_names_.resize(num_inputs_);
        for (size_t i = 0; i < num_inputs_; i++) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator_);
            input_names_[i] = input_name.get();
            auto type_info = session_->GetInputTypeInfo(i);
            input_shape_ = type_info.GetTensorTypeAndShapeInfo().GetShape();
        }
        num_outputs_ = session_->GetOutputCount();
        output_names_.resize(num_outputs_);
        for (size_t i = 0; i < num_outputs_; i++) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_[i] = output_name.get();
        }

        left_leg_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_left_leg", 1,
            std::bind(&Inference::subs_left_leg_callback, this, std::placeholders::_1));
        right_leg_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_right_leg", 1,
            std::bind(&Inference::subs_right_leg_callback, this, std::placeholders::_1));
        left_arm_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_left_arm", 1,
            std::bind(&Inference::subs_left_arm_callback, this, std::placeholders::_1));
        right_arm_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_right_arm", 1,
            std::bind(&Inference::subs_right_arm_callback, this, std::placeholders::_1));
        IMU_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/IMU_data", 1, std::bind(&Inference::subs_IMU_callback, this, std::placeholders::_1));
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 1, std::bind(&Inference::subs_joy_callback, this, std::placeholders::_1));
        left_leg_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_left_leg", 1);
        right_leg_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_right_leg", 1);
        left_arm_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_left_arm", 1);
        right_arm_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_right_arm", 1);
        timer_ = this->create_wall_timer(std::chrono::milliseconds((int)(dt_ * 1000)),
                                         std::bind(&Inference::inference, this));
    }
    ~Inference() {}

   private:
    std::string model_name_, model_path_;
    int frame_stack_;
    int decimation_;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_, output_names_;
    size_t num_inputs_, num_outputs_;
    std::vector<int64_t> input_shape_;
    int intra_threads_;
    Ort::AllocatorWithDefaultOptions allocator_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_leg_publisher_, right_leg_publisher_,
        left_arm_publisher_, right_arm_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_leg_subscription_,
        right_leg_subscription_, left_arm_subscription_, right_arm_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr  IMU_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    int step_;
    std::vector<float> obs_, act_, last_act_, last_output_;
    float act_alpha_, gyro_alpha_, angle_alpha_;
    std::deque<std::vector<float>> hist_obs_;
    std::vector<float> left_leg_obs_, right_leg_obs_, left_arm_obs_, right_arm_obs_, imu_obs_;
    float dt_;
    float vx_, vy_, dyaw_;
    float obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_, obs_scales_dof_vel_,
        obs_scales_gravity_b_, clip_observations_;
    float action_scale_, clip_actions_;
    std::vector<float> motor_lower_limit_, motor_higher_limit_, motor_default_angle_;
    std::vector<int> usd2urdf_;
    std::shared_mutex infer_mutex_;
    float last_roll_, last_pitch_, last_yaw_;

    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
        std::unique_lock lock(infer_mutex_);
        vx_ = msg->axes[3] * 0.2;
        vy_ = msg->axes[2] * 0.2;
        if (msg->buttons[6] == 1) {
            dyaw_ = msg->buttons[6] * 0.4;
        } else if (msg->buttons[7] == 1) {
            dyaw_ = -msg->buttons[7] * 0.4;
        } else {
            dyaw_ = 0.0;
        }
    }

    void subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
        std::unique_lock lock(infer_mutex_);
        for (int i = 0; i < 6; i++) {
            left_leg_obs_[i] = msg->position[i];
            left_leg_obs_[6 + i] = msg->velocity[i];
        }
    }

    void subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
        std::unique_lock lock(infer_mutex_);
        for (int i = 0; i < 7; i++) {
            right_leg_obs_[i] = msg->position[i];
            right_leg_obs_[7 + i] = msg->velocity[i];
        }
    }

    void subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
        std::unique_lock lock(infer_mutex_);
        for (int i = 0; i < 5; i++) {
            left_arm_obs_[i] = msg->position[i];
            left_arm_obs_[5 + i] = msg->velocity[i];
        }
    }

    void subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
        std::unique_lock lock(infer_mutex_);
        for (int i = 0; i < 5; i++) {
            right_arm_obs_[i] = msg->position[i];
            right_arm_obs_[5 + i] = msg->velocity[i];
        }
    }

    void subs_IMU_callback(const std::shared_ptr<sensor_msgs::msg::Imu> msg) {
        std::unique_lock lock(infer_mutex_);
        imu_obs_[0] = msg->orientation.w;
        imu_obs_[1] = msg->orientation.x;
        imu_obs_[2] = msg->orientation.y;
        imu_obs_[3] = msg->orientation.z;
        imu_obs_[4] = gyro_alpha_ * msg->angular_velocity.x + (1 - gyro_alpha_) * imu_obs_[4];
        imu_obs_[5] = gyro_alpha_ * msg->angular_velocity.y + (1 - gyro_alpha_) * imu_obs_[5];
        imu_obs_[6] = gyro_alpha_ * msg->angular_velocity.z + (1 - gyro_alpha_) * imu_obs_[6];
    }

    void publish_joint_states() {
        auto left_leg_message = sensor_msgs::msg::JointState();
        left_leg_message.header.stamp = this->now();
        left_leg_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
        left_leg_message.position = {act_[0], act_[1], act_[2], act_[3], act_[4], act_[5]};
        left_leg_message.velocity = {0, 0, 0, 0, 0, 0};
        left_leg_message.effort = {0, 0, 0, 0, 0, 0};
        left_leg_publisher_->publish(left_leg_message);

        auto right_leg_message = sensor_msgs::msg::JointState();
        right_leg_message.header.stamp = this->now();
        right_leg_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
        right_leg_message.position = {act_[6], act_[7], act_[8], act_[9], act_[10], act_[11], act_[12]};
        right_leg_message.velocity = {0, 0, 0, 0, 0, 0, 0};
        right_leg_message.effort = {0, 0, 0, 0, 0, 0, 0};
        right_leg_publisher_->publish(right_leg_message);

        auto left_arm_message = sensor_msgs::msg::JointState();
        left_arm_message.header.stamp = this->now();
        left_arm_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
        left_arm_message.position = {act_[13], act_[14], act_[15], act_[16], act_[17]};
        left_arm_message.velocity = {0, 0, 0, 0, 0};
        left_arm_message.effort = {0, 0, 0, 0, 0};
        left_arm_publisher_->publish(left_arm_message);

        auto right_arm_message = sensor_msgs::msg::JointState();
        right_arm_message.header.stamp = this->now();
        right_arm_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
        right_arm_message.position = {act_[18], act_[19], act_[20], act_[21], act_[22]};
        right_arm_message.velocity = {0, 0, 0, 0, 0};
        right_arm_message.effort = {0, 0, 0, 0, 0};
        right_arm_publisher_->publish(right_arm_message);
    }

    void get_gravity_b() {
        float w, x, y, z;
        w = imu_obs_[0];
        x = imu_obs_[1];
        y = imu_obs_[2];
        z = imu_obs_[3];

        Eigen::Quaternionf q_b2w(w, x, y, z);
        Eigen::Vector3f gravity_w(0.0f, 0.0f, -1.0f);
        Eigen::Quaternionf q_w2b = q_b2w.inverse();
        Eigen::Vector3f gravity_b = q_w2b * gravity_w;

        obs_[3] = gravity_b.x() * obs_scales_gravity_b_;
        obs_[4] = gravity_b.y() * obs_scales_gravity_b_;
        obs_[5] = gravity_b.z() * obs_scales_gravity_b_;

        // RCLCPP_INFO(this->get_logger(), "gravity_b: %f %f %f", obs_[44], obs_[45], obs_[46]);
    }

    void inference() {
        if (step_ % decimation_ == 0) {
            {
                std::shared_lock lock(infer_mutex_);
                for (int i = 0; i < 3; i++) {
                    obs_[i] = imu_obs_[4 + i] * obs_scales_ang_vel_;
                }
                get_gravity_b();
                obs_[6] = vx_ * obs_scales_lin_vel_;
                obs_[7] = vy_ * obs_scales_lin_vel_;
                obs_[8] = dyaw_ * obs_scales_ang_vel_;
                // RCLCPP_INFO(this->get_logger(), "obs_[4]: %f", obs_[4]);

                std::vector<float> joint_obs;
                joint_obs.resize(46);
                for (int i = 0; i < 6; i++) {
                    joint_obs[i] = left_leg_obs_[i] * obs_scales_dof_pos_;
                    joint_obs[23 + i] = left_leg_obs_[6 + i] * obs_scales_dof_vel_;
                }
                for (int i = 0; i < 7; i++) {
                    joint_obs[6 + i] = right_leg_obs_[i] * obs_scales_dof_pos_;
                    joint_obs[29 + i] = right_leg_obs_[7 + i] * obs_scales_dof_vel_;
                }
                for (int i = 0; i < 5; i++) {
                    joint_obs[13 + i] = left_arm_obs_[i] * obs_scales_dof_pos_;
                    joint_obs[36 + i] = left_arm_obs_[5 + i] * obs_scales_dof_vel_;
                }
                for (int i = 0; i < 5; i++) {
                    joint_obs[18 + i] = right_arm_obs_[i] * obs_scales_dof_pos_;
                    joint_obs[41 + i] = right_arm_obs_[5 + i] * obs_scales_dof_vel_;
                }
                for (int i = 0; i < 23; i++) {
                    obs_[9 + i] = joint_obs[usd2urdf_[i]];
                    obs_[32 + i] = joint_obs[23 + usd2urdf_[i]];
                }

                for (int i = 0; i < 23; i++) {
                    obs_[55 + i] = last_output_[i];
                }
                std::transform(obs_.begin(), obs_.end(), obs_.begin(), [this](float val) {
                    return std::clamp(val, -clip_observations_, clip_observations_);
                });
                hist_obs_.pop_front();
                hist_obs_.push_back(obs_);
            }
            std::vector<float> input(78 * frame_stack_);
            for (int i = 0; i < frame_stack_; i++) {
                std::copy(hist_obs_[i].begin(), hist_obs_[i].end(), input.begin() + i * 78);
            }
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            std::vector<const char *> input_names_raw(num_inputs_);
            for (size_t i = 0; i < num_inputs_; i++) {
                input_names_raw[i] = input_names_[i].c_str();
            }
            std::vector<const char *> output_names_raw(num_outputs_);
            for (size_t i = 0; i < num_outputs_; i++) {
                output_names_raw[i] = output_names_[i].c_str();
            }
            Ort::Value input_tensor =
                Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(input.data()), input.size(),
                                                input_shape_.data(), input_shape_.size());

            auto output_tensors =
                session_->Run(Ort::RunOptions{nullptr}, input_names_raw.data(), &input_tensor, 1,
                              output_names_raw.data(), output_names_raw.size());

            std::vector<float> output;
            for (auto &tensor : output_tensors) {
                float *data = tensor.GetTensorMutableData<float>();
                size_t count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
                output.insert(output.end(), data, data + count);
            }
            act_.resize(output.size());
            for (int i = 0; i < output.size(); i++) {
                output[i] = std::clamp(output[i], -clip_actions_, clip_actions_);
                act_[usd2urdf_[i]] = output[i];
                act_[usd2urdf_[i]] = act_[usd2urdf_[i]] * action_scale_;
                act_[usd2urdf_[i]] =
                    std::max(motor_lower_limit_[usd2urdf_[i]],
                             std::min(act_[usd2urdf_[i]], motor_higher_limit_[usd2urdf_[i]]));
            }
            last_output_ = output;
        }
        for (size_t i = 0; i < act_.size(); i++) {
            act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
        }
        publish_joint_states();
        last_act_ = act_;
        step_ += 1;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Inference>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}