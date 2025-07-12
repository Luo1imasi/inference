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
        joint_lower_limit_.resize(23);
        joint_higher_limit_.resize(23);
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
            "joint_lower_limit",
            std::vector<float>{-0.2,  -0.2, -2.5,  0.0,  -0.6,  -0.5,  -1.0, -2.0,  -2.5, 0.0,   -0.6, -0.5,
                               -2.62, -2.0, -0.25, -2.6, -1.00, -1.57, -2.0, -2.25, -2.6, -1.00, -1.57});
        this->declare_parameter<std::vector<float>>(
            "joint_higher_limit",
            std::vector<float>{1.0,  2.0, 0.8,  2.5, 0.6,  0.5,  0.2, 0.2,  0.8, 2.5,  0.6, 0.5,
                               2.62, 2.0, 2.25, 2.6, 1.57, 1.57, 2.0, 0.25, 2.6, 1.57, 1.57});
        this->declare_parameter<std::vector<long int>>(
            "usd2urdf", std::vector<long int>{0, 6,  12, 1, 7,  13, 18, 2, 8,  14, 19, 3,
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
        this->get_parameter("joint_lower_limit", tmp);
        std::transform(tmp.begin(), tmp.end(), joint_lower_limit_.begin(),
                       [](double val) { return static_cast<float>(val); });
        this->get_parameter("joint_higher_limit", tmp);
        std::transform(tmp.begin(), tmp.end(), joint_higher_limit_.begin(),
                       [](double val) { return static_cast<float>(val); });
        this->get_parameter("usd2urdf", usd2urdf_);

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
            "joint_lower_limit: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
            "%f, %f, %f, %f",
            joint_lower_limit_[0], joint_lower_limit_[1], joint_lower_limit_[2], joint_lower_limit_[3],
            joint_lower_limit_[4], joint_lower_limit_[5], joint_lower_limit_[6], joint_lower_limit_[7],
            joint_lower_limit_[8], joint_lower_limit_[9], joint_lower_limit_[10], joint_lower_limit_[11],
            joint_lower_limit_[12], joint_lower_limit_[13], joint_lower_limit_[14], joint_lower_limit_[15],
            joint_lower_limit_[16], joint_lower_limit_[17], joint_lower_limit_[18], joint_lower_limit_[19],
            joint_lower_limit_[20], joint_lower_limit_[21], joint_lower_limit_[22]);
        RCLCPP_INFO(this->get_logger(),
                    "joint_higher_limit: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                    "%f, %f, %f, %f, %f, %f",
                    joint_higher_limit_[0], joint_higher_limit_[1], joint_higher_limit_[2],
                    joint_higher_limit_[3], joint_higher_limit_[4], joint_higher_limit_[5],
                    joint_higher_limit_[6], joint_higher_limit_[7], joint_higher_limit_[8],
                    joint_higher_limit_[9], joint_higher_limit_[10], joint_higher_limit_[11],
                    joint_higher_limit_[12], joint_higher_limit_[13], joint_higher_limit_[14],
                    joint_higher_limit_[15], joint_higher_limit_[16], joint_higher_limit_[17],
                    joint_higher_limit_[18], joint_higher_limit_[19], joint_higher_limit_[20],
                    joint_higher_limit_[21], joint_higher_limit_[22]);
        RCLCPP_INFO(this->get_logger(),
                    "usd2urdf: %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, "
                    "%ld, %ld, %ld, %ld",
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
    std::vector<float> joint_lower_limit_, joint_higher_limit_;
    std::vector<long int> usd2urdf_;
    std::shared_mutex infer_mutex_;
    float last_roll_, last_pitch_, last_yaw_;

    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_IMU_callback(const std::shared_ptr<sensor_msgs::msg::Imu> msg);
    void publish_joint_states();
    void get_gravity_b();
    void inference();
};