#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>

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
        this->declare_parameter<int>("fps", 50);
        this->declare_parameter<double>("vx", 0.0);
        this->declare_parameter<double>("vy", 0.0);
        this->declare_parameter<double>("dyaw", 0.0);
        this->declare_parameter<double>("cycle_time", 0.0);
        this->declare_parameter<double>("obs_scales_lin_vel", 0.0);
        this->declare_parameter<double>("obs_scales_ang_vel", 0.0);
        this->declare_parameter<double>("obs_scales_dof_pos", 0.0);
        this->declare_parameter<double>("obs_scales_dof_vel", 0.0);
        this->declare_parameter<std::vector<double>>("motor_lower_limit", std::vector<double>{-0.15,-1.0,-1.0,-0.4,-1.0,-0.5,-0.7,-1.2,-1.0,-1.5,-1.0,-0.5});
        this->declare_parameter<std::vector<double>>("motor_higher_limit", std::vector<double>{0.7, 1.2, 1.0, 1.5, 1.0, 0.5, 0.15, 1.0, 1.0, 0.4, 1.0, 0.5});

        this->get_parameter("model_name", model_name_);
        this->get_parameter("fps", fps_);
        this->get_parameter("vx", vx_);
        this->get_parameter("vy", vy_);
        this->get_parameter("dyaw", dyaw_);
        this->get_parameter("cycle_time", cycle_time_);
        this->get_parameter("obs_scales_lin_vel", obs_scales_lin_vel_);
        this->get_parameter("obs_scales_ang_vel", obs_scales_ang_vel_);
        this->get_parameter("obs_scales_dof_pos", obs_scales_dof_pos_);
        this->get_parameter("obs_scales_dof_vel", obs_scales_dof_vel_);
        this->get_parameter("motor_lower_limit", motor_lower_limit_);
        this->get_parameter("motor_higher_limit", motor_lower_limit_);

        model_path_ = std::string(ROOT_DIR) + "models/" + model_name_;
        RCLCPP_INFO(this->get_logger(), "model_path: %s", model_path_.c_str());
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

        left_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_left", 1, std::bind(&Inference::subs_left_callback, this, std::placeholders::_1));
        right_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_right", 1, std::bind(&Inference::subs_right_callback, this, std::placeholders::_1));
        IMU_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/IMU_data", 1, std::bind(&Inference::subs_IMU_callback, this, std::placeholders::_1));
        left_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_left", 1);
        right_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_right", 1);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(10),
                                         std::bind(&Inference::inference, this));
    }
    ~Inference() {}

   private:
    std::string model_name_, model_path_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_publisher_, right_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_subscription_, right_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr  IMU_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    int step_;
    std::vector<double> obs_, act_;
    std::vector<double> left_obs_, right_obs_, imu_obs_, left_act_, right_act_;
    int fps_;
    double vx_, vy_, dyaw_;
    double cycle_time_, obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_, obs_scales_dof_vel_;
    std::vector<double> motor_lower_limit_, motor_higher_limit_;

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

    void quaternion_to_euler(){
        double w, x, y, z;
        w = imu_obs_[0];
        x = imu_obs_[1];
        y = imu_obs_[2];
        z = imu_obs_[3];

        double t0 = 2.0 * (w * x + y * z);
        double t1 = 1.0 - 2.0 * (x * x + y * y);
        obs_[44] = std::atan2(t0, t1);

        double t2 = 2.0 * (w * y - z * x);
        t2 = std::max(-1.0, std::min(1.0, t2));
        obs_[45] = std::asin(t2);

        double t3 = 2.0 * (w * z + x * y);
        double t4 = 1.0 - 2.0 * (y * y + z * z);
        obs_[46] = std::atan2(t3, t4);
    }

    void inference() {
        step_ += 1;
        quaternion_to_euler();
        obs_[0] = cos(2 * PI * step_ * 1 / fps_ / cycle_time_);
        obs_[1] = sin(2 * PI * step_ * 1 / fps_ / cycle_time_); 
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
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Inference>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}