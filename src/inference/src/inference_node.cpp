#include "inference_node.hpp"

void Inference::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
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

void Inference::subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    std::unique_lock lock(infer_mutex_);
    for (int i = 0; i < 6; i++) {
        left_leg_obs_[i] = msg->position[i];
        left_leg_obs_[6 + i] = msg->velocity[i];
    }
}

void Inference::subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    std::unique_lock lock(infer_mutex_);
    for (int i = 0; i < 7; i++) {
        right_leg_obs_[i] = msg->position[i];
        right_leg_obs_[7 + i] = msg->velocity[i];
    }
}

void Inference::subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    std::unique_lock lock(infer_mutex_);
    for (int i = 0; i < 5; i++) {
        left_arm_obs_[i] = msg->position[i];
        left_arm_obs_[5 + i] = msg->velocity[i];
    }
}

void Inference::subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    std::unique_lock lock(infer_mutex_);
    for (int i = 0; i < 5; i++) {
        right_arm_obs_[i] = msg->position[i];
        right_arm_obs_[5 + i] = msg->velocity[i];
    }
}

void Inference::subs_IMU_callback(const std::shared_ptr<sensor_msgs::msg::Imu> msg) {
    std::unique_lock lock(infer_mutex_);
    imu_obs_[0] = msg->orientation.w;
    imu_obs_[1] = msg->orientation.x;
    imu_obs_[2] = msg->orientation.y;
    imu_obs_[3] = msg->orientation.z;
    imu_obs_[4] = gyro_alpha_ * msg->angular_velocity.x + (1 - gyro_alpha_) * imu_obs_[4];
    imu_obs_[5] = gyro_alpha_ * msg->angular_velocity.y + (1 - gyro_alpha_) * imu_obs_[5];
    imu_obs_[6] = gyro_alpha_ * msg->angular_velocity.z + (1 - gyro_alpha_) * imu_obs_[6];
}

void Inference::publish_joint_states() {
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

void Inference::get_gravity_b() {
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

void Inference::inference() {
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

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_raw.data(), &input_tensor,
                                            1, output_names_raw.data(), output_names_raw.size());

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
            act_[usd2urdf_[i]] = std::max(joint_lower_limit_[usd2urdf_[i]],
                                          std::min(act_[usd2urdf_[i]], joint_higher_limit_[usd2urdf_[i]]));
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

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Inference>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}