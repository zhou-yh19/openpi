#!/usr/bin/env python3
"""
Arm velocity controller node for Teleavatar.

Subscribes to:
  - /left_arm/model_joint_cmd (position commands from policy)
  - /left_arm/joint_states (actual joint states)
  - /right_arm/model_joint_cmd
  - /right_arm/joint_states

Publishes to:
  - /left_arm/joint_cmd (velocity commands at 100Hz)
  - /right_arm/joint_cmd (velocity commands at 100Hz)

Control law: v = kp * (des_q - state_q) + feedforward
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ArmVelocityController(Node):
    """PD controller with feedforward for arm velocity control."""

    def __init__(self):
        super().__init__('arm_velocity_controller')

        # Control parameters
        self.arm_ctrl_dt = 1.0 / 100.0  # 100 Hz control loop
        self.num_joints = 7

        # PD gains (feedback)
        self.kp_err_left = np.array([7.0, 7.0, 10.0, 10.0, 10.0, 8.0, 8.0])
        self.kp_err_right = np.array([7.0, 7.0, 10.0, 10.0, 10.0, 8.0, 8.0])

        # State storage - Left arm
        self.left_des_q = None
        self.left_state_q = None

        # State storage - Right arm
        self.right_des_q = None
        self.right_state_q = None

        # Subscribers for model commands (from policy)
        self.create_subscription(
            JointState,
            '/left_arm/model_joint_cmd',
            self.left_model_cmd_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/right_arm/model_joint_cmd',
            self.right_model_cmd_callback,
            10
        )

        # Subscribers for joint states (from robot)
        self.create_subscription(
            JointState,
            '/left_arm/joint_states',
            self.left_state_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/right_arm/joint_states',
            self.right_state_callback,
            10
        )

        # Publishers for velocity commands
        self.left_cmd_pub = self.create_publisher(
            JointState,
            '/left_arm/joint_cmd',
            10
        )
        self.right_cmd_pub = self.create_publisher(
            JointState,
            '/right_arm/joint_cmd',
            10
        )

        # Control loop timer (100 Hz)
        self.create_timer(self.arm_ctrl_dt, self.control_loop)

        self.get_logger().info('Arm velocity controller initialized (100Hz)')

    def left_model_cmd_callback(self, msg: JointState):
        """Receive desired position from policy for left arm."""
        if len(msg.position) >= self.num_joints:
            self.left_des_q = np.array(msg.position[:self.num_joints])

    def right_model_cmd_callback(self, msg: JointState):
        """Receive desired position from policy for right arm."""
        if len(msg.position) >= self.num_joints:
            self.right_des_q = np.array(msg.position[:self.num_joints])

    def left_state_callback(self, msg: JointState):
        """Receive actual joint states for left arm."""
        if len(msg.position) >= self.num_joints:
            self.left_state_q = np.array(msg.position[:self.num_joints])

    def right_state_callback(self, msg: JointState):
        """Receive actual joint states for right arm."""
        if len(msg.position) >= self.num_joints:
            self.right_state_q = np.array(msg.position[:self.num_joints])

    def get_target_v(self, des_q, state_q, kp_err):
        """
        Compute target velocity using simple PD control.

        Args:
            des_q: Desired position
            state_q: Current position
            kp_err: Proportional gains

        Returns:
            target_velocity
        """
        vel_fb = kp_err * (des_q - state_q)
        return vel_fb

    def control_loop(self):
        """Main control loop running at 100 Hz."""
        timestamp = self.get_clock().now().to_msg()

        # Control left arm
        if self.left_des_q is not None and self.left_state_q is not None:
            left_vel = self.get_target_v(
                self.left_des_q,
                self.left_state_q,
                self.kp_err_left
            )

            # Publish left arm command (position + velocity)
            left_msg = JointState()
            left_msg.header.stamp = timestamp
            left_msg.position = self.left_des_q.tolist()
            left_msg.velocity = left_vel.tolist()
            self.left_cmd_pub.publish(left_msg)

        # Control right arm
        if self.right_des_q is not None and self.right_state_q is not None:
            right_vel = self.get_target_v(
                self.right_des_q,
                self.right_state_q,
                self.kp_err_right
            )

            # Publish right arm command (position + velocity)
            right_msg = JointState()
            right_msg.header.stamp = timestamp
            right_msg.position = self.right_des_q.tolist()
            right_msg.velocity = right_vel.tolist()
            self.right_cmd_pub.publish(right_msg)


def main(args=None):
    rclpy.init(args=args)
    controller = ArmVelocityController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
