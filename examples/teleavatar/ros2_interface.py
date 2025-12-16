#!/usr/bin/env python3
"""
ROS2 interface wrapper for Teleavatar robot.
Handles subscribing to sensor topics and publishing actions.
"""

import logging
import time
from threading import Lock
from typing import Dict, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState


class TeleavatarROS2Interface(Node):
    """Thread-safe ROS2 interface for Teleavatar robot sensors and actuators."""

    def __init__(self, node_name: str = "teleavatar_openpi_interface"):
        super().__init__(node_name)

        self.logger = self.get_logger()
        self.cv_bridge = CvBridge()
        self.lock = Lock()

        # Storage for latest sensor data
        self.latest_images: Dict[str, np.ndarray] = {}
        self.latest_joint_states: Dict[str, JointState] = {}
        self.image_timestamps: Dict[str, float] = {}
        self.joint_timestamps: Dict[str, float] = {}

        self.left_joint_names = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']
        self.right_joint_names = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
        self.left_gripper_names = ['l_joint8']
        self.right_gripper_names = ['r_joint8']

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        self.logger.info("TeleavatarROS2Interface initialized (waiting for sensor data in background)")

    def _setup_subscribers(self):
        """Setup ROS2 subscribers for images and joint states."""
        # Image subscribers - explicit subscriptions to avoid lambda closure issues
        self.create_subscription(
            Image,
            '/left/image_raw',
            lambda msg: self._image_callback(msg, 'left_color'),
            10
        )
        self.create_subscription(
            Image,
            '/right/image_raw',
            lambda msg: self._image_callback(msg, 'right_color'),
            10
        )
        self.create_subscription(
            Image,
            '/head/image_raw',   # use chest_camera image
            lambda msg: self._image_callback(msg, 'head_camera'),
            10
        )

        # Joint state subscribers - explicit subscriptions
        self.create_subscription(
            JointState,
            '/left_arm/joint_states',
            lambda msg: self._joint_state_callback(msg, 'left_arm'),
            10
        )
        self.create_subscription(
            JointState,
            '/right_arm/joint_states',
            lambda msg: self._joint_state_callback(msg, 'right_arm'),
            10
        )
        self.create_subscription(
            JointState,
            '/left_gripper/joint_states',
            lambda msg: self._joint_state_callback(msg, 'left_gripper'),
            10
        )
        self.create_subscription(
            JointState,
            '/right_gripper/joint_states',
            lambda msg: self._joint_state_callback(msg, 'right_gripper'),
            10
        )

        self.logger.info("ROS2 subscribers initialized")

    def _setup_publishers(self):
        """Setup ROS2 publishers for action commands."""
        self.action_publishers = {
            'left_arm': self.create_publisher(JointState, '/left_arm/model_joint_cmd', 10),
            'right_arm': self.create_publisher(JointState, '/right_arm/model_joint_cmd', 10),
            'left_gripper': self.create_publisher(JointState, '/left_gripper/joint_cmd', 10),
            'right_gripper': self.create_publisher(JointState, '/right_gripper/joint_cmd', 10),
        }
        self.logger.info("ROS2 publishers initialized")

    def _image_callback(self, msg: Image, camera_name: str):
        """Callback for image messages."""
        try:
            # Convert ROS Image to numpy array (RGB format)
            # self.logger.info(f"Received image from {camera_name} at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            with self.lock:
                self.latest_images[camera_name] = cv_image
                self.image_timestamps[camera_name] = time.time()
        except Exception as e:
            self.logger.error(f"Failed to process image from {camera_name}: {e}")

    def _joint_state_callback(self, msg: JointState, joint_group: str):
        """Callback for joint state messages."""
        # self.logger.info(f"Received joint state from {joint_group} at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        with self.lock:
            self.latest_joint_states[joint_group] = msg
            self.joint_timestamps[joint_group] = time.time()

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for initial sensor data to arrive.

        NOTE: This should be called AFTER the ROS2 node starts spinning,
        otherwise callbacks will never be triggered!

        Returns:
            True if all data received, False if timeout
        """
        required_images = ['left_color', 'right_color', 'head_camera']
        required_joints = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']

        start_time = time.time()
        self.logger.info("Waiting for initial sensor data...")

        last_status_time = start_time
        while time.time() - start_time < timeout:
            with self.lock:
                images_ready = all(cam in self.latest_images for cam in required_images)
                joints_ready = all(joint in self.latest_joint_states for joint in required_joints)

                # Log progress every 2 seconds
                if time.time() - last_status_time > 2.0:
                    have_images = [cam for cam in required_images if cam in self.latest_images]
                    have_joints = [joint for joint in required_joints if joint in self.latest_joint_states]
                    self.logger.info(f"  Progress: images={have_images}, joints={have_joints}")
                    last_status_time = time.time()

                if images_ready and joints_ready:
                    self.logger.info("✓ All sensor data received!")
                    return True

            time.sleep(0.1)

        # Timeout - log what's missing
        with self.lock:
            missing_images = [cam for cam in required_images if cam not in self.latest_images]
            missing_joints = [joint for joint in required_joints if joint not in self.latest_joint_states]

        self.logger.error(
            f"✗ Timeout waiting for sensor data after {timeout}s. "
            f"Missing: images={missing_images}, joints={missing_joints}"
        )
        return False

    def get_observation(self) -> Optional[Dict]:
        """Get current observation from all sensors.

        Returns:
            Dictionary with 'images' and 'state' keys, or None if data is incomplete.
        """
        with self.lock:
            required_images = ['left_color', 'right_color', 'head_camera']
            required_joints = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']

            # Check if we have all required data
            if not all(cam in self.latest_images for cam in required_images):
                return None
            if not all(joint in self.latest_joint_states for joint in required_joints):
                return None

            # Build 48-dimensional state vector
            # Layout: [positions(16), velocities(16), efforts(16)]
            state_48d = np.zeros(48, dtype=np.float32)

            # Extract joint data
            left_arm = self.latest_joint_states['left_arm']
            right_arm = self.latest_joint_states['right_arm']
            left_gripper = self.latest_joint_states['left_gripper']
            right_gripper = self.latest_joint_states['right_gripper']

            # Positions (indices 0-15)
            state_48d[0:7] = self._extract_joint_field(left_arm, 'position', 7)
            state_48d[7] = self._extract_joint_field(left_gripper, 'position', 1)[0]
            state_48d[8:15] = self._extract_joint_field(right_arm, 'position', 7)
            state_48d[15] = self._extract_joint_field(right_gripper, 'position', 1)[0]

            # Velocities (indices 16-31)
            state_48d[16:23] = self._extract_joint_field(left_arm, 'velocity', 7)
            state_48d[23] = self._extract_joint_field(left_gripper, 'velocity', 1)[0]
            state_48d[24:31] = self._extract_joint_field(right_arm, 'velocity', 7)
            state_48d[31] = self._extract_joint_field(right_gripper, 'velocity', 1)[0]

            # Efforts (indices 32-47)
            state_48d[32:39] = self._extract_joint_field(left_arm, 'effort', 7)
            state_48d[39] = self._extract_joint_field(left_gripper, 'effort', 1)[0]
            state_48d[40:47] = self._extract_joint_field(right_arm, 'effort', 7)
            state_48d[47] = self._extract_joint_field(right_gripper, 'effort', 1)[0]

            return {
                'images': {
                    'left_color': self.latest_images['left_color'].copy(),
                    'right_color': self.latest_images['right_color'].copy(),
                    'head_camera': self.latest_images['head_camera'].copy(),
                },
                'state': state_48d,
            }

    def _extract_joint_field(self, msg: JointState, field: str, num_joints: int) -> np.ndarray:
        """Extract joint data field (position/velocity/effort) from JointState message."""
        data = getattr(msg, field, [])

        if len(data) >= num_joints:
            return np.array(data[:num_joints], dtype=np.float32)
        else:
            # Pad with zeros if not enough data
            result = np.zeros(num_joints, dtype=np.float32)
            result[:len(data)] = data
            return result

    def publish_action(self, actions: np.ndarray):
        """Publish 16-dimensional action to ROS topics.

        Publishes position commands to model_joint_cmd topics for arms.
        A separate control node will subscribe to these and compute velocities
        using PD control + feedforward.

        Args:
            actions: 16-dim array [left_arm_pos(7), left_gripper_effort(1),
                                   right_arm_pos(7), right_gripper_effort(1)]
        """
        if actions.shape != (16,):
            self.logger.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        timestamp = self.get_clock().now().to_msg()

        # Left arm (position command)
        left_arm_msg = JointState()
        left_arm_msg.header.stamp = timestamp
        left_arm_msg.header.frame_id = 'left_arm'
        left_arm_msg.name = self.left_joint_names
        left_arm_msg.position = actions[0:7].tolist()
        left_arm_msg.velocity = np.zeros(7).tolist()
        left_arm_msg.effort = np.zeros(7).tolist()
        self.action_publishers['left_arm'].publish(left_arm_msg)

        # Left gripper (effort)
        left_gripper_msg = JointState()
        left_gripper_msg.header.stamp = timestamp
        left_gripper_msg.header.frame_id = 'left_gripper'
        left_gripper_msg.name = self.left_gripper_names
        left_gripper_msg.position = [0.0]
        left_gripper_msg.velocity = [0.0]
        left_gripper_msg.effort = [float(actions[7])]
        self.action_publishers['left_gripper'].publish(left_gripper_msg)

        # Right arm (position command)
        right_arm_msg = JointState()
        right_arm_msg.header.stamp = timestamp
        right_arm_msg.header.frame_id = 'right_arm'
        right_arm_msg.name = self.right_joint_names
        right_arm_msg.position = actions[8:15].tolist()
        right_arm_msg.velocity = np.zeros(7).tolist()
        right_arm_msg.effort = np.zeros(7).tolist()
        self.action_publishers['right_arm'].publish(right_arm_msg)

        # Right gripper (effort)
        right_gripper_msg = JointState()
        right_gripper_msg.header.stamp = timestamp
        right_gripper_msg.header.frame_id = 'right_gripper'
        right_gripper_msg.name = self.right_gripper_names
        right_gripper_msg.position = [0.0]
        right_gripper_msg.velocity = [0.0]
        right_gripper_msg.effort = [float(actions[15])]
        self.action_publishers['right_gripper'].publish(right_gripper_msg)
