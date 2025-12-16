#!/usr/bin/env python3
"""
ROS2 interface wrapper for Teleavatar robot with end-effector pose support.
Handles subscribing to sensor topics (including end-effector poses) and publishing actions.
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
from geometry_msgs.msg import Pose

class TeleavatarEndEffectorROS2Interface(Node):
    """Thread-safe ROS2 interface for Teleavatar robot with end-effector representation."""

    def __init__(self, node_name: str = "teleavatar_endeffector_openpi_interface"):
        super().__init__(node_name)

        self.logger = self.get_logger()
        self.cv_bridge = CvBridge()
        self.lock = Lock()

        # Storage for latest sensor data
        self.latest_images: Dict[str, np.ndarray] = {}
        self.latest_joint_states: Dict[str, JointState] = {}
        self.latest_ee_poses: Dict[str, Pose] = {}
        self.image_timestamps: Dict[str, float] = {}
        self.joint_timestamps: Dict[str, float] = {}
        self.ee_pose_timestamps: Dict[str, float] = {}

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        self.logger.info("TeleavatarEndEffectorROS2Interface initialized (waiting for sensor data in background)")

    def _setup_subscribers(self):
        """Setup ROS2 subscribers for images, joint states, and end-effector poses."""
        # Image subscribers
        self.create_subscription(
            Image,
            '/left/color/image_raw',
            lambda msg: self._image_callback(msg, 'left_color'),
            10
        )
        self.create_subscription(
            Image,
            '/right/color/image_raw',
            lambda msg: self._image_callback(msg, 'right_color'),
            10
        )
        self.create_subscription(
            Image,
            '/head/image_raw',
            lambda msg: self._image_callback(msg, 'head_camera'),
            10
        )

        # Joint state subscribers (for gripper efforts)
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

        # End-effector CURRENT pose subscribers
        # Subscribe to current end-effector poses (input to the model)
        # TODO: Update these topic names to match your actual ROS2 topics
        self.create_subscription(
            Pose,
            '/left_arm/current_ee_pose',
            lambda msg: self._ee_pose_callback(msg, 'left_ee'),
            10
        )
        self.create_subscription(
            Pose,
            '/right_arm/current_ee_pose',
            lambda msg: self._ee_pose_callback(msg, 'right_ee'),
            10
        )

        self.logger.info("ROS2 subscribers initialized (including current end-effector poses)")

    def _setup_publishers(self):
        """Setup ROS2 publishers for end-effector action commands."""
        # Publish TARGET end-effector poses (output from the model)
        # TODO: Update these topic names to match your actual ROS2 topics
        self.action_publishers = {
            'left_ee': self.create_publisher(Pose, '/left_target_ee_pose', 10),
            'right_ee': self.create_publisher(Pose, '/right_target_ee_pose', 10),
            'left_gripper': self.create_publisher(JointState, '/left_gripper/joint_cmd', 10),
            'right_gripper': self.create_publisher(JointState, '/right_gripper/joint_cmd', 10),
        }
        self.logger.info("ROS2 publishers initialized (target end-effector pose control)")

    def _image_callback(self, msg: Image, camera_name: str):
        """Callback for image messages."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self.lock:
                self.latest_images[camera_name] = cv_image
                self.image_timestamps[camera_name] = time.time()
        except Exception as e:
            self.logger.error(f"Failed to process image from {camera_name}: {e}")

    def _joint_state_callback(self, msg: JointState, joint_group: str):
        """Callback for joint state messages."""
        with self.lock:
            self.latest_joint_states[joint_group] = msg
            self.joint_timestamps[joint_group] = time.time()

    def _ee_pose_callback(self, msg: Pose, ee_name: str):
        """Callback for end-effector pose messages."""
        with self.lock:
            self.latest_ee_poses[ee_name] = msg
            self.ee_pose_timestamps[ee_name] = time.time()

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for initial sensor data to arrive.

        Returns:
            True if all data received, False if timeout
        """
        required_images = ['left_color', 'right_color', 'head_camera']
        required_joints = ['left_gripper', 'right_gripper']  # Only need grippers for effort
        required_ee_poses = ['left_ee', 'right_ee']

        start_time = time.time()
        self.logger.info("Waiting for initial sensor data (including end-effector poses)...")

        last_status_time = start_time
        while time.time() - start_time < timeout:
            with self.lock:
                images_ready = all(cam in self.latest_images for cam in required_images)
                joints_ready = all(joint in self.latest_joint_states for joint in required_joints)
                ee_poses_ready = all(ee in self.latest_ee_poses for ee in required_ee_poses)

                # Log progress every 2 seconds
                if time.time() - last_status_time > 2.0:
                    have_images = [cam for cam in required_images if cam in self.latest_images]
                    have_joints = [joint for joint in required_joints if joint in self.latest_joint_states]
                    have_ee_poses = [ee for ee in required_ee_poses if ee in self.latest_ee_poses]
                    self.logger.info(f"  Progress: images={have_images}, joints={have_joints}, ee_poses={have_ee_poses}")
                    last_status_time = time.time()

                if images_ready and joints_ready and ee_poses_ready:
                    self.logger.info("✓ All sensor data received!")
                    return True

            time.sleep(0.1)

        # Timeout - log what's missing
        with self.lock:
            missing_images = [cam for cam in required_images if cam not in self.latest_images]
            missing_joints = [joint for joint in required_joints if joint not in self.latest_joint_states]
            missing_ee_poses = [ee for ee in required_ee_poses if ee not in self.latest_ee_poses]

        self.logger.error(
            f"✗ Timeout waiting for sensor data after {timeout}s. "
            f"Missing: images={missing_images}, joints={missing_joints}, ee_poses={missing_ee_poses}"
        )
        return False

    def get_observation(self) -> Optional[Dict]:
        """Get current observation from all sensors.

        Returns:
            Dictionary with 'images' and 'state' keys, where state is 62-dimensional:
            [joint_positions(16), joint_velocities(16), joint_efforts(16), 
             left_ee_pose(7), right_ee_pose(7)]
        """
        with self.lock:
            required_images = ['left_color', 'right_color', 'head_camera']
            required_joints = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']
            required_ee_poses = ['left_ee', 'right_ee']

            # Check if we have all required data
            if not all(cam in self.latest_images for cam in required_images):
                return None
            if not all(joint in self.latest_joint_states for joint in required_joints):
                return None
            if not all(ee in self.latest_ee_poses for ee in required_ee_poses):
                return None

            # Build 62-dimensional state vector
            # Layout: [joint_positions(16), joint_velocities(16), joint_efforts(16), 
            #          left_ee_pose(7), right_ee_pose(7)]
            state_62d = np.zeros(62, dtype=np.float32)

            # Extract joint data
            left_arm = self.latest_joint_states['left_arm']
            right_arm = self.latest_joint_states['right_arm']
            left_gripper = self.latest_joint_states['left_gripper']
            right_gripper = self.latest_joint_states['right_gripper']

            # Joint positions (indices 0-15)
            state_62d[0:7] = self._extract_joint_field(left_arm, 'position', 7)
            state_62d[7] = self._extract_joint_field(left_gripper, 'position', 1)[0]
            state_62d[8:15] = self._extract_joint_field(right_arm, 'position', 7)
            state_62d[15] = self._extract_joint_field(right_gripper, 'position', 1)[0]

            # Joint velocities (indices 16-31)
            state_62d[16:23] = self._extract_joint_field(left_arm, 'velocity', 7)
            state_62d[23] = self._extract_joint_field(left_gripper, 'velocity', 1)[0]
            state_62d[24:31] = self._extract_joint_field(right_arm, 'velocity', 7)
            state_62d[31] = self._extract_joint_field(right_gripper, 'velocity', 1)[0]

            # Joint efforts (indices 32-47)
            state_62d[32:39] = self._extract_joint_field(left_arm, 'effort', 7)
            state_62d[39] = self._extract_joint_field(left_gripper, 'effort', 1)[0]
            state_62d[40:47] = self._extract_joint_field(right_arm, 'effort', 7)
            state_62d[47] = self._extract_joint_field(right_gripper, 'effort', 1)[0]

            # End-effector poses (indices 48-61)
            left_ee_pose = self.latest_ee_poses['left_ee']
            right_ee_pose = self.latest_ee_poses['right_ee']

            # Left EE pose: (x, y, z, qx, qy, qz, qw) - indices 48-54
            state_62d[48] = left_ee_pose.position.x
            state_62d[49] = left_ee_pose.position.y
            state_62d[50] = left_ee_pose.position.z
            state_62d[51] = left_ee_pose.orientation.x
            state_62d[52] = left_ee_pose.orientation.y
            state_62d[53] = left_ee_pose.orientation.z
            state_62d[54] = left_ee_pose.orientation.w

            # Right EE pose: (x, y, z, qx, qy, qz, qw) - indices 55-61
            state_62d[55] = right_ee_pose.position.x
            state_62d[56] = right_ee_pose.position.y
            state_62d[57] = right_ee_pose.position.z
            state_62d[58] = right_ee_pose.orientation.x
            state_62d[59] = right_ee_pose.orientation.y
            state_62d[60] = right_ee_pose.orientation.z
            state_62d[61] = right_ee_pose.orientation.w

            return {
                'images': {
                    'left_color': self.latest_images['left_color'].copy(),
                    'right_color': self.latest_images['right_color'].copy(),
                    'head_camera': self.latest_images['head_camera'].copy(),
                },
                'state': state_62d,
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
        """Publish 16-dimensional TARGET action to ROS topics.

        Args:
            actions: 16-dim array [left_ee_target_pose(7), left_gripper_effort(1),
                                   right_ee_target_pose(7), right_gripper_effort(1)]
                     These are the TARGET poses predicted by the model
        """
        if actions.shape != (16,):
            self.logger.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        timestamp = self.get_clock().now().to_msg()

        # Left end-effector TARGET pose (predicted by the model)
        left_ee_msg = Pose()
        left_ee_msg.position.x = float(actions[0])
        left_ee_msg.position.y = float(actions[1])
        left_ee_msg.position.z = float(actions[2])
        left_ee_msg.orientation.x = float(actions[3])
        left_ee_msg.orientation.y = float(actions[4])
        left_ee_msg.orientation.z = float(actions[5])
        left_ee_msg.orientation.w = float(actions[6])
        self.action_publishers['left_ee'].publish(left_ee_msg)

        # Left gripper (effort)
        left_gripper_msg = JointState()
        left_gripper_msg.header.stamp = timestamp
        left_gripper_msg.header.frame_id = 'left_gripper'
        left_gripper_msg.name = ['l_joint8']
        left_gripper_msg.position = [0.0]
        left_gripper_msg.velocity = [0.0]
        left_gripper_msg.effort = [float(actions[7])]
        self.action_publishers['left_gripper'].publish(left_gripper_msg)

        # Right end-effector TARGET pose (predicted by the model)
        right_ee_msg = Pose()
        right_ee_msg.position.x = float(actions[8])
        right_ee_msg.position.y = float(actions[9])
        right_ee_msg.position.z = float(actions[10])
        right_ee_msg.orientation.x = float(actions[11])
        right_ee_msg.orientation.y = float(actions[12])
        right_ee_msg.orientation.z = float(actions[13])
        right_ee_msg.orientation.w = float(actions[14])
        self.action_publishers['right_ee'].publish(right_ee_msg)

        # Right gripper (effort)
        right_gripper_msg = JointState()
        right_gripper_msg.header.stamp = timestamp
        right_gripper_msg.header.frame_id = 'right_gripper'
        right_gripper_msg.name = ['r_joint8']
        right_gripper_msg.position = [0.0]
        right_gripper_msg.velocity = [0.0]
        right_gripper_msg.effort = [float(actions[15])]
        self.action_publishers['right_gripper'].publish(right_gripper_msg)

