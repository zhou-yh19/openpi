#!/usr/bin/env python3
"""
ROS2 to WebSocket bridge for deploying OpenPI policy on Teleavatar robot.

This script:
1. Subscribes to ROS2 topics for images and joint states
2. Sends observations to the policy server via WebSocket
3. Publishes predicted actions back to ROS2 topics

Usage:
    python deploy_policy_bridge.py --server-url ws://localhost:8000
"""

import argparse
import logging
import time
from collections import deque
from threading import Lock
import numpy as np
import websocket

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

# OpenPI client
from openpi_client import msgpack_numpy


class TeleavatarPolicyBridge(Node):
    """Bridge between ROS2 and OpenPI policy server."""

    def __init__(self, server_url: str, control_frequency: float = 30.0):
        super().__init__('teleavatar_policy_bridge')

        self.server_url = server_url
        self.control_frequency = control_frequency
        self.logger = self.get_logger()

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # Thread-safe storage for latest observations
        self.lock = Lock()
        self.latest_images = {}
        self.latest_joint_states = {}

        # WebSocket connection
        self.ws = None
        self.packer = msgpack_numpy.Packer()
        self.connected = False

        # Statistics
        self.stats = {
            'inference_count': 0,
            'total_latency_ms': 0.0,
            'last_inference_time': None
        }

        # Setup ROS2 subscribers
        self._setup_subscribers()

        # Setup ROS2 publishers
        self._setup_publishers()

        # Connect to policy server
        self._connect_to_server()

        # Start control loop timer
        self.timer = self.create_timer(1.0 / control_frequency, self.control_loop)

        self.logger.info(f"Policy bridge initialized (control frequency: {control_frequency} Hz)")

    def _setup_subscribers(self):
        """Setup ROS2 subscribers for images and joint states."""
        # Image subscribers (uncompressed sensor_msgs/Image)
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
            '/xr_video_topic/image_raw',
            lambda msg: self._image_callback(msg, 'head_camera'),
            10
        )

        # Joint state subscribers
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
            'left_arm': self.create_publisher(JointState, '/left_arm/joint_cmd', 10),
            'right_arm': self.create_publisher(JointState, '/right_arm/joint_cmd', 10),
            'left_gripper': self.create_publisher(JointState, '/left_gripper/joint_cmd', 10),
            'right_gripper': self.create_publisher(JointState, '/right_gripper/joint_cmd', 10),
        }

        self.logger.info("ROS2 publishers initialized")

    def _image_callback(self, msg: Image, camera_name: str):
        """Callback for image messages."""
        try:
            # Convert ROS Image to numpy array (OpenCV format)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            with self.lock:
                self.latest_images[camera_name] = cv_image

        except Exception as e:
            self.logger.error(f"Failed to process image from {camera_name}: {e}")

    def _joint_state_callback(self, msg: JointState, joint_group: str):
        """Callback for joint state messages."""
        with self.lock:
            self.latest_joint_states[joint_group] = msg

    def _connect_to_server(self):
        """Connect to the policy WebSocket server."""
        try:
            self.logger.info(f"Connecting to policy server at {self.server_url}...")
            self.ws = websocket.create_connection(self.server_url, timeout=10)

            # Receive metadata from server
            metadata = msgpack_numpy.unpackb(self.ws.recv())
            self.logger.info(f"Connected! Server metadata: {metadata}")

            self.connected = True

        except Exception as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            self.connected = False
            raise

    def _build_observation(self) -> dict | None:
        """Build observation dict from latest ROS data."""
        with self.lock:
            # Check if we have all required data
            required_images = ['left_color', 'right_color', 'head_camera']
            required_joints = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']

            if not all(cam in self.latest_images for cam in required_images):
                missing = [cam for cam in required_images if cam not in self.latest_images]
                self.logger.warning(f"Missing images: {missing}")
                return None

            if not all(joint in self.latest_joint_states for joint in required_joints):
                missing = [joint for joint in required_joints if joint not in self.latest_joint_states]
                self.logger.warning(f"Missing joint states: {missing}")
                return None

            # Build 48-dimensional state vector that matches the dataset format
            # Layout: positions[16] + velocities[16] + efforts[16]
            state_48d = np.zeros(48, dtype=np.float32)

            # Extract left arm (7 joints)
            left_arm = self.latest_joint_states['left_arm']
            state_48d[0:7] = self._extract_joint_data(left_arm, 7, 'position')
            state_48d[16:23] = self._extract_joint_data(left_arm, 7, 'velocity')
            state_48d[32:39] = self._extract_joint_data(left_arm, 7, 'effort')

            # Extract left gripper (1 joint)
            left_gripper = self.latest_joint_states['left_gripper']
            state_48d[7] = self._extract_joint_data(left_gripper, 1, 'position')[0]
            state_48d[23] = self._extract_joint_data(left_gripper, 1, 'velocity')[0]
            state_48d[39] = self._extract_joint_data(left_gripper, 1, 'effort')[0]

            # Extract right arm (7 joints)
            right_arm = self.latest_joint_states['right_arm']
            state_48d[8:15] = self._extract_joint_data(right_arm, 7, 'position')
            state_48d[24:31] = self._extract_joint_data(right_arm, 7, 'velocity')
            state_48d[40:47] = self._extract_joint_data(right_arm, 7, 'effort')

            # Extract right gripper (1 joint)
            right_gripper = self.latest_joint_states['right_gripper']
            state_48d[15] = self._extract_joint_data(right_gripper, 1, 'position')[0]
            state_48d[31] = self._extract_joint_data(right_gripper, 1, 'velocity')[0]
            state_48d[47] = self._extract_joint_data(right_gripper, 1, 'effort')[0]

            # Build observation dict with flat keys (using slashes, not nested dicts)
            obs = {
                'observation/images/left_color': self.latest_images['left_color'].copy(),
                'observation/images/right_color': self.latest_images['right_color'].copy(),
                'observation/images/head_camera': self.latest_images['head_camera'].copy(),
                'observation/state': state_48d,
            }

            return obs

    def _extract_joint_data(self, msg: JointState, num_joints: int, field: str) -> np.ndarray:
        """Extract joint data from JointState message."""
        data = getattr(msg, field, [])

        # Ensure correct length
        if len(data) >= num_joints:
            return np.array(data[:num_joints], dtype=np.float32)
        else:
            # Pad with zeros if not enough data
            result = np.zeros(num_joints, dtype=np.float32)
            result[:len(data)] = data
            return result

    def control_loop(self):
        """Main control loop: get observation, query policy, publish actions."""
        if not self.connected:
            self.logger.warning("Not connected to policy server")
            return

        try:
            # Build observation
            start_time = time.time()
            obs = self._build_observation()

            if obs is None:
                return

            # Send observation to policy server
            self.ws.send(self.packer.pack(obs))

            # Receive action from policy server
            action_response = msgpack_numpy.unpackb(self.ws.recv())

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Extract actions (shape: [action_horizon, 16])
            actions = action_response['actions']
            server_timing = action_response.get('server_timing', {})

            # Use first action in the sequence (action_horizon dimension)
            current_action = actions[0]  # Shape: (16,)

            # Publish actions to ROS
            self._publish_actions(current_action)

            # Update statistics
            self.stats['inference_count'] += 1
            self.stats['total_latency_ms'] += latency_ms
            self.stats['last_inference_time'] = time.time()

            # Log statistics every 30 inferences
            if self.stats['inference_count'] % 30 == 0:
                avg_latency = self.stats['total_latency_ms'] / self.stats['inference_count']
                infer_ms = server_timing.get('infer_ms', 0)
                self.logger.info(
                    f"Stats - Count: {self.stats['inference_count']}, "
                    f"Avg latency: {avg_latency:.1f}ms, "
                    f"Server infer: {infer_ms:.1f}ms"
                )

        except Exception as e:
            self.logger.error(f"Error in control loop: {e}")
            self.connected = False
            # Try to reconnect
            try:
                self._connect_to_server()
            except:
                pass

    def _publish_actions(self, actions: np.ndarray):
        """Publish actions to ROS topics.

        Actions format (16-dim):
        - [0:7]: Left arm joint positions
        - [7:8]: Left gripper position
        - [8:15]: Right arm joint positions
        - [15:16]: Right gripper position
        """
        timestamp = self.get_clock().now().to_msg()

        # Left arm
        left_arm_msg = JointState()
        left_arm_msg.header.stamp = timestamp
        left_arm_msg.position = actions[0:7].tolist()
        self.action_publishers['left_arm'].publish(left_arm_msg)

        # Left gripper
        left_gripper_msg = JointState()
        left_gripper_msg.header.stamp = timestamp
        left_gripper_msg.position = [float(actions[7])]
        self.action_publishers['left_gripper'].publish(left_gripper_msg)

        # Right arm
        right_arm_msg = JointState()
        right_arm_msg.header.stamp = timestamp
        right_arm_msg.position = actions[8:15].tolist()
        self.action_publishers['right_arm'].publish(right_arm_msg)

        # Right gripper
        right_gripper_msg = JointState()
        right_gripper_msg.header.stamp = timestamp
        right_gripper_msg.position = [float(actions[15])]
        self.action_publishers['right_gripper'].publish(right_gripper_msg)

    def cleanup(self):
        """Cleanup resources."""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        self.logger.info("Bridge shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="ROS2 to OpenPI policy server bridge")
    parser.add_argument(
        '--server-url',
        type=str,
        default='ws://localhost:8000',
        help='WebSocket URL of the policy server'
    )
    parser.add_argument(
        '--control-frequency',
        type=float,
        default=30.0,
        help='Control loop frequency in Hz'
    )

    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    try:
        # Create bridge node
        bridge = TeleavatarPolicyBridge(
            server_url=args.server_url,
            control_frequency=args.control_frequency
        )

        # Spin
        rclpy.spin(bridge)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'bridge' in locals():
            bridge.cleanup()
            bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()