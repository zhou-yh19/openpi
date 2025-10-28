#!/usr/bin/env python3
"""
Dataset interface wrapper for loading observations from LeRobot format dataset.
This provides the same interface as TeleavatarROS2Interface but reads from disk instead of ROS2 topics.
Can optionally publish actions to ROS2 topics for real robot control.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import lerobot.datasets.lerobot_dataset as lerobot_dataset

# ROS2 imports (optional, only needed if publishing actions)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Dummy base class if ROS2 not available


class TeleavatarDatasetInterface:
    """Interface for reading Teleavatar observations from LeRobot format dataset."""

    def __init__(
        self,
        dataset_path: str = "/home/caslx/Robotics/openpi/datasets",
        episode_index: int = 0,
        start_frame: int = 0,
        enable_ros2_publishing: bool = False,
        ros2_node: Optional[Node] = None,
    ):
        """Initialize dataset interface.

        Args:
            dataset_path: Path to the LeRobot format dataset directory
            episode_index: Which episode to load (default: 0)
            start_frame: Starting frame index within the episode (default: 0)
            enable_ros2_publishing: If True, publish actions to ROS2 topics (default: False)
            ros2_node: Existing ROS2 node to use (if None and enable_ros2_publishing=True, will create one)
        """
        self.logger = logging.getLogger(__name__)
        # Convert to absolute path
        self.dataset_path = Path(dataset_path).resolve()

        # Load dataset metadata
        self.logger.info(f"Loading dataset from: {self.dataset_path}")
        self.metadata = lerobot_dataset.LeRobotDatasetMetadata(str(self.dataset_path))
        self.logger.info(f"Dataset info: {self.metadata.total_episodes} episodes, {self.metadata.total_frames} frames")

        # Load dataset (without delta timestamps since we're doing inference frame-by-frame)
        self.dataset = lerobot_dataset.LeRobotDataset(str(self.dataset_path))

        # Load episode metadata first
        import json
        episodes_file = self.dataset_path / "meta" / "episodes.jsonl"
        self.episodes_metadata = []
        with open(episodes_file, 'r') as f:
            for line in f:
                self.episodes_metadata.append(json.loads(line))

        # Episode and frame tracking
        self.episode_index = episode_index
        self.current_frame_in_episode = start_frame
        self.current_global_frame = 0

        # Find the global frame index for the start of this episode
        self._init_episode_boundaries()
        self.current_global_frame = self.episode_start_frames[episode_index] + start_frame

        episode_info = self.episodes_metadata[episode_index]
        self.episode_length = episode_info['length']
        self.episode_task = episode_info['tasks'][0] if episode_info['tasks'] else "No task"

        self.logger.info(f"Initialized at episode {episode_index}, frame {start_frame}/{self.episode_length}")
        self.logger.info(f"Task: {self.episode_task}")

        # ROS2 publishing setup (optional)
        self.enable_ros2_publishing = enable_ros2_publishing
        self.ros2_node = ros2_node
        self.action_publishers = {}
        
        if self.enable_ros2_publishing:
            if not ROS2_AVAILABLE:
                raise RuntimeError("ROS2 is not available. Cannot enable ROS2 publishing.")
            self._setup_ros2_publishers()
            self.logger.info("ROS2 action publishing enabled")

    def _init_episode_boundaries(self):
        """Calculate the starting global frame index for each episode."""
        self.episode_start_frames = [0]
        cumulative_frames = 0

        for ep_meta in self.episodes_metadata:
            cumulative_frames += ep_meta['length']
            self.episode_start_frames.append(cumulative_frames)

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Compatibility method with ROS2 interface. Always returns True for dataset.

        Args:
            timeout: Unused, kept for API compatibility

        Returns:
            True (data is always available from dataset)
        """
        self.logger.info("Dataset interface: initial data ready (loaded from disk)")
        return True

    def get_observation(self) -> Optional[Dict]:
        """Get current observation from dataset.

        Returns observation in the same format as TeleavatarROS2Interface:
            {
                'images': {
                    'left_color': np.ndarray (H, W, 3),
                    'right_color': np.ndarray (H, W, 3),
                    'head_camera': np.ndarray (H, W, 3),
                },
                'state': np.ndarray (48,) - [positions(16), velocities(16), efforts(16)]
            }

        Returns:
            Dictionary with observation data, or None if end of episode reached.
        """
        # Check if we've reached the end of the current episode
        if self.current_frame_in_episode >= self.episode_length:
            self.logger.info(f"Reached end of episode {self.episode_index}")
            return None

        # Check if global frame index is valid
        if self.current_global_frame >= len(self.dataset):
            self.logger.error(f"Global frame index {self.current_global_frame} exceeds dataset size {len(self.dataset)}")
            return None

        try:
            # Load frame from dataset
            frame = self.dataset[self.current_global_frame]

            # Extract images
            # Dataset uses: observation.images.left_wrist, right_wrist, head
            # We need to map to: left_color, right_color, head_camera
            images = {
                'left_color': self._get_image_array(frame, 'observation.images.left_wrist'),
                'right_color': self._get_image_array(frame, 'observation.images.right_wrist'),
                'head_camera': self._get_image_array(frame, 'observation.images.head'),
            }

            # Extract state
            # Dataset has 62-dim state: [positions(16), velocities(16), efforts(16), end_effector(14)]
            # We need 48-dim state: [positions(16), velocities(16), efforts(16)]
            state_62d = frame['observation.state']
            
            if isinstance(state_62d, np.ndarray):
                state_array = state_62d
            else:
                # Handle torch tensor
                state_array = state_62d.numpy()

            # Extract first 48 dimensions (positions + velocities + efforts)
            state_48d = state_array[:48].astype(np.float32)

            # Move to next frame
            self.current_frame_in_episode += 1
            self.current_global_frame += 1

            return {
                'images': images,
                'state': state_48d,
            }

        except Exception as e:
            self.logger.error(f"Error loading frame {self.current_global_frame}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_image_array(self, frame: Dict, key: str) -> np.ndarray:
        """Extract image array from frame data.

        Args:
            frame: Dataset frame dictionary
            key: Key for the image (e.g., 'observation.images.left_wrist')

        Returns:
            Image as numpy array (H, W, 3) in RGB format
        """
        img = frame[key]

        # Convert to numpy if it's a torch tensor
        if hasattr(img, 'numpy'):
            img_array = img.numpy()
        else:
            img_array = np.array(img)

        # Ensure uint8 format
        if img_array.dtype != np.uint8:
            # If float in [0, 1], scale to [0, 255]
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        return img_array

    def reset(self, episode_index: Optional[int] = None, start_frame: int = 0):
        """Reset to a different episode or frame.

        Args:
            episode_index: Episode to reset to (None = stay on current episode)
            start_frame: Starting frame within the episode
        """
        if episode_index is not None:
            if episode_index < 0 or episode_index >= self.metadata.total_episodes:
                self.logger.error(f"Invalid episode index {episode_index}. Must be in [0, {self.metadata.total_episodes})")
                return

            self.episode_index = episode_index
            episode_info = self.episodes_metadata[episode_index]
            self.episode_length = episode_info['length']
            self.episode_task = episode_info['tasks'][0] if episode_info['tasks'] else "No task"

        self.current_frame_in_episode = start_frame
        self.current_global_frame = self.episode_start_frames[self.episode_index] + start_frame

        self.logger.info(f"Reset to episode {self.episode_index}, frame {start_frame}/{self.episode_length}")
        self.logger.info(f"Task: {self.episode_task}")

    def get_episode_info(self) -> Dict:
        """Get information about the current episode.

        Returns:
            Dictionary with episode metadata
        """
        return {
            'episode_index': self.episode_index,
            'episode_length': self.episode_length,
            'current_frame': self.current_frame_in_episode,
            'task': self.episode_task,
            'global_frame': self.current_global_frame,
        }

    def _setup_ros2_publishers(self):
        """Setup ROS2 publishers for action commands (copied from ros2_interface.py)."""
        if self.ros2_node is None:
            raise RuntimeError("ROS2 node is required for publishing actions")
        
        # Joint names (same as in ros2_interface.py)
        self.left_joint_names = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']
        self.right_joint_names = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
        self.left_gripper_names = ['l_joint8']
        self.right_gripper_names = ['r_joint8']
        
        # Create publishers
        self.action_publishers = {
            'left_arm': self.ros2_node.create_publisher(JointState, '/left_arm/ik_joint_cmd', 10),
            'right_arm': self.ros2_node.create_publisher(JointState, '/right_arm/ik_joint_cmd', 10),
            'left_gripper': self.ros2_node.create_publisher(JointState, '/left_gripper/joint_cmd', 10),
            'right_gripper': self.ros2_node.create_publisher(JointState, '/right_gripper/joint_cmd', 10),
        }
        self.logger.info("ROS2 action publishers initialized")

    def publish_action(self, actions: np.ndarray):
        """Publish 16-dimensional action to ROS2 topics.
        
        Copied from ros2_interface.py to maintain identical behavior.
        Publishes position commands to model_joint_cmd topics for arms.
        A separate control node will subscribe to these and compute velocities
        using PD control + feedforward.

        Args:
            actions: 16-dim array [left_arm_pos(7), left_gripper_effort(1),
                                   right_arm_pos(7), right_gripper_effort(1)]
        """
        if not self.enable_ros2_publishing:
            # Silently skip if ROS2 publishing is not enabled
            return
        
        if actions.shape != (16,):
            self.logger.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        timestamp = self.ros2_node.get_clock().now().to_msg()

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
        #right_arm_msg.position[2]=-actions[9]
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


if __name__ == "__main__":
    """Test the dataset interface."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    # Create interface
    interface = TeleavatarDatasetInterface(
        dataset_path="/home/caslx/Robotics/openpi/datasets",
        episode_index=0,
        start_frame=0,
    )

    # Wait for data (always succeeds for dataset)
    if not interface.wait_for_initial_data():
        print("Failed to load initial data")
        exit(1)

    # Get a few observations
    print("\nReading first 5 frames:")
    for i in range(5):
        obs = interface.get_observation()
        if obs is None:
            print(f"Frame {i}: No more data")
            break

        print(f"\nFrame {i}:")
        print(f"  State shape: {obs['state'].shape}")
        print(f"  State (first 8 values): {obs['state'][:8]}")
        print(f"  Images:")
        for cam_name, img in obs['images'].items():
            print(f"    {cam_name}: {img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

    # Show episode info
    print("\nEpisode info:")
    info = interface.get_episode_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

