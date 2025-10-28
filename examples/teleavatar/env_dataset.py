#!/usr/bin/env python3
"""
Environment wrapper for Teleavatar robot using LeRobot dataset interface.
Can read observations from dataset while optionally publishing actions to ROS2 for real robot control.
"""

import logging
import threading
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.teleavatar import dataset_interface

# Optional ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class TeleavatarDatasetEnvironment(_environment.Environment):
    """Environment for Teleavatar that reads observations from LeRobot dataset."""

    def __init__(
        self,
        prompt: str = "pick a toy and put it in the basket using left gripper",
        dataset_path: str = "/home/caslx/Robotics/openpi/datasets",
        episode_index: int = 0,
        start_frame: int = 0,
        enable_ros2_publishing: bool = False,
    ):
        """Initialize Teleavatar dataset environment.

        Args:
            prompt: Default language instruction for the policy
            dataset_path: Path to the LeRobot format dataset
            episode_index: Which episode to play back (default: 0)
            start_frame: Starting frame within the episode (default: 0)
            enable_ros2_publishing: If True, publish actions to ROS2 topics (default: False)

        Note: Images are NOT resized here - they are kept at original resolution
        to match training data format (480×848 for stereo, 1080×1920 for head).
        """
        self._prompt = prompt
        self._episode_complete = False
        self._ros_node: Optional[Node] = None
        self._ros_thread: Optional[threading.Thread] = None

        # Initialize ROS2 if needed
        if enable_ros2_publishing:
            if not ROS2_AVAILABLE:
                raise RuntimeError("ROS2 is not available. Cannot enable action publishing.")
            self._init_ros2()

        # Initialize dataset interface
        self._dataset_interface = dataset_interface.TeleavatarDatasetInterface(
            dataset_path=dataset_path,
            episode_index=episode_index,
            start_frame=start_frame,
            enable_ros2_publishing=enable_ros2_publishing,
            ros2_node=self._ros_node,
        )

        # Wait for data (always succeeds for dataset)
        if not self._dataset_interface.wait_for_initial_data():
            raise RuntimeError("Failed to load dataset")

        mode_str = "with ROS2 action publishing" if enable_ros2_publishing else "without ROS2 (playback only)"
        logging.info(f"TeleavatarDatasetEnvironment initialized {mode_str}")
        logging.info(f"Prompt: '{prompt}'")
        info = self._dataset_interface.get_episode_info()
        logging.info(f"Playing episode {info['episode_index']}: {info['task']}")
        logging.info(f"Episode length: {info['episode_length']} frames")

    def _init_ros2(self):
        """Initialize ROS2 in a background thread for action publishing."""
        import time

        # Event to signal when executor starts spinning
        spin_started = threading.Event()

        def ros_spin():
            rclpy.init()
            self._ros_node = rclpy.create_node('teleavatar_dataset_action_publisher')

            # Spin in background
            executor = rclpy.executors.MultiThreadedExecutor()
            executor.add_node(self._ros_node)

            # Signal that spinning is about to start
            spin_started.set()

            try:
                executor.spin()
            finally:
                executor.shutdown()
                self._ros_node.destroy_node()
                rclpy.shutdown()

        self._ros_thread = threading.Thread(target=ros_spin, daemon=True)
        self._ros_thread.start()

        # Wait for ROS2 node to be created
        timeout = 10.0
        start_time = time.time()
        while self._ros_node is None and time.time() - start_time < timeout:
            time.sleep(0.1)

        if self._ros_node is None:
            raise RuntimeError("Failed to initialize ROS2 node within timeout")

        logging.info("ROS2 node created, waiting for executor to start spinning...")

        # Wait for executor to start spinning
        if not spin_started.wait(timeout=5.0):
            raise RuntimeError("ROS2 executor failed to start spinning")

        logging.info("ROS2 executor started for action publishing")

    @override
    def reset(self) -> None:
        """Reset the environment to the beginning of the current episode."""
        self._dataset_interface.reset(start_frame=0)
        self._episode_complete = False
        info = self._dataset_interface.get_episode_info()
        logging.info(f"Environment reset to episode {info['episode_index']}, frame 0")

    def reset_to_episode(self, episode_index: int, start_frame: int = 0):
        """Reset to a specific episode and frame.

        Args:
            episode_index: Episode to reset to
            start_frame: Starting frame within the episode
        """
        self._dataset_interface.reset(episode_index=episode_index, start_frame=start_frame)
        self._episode_complete = False
        info = self._dataset_interface.get_episode_info()
        logging.info(f"Environment reset to episode {info['episode_index']}, frame {start_frame}")

    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete (reached end of dataset episode)."""
        return self._episode_complete

    @override
    def get_observation(self) -> dict:
        """Get current observation from dataset.

        Returns:
            Dictionary with keys:
                - 'observation/state': 48-dim proprioceptive state
                - 'observation/images/left_color': Left wrist camera image
                - 'observation/images/right_color': Right wrist camera image
                - 'observation/images/head_camera': Head camera image
                - 'prompt': Language instruction

        Note: Images are kept at original resolution to match training data:
            - left_color, right_color: 480×848×3 (H,W,C)
            - head_camera: 1080×1920×3 (H,W,C)
        The policy's _parse_image will handle any format conversion if needed.
        """
        # Get raw observation from dataset
        raw_obs = self._dataset_interface.get_observation()
        
        if raw_obs is None:
            # Mark episode as complete when we run out of data
            self._episode_complete = True
            info = self._dataset_interface.get_episode_info()
            logging.info(f"Reached end of episode {info['episode_index']} (frame {info['current_frame']})")
            
            # Return a dummy observation (runtime will check is_episode_complete)
            # This should not be used by the policy
            return {
                'observation/state': np.zeros(48, dtype=np.float32),
                'observation/images/left_color': np.zeros((480, 848, 3), dtype=np.uint8),
                'observation/images/right_color': np.zeros((480, 848, 3), dtype=np.uint8),
                'observation/images/head_camera': np.zeros((1080, 1920, 3), dtype=np.uint8),
                'prompt': self._prompt,
            }

        # Process images: keep original resolution AND keep (H, W, C) format
        # Policy's _parse_image will handle format conversion if needed
        # Return with the exact keys expected by teleavatar_policy.py
        return {
            'observation/state': raw_obs['state'],
            'observation/images/left_color': image_tools.convert_to_uint8(raw_obs['images']['left_color']),
            'observation/images/right_color': image_tools.convert_to_uint8(raw_obs['images']['right_color']),
            'observation/images/head_camera': image_tools.convert_to_uint8(raw_obs['images']['head_camera']),
            'prompt': self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        """Apply action to the robot.
        
        If ROS2 publishing is enabled, publishes actions to ROS2 topics.
        Otherwise, this is a no-op.

        Args:
            action: Dictionary containing 'actions' key with 16-dim action array
        """
        if 'actions' not in action:
            logging.warning(f"Action dict must contain 'actions' key, got: {action.keys()}")
            return

        actions = action['actions']
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=np.float32)

        # Ensure correct shape
        if actions.shape != (16,):
            logging.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        # Publish to ROS2 (will no-op if publishing is disabled)
        self._dataset_interface.publish_action(actions)

    def set_prompt(self, prompt: str):
        """Update the language instruction prompt.

        Args:
            prompt: New language instruction
        """
        self._prompt = prompt
        logging.info(f"Updated prompt to: '{prompt}'")

    def get_episode_info(self) -> dict:
        """Get information about the current episode.

        Returns:
            Dictionary with episode metadata
        """
        return self._dataset_interface.get_episode_info()

    def __del__(self):
        """Cleanup when environment is destroyed."""
        if self._ros_thread is not None and self._ros_thread.is_alive():
            logging.info("Shutting down ROS2 thread...")

