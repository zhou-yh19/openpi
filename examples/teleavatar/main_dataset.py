#!/usr/bin/env python3
"""
Main entry point for running Teleavatar policy inference with LeRobot dataset playback.

This script uses dataset_interface to read observations from a LeRobot dataset
instead of from live ROS2 topics, enabling offline inference testing.

Usage:
    # Start policy server first (in another terminal):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi0_teleavatar_low_mem_finetune \
        --policy.dir=pi0_teleavatar_low_mem_finetune_new_data/pi0_lora_with_joint_positions_and_gripper_efforts_new_data/29999

    # Then run this script:
    python examples/teleavatar/main_dataset.py \
        --remote-host 127.0.0.1 \
        --dataset-path datasets \
        --episode-index 0 \
        --prompt "Pick up the toy and drop it in the basket on the left"
"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import sys
sys.path.append('/home/caslx/Robotics/openpi')
from examples.teleavatar import env_dataset as _env_dataset


@dataclasses.dataclass
class Args:
    """Command-line arguments for Teleavatar dataset inference."""

    # Remote policy server settings
    remote_host: str = "127.0.0.1"
    """IP address of the policy server"""

    remote_port: int = 8000
    """Port of the policy server"""

    # Dataset settings
    dataset_path: str = "/home/caslx/Robotics/openpi/datasets"
    """Path to the LeRobot format dataset directory"""

    episode_index: int = 0
    """Episode index to play back (default: 0)"""

    start_frame: int = 0
    """Starting frame within the episode (default: 0)"""

    # Control settings
    control_frequency: float = 20.0
    """Control loop frequency in Hz (should match dataset FPS for realistic timing)"""

    action_horizon: int = 10
    """Number of actions in each chunk returned by policy (default: 10)"""

    open_loop_horizon: int = 8
    """Number of actions to execute before querying policy again (default: 8)"""

    # Task settings
    prompt: str = "Pick up the toy and drop it in the basket on the left"
    """Language instruction for the robot"""

    # ROS2 action publishing
    enable_ros2_publishing: bool = True
    """If True, publish predicted actions to ROS2 topics for real robot control (default: False)"""

    # Episode settings
    num_episodes: int = 1
    """Number of episodes to run (for dataset mode, this is number of episode replays)"""

    max_episode_steps: int = 0
    """Maximum steps per episode (0 = unlimited, will use dataset episode length)"""


def main(args: Args) -> None:
    """Main function to run Teleavatar dataset inference."""

    logging.info("=" * 60)
    logging.info("Teleavatar OpenPI Dataset Inference")
    logging.info("=" * 60)
    logging.info(f"Policy server: ws://{args.remote_host}:{args.remote_port}")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Episode: {args.episode_index}, starting at frame {args.start_frame}")
    logging.info(f"Control frequency: {args.control_frequency} Hz")
    logging.info(f"Action horizon: {args.action_horizon} steps")
    logging.info(f"Open-loop horizon: {args.open_loop_horizon} steps")
    logging.info(f"Prompt: '{args.prompt}'")
    logging.info(f"ROS2 action publishing: {'ENABLED' if args.enable_ros2_publishing else 'DISABLED'}")
    if args.enable_ros2_publishing:
        logging.info("  ⚠ Actions will be published to ROS2 topics - robot will move!")
    logging.info("=" * 60)

    # Validate settings
    if args.open_loop_horizon > args.action_horizon:
        logging.warning(
            f"open_loop_horizon ({args.open_loop_horizon}) > action_horizon ({args.action_horizon}). "
            f"This means the policy will be queried before the previous chunk is exhausted."
        )

    # Create WebSocket client policy
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )

    # Get and log server metadata
    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Connected to policy server. Metadata: {metadata}")

    # Create Teleavatar dataset environment
    environment = _env_dataset.TeleavatarDatasetEnvironment(
        prompt=args.prompt,
        dataset_path=args.dataset_path,
        episode_index=args.episode_index,
        start_frame=args.start_frame,
        enable_ros2_publishing=args.enable_ros2_publishing,
    )

    # Log episode info
    episode_info = environment.get_episode_info()
    logging.info(f"Episode info: {episode_info}")

    # Create policy agent with action chunking
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.open_loop_horizon,  # Execute this many actions before querying
        )
    )

    # Create runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],  # Could add video recording, logging, etc.
        max_hz=args.control_frequency,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps if args.max_episode_steps > 0 else episode_info['episode_length'],
    )

    # Run!
    logging.info("\nStarting dataset inference...")
    logging.info("Press Ctrl+C to stop\n")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("\n\nStopping inference (Ctrl+C pressed)...")
    finally:
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )

    # Parse arguments and run
    args: Args = tyro.cli(Args)
    main(args)

