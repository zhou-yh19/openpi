#!/usr/bin/env python3
"""
Main entry point for running Teleavatar robot with OpenPI policy using end-effector representation.

This script uses end-effector poses (position + quaternion) instead of joint angles.

Usage:
    # Start policy server first (in another terminal):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi0_teleavatar_low_mem_finetune \
        --policy.dir=pi0_teleavatar_low_mem_finetune/pi0_lora_with_joint_positions_and_gripper_efforts/29999

    # Then run this script:
    python examples/teleavatar/main_endeffector.py --remote-host 192.168.1.100 --prompt "pick up the red cube"
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
from examples.teleavatar import env_endeffector as _env


@dataclasses.dataclass
class Args:
    """Command-line arguments for Teleavatar deployment with end-effector control."""

    # Remote policy server settings
    remote_host: str = "0.0.0.0"
    """IP address of the policy server (e.g., '192.168.1.100')"""

    remote_port: int = 8000
    """Port of the policy server"""

    # Control settings
    control_frequency: float = 20.0
    """Control loop frequency in Hz (default: 20 Hz)"""

    action_horizon: int = 10
    """Number of actions in each chunk returned by policy (default: 10)"""

    open_loop_horizon: int = 8
    """Number of actions to execute before querying policy again (default: 8)"""

    # Task settings
    prompt: str = "Pick up the cube and drop it in the box on the left"
    """Language instruction for the robot"""

    # Episode settings
    num_episodes: int = 1
    """Number of episodes to run"""

    max_episode_steps: int = 600
    """Maximum steps per episode (0 = unlimited)"""


def main(args: Args) -> None:
    """Main function to run Teleavatar with end-effector policy inference."""

    logging.info("=" * 60)
    logging.info("Teleavatar OpenPI Deployment (End-Effector Control)")
    logging.info("=" * 60)
    logging.info(f"Policy server: ws://{args.remote_host}:{args.remote_port}")
    logging.info(f"Control frequency: {args.control_frequency} Hz")
    logging.info(f"Action horizon: {args.action_horizon} steps")
    logging.info(f"Open-loop horizon: {args.open_loop_horizon} steps")
    logging.info(f"Prompt: '{args.prompt}'")
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

    # Create Teleavatar environment with end-effector representation
    environment = _env.TeleavatarEndEffectorEnvironment(
        prompt=args.prompt,
    )

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
        max_episode_steps=args.max_episode_steps,
    )

    # Run!
    logging.info("\nStarting robot control loop with end-effector representation...")
    logging.info("Press Ctrl+C to stop\n")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("\n\nStopping robot (Ctrl+C pressed)...")
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

