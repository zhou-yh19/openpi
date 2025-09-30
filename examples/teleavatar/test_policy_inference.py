#!/usr/bin/env python3
"""
Simple test to verify policy inference without ROS.
Tests the data format expected by the policy server.
"""

import numpy as np
import asyncio
from websockets.sync.client import connect
from openpi_client import msgpack_numpy


def test_policy_server(server_url='ws://localhost:8000'):
    """Test connecting to policy server and getting predictions."""

    print(f"Connecting to {server_url}...")
    ws = connect(server_url, open_timeout=10)

    # Receive metadata
    metadata = msgpack_numpy.unpackb(ws.recv())
    print(f"✓ Connected! Server metadata: {metadata}")

    # Create test observation with 48-dim state
    # Use flat keys with slashes (not nested dicts)
    obs = {
        'observation/images/left_color': np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8),
        'observation/images/right_color': np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8),
        'observation/images/head_camera': np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
        'observation/state': np.random.rand(48).astype(np.float32),
    }

    print(f"\nSending observation:")
    print(f"  - Images: left_color {obs['observation/images/left_color'].shape}")
    print(f"  -         right_color {obs['observation/images/right_color'].shape}")
    print(f"  -         head_camera {obs['observation/images/head_camera'].shape}")
    print(f"  - State: {obs['observation/state'].shape}")

    # Send observation
    packer = msgpack_numpy.Packer()
    ws.send(packer.pack(obs))
    print("✓ Observation sent")

    # Receive action
    print("\nWaiting for action...")
    response = ws.recv()

    # Check if response is an error message (string) or action data (bytes)
    if isinstance(response, str):
        print(f"\n✗ Server returned error:\n{response}")
        ws.close()
        return False

    action_response = msgpack_numpy.unpackb(response)

    actions = action_response['actions']
    server_timing = action_response.get('server_timing', {})

    print(f"✓ Received action!")
    print(f"  - Action shape: {actions.shape}")
    print(f"  - Server inference time: {server_timing.get('infer_ms', 0):.1f}ms")
    print(f"  - First action: {actions[0]}")  # First timestep

    ws.close()
    print("\n✓ Test passed!")

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test policy server inference')
    parser.add_argument('--server-url', type=str, default='ws://localhost:8000',
                       help='WebSocket URL of policy server')
    args = parser.parse_args()

    try:
        test_policy_server(args.server_url)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()