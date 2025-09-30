import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_teleavatar_example() -> dict:
    """Creates a random input example for the Teleavatar policy."""
    return {
        "observation/state": np.random.rand(48),
        "observation/images/left_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/right_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/head_camera": np.random.randint(256, size=(1080, 1920, 3), dtype=np.uint8),
        "actions": np.random.rand(48),
        "prompt": "pick a cube and place it on another cube",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format following LeRobot conventions."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TeleavatarInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For the teleavatar robot, this extracts:
    - State: 48 dimensions  (positions and velocities of arms and grippers)
    - Images: 3 camera feeds (left_color, right_color, head_color)
    - Actions: 16 dimensions (joint positions for joints 1-7 of both arms, and gripper efforts for both grippers)
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # For teleavatar, we have 3 color cameras with different resolutions.
        left_color = _parse_image(data["observation/images/left_color"])
        right_color = _parse_image(data["observation/images/right_color"])
        head_color = _parse_image(data["observation/images/head_camera"])

        # Resize head camera to match stereo camera resolution for consistency
        if head_color.shape[:2] != (480, 848):
            import cv2
            head_color = cv2.resize(head_color, (848, 480))

        state_16d = np.concatenate([
            data["observation/state"][0:7],   # Left arm joint positions (7 values)
            data["observation/state"][39:40],  # Left gripper effort (1 value)
            data["observation/state"][8:15],  # Right arm joint positions (7 values)
            data["observation/state"][47:48],  # Right gripper effort (1 value)
        ], axis=0)  # Concatenate along the feature dimension

        # Create inputs dict. Do not change the keys in the dict below.
        # Pi0 models support three image inputs: one third-person view and two wrist views.
        # Map teleavatar cameras to the expected model inputs.
        inputs = {
            "state": state_16d,
            "image": {
                "base_0_rgb": left_color,       # Left stereo camera as base view
                "left_wrist_0_rgb": right_color,  # Right stereo camera as left wrist
                "right_wrist_0_rgb": head_color,  # Head camera as right wrist
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.True_,
            },
        }

        # Extract the specific 16 actions for training from the full 62-dim actions.
        # Actions are only available during training.
        if "action" in data:
            # Extract the 16 actions we want to train on:
            # - Left arm joint positions (joints 1-7): indices 0-6
            # - Right arm joint positions (joints 1-7): indices 8-14
            # - Left gripper effort: index 39
            # - Right gripper effort: index 47
            # Extract the 16 actions we want from the full action array.
            # data["action"] has shape [action_horizon, 48]
            action_data = data["action"]

            # Extract specific indices for each timestep
            selected_actions = np.concatenate([
                action_data[:, 0:7],   # Left arm joint positions (7 values)
                action_data[:, 39:40],  # Left gripper effort (1 value)
                action_data[:, 8:15],  # Right arm joint positions (7 values)
                action_data[:, 47:48],  # Right gripper effort (1 value)
            ], axis=1)  # Concatenate along action dimension, not time dimension

            inputs["actions"] = selected_actions

        # Pass the prompt (aka language instruction) to the model.
        # For teleavatar, we use a default prompt since the dataset doesn't have task descriptions.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = "pick a toy and put it in the basket using left gripper"  # Default task for teleavatar

        return inputs


@dataclasses.dataclass(frozen=True)
class TeleavatarOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is
    used for inference only.

    For teleavatar, we return 16 actions:
    - Joint positions for joints 1-7 for both arms (14 values)
    - Joint efforts for left and right grippers (2 values)

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 actions for teleavatar.
        # Since the model may output more dimensions due to padding, we extract just what we need.
        # For your own dataset, replace `16` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :16])}