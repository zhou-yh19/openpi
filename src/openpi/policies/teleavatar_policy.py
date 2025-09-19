import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_teleavatar_example() -> dict:
    """Creates a random input example for the Teleavatar policy."""
    return {
        "observation/state": np.random.rand(62),  # Full 62-dim from dataset
        "observation/images/left_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/right_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/head_camera": np.random.randint(256, size=(1080, 1920, 3), dtype=np.uint8),
        "actions": np.random.rand(62),  # Full 62-dim from dataset
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
    - State: 48 dimensions from the full 62-dim state (positions + velocities + efforts, no EE poses)
    - Images: 3 camera feeds (left_color, right_color, head_color)
    - Actions: 16 dimensions from the full 62-dim actions (joint velocities + gripper efforts)

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
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

        # Extract only the first 48 dimensions from the full 62-dim state
        # (positions + velocities + efforts, excluding end-effector poses)
        state_48d = data["observation/state"][:48]

        # Create inputs dict. Do not change the keys in the dict below.
        # Pi0 models support three image inputs: one third-person view and two wrist views.
        # Map teleavatar cameras to the expected model inputs.
        inputs = {
            "state": state_48d,
            "image": {
                "base_0_rgb": left_color,       # Left stereo camera as base view
                "left_wrist_0_rgb": right_color,  # Right stereo camera as left wrist
                "right_wrist_0_rgb": head_color,  # Head camera as right wrist
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.True_,
            },
        }

        # Extract the specific 16 actions for training from the full 62-dim actions.
        # Actions are only available during training.
        if "action" in data:
            # Extract the 16 actions we want to train on:
            # - Left arm joint velocities (joints 1-7): indices 16-22
            # - Right arm joint velocities (joints 1-7): indices 24-30
            # - Left gripper effort: index 39
            # - Right gripper effort: index 47
            # Extract the 16 actions we want from the 62-dim action space
            # data["action"] has shape [action_horizon, 62]
            action_data = data["action"]  # Shape: [action_horizon, 62]

            # Extract specific indices for each timestep
            selected_actions = np.concatenate([
                action_data[:, 16:23],  # Left arm joint velocities (7 values)
                action_data[:, 24:31],  # Right arm joint velocities (7 values)
                action_data[:, 39:40],  # Left gripper effort (1 value)
                action_data[:, 47:48],  # Right gripper effort (1 value)
            ], axis=1)  # Concatenate along action dimension, not time dimension

            inputs["actions"] = selected_actions

        # Pass the prompt (aka language instruction) to the model.
        # For teleavatar, we use a default prompt since the dataset doesn't have task descriptions.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = "pick a cube and place it on another cube"  # Default task for teleavatar

        return inputs


@dataclasses.dataclass(frozen=True)
class TeleavatarOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is
    used for inference only.

    For teleavatar, we return 16 actions:
    - Joint velocities for joints 1-7 for both arms (14 values)
    - Joint efforts for left and right grippers (2 values)

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 actions for teleavatar.
        # Since the model may output more dimensions due to padding, we extract just what we need.
        # For your own dataset, replace `16` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :16])}