import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_teleavatar_endeffector_example() -> dict:
    """Creates a random input example for the Teleavatar policy with end-effector representation."""
    return {
        "observation/state": np.random.rand(62),  # 62-dim state with end-effector poses
        "observation/images/left_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/right_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/head_camera": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "actions": np.random.rand(62),  # 62-dim actions with end-effector poses
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
class TeleavatarEndEffectorInputs(transforms.DataTransformFn):
    """
    Converts inputs to the model format for Teleavatar robot using end-effector representation.

    **Input format (62-dim observation/state from LeRobot dataset):**
    Layout: [joint_positions(16), joint_velocities(16), joint_efforts(16), 
             left_ee_pose(7), right_ee_pose(7)]
    - Indices 0-15: Joint positions (7 left arm, 1 left gripper, 7 right arm, 1 right gripper)
    - Indices 16-31: Joint velocities (same layout)
    - Indices 32-47: Joint efforts (same layout)
    - Indices 48-54: Left end-effector pose (x, y, z, qx, qy, qz, qw) - CURRENT pose
    - Indices 55-61: Right end-effector pose (x, y, z, qx, qy, qz, qw) - CURRENT pose

    **Model state format (16-dim):**
    We extract current end-effector poses as input:
    [left_ee_pose(7), left_gripper_effort(1), right_ee_pose(7), right_gripper_effort(1)]
    - Indices 0-6: Left end-effector CURRENT pose (from input[48:55])
    - Index 7: Left gripper effort (from input[39])
    - Indices 8-14: Right end-effector CURRENT pose (from input[55:62])
    - Index 15: Right gripper effort (from input[47])
    
    **Model output (16-dim):**
    The model predicts TARGET end-effector poses:
    [left_ee_target_pose(7), left_gripper_effort(1), right_ee_target_pose(7), right_gripper_effort(1)]
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W) during training, but runtime sends uint8 (H,W,C)
        left_color = _parse_image(data["observation/images/left_color"])
        right_color = _parse_image(data["observation/images/right_color"])
        head_color = _parse_image(data["observation/images/head_camera"])

        # Resize head camera to match stereo camera resolution for consistency
        if head_color.shape[:2] != (480, 848):
            import cv2
            head_color = cv2.resize(head_color, (848, 480))

        # Extract 16-dim state from extended observation
        # Using end-effector representation instead of joint angles
        state_14d = np.concatenate([
            data["observation/state"][48:55],  # Left arm end-effector (x,y,z,qx,qy,qz,qw)
            data["observation/state"][55:62],  # Right arm end-effector (x,y,z,qx,qy,qz,qw)
        ], axis=0)

        # Create inputs dict. Do not change the keys in the dict below.
        # Pi0 models support three image inputs: one third-person view and two wrist views.
        # Map teleavatar cameras to the expected model inputs.
        inputs = {
            "state": state_14d,
            "image": {
                "base_0_rgb": head_color,       
                "left_wrist_0_rgb": left_color,  
                "right_wrist_0_rgb": right_color,  
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.True_,
            },
        }

        # Extract 16-dim TARGET actions from 62-dim action space during training
        # Actions are only available during training, not during inference
        if "action" in data:
            # data["action"] has shape [action_horizon, 62]
            # Layout: [joint_positions(16), joint_velocities(16), joint_efforts(16), 
            #          left_ee_target_pose(7), right_ee_target_pose(7)]
            action_data = data["action"]

            # Extract TARGET end-effector poses (indices 48-61)
            selected_actions = np.concatenate([
                action_data[:, 48:55],  # Left arm TARGET end-effector pose
                action_data[:, 39:40],  # Left gripper effort (index 39 = 32+7)
                action_data[:, 55:62],  # Right arm TARGET end-effector pose
                action_data[:, 47:48],  # Right gripper effort (index 47 = 32+15)
            ], axis=1)  # Concatenate along action dimension

            inputs["actions"] = selected_actions

        # Pass the prompt (aka language instruction) to the model.
        # For teleavatar, we use a default prompt since the dataset doesn't have task descriptions.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = "pick a toy and put it in the basket using left gripper"  # Default task for teleavatar

        return inputs


@dataclasses.dataclass(frozen=True)
class TeleavatarEndEffectorOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is
    used for inference only.

    For teleavatar with end-effector representation, we return 16 actions:
    - End-effector TARGET pose (x,y,z,qx,qy,qz,qw) for both arms (14 values)
    - Gripper efforts for left and right grippers (2 values)
    
    **Why extract only 16 dimensions?**
    - Model is configured with action_dim=32 to match pre-trained pi0_base weights
    - But we only need 16 dimensions for our robot (2 arms × 7 DOF + 2 grippers)
    - The extra dimensions (16-31) are padding and should be discarded
    - This allows us to leverage pre-trained weights while adapting to our robot
    """

    def __call__(self, data: dict) -> dict:
        # Extract only the first 16 actions from model output (action_dim=32)
        # Model output has padding dimensions that we don't need
        # Output format: [left_ee_target_pose(7), left_gripper_effort(1), 
        #                 right_ee_target_pose(7), right_gripper_effort(1)]
        return {"actions": np.asarray(data["actions"][:, :16])}

