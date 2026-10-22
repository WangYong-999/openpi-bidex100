"""BiDex-100 58-D transforms, ported from the internal mobile branch."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class BiDexJointInputs(transforms.DataTransformFn):
    """Inputs for the Astribot kitchen task policy (arm + dex hand, 58D).

    Expected inputs (from RepackTransform):
    - images: dict with cam_high, cam_left_wrist, cam_right_wrist
    - state_arm: arm_joint_position [14] (right 7 + left 7)
    - state_dex: dex_joint_position [44] (right 22 + left 22)

    Actions:
    - action_arm: arm_joint_position [14]
    - action_dex: dex_joint_position [44]

    Assembled:
    - state: [58] = arm(14) + dex(44)
    - actions: [action_horizon, 58] = same layout
    """

    adapt_to_pi: bool = True

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    @staticmethod
    def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            return einops.rearrange(img, "c h w -> h w c")

        images = data["images"]
        images_dict = {name: convert_image(img) for name, img in images.items()}
        data["images"] = images_dict
        return data

    def __call__(self, data: dict) -> dict:
        data = self._decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # Construct state: arm (14D) + dex (44D) = 58D
        state_arm = np.asarray(data["state_arm"]).reshape(-1)  # 14D
        state_dex = np.asarray(data["state_dex"]).reshape(-1)  # 44D
        state = np.concatenate([state_arm, state_dex])  # 58D

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "action_arm" in data:
            action_arm = np.asarray(data["action_arm"])  # [chunk, 14]
            action_dex = np.asarray(data["action_dex"])  # [chunk, 44]
            actions = np.concatenate([action_arm, action_dex], axis=1)  # [chunk, 58]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BiDexJointOutputs(transforms.DataTransformFn):
    """Outputs for the Astribot kitchen task policy (arm + dex hand, 58D)."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :58])

        actions_arm = actions[:, 0:14]  # 14D
        actions_dex = actions[:, 14:58]  # 44D

        return {
            "actions": actions,
            "actions_arm": actions_arm,  # [chunk, 14]
            "actions_dex": actions_dex,  # [chunk, 44]
        }


@dataclasses.dataclass(frozen=True)
class BiDexEEInputs(transforms.DataTransformFn):
    """Inputs for the BiDex-100 ee-pose control policy (ee + dex, 58D).

    Expected inputs (from RepackTransform):
    - images: dict with cam_high, cam_left_wrist, cam_right_wrist (CHW uint8)
    - state_ee: observation.states.abs_end_effector [14]
      ([right_xyz(3)+right_quat_xyzw(4), left_xyz(3)+left_quat_xyzw(4)])
    - state_dex: observation.states.dex_joint_position [44]

    Actions (ABSOLUTE, no DeltaActions — quaternions can't be element-wise deltas):
    - action_ee: actions.abs_end_effector [14]
    - action_dex: actions.dex_joint_position [44]

    Assembled:
    - state: [58] = ee(14) + dex(44)
    - actions: [action_horizon, 58] = same layout
    """

    adapt_to_pi: bool = True

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    @staticmethod
    def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            return einops.rearrange(img, "c h w -> h w c")

        images = data["images"]
        images_dict = {name: convert_image(img) for name, img in images.items()}
        data["images"] = images_dict
        return data

    def __call__(self, data: dict) -> dict:
        data = self._decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # Construct state: ee (14D) + dex (44D) = 58D
        state_ee = np.asarray(data["state_ee"]).reshape(-1)  # 14D
        state_dex = np.asarray(data["state_dex"]).reshape(-1)  # 44D
        state = np.concatenate([state_ee, state_dex])  # 58D

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "action_ee" in data:
            action_ee = np.asarray(data["action_ee"])  # [chunk, 14]
            action_dex = np.asarray(data["action_dex"])  # [chunk, 44]
            actions = np.concatenate([action_ee, action_dex], axis=1)  # [chunk, 58]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BiDexEEOutputs(transforms.DataTransformFn):
    """Outputs for the BiDex-100 ee-pose control policy (58D).

    Actions are ABSOLUTE ee (pos+quat) + absolute dex. No AbsoluteActions
    inversion is applied (use_delta_joint_actions=False upstream).
    """

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :58])

        actions_ee = actions[:, 0:14]  # 14D
        actions_dex = actions[:, 14:58]  # 44D

        return {
            "actions": actions,
            "actions_ee": actions_ee,  # [chunk, 14]
            "actions_dex": actions_dex,  # [chunk, 44]
        }
