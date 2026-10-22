"""BiDex-100 co-training configs, selected from the internal OpenPI integration.

The public release uses <root>/<capability>/<task>/meta/info.json. Set
BIDEX_DATA_ROOT to the downloaded datasets/lerobotv21 directory before training.
No dataset is required to import configs or load an evaluation checkpoint.
"""

import dataclasses
import os
from pathlib import Path

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import bidex_policy
from openpi.training import config
from openpi.training import optimizer
from openpi.training import weight_loaders

SUITES = ("Generalization", "Instruction_Following", "Limited_Demonstrations", "Long_Horizon", "Memory")


def task_paths(suite=None):
    root = Path(os.environ.get("BIDEX_DATA_ROOT", "../BiDex-100/downloads/BiDex-100/datasets/lerobotv21"))
    return tuple(
        str(info.parent.parent.resolve())
        for name in ((suite,) if suite else SUITES)
        for info in sorted((root / name).glob("*/meta/info.json"))
    )


@dataclasses.dataclass(frozen=True)
class BiDexDataConfig(config.DataConfigFactory):
    mode: str = "joint"
    suite: str | None = None

    def create(self, assets_dirs, model_config):
        arm = {"joint": "arm_joint_position", "robot_ee": "abs_end_effector", "camera_ee": "abs_end_effector_camera"}[
            self.mode
        ]
        repack = {
            "images": {
                "cam_high": "observation.images.head_color",
                "cam_left_wrist": "observation.images.left_wrist_color",
                "cam_right_wrist": "observation.images.right_wrist_color",
            },
            "state_dex": "observation.states.dex_joint_position",
            "action_dex": "actions.dex_joint_position",
            "prompt": "prompt",
        }
        key = "arm" if self.mode == "joint" else "ee"
        repack[f"state_{key}"] = f"observation.states.{arm}"
        repack[f"action_{key}"] = f"actions.{arm}"
        inputs = bidex_policy.BiDexJointInputs if self.mode == "joint" else bidex_policy.BiDexEEInputs
        outputs = bidex_policy.BiDexJointOutputs if self.mode == "joint" else bidex_policy.BiDexEEOutputs
        data_transforms = transforms.Group(inputs=[inputs(adapt_to_pi=False)], outputs=[outputs(adapt_to_pi=False)])
        if self.mode == "joint":
            mask = transforms.make_bool_mask(14, 44)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)], outputs=[transforms.AbsoluteActions(mask)]
            )
        paths = task_paths(self.suite)
        # Evaluators load configs without training data; training must fail clearly.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repo_id="BiDex-100",
            bidex_dataset_paths=paths,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(repack)]),
            data_transforms=data_transforms,
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=(f"actions.{arm}", "actions.dex_joint_position"),
            prompt_from_task=True,
        )


def get_bidex_configs():
    configs = []
    for suite in (None, *SUITES):
        for mode in ("joint", "robot_ee", "camera_ee"):
            name = f"BiDex-{suite or 'All'}-{mode}-1"
            asset_id = f"bidex_{(suite or 'all').lower()}_{mode}"
            item = config.TrainConfig(
                name=name,
                model=pi0_config.Pi0Config(pi05=True, max_token_len=512, action_dim=58, action_horizon=50),
                data=BiDexDataConfig(
                    repo_id="BiDex-100",
                    mode=mode,
                    suite=suite,
                    assets=config.AssetsConfig(assets_dir="./assets", asset_id=asset_id),
                ),
                batch_size=256,
                num_workers=8,
                num_train_steps=60_000,
                lr_schedule=optimizer.CosineDecaySchedule(
                    warmup_steps=100, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
                ),
                optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
                ema_decay=0.999,
                save_interval=10_000,
                weight_loader=weight_loaders.PartialCheckpointWeightLoader(
                    "gs://openpi-assets/checkpoints/pi05_base/params"
                ),
            )
            configs.append(item)
            # Required by the benchmark launcher and pre-rename checkpoints.
            # This is a config alias for BiDex-100, not another benchmark.
            legacy_assets = dataclasses.replace(item.data.assets, asset_id=asset_id.replace("bidex", "dexsim", 1))
            configs.append(
                dataclasses.replace(
                    item,
                    name=name.replace("BiDex", "DexSim", 1),
                    data=dataclasses.replace(item.data, assets=legacy_assets),
                )
            )
    return configs
