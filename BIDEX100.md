# BiDex-100 · π0.5

[Document](https://bidex-100.github.io/BiDex-100/document.html) · [Benchmark code](https://github.com/robbyant-research/BiDex-100)


### Create the environment

```bash
git clone https://github.com/WangYong-999/openpi-bidex100.git
cd openpi-bidex100
python -m pip install -U uv
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
```

The repository lockfile selects OpenPI's LeRobot revision and JAX dependencies.
Use this environment for both the trainer and policy server. The simulation
workers can use the separate BiDex-100 environment.

### Prepare data and train

Set an absolute path to the downloaded **LeRobot v2.1** root. It contains the
five capability directories, each with task-level `meta/info.json` files.

```bash
export BIDEX_DATA_ROOT=/absolute/path/to/BiDex-100/downloads/BiDex-100/datasets/lerobotv21

uv run scripts/compute_norm_stats.py --config-name BiDex-All-joint-1
uv run scripts/train.py BiDex-All-joint-1 \
  --exp-name bidex_all_joint \
  --batch-size 256 \
  --num-train-steps 60000 \
  --fsdp-devices 8
```

The 58-D projection layers are initialized for BiDex-100 while compatible weights
are transferred from `gs://openpi-assets/checkpoints/pi05_base/params`. The joint
transform trains on deltas and restores absolute targets for evaluation. For a
single GPU, use `--fsdp-devices 1` and choose a batch size that fits memory.
The checkpoint path is `checkpoints/BiDex-All-joint-1/bidex_all_joint/60000`.

### Evaluate

From the **BiDex-100** repository, set `MODEL_REPO` to the absolute model clone:

```bash
export MODEL_REPO=/absolute/path/to/openpi-bidex100
export MODEL_PY="$MODEL_REPO/.venv/bin/python"
export MODEL_CKPT="$MODEL_REPO/checkpoints/BiDex-All-joint-1/bidex_all_joint/60000"
export BATCHED_CONFIG=BiDex-All-joint-1
export BIDEX_LEADERBOARD_ROOT=/absolute/path/to/evaluation-output

MID_OVERRIDE=pi05_smoke MODEL_GPUS=0 RENDER_GPUS=0 \
  bash scripts/evaluation/run_vec_models_eval.sh pi05 smoke
```
