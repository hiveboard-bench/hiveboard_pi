# Hiveboard pi0.5

Train pi0.5 LoRA on DataHive recordings or the original Oopsie Hiveboard dataset.
DataHive is the default data source. Both use separate datasets and normalization assets.

## Requirements

- Linux with an NVIDIA GPU and a CUDA 12 compatible driver. The reference run used an L40S with 48 GB VRAM.
- `uv`, `git`, `make`, and `ffmpeg`.
- DataHive recordings or the original Oopsie Hiveboard dataset, with absolute Cartesian targets, measured gripper width, and front/wrist RGB videos.

Joint-only actions are not supported. Positions and gripper widths must use meters; quaternions must use xyzw order. States and commands must share a clock and run at least at 10 Hz. Videos must start with the first recorded state.

## Setup

```bash
cd ~/Desktop/hiveboard_pi
make setup
make wandb
```

Enter your W&B entity and API key when prompted. Base model weights download automatically when needed.

## Prepare your recordings

Keep the DataHive folder structure:

```text
samples/
  <session_id>/
    trials_<session_id>.csv
    episodes/
      <episode_id>.h5
      <episode_id>_cam_front.mp4
      <episode_id>_cam_wrist.mp4
```

```bash
make prepare INPUT_DIR=/path/to/samples
```

Only episodes annotated as `success` are selected. The command creates reproducible train/validation/test splits, resamples to 10 Hz, and converts the data to LeRobot. Exclusions are listed in `manifests/datahive.report.json`.

If your camera names differ:

```bash
make prepare INPUT_DIR=/path/to/samples FRONT_CAMERA=external WRIST_CAMERA=wrist
```

For binary gripper commands, specify which value means open and the open width in meters. For example, **if your collector uses these values**:

```bash
make prepare INPUT_DIR=/path/to/samples BINARY_OPEN_VALUE=1 GRIPPER_OPEN_WIDTH=0.08
```

If the measured gripper state is an opening fraction instead of meters, also set `STATE_GRIPPER_UNIT=opening_fraction`.

To save your settings, copy `config/datahive.example.mk` to `local.mk` and edit it. Otherwise, the default input folder is `~/DataHive/samples`.

## Train

```bash
make check
make smoke EXP_NAME=my-run STEPS=10
make train EXP_NAME=my-run
```

Training uses batch 32 and 30,000 steps. Normalization is computed from your own data. The smoke test uses a separate experiment.

Resume an interrupted run or evaluate its final checkpoint:

```bash
make resume EXP_NAME=my-run
make evaluate EXP_NAME=my-run
```

Use the same experiment name when resuming. Evaluation requires validation/test data and checkpoint `29999`.

## Output

- Converted data: `lerobot/local/fr3_datahive_<split>/`
- Checkpoints: `openpi/checkpoints/pi05_fr3_lora_datahive_state/<EXP_NAME>/<step>/`
- Evaluation results: `logs/<EXP_NAME>-eval.json`

Existing datasets are not overwritten. For a different collection, use a fresh copy of this package. Changing `EXP_NAME` starts another run on the same converted data.

## Other commands

```bash
make help
make verify
make test DATAHIVE_ROOT=/path/to/DataHive
```

Tests require a [DataHive source checkout](https://github.com/hiveboard-bench/DataHive).
They run on CPU and generate their own episodes and videos in temporary folders.

## Original Oopsie dataset

The original Oopsie HDF5 reader, fixed train/validation/test splits, and reference
normalization checks are still supported. Keep the original HDF5 filenames and
their front/wrist MP4 files together in the input folder.

```bash
make prepare SOURCE=oopsie INPUT_DIR=/path/to/dataset_hiveboard
make train SOURCE=oopsie EXP_NAME=oopsie-run
make resume SOURCE=oopsie EXP_NAME=oopsie-run
make evaluate SOURCE=oopsie EXP_NAME=oopsie-run
```

Use `SOURCE=oopsie` for every command related to this dataset. `SOURCE=reference`
remains an equivalent alias. This mode uses the original 447-episode manifest
(359 train / 44 validation / 44 test), config `pi05_fr3_lora_hiveboard_state`, and
dataset prefix `local/fr3_hiveboard`. DataHive uses config
`pi05_fr3_lora_datahive_state` and prefix `local/fr3_datahive`.

## Source and licenses

The OpenPI source is included at a pinned revision, with project-specific FR3
integration. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
Datasets, checkpoints, environments, logs, and local credentials are excluded by
`.gitignore`.
