#!/usr/bin/env bash
set -euo pipefail

root="${PI05_ROOT:-$HOME/pi05-lora}"
openpi="$root/openpi"
wandb_env="$root/.secrets/wandb.env"
config="${1:?usage: $0 CONFIG [EXP_NAME] [STEPS]}"
experiment="${2:-fr3-lora-smoke}"
steps="${3:-10}"

if [[ ! "$steps" =~ ^[0-9]+$ ]] || (( steps < 1 || steps > 50 )); then
  echo "smoke-test steps must be an integer from 1 to 50" >&2
  exit 2
fi
if [[ ! -r "$wandb_env" ]]; then
  echo "missing $wandb_env; run training/openpi/configure_wandb.sh first" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$wandb_env"
export HF_LEROBOT_HOME="$root/lerobot"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XLA_PYTHON_CLIENT_PREALLOCATE="true"

cd "$openpi"
repo_id="$(uv run python -c "from openpi.training import config; print(config.get_config('$config').data.repo_id)")"
if [[ ! -f "$openpi/assets/$config/$repo_id/norm_stats.json" ]]; then
  echo "normalization statistics are missing; run training/openpi/run_training.sh once or compute them" >&2
  exit 1
fi
if [[ ! -d "$HF_LEROBOT_HOME/$repo_id" ]]; then
  echo "training dataset $repo_id is missing under $HF_LEROBOT_HOME" >&2
  exit 1
fi

mkdir -p "$root/logs"
log_file="$root/logs/${experiment}.log"

echo "Smoke test: config=$config experiment=$experiment steps=$steps"
echo "W&B: entity=$WANDB_ENTITY project=$WANDB_PROJECT"
echo "Log: $log_file"

uv run scripts/train.py "$config" \
  --exp-name "$experiment" \
  --num-train-steps "$steps" \
  --log-interval 1 \
  --save-interval "$steps" \
  --overwrite \
  2>&1 | tee "$log_file"

checkpoint_dir="$openpi/checkpoints/$config/$experiment"
checkpoint_path="$checkpoint_dir/$((steps - 1))"
if [[ ! -d "$checkpoint_path/params" ]]; then
  echo "training finished without the expected checkpoint parameters: $checkpoint_path/params" >&2
  exit 1
fi

echo "Smoke test completed successfully."
echo "Checkpoint: $checkpoint_path"
