#!/usr/bin/env bash
set -euo pipefail

usage="usage: $0 EXP_NAME CONFIG [--resume | --overwrite]"
if [[ $# -lt 2 || "$2" == --* ]]; then
  echo "$usage" >&2
  exit 2
fi
experiment="$1"
config="$2"
shift 2
resume=0
train_args=()
for arg in "$@"; do
  case "$arg" in
    --resume) resume=1; train_args+=("$arg") ;;
    --overwrite) train_args+=("$arg") ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
done
if [[ "$resume" == 1 && ${#train_args[@]} -gt 1 ]]; then
  echo "--resume and --overwrite are mutually exclusive" >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${PI05_ROOT:-$HOME/pi05-lora}"
openpi="$root/openpi"
wandb_env="$root/.secrets/wandb.env"

if [[ ! -r "$wandb_env" ]]; then
  echo "missing $wandb_env; run training/openpi/configure_wandb.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$wandb_env"
export HF_LEROBOT_HOME="$root/lerobot"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

cd "$openpi"
guard=(uv run python "$repo_root/training/openpi/norm_stats_guard.py" --config-name "$config" --experiment "$experiment")
if [[ "$resume" == 1 ]]; then
  "${guard[@]}" --check
else
  uv run python "$repo_root/training/openpi/compute_norm_stats_without_video_decode.py" --config-name "$config"
  "${guard[@]}" --record
  run_info="$(uv run python "$repo_root/training/openpi/wandb_run_info.py" --config-name "$config" --experiment "$experiment")"
  eval "$run_info"
  printf 'W&B description:\n%s\n' "$WANDB_NOTES"
fi
uv run scripts/train.py "$config" --exp-name "$experiment" ${train_args[@]+"${train_args[@]}"}
