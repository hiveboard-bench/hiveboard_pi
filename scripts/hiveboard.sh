#!/usr/bin/env bash
set -euo pipefail
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PI05_ROOT="${PI05_ROOT:-$bundle}"
export HF_LEROBOT_HOME="$PI05_ROOT/lerobot"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
openpi="$PI05_ROOT/openpi"
export SOURCE="${SOURCE:-datahive}"
# Keep one implementation for the original Oopsie dataset and its reference alias.
[[ "$SOURCE" != oopsie ]] || export SOURCE=reference
if [[ "$SOURCE" == datahive ]]; then
  config=pi05_fr3_lora_datahive_state
  prefix=local/fr3_datahive
  export MANIFEST="${MANIFEST:-$PI05_ROOT/manifests/datahive.jsonl}"
elif [[ "$SOURCE" == reference ]]; then
  config=pi05_fr3_lora_hiveboard_state
  prefix=local/fr3_hiveboard
  export MANIFEST="${MANIFEST:-$bundle/training/openpi/experiments/manifests/hiveboard/success_and_suboptimal.jsonl}"
else
  echo 'SOURCE must be datahive, oopsie or reference' >&2; exit 2
fi
experiment="${EXP_NAME:-datahive-state}"
export MANIFEST="$(realpath -m -- "$MANIFEST")"
[[ -z "${INPUT_DIR:-}" ]] || export INPUT_DIR="$(realpath -m -- "$INPUT_DIR")"
[[ -z "${TASK_MAP:-}" ]] || export TASK_MAP="$(realpath -m -- "$TASK_MAP")"
manifest="$MANIFEST"
reader_args=(--front-camera "${FRONT_CAMERA:-front}" --wrist-camera "${WRIST_CAMERA:-wrist}" --state-gripper-unit "${STATE_GRIPPER_UNIT:-width_m}")
[[ -z "${GRIPPER_OPEN_WIDTH:-}" ]] || reader_args+=(--gripper-open-width "$GRIPPER_OPEN_WIDTH")
[[ -z "${BINARY_OPEN_VALUE:-}" ]] || reader_args+=(--binary-open-value "$BINARY_OPEN_VALUE")

case "${1:-}" in
  setup)
    command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
    cd "$bundle"
    sha256sum --quiet -c provenance/SHA256SUMS
    cd "$openpi"
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
    uv run --frozen python "$bundle/scripts/verify_run.py"
    ;;
  manifest)
    input="${INPUT_DIR:?Set INPUT_DIR=/path/to/samples}"
    if [[ "$SOURCE" == reference ]]; then
      python3 "$bundle/scripts/verify_run.py" --raw "$input"
    else
      cd "$openpi"
      task_args=()
      [[ -z "${TASK_MAP:-}" ]] || task_args+=(--task-map "$TASK_MAP")
      uv run --frozen python "$bundle/training/openpi/data/build_datahive_manifest.py" \
        --input-root "$input" --output "$manifest" "${reader_args[@]}" "${task_args[@]}"
    fi
    ;;
  prepare)
    input="${INPUT_DIR:?Set INPUT_DIR=/path/to/dataset_hiveboard}"
    python3 "$bundle/scripts/verify_run.py" --raw "$input"
    # Refuse partial conversion up front; the historical converter never overwrites by default.
    for split in train validation test; do
      if [[ -e "$HF_LEROBOT_HOME/${prefix}_$split" ]]; then
        echo "Output already exists: $HF_LEROBOT_HOME/${prefix}_$split. Run make check or choose a separate PI05_ROOT." >&2
        exit 1
      fi
    done
    cd "$openpi"
    for split in train validation test; do
      if ! python3 -c 'import json,sys; sys.exit(not any(json.loads(line)["split"]==sys.argv[2] for line in open(sys.argv[1]) if line.strip()))' "$manifest" "$split"; then
        echo "Split $split is empty; skipping conversion."
        continue
      fi
      uv run --frozen python "$bundle/training/openpi/data/convert_hiveboard_to_lerobot.py" \
        --manifest "$manifest" --output-root "$HF_LEROBOT_HOME" --input-root "$input" \
        --repo-id "${prefix}_$split" --split "$split" --source-format "$SOURCE" "${reader_args[@]}"
    done
    uv run --frozen python "$bundle/scripts/verify_run.py" --datasets
    ;;
  verify|check)
    cd "$bundle"
    sha256sum --quiet -c provenance/SHA256SUMS
    cd "$openpi"
    args=()
    [[ "$1" != check ]] || args+=(--datasets)
    uv run --frozen python "$bundle/scripts/verify_run.py" "${args[@]}"
    ;;
  stats)
    cd "$openpi"
    uv run --frozen python "$bundle/scripts/verify_run.py" --datasets --training-ready
    uv run --frozen python "$bundle/training/openpi/compute_norm_stats_without_video_decode.py" --config-name "$config"
    uv run --frozen python "$bundle/scripts/verify_run.py" --datasets --stats
    ;;
  smoke)
    bash "$bundle/training/openpi/run_smoke_test.sh" "$config" "${experiment}-smoke" "${STEPS:-10}"
    ;;
  evaluate)
    cd "$openpi"
    checkpoint="checkpoints/$config/$experiment"
    [[ -d "$checkpoint/29999" ]] || { echo "Final checkpoint not found: $openpi/$checkpoint/29999" >&2; exit 1; }
    mkdir -p "$PI05_ROOT/logs"
    uv run --frozen python "$bundle/training/openpi/evaluate_checkpoints.py" \
      --config-name "$config" --dataset-prefix "$prefix" --checkpoint-root "$checkpoint" --steps 29999 \
      --out "$PI05_ROOT/logs/${experiment}-eval.json"
    ;;
  *) echo "Unknown command: ${1:-}" >&2; exit 2 ;;
esac
