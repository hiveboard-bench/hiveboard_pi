#!/usr/bin/env python3
"""Print shell exports that give a new OpenPI run its W&B description, tags and group.

``run_training.sh`` evaluates the output before ``scripts/train.py``; ``wandb.init`` reads
``WANDB_NOTES`` (Markdown), ``WANDB_TAGS`` and ``WANDB_RUN_GROUP`` from the environment.
The notes summarize the dataset as converted, the model and optimizer settings, the data
pipeline, and the exact code and statistics the run starts from. Run it in the OpenPI
environment with ``HF_LEROBOT_HOME`` set.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

from openpi.training import config as _config

PIPELINE = (
    "actions anchored to the observation state (`Fr3Inputs.anchored_actions`); idle leading "
    "frames dropped (7+ unchanged commands); state quaternion kept on the home hemisphere"
)


def _git(path: Path) -> str:
    """Short commit of the repository containing ``path``, marked when the tree is dirty."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if dirty else "")


def _dataset_summary(root: Path) -> tuple[str, list[str]]:
    """One line of episode/frame totals and one line per task for a converted LeRobot dataset."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return f"`{root.name}` not converted", []
    info = json.loads(info_path.read_text())
    episodes, frames = collections.Counter(), collections.Counter()
    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        for line in episodes_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                for task in row.get("tasks", []):
                    episodes[task] += 1
                    frames[task] += int(row.get("length", 0))
    total = (
        f"{info.get('total_episodes', '?')} episodes, {info.get('total_frames', '?')} frames "
        f"@ {info.get('fps', '?')} Hz"
    )
    tasks = [
        f"  - {task}: {episodes[task]} episodes, {frames[task]} frames" for task in sorted(episodes)
    ]
    return total, tasks


def notes(config_name: str, experiment: str) -> tuple[str, list[str], str]:
    """Return the Markdown notes, tags and group of a new run."""
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = data_config.repo_id
    home = Path(os.environ["HF_LEROBOT_HOME"])
    model, schedule = config.model, config.lr_schedule
    state_input = bool(getattr(model, "discrete_state_input", False))
    train_total, train_tasks = _dataset_summary(home / repo_id)
    held_out = []
    if repo_id.endswith("_train"):
        for split in ("validation", "test"):
            total, _ = _dataset_summary(home / f"{repo_id[: -len('_train')]}_{split}")
            held_out.append(f"{split} {total}")
    record_path = config.assets_dirs / repo_id / "runs" / f"{experiment}.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    weights = getattr(config.weight_loader, "params_path", type(config.weight_loader).__name__)
    lines = [
        f"**{config_name}** / `{experiment}` started {datetime.date.today().isoformat()}",
        "",
        f"**Dataset** `{repo_id}`: {train_total}",
        *train_tasks,
        f"**Held out**: {'; '.join(held_out) or 'none'}",
        "",
        f"**Model**: pi0.5 LoRA ({getattr(model, 'paligemma_variant', '?')} + "
        f"{getattr(model, 'action_expert_variant', '?')}), action horizon "
        f"{getattr(model, 'action_horizon', '?')}, state input "
        f"{'discretized into the prompt' if state_input else 'none (images + prompt only)'}",
        f"**Init**: `{weights}`",
        f"**Training**: batch {config.batch_size}, {config.num_train_steps} steps, cosine LR "
        f"warmup {getattr(schedule, 'warmup_steps', '?')} -> peak "
        f"{getattr(schedule, 'peak_lr', '?')} -> {getattr(schedule, 'decay_lr', '?')} at "
        f"{getattr(schedule, 'decay_steps', '?')}, grad clip "
        f"{getattr(config.optimizer, 'clip_gradient_norm', '?')}, EMA {config.ema_decay}, "
        f"seed {config.seed}, checkpoints every {config.save_interval} (kept every "
        f"{config.keep_period})",
        f"**Pipeline**: {PIPELINE}",
        f"**Code**: hiveboard_dataset `{_git(Path(__file__).resolve().parent)}`, openpi "
        f"`{_git(Path.cwd())}`",
        f"**Norm stats**: sha256 `{record.get('norm_stats_sha256', '?')[:12]}`, dataset "
        f"fingerprint `{record.get('dataset_fingerprint', '?')[:12]}`",
    ]
    tags = [config_name, "state" if state_input else "no-state", repo_id.split("/")[-1], "lora"]
    return "\n".join(lines), tags, repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    text, tags, group = notes(args.config_name, args.experiment)
    print(f"export WANDB_NOTES={shlex.quote(text)}")
    print(f"export WANDB_TAGS={shlex.quote(','.join(tags))}")
    print(f"export WANDB_RUN_GROUP={shlex.quote(group)}")


if __name__ == "__main__":
    main()
