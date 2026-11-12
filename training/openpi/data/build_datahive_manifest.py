#!/usr/bin/env python3
"""Build deterministic training splits from annotated DataHive sessions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import h5py
import yaml

from datahive_adapter import READER_DEFAULTS, attr, read_episode

BUNDLE = Path(__file__).resolve().parents[3]


def build(
    input_root: Path,
    output: Path,
    *,
    seed: int = 42,
    task_map: Path | None = None,
    **options,
) -> dict:
    options = {**READER_DEFAULTS, **options}
    input_root = input_root.resolve()
    registry = (
        yaml.safe_load((task_map or input_root / "attachments.yaml").read_text())
        if (task_map or (input_root / "attachments.yaml").is_file())
        else yaml.safe_load((BUNDLE / "config/datahive_tasks.yaml").read_text())
    )
    rows, excluded = [], []
    files = sorted(input_root.rglob("*.h5"))
    for path in files:
        try:
            with h5py.File(path, "r") as episode:
                episode_id = str(attr(episode, "episode_id", ""))
                trial_id = str(attr(episode, "trial_id", ""))
                session = str(attr(episode, "session_id", ""))
                if not episode_id or not trial_id or not session:
                    raise ValueError("episode_id, trial_id and session_id must be set")
                if path.parent.name != "episodes" or path.parent.parent.name != session:
                    raise ValueError(
                        "expected samples/<session_id>/episodes/<episode_id>.h5"
                    )
                task_ids = attr(episode, "task_ids", [])
                lab, platform = attr(episode, "lab_id"), attr(episode, "platform_id")
            csv_path = path.parent.parent / f"trials_{session}.csv"
            if not csv_path.exists():
                csv_path = path.parent.parent / "trials.csv"
            if not csv_path.exists():
                raise ValueError("session annotation CSV not found")
            with csv_path.open(newline="", encoding="utf-8-sig") as stream:
                matches = [
                    r for r in csv.DictReader(stream) if r.get("trial_id") == trial_id
                ]
            if len(matches) != 1:
                raise ValueError("trial_id must identify exactly one CSV annotation")
            annotation = matches[0]
            if annotation.get("outcome") != "success":
                raise ValueError(
                    f"excluded outcome: {annotation.get('outcome') or 'not annotated'}"
                )
            if (
                annotation.get("lab_id") != lab
                or annotation.get("platform_id") != platform
            ):
                raise ValueError("CSV lab_id/platform_id do not match the episode")
            attachment = annotation.get("attachment_id", "")
            if task_ids != [attachment]:
                raise ValueError("task_ids must match the annotated attachment_id")
            entry = registry.get(attachment)
            prompt = entry if isinstance(entry, str) else (entry or {}).get("success")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"missing prompt for attachment_id={attachment!r}; provide TASK_MAP"
                )
            data = read_episode(path, **options)
            rows.append(
                {
                    "episode_id": episode_id,
                    "trial_id": trial_id,
                    "session": session,
                    "path": str(path.relative_to(input_root)),
                    "task": prompt.strip(),
                    "attachment_id": attachment,
                    "outcome": "success",
                    "frames": len(data.state),
                    "source_format": "datahive",
                    "fps": data.fps,
                    "hdf5_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        except (ValueError, KeyError, OSError, TypeError) as error:
            excluded.append(
                {"path": str(path.relative_to(input_root)), "reason": str(error)}
            )
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    rng = random.Random(seed)
    for task in sorted(by_task):
        group = by_task[task]
        rng.shuffle(group)
        held = max(1, int(len(group) * 0.1)) if len(group) >= 3 else 0
        for index, row in enumerate(group):
            row["split"] = (
                "validation"
                if index < held
                else "test"
                if index < 2 * held
                else "train"
            )
    rows.sort(key=lambda r: r["path"])
    manifest = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    counts, frames = Counter(), Counter()
    for row in rows:
        counts[row["split"]] += 1
        frames[row["split"]] += row["frames"]
    report = {
        "source_format": "datahive",
        "input_root": str(input_root),
        "seed": seed,
        "fps": 10,
        "scanned": len(files),
        "selected": len(rows),
        "excluded": excluded,
        "splits": {
            split: {"episodes": counts[split], "frames": frames[split]}
            for split in ("train", "validation", "test")
        },
        "reader_options": options,
        "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text() != manifest:
        raise ValueError(
            f"existing manifest differs from the current data: {output}; preserve it and choose another MANIFEST"
        )
    output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if not rows:
        raise ValueError(
            f"no compatible episodes annotated as success; see {output.with_suffix('.report.json')}"
        )
    output.write_text(manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-map", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--front-camera", default="front")
    parser.add_argument("--wrist-camera", default="wrist")
    parser.add_argument("--gripper-open-width", type=float)
    parser.add_argument("--binary-open-value", type=int, choices=(0, 1))
    parser.add_argument(
        "--state-gripper-unit",
        choices=("width_m", "opening_fraction"),
        default="width_m",
    )
    args = vars(parser.parse_args())
    try:
        report = build(**args)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
