#!/usr/bin/env python3
"""Verify the training config and either a DataHive manifest or historical splits."""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parents[1]
REFERENCE_COUNTS = {"train": (359, 69667), "validation": (44, 8595), "test": (44, 8520)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--datasets", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--training-ready", action="store_true")
    parser.add_argument(
        "--source",
        choices=("datahive", "reference"),
        default=os.environ.get("SOURCE", "datahive"),
    )
    args = parser.parse_args()
    datahive = args.source == "datahive"
    prefix = "local/fr3_datahive" if datahive else "local/fr3_hiveboard"
    config_name = (
        "pi05_fr3_lora_datahive_state" if datahive else "pi05_fr3_lora_hiveboard_state"
    )
    root = Path(os.environ.get("PI05_ROOT", str(BUNDLE)))
    default_manifest = (
        root / "manifests/datahive.jsonl"
        if datahive
        else BUNDLE
        / "training/openpi/experiments/manifests/hiveboard/success_and_suboptimal.jsonl"
    )
    manifest = Path(os.environ.get("MANIFEST", str(default_manifest)))
    rows = []
    counts = REFERENCE_COUNTS
    if manifest.exists():
        rows = [
            json.loads(line)
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]
        require(bool(rows), "The manifest is empty.")
        require(
            len({row["path"] for row in rows}) == len(rows),
            "The manifest contains duplicate episodes.",
        )
        require(
            all(r["split"] in {"train", "validation", "test"} for r in rows),
            "Unknown split in the manifest.",
        )
        if datahive:
            report_path = manifest.with_suffix(".report.json")
            require(
                report_path.exists(),
                f"Report not found: {report_path}; run make manifest.",
            )
            report = json.loads(report_path.read_text())
            require(
                hashlib.sha256(manifest.read_bytes()).hexdigest()
                == report["manifest_sha256"],
                "The manifest changed after it was generated.",
            )
            require(
                report["source_format"] == "datahive" and report["fps"] == 10,
                "Incompatible report.",
            )
            counts = {
                split: (value["episodes"], value["frames"])
                for split, value in report["splits"].items()
            }
            for split, (episodes, frames) in counts.items():
                selected = [r for r in rows if r["split"] == split]
                require(
                    len(selected) == episodes
                    and sum(r["frames"] for r in selected) == frames,
                    f"Manifest and report disagree for split {split}.",
                )
            require(counts["train"][0] > 0, "No training episodes selected.")
            require(
                all(
                    r["source_format"] == "datahive" and r["outcome"] == "success"
                    for r in rows
                ),
                "The DataHive manifest must contain only successful DataHive episodes.",
            )
        else:
            require(
                Counter(row["split"] for row in rows)
                == Counter({k: v[0] for k, v in counts.items()}),
                "The manifest does not match the reference splits 359/44/44.",
            )
    elif args.raw or args.datasets or args.stats or args.training_ready or not datahive:
        raise SystemExit(
            f"Manifest not found: {manifest}. Run make manifest or make prepare."
        )
    if args.raw is not None:
        missing = [
            row["path"] for row in rows if not (args.raw / row["path"]).is_file()
        ]
        require(not missing, f"Missing episodes: {missing[:10]}")
        print(f"{len(rows)} episodes found in {args.raw}.")
        return

    import wandb
    from openpi.training import config

    originals = json.loads(
        gzip.decompress((BUNDLE / "provenance/original_reference.json.gz").read_bytes())
    )
    expected = originals["wandb_run.json"]["config"]
    expected["data"]["repo_id"] = f"{prefix}_train"
    expected["name"] = config_name
    cfg = config.get_config(config_name)
    actual = wandb.util.json_friendly_val(
        dataclasses.asdict(dataclasses.replace(cfg, exp_name="hiveboard-state"))
    )
    different = [
        key
        for key in set(actual) | set(expected)
        if actual.get(key) != expected.get(key)
    ]
    require(not different, "Configuration differs from W&B: " + ", ".join(different))
    print(
        f"{config_name}: model and optimizer match 0pczxy2h; separate dataset {prefix}_train."
    )
    if args.training_ready or args.stats:
        require(
            counts["train"][1] >= cfg.batch_size,
            f"Training data contains {counts['train'][1]} frames; batch {cfg.batch_size} requires at least {cfg.batch_size}. Collect more data.",
        )
    if rows:
        print(f"Manifest {args.source}: {len(rows)} episodes, splits {counts}.")

    home = Path(os.environ.get("HF_LEROBOT_HOME", str(root / "lerobot")))
    if args.datasets:
        for split, (episodes, frames) in counts.items():
            if episodes == 0:
                print(f"{split}: empty (unavailable for evaluation).")
                continue
            dataset = home / f"{prefix}_{split}"
            path = dataset / "meta/info.json"
            require(path.is_file(), f"Dataset not found: {path}. Run make prepare.")
            info = json.loads(path.read_text())
            require(
                (info["total_episodes"], info["total_frames"], info["fps"])
                == (episodes, frames, 10),
                f"Dataset {split} differs from the manifest.",
            )
            if datahive:
                source_path = dataset / "meta/hiveboard_source.json"
                require(
                    source_path.is_file(), f"Missing manifest binding: {source_path}"
                )
                source = json.loads(source_path.read_text())
                require(
                    source["manifest_sha256"] == report["manifest_sha256"]
                    and source["reader_options"] == report["reader_options"],
                    f"Dataset {split} belongs to a different manifest or camera/gripper mapping.",
                )
            stored = [
                json.loads(line)
                for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()
                if line.strip()
            ]
            selected = [r for r in rows if r["split"] == split]
            require(
                len(stored) == len(selected), f"Inconsistent episode metadata: {split}"
            )
            for row, saved in zip(selected, stored):
                require(
                    saved["tasks"] == [row["task"]],
                    f"Prompt differs from the manifest: {row['path']}",
                )
                if datahive:
                    require(
                        saved["length"] == row["frames"],
                        f"Frame count differs from the manifest: {row['path']}",
                    )
            print(f"{split}: {episodes} episodes, {frames} frames, 10 Hz — OK")

    if args.stats:
        sys.path.insert(0, str(BUNDLE / "training/openpi"))
        from norm_stats_guard import dataset_fingerprint

        stats = cfg.assets_dirs / f"{prefix}_train/norm_stats.json"
        require(stats.is_file(), f"Normalization statistics not found: {stats}")
        stats_hash = hashlib.sha256(stats.read_bytes()).hexdigest()
        if not datahive:
            record = json.loads((BUNDLE / "provenance/dataset_record.json").read_text())
            require(
                stats_hash == record["norm_stats_sha256"],
                "Normalization differs from the reference run.",
            )
        print(f"Normalization {args.source}: {stats_hash}")
        print(f"Fingerprint local: {dataset_fingerprint(home / f'{prefix}_train')}")
        print(
            "Resume checks the dataset fingerprint and normalization recorded when this experiment started."
        )


if __name__ == "__main__":
    main()
