#!/usr/bin/env python3
"""Tie each training run to the dataset and normalization statistics it started with.

OpenPI reads ``assets/<config>/<repo_id>/norm_stats.json`` again on every start, resumes
included, and stores it in each checkpoint, so statistics recomputed for a changed
dataset would silently renormalize a resumed run. ``--record`` stores the dataset
fingerprint and the statistics hash for one experiment; ``--check`` refuses to resume
when either changed. Run it in the OpenPI environment with ``HF_LEROBOT_HOME`` set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from openpi.training import config as _config


def _paths(config_name: str) -> tuple[str, Path, Path]:
    """Resolve the repo id, LeRobot dataset root and statistics directory of a config."""
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise SystemExit(f"{config_name} has no LeRobot repo_id")
    if data_config.asset_id != data_config.repo_id:
        raise SystemExit(
            f"{config_name} loads statistics from asset {data_config.asset_id!r}, not the "
            f"{data_config.repo_id!r} statistics compute_norm_stats writes"
        )
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home:
        raise SystemExit("HF_LEROBOT_HOME must be set")
    return (
        data_config.repo_id,
        Path(home) / data_config.repo_id,
        config.assets_dirs / data_config.repo_id,
    )


def dataset_fingerprint(root: Path) -> str:
    """SHA-256 over the metadata and parquet frames that norm statistics are computed from."""
    files = sorted(
        path for folder in ("meta", "data") for path in (root / folder).rglob("*") if path.is_file()
    )
    if not any(path.suffix == ".parquet" for path in files):
        raise SystemExit(f"no LeRobot parquet data under {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--experiment", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true", help="store the state a new run starts from")
    mode.add_argument("--check", action="store_true", help="verify a resumed run is unchanged")
    args = parser.parse_args()

    repo_id, dataset_root, stats_dir = _paths(args.config_name)
    stats_path = stats_dir / "norm_stats.json"
    if not stats_path.is_file():
        raise SystemExit(f"missing {stats_path}")
    current = {
        "config": args.config_name,
        "repo_id": repo_id,
        "dataset_fingerprint": dataset_fingerprint(dataset_root),
        "norm_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }
    record_path = stats_dir / "runs" / f"{args.experiment}.json"
    print(f"norm stats: {stats_path}")
    print(f"dataset:    {dataset_root} ({current['dataset_fingerprint'][:12]})")

    if args.record:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"recorded:   {record_path}")
        return
    if not record_path.is_file():
        raise SystemExit(
            f"no record for experiment {args.experiment!r} at {record_path}; it was not started "
            "by run_training.sh, so its statistics cannot be verified"
        )
    recorded = json.loads(record_path.read_text())
    changed = [key for key in current if recorded.get(key) != current[key]]
    if changed:
        raise SystemExit(
            f"refusing to resume {args.experiment!r}: {', '.join(changed)} changed since the run "
            f"started ({record_path}); resuming would train the remaining steps with different "
            "normalization. Start a new experiment instead."
        )
    print(f"verified:   {record_path}")


if __name__ == "__main__":
    main()
