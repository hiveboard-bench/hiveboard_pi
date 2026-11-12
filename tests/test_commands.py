"""Check shell orchestration without downloading weights or starting GPU jobs."""

from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("save_parameters", [False, True])
def test_smoke_requires_saved_parameters(tmp_path, save_parameters):
    root = tmp_path / "workspace"
    repo = "local/fr3_datahive_train"
    stats = root / "openpi/assets/synthetic_config" / repo / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}")
    (root / "lerobot" / repo).mkdir(parents=True)
    secrets = root / ".secrets/wandb.env"
    secrets.parent.mkdir()
    secrets.write_text(
        "export WANDB_ENTITY=synthetic\nexport WANDB_PROJECT=synthetic\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
if "-c" in sys.argv:
    print("local/fr3_datahive_train")
else:
    checkpoint = Path(os.environ["PI05_ROOT"]) / "openpi/checkpoints/synthetic_config/synthetic-smoke/1"
    checkpoint.mkdir(parents=True)
    if os.environ["SAVE_PARAMETERS"] == "1":
        (checkpoint / "params").mkdir()
""")
    uv.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "training/openpi/run_smoke_test.sh"),
            "synthetic_config",
            "synthetic-smoke",
            "2",
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PI05_ROOT": str(root),
            "SAVE_PARAMETERS": str(int(save_parameters)),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if save_parameters else 1)
    if not save_parameters:
        assert "without the expected checkpoint parameters" in result.stderr


def test_make_test_reports_missing_datahive_checkout(tmp_path):
    result = subprocess.run(
        ["make", "test", f"DATAHIVE_ROOT={tmp_path / 'missing'}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "DataHive checkout not found" in result.stderr


@pytest.mark.parametrize(
    "source,config",
    [
        ("datahive", "pi05_fr3_lora_datahive_state"),
        ("oopsie", "pi05_fr3_lora_hiveboard_state"),
        ("reference", "pi05_fr3_lora_hiveboard_state"),
    ],
)
def test_source_selects_the_matching_training_config(source, config):
    result = subprocess.run(
        ["make", "-n", "train", f"SOURCE={source}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f'"{config}"' in result.stdout


def test_oopsie_alias_matches_reference_raw_manifest(tmp_path):
    import json
    import sys

    manifest = (
        ROOT
        / "training/openpi/experiments/manifests/hiveboard/success_and_suboptimal.jsonl"
    )
    for row in map(json.loads, manifest.read_text().splitlines()):
        path = tmp_path / row["path"]
        path.touch()
    outputs = []
    for source in ("oopsie", "reference"):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify_run.py"),
                "--source",
                source,
                "--raw",
                str(tmp_path),
            ],
            env={
                key: value
                for key, value in os.environ.items()
                if key not in {"SOURCE", "MANIFEST"}
            },
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert "447 episodes" in outputs[0]
