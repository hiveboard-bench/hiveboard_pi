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
