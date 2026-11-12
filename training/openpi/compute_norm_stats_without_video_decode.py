#!/usr/bin/env python3
"""Run OpenPI's official norm-stat computation without decoding unused videos.

The upstream script computes statistics only for ``state`` and ``actions``, but
LeRobot decodes every camera frame before those two keys are selected.  This
wrapper substitutes tiny image placeholders for that one preprocessing pass.
It does not alter the numeric samples, action horizons, RunningStats algorithm,
or the saved OpenPI normalization format.  Training still reads real videos.
"""

from __future__ import annotations

import sys

import torch
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _placeholder_videos(self, query_timestamps, ep_idx):
    del self, ep_idx
    # These images are discarded by the upstream norm-stat loop. A valid CHW
    # shape is enough for the FR3 input transform, at negligible I/O cost.
    return {key: torch.zeros((3, 1, 1), dtype=torch.uint8) for key in query_timestamps}


LeRobotDataset._query_videos = _placeholder_videos

# OpenPI keeps this supported command as a script rather than an importable
# package module. Import it after installing the monkeypatch so spawned data
# workers execute the same lightweight path.
sys.path.insert(0, "scripts")
import compute_norm_stats  # noqa: E402

if __name__ == "__main__":
    tyro.cli(compute_norm_stats.main)
