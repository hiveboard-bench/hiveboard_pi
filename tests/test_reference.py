"""Keep the original Oopsie Cartesian reader and video-copy path covered."""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/openpi/data"))
from convert_hiveboard_to_lerobot import convert


@pytest.mark.parametrize("source", ["oopsie", "reference"])
def test_original_format_conversion_preserves_targets_and_videos(tmp_path, source):
    count = 20
    pose = np.tile([0.4, 0, 0.2, 1, 0, 0, 0], (count, 1))
    target = pose.copy()
    target[:, 0] += np.arange(count) * 0.001
    episode_path = tmp_path / "original.h5"
    with h5py.File(episode_path, "w") as episode:
        episode.attrs["robot_profile"] = json.dumps({"control_freq": 10})
        episode.create_dataset(
            "observations/robot_states/cartesian_position", data=pose
        )
        episode.create_dataset(
            "observations/robot_states/gripper_position", data=np.full((count, 1), 0.08)
        )
        episode.create_dataset("actions/cartesian_position", data=target)
        episode.create_dataset(
            "actions/gripper_position", data=np.full((count, 1), 0.08)
        )
        for name in ("front", "wrist"):
            video = tmp_path / f"{name}.mp4"
            out = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32)
            )
            assert out.isOpened()
            for i in range(count):
                out.write(np.full((32, 32, 3), i, dtype=np.uint8))
            out.release()
            episode.create_dataset(f"observations/video_paths/{name}", data=video.name)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": "original.h5",
                "episode_id": "original",
                "split": "train",
                "task": "synthetic task",
            }
        )
        + "\n"
    )
    repo_id = f"local/original_{source}"
    convert(
        manifest,
        tmp_path / "lerobot",
        input_root=tmp_path,
        repo_id=repo_id,
        source_format=source,
    )
    output = tmp_path / "lerobot" / repo_id
    info = json.loads((output / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["fps"]) == (1, count, 10)
    table = pq.read_table(next(output.rglob("*.parquet")))
    np.testing.assert_allclose(
        np.array(table["actions"].to_pylist())[:, :7], target, atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(table["actions"].to_pylist())[:, 7], 0.08, atol=1e-6
    )
    for feature, camera in (("front_image", "front"), ("wrist_image", "wrist")):
        converted = next(output.rglob(f"{feature}/episode_*.mp4"))
        assert (
            hashlib.sha256(converted.read_bytes()).digest()
            == hashlib.sha256((tmp_path / f"{camera}.mp4").read_bytes()).digest()
        )
