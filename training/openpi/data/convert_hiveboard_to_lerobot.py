#!/usr/bin/env python3
"""Convert an FR3 manifest to LeRobot with front+wrist video and absolute Cartesian targets.

The stored actions are the recorded targets ``[xyz, xyzw, gripper width]``. OpenPI's
``Fr3Inputs`` turns each sampled chunk into actions relative to the observation's state.
Leading stationary frames are dropped as in OpenPI's DROID idle filter.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

import cv2
import h5py
import numpy as np


def _load_shared():
    """Import tools/hiveboard_episode.py by path; the script directory is not its package."""
    module = sys.modules.get("hiveboard_episode")
    if module is None:
        path = Path(__file__).resolve().parents[3] / "tools" / "hiveboard_episode.py"
        spec = importlib.util.spec_from_file_location("hiveboard_episode", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


_shared = _load_shared()
gripper_actions = _shared.gripper_actions
motion_start = _shared.motion_start
state_quaternions = _shared.state_quaternions

STATE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width_m")
ACTION_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper_target_m")


def episode_arrays(episode: h5py.File) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the trimmed state and absolute-target action arrays, and the first kept frame."""
    state_pose = np.asarray(
        episode["observations/robot_states/cartesian_position"], dtype=np.float32
    )
    measured = np.asarray(
        episode["observations/robot_states/gripper_position"], dtype=np.float32
    )
    target_pose = np.asarray(episode["actions/cartesian_position"], dtype=np.float32)
    gripper = gripper_actions(episode["actions/gripper_position"][:], measured)
    start = motion_start(target_pose, gripper)
    state = np.concatenate([state_pose, measured.reshape(len(measured), -1)], axis=1)
    state[:, 3:7] = state_quaternions(state[:, 3:7])
    actions = np.concatenate([target_pose, gripper], axis=1).astype(np.float32)
    return state[start:], actions[start:], start


def video_frames(path: Path, expected: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != expected:
        raise ValueError(f"{path} has {len(frames)} frames; expected {expected}")
    return frames


def convert(
    manifest: Path,
    output_root: Path,
    *,
    input_root: Path,
    repo_id: str = "local/fr3_hiveboard",
    split: str = "train",
    limit_episodes: int | None = None,
    video_codec: str = "h264",
    reuse_source_videos: bool = True,
    overwrite: bool = False,
    source_format: str = "reference",
    front_camera: str = "front",
    wrist_camera: str = "wrist",
    gripper_open_width: float | None = None,
    binary_open_value: int | None = None,
    state_gripper_unit: str = "width_m",
) -> None:
    """Write a selected manifest split as a front/wrist LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
    from lerobot.common.datasets import video_utils

    if source_format == "oopsie":
        source_format = "reference"
    if source_format not in {"reference", "datahive"}:
        raise ValueError("source_format must be datahive, oopsie or reference")
    from datahive_adapter import read_episode, selected_frames

    reader_options = dict(
        front_camera=front_camera,
        wrist_camera=wrist_camera,
        gripper_open_width=gripper_open_width,
        binary_open_value=binary_open_value,
        state_gripper_unit=state_gripper_unit,
    )

    rows = [
        json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
    ]
    rows = [row for row in rows if row["split"] == split]
    if limit_episodes is not None:
        if limit_episodes < 1:
            raise ValueError("limit_episodes must be positive")
        rows = rows[:limit_episodes]
    if not rows:
        raise ValueError(f"manifest has no {split!r} episodes")
    if video_codec not in {"h264", "hevc", "libsvtav1"}:
        raise ValueError("video_codec must be h264, hevc, or libsvtav1")
    encode = functools.partial(
        video_utils.encode_video_frames,
        vcodec=video_codec,
        crf=23 if video_codec == "h264" else 30,
    )
    copied_videos: dict[int, dict[str, Path]] = {}

    def encode_or_copy_video(imgs_dir, video_path, fps, **kwargs):
        """Copy an untrimmed episode's source MP4; encode the frames of every other episode."""
        video_path = Path(video_path)
        match = re.fullmatch(r"episode_(\d+)\.mp4", video_path.name)
        feature = video_path.parent.name
        if match is None or feature not in {"front_image", "wrist_image"}:
            raise ValueError(f"unexpected LeRobot video path: {video_path}")
        sources = copied_videos.get(int(match.group(1)))
        if sources is None:
            return encode(imgs_dir, video_path, fps, **kwargs)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources[feature], video_path)

    original_encoder = lerobot_dataset_module.encode_video_frames
    original_compute_stats = lerobot_dataset_module.compute_episode_stats
    try:
        lerobot_dataset_module.encode_video_frames = encode_or_copy_video
        output_path = output_root.resolve() / repo_id
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(f"output exists: {output_path}")
            shutil.rmtree(output_path)

        first_path = input_root.resolve() / rows[0]["path"]
        if source_format == "datahive":
            first = read_episode(first_path, **reader_options)
            fps, width, height = first.fps, first.width, first.height
        else:
            with h5py.File(first_path, "r") as episode:
                fps = int(json.loads(episode.attrs["robot_profile"])["control_freq"])
                raw = episode["observations/video_paths/front"][()]
                video = (
                    first_path.parent
                    / Path(raw.decode() if isinstance(raw, bytes) else str(raw)).name
                )
            probe = cv2.VideoCapture(str(video))
            width, height = (
                int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            probe.release()
        image = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_path,
            robot_type="franka_research_3",
            fps=fps,
            features={
                "front_image": image,
                "wrist_image": image,
                "state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": list(STATE_NAMES),
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": list(ACTION_NAMES),
                },
            },
            image_writer_threads=8,
            image_writer_processes=0,
        )
        save_image = dataset._save_image
        copying = [False]
        dataset._save_image = (
            lambda image, path: None if copying[0] else save_image(image, path)
        )

        def compute_numeric_episode_stats(episode_buffer, features):
            """Skip image statistics; copied episodes only carry placeholder frames."""
            numeric = {
                key: value
                for key, value in features.items()
                if value["dtype"] not in {"image", "video"}
            }
            numeric_buffer = {
                key: value for key, value in episode_buffer.items() if key in numeric
            }
            return original_compute_stats(numeric_buffer, numeric)

        lerobot_dataset_module.compute_episode_stats = compute_numeric_episode_stats
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        trimmed_episodes = trimmed_frames = 0
        for episode_index, row in enumerate(rows):
            path = input_root.resolve() / row["path"]
            data = None
            if source_format == "datahive":
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["hdf5_sha256"]:
                    raise ValueError(f"HDF5 changed since manifest creation: {path}")
                data = read_episode(path, **reader_options)
                if (data.fps, data.width, data.height) != (fps, width, height):
                    raise ValueError(f"episode dimensions/fps differ: {path}")
                state, actions, start, sources = (
                    data.state,
                    data.actions,
                    data.start,
                    data.videos,
                )
                if len(state) != row["frames"]:
                    raise ValueError(f"episode changed since manifest creation: {path}")
            else:
                with h5py.File(path, "r") as episode:
                    state, actions, start = episode_arrays(episode)
                    sources = {}
                    for feature, camera in (
                        ("front_image", "front"),
                        ("wrist_image", "wrist"),
                    ):
                        raw = episode[f"observations/video_paths/{camera}"][()]
                        sources[feature] = (
                            path.parent
                            / Path(
                                raw.decode() if isinstance(raw, bytes) else str(raw)
                            ).name
                        )
            copying[0] = (
                source_format == "reference" and reuse_source_videos and start == 0
            )
            if copying[0]:
                copied_videos[episode_index] = sources
                frames = None
            elif data is not None:
                frames = {
                    feature: selected_frames(source, data.video_indices[feature])
                    for feature, source in sources.items()
                }
            else:
                frames = {
                    feature: video_frames(source, start + len(actions))[start:]
                    for feature, source in sources.items()
                }
            for index in range(len(actions)):
                dataset.add_frame(
                    {
                        "front_image": placeholder
                        if frames is None
                        else next(frames["front_image"])
                        if data is not None
                        else frames["front_image"][index],
                        "wrist_image": placeholder
                        if frames is None
                        else next(frames["wrist_image"])
                        if data is not None
                        else frames["wrist_image"][index],
                        "state": state[index],
                        "actions": actions[index],
                        "task": row["task"],
                    }
                )
            dataset.save_episode()
            trimmed_episodes += int(start > 0)
            trimmed_frames += start
            print(
                f"converted {row['episode_id']}: {len(actions)} frames, {start} idle leading frames dropped"
            )
        print(
            f"wrote {len(rows)} {split} episodes to {output_path}; dropped {trimmed_frames} idle "
            f"leading frames from {trimmed_episodes} episodes"
        )
        if source_format == "datahive":
            (output_path / "meta/hiveboard_source.json").write_text(
                json.dumps(
                    {
                        "source_format": "datahive",
                        "manifest_sha256": hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest(),
                        "reader_options": reader_options,
                        "split": split,
                    },
                    indent=2,
                )
                + "\n"
            )

    finally:
        # Keep repeated conversions independent, including failed conversions.
        lerobot_dataset_module.encode_video_frames = original_encoder
        lerobot_dataset_module.compute_episode_stats = original_compute_stats


if __name__ == "__main__":
    import tyro

    tyro.cli(convert)
