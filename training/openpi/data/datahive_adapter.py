"""Read DataHive v0/v1 Cartesian episodes without modifying the source files.

This adapter targets the existing FR3 policy: absolute base-frame Cartesian poses,
one measured gripper width and two RGB cameras. Joint/velocity-only actions are
deliberately rejected: they do not contain the targets this policy learns.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

READER_DEFAULTS = {
    "front_camera": "front",
    "wrist_camera": "wrist",
    "gripper_open_width": None,
    "binary_open_value": None,
    "state_gripper_unit": "width_m",
}


def attr(episode: h5py.File, name: str, default=None):
    value = episode.attrs.get(name, default)
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str) and value.startswith(("[", "{")):
        return json.loads(value)
    return value


def group(episode: h5py.File, canonical: str, legacy: str) -> h5py.Group:
    name = canonical if canonical in episode else legacy
    if name not in episode:
        raise ValueError(f"missing group: {canonical}")
    return episode[name]


def timestamps(stream: h5py.Group) -> np.ndarray:
    if "timestamp" not in stream:
        raise ValueError(f"{stream.name}: missing timestamp")
    values = np.asarray(stream["timestamp"], dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(
            f"{stream.name}: timestamps must be finite and strictly increasing"
        )
    dt = np.diff(values)
    if dt.max() > 3.1 * np.median(dt):
        raise ValueError(f"{stream.name}: timestamp gaps exceed 3 sampling periods")
    return values


def values(stream: h5py.Group, name: str, length: int) -> np.ndarray:
    if name not in stream:
        raise ValueError(f"missing field: {stream.name}/{name}")
    result = np.asarray(stream[name], dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or len(result) != length or not np.isfinite(result).all():
        raise ValueError(
            f"{stream.name}/{name}: invalid shape/length or non-finite values"
        )
    return result


def poses(value: np.ndarray, representation: str, label: str) -> np.ndarray:
    widths = {"quat": 4, "rotvec": 3, "matrix": 9}
    width = 3 if representation.startswith("euler_") else widths.get(representation)
    if width is None or value.shape[1] != 3 + width:
        raise ValueError(
            f"{label}: unsupported representation {representation!r} or dimensions"
        )
    orientation = value[:, 3:]
    if representation == "quat":
        norm = np.linalg.norm(orientation, axis=1)
        if np.any(np.abs(norm - 1) > 0.05):
            raise ValueError(
                f"{label}: invalid xyzw quaternion (norm must be close to 1)"
            )
        rotation = Rotation.from_quat(orientation)
    elif representation == "rotvec":
        rotation = Rotation.from_rotvec(orientation)
    elif representation == "matrix":
        matrix = orientation.reshape(-1, 3, 3)
        if not np.allclose(
            matrix @ matrix.transpose(0, 2, 1), np.eye(3), atol=1e-4
        ) or not np.allclose(np.linalg.det(matrix), 1, atol=1e-4):
            raise ValueError(f"{label}: invalid rotation matrix")
        rotation = Rotation.from_matrix(matrix)
    else:
        order = representation.removeprefix("euler_")
        if len(order) != 3 or any(c not in "xyzXYZ" for c in order):
            raise ValueError(f"{label}: invalid Euler sequence")
        rotation = Rotation.from_euler(order, orientation)
    return np.concatenate([value[:, :3], rotation.as_quat()], axis=1)


def video_info(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"missing or unreadable video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count, width, height = [
            int(capture.get(key))
            for key in (
                cv2.CAP_PROP_FRAME_COUNT,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
            )
        ]
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0 or min(count, width, height) <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return fps, count, width, height


@dataclass
class EpisodeData:
    state: np.ndarray
    actions: np.ndarray
    videos: dict[str, Path]
    video_indices: dict[str, np.ndarray]
    start: int
    fps: int
    width: int
    height: int


def read_episode(
    path: Path,
    *,
    front_camera: str = "front",
    wrist_camera: str = "wrist",
    target_fps: int = 10,
    gripper_open_width: float | None = None,
    binary_open_value: int | None = None,
    state_gripper_unit: str = "width_m",
) -> EpisodeData:
    # Import the numeric helpers shared with the historical converter.
    import importlib.util
    import sys

    shared = sys.modules.get("hiveboard_episode")
    if shared is None:
        spec = importlib.util.spec_from_file_location(
            "hiveboard_episode",
            Path(__file__).resolve().parents[3] / "tools/hiveboard_episode.py",
        )
        shared = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = shared
        spec.loader.exec_module(shared)
    if target_fps != 10:
        raise ValueError("this FR3 policy requires target_fps=10")
    if gripper_open_width is not None and (
        not np.isfinite(gripper_open_width) or gripper_open_width <= 0
    ):
        raise ValueError("gripper_open_width must be finite and positive")
    if front_camera == wrist_camera:
        raise ValueError(
            "front_camera and wrist_camera must identify two different cameras"
        )
    with h5py.File(path, "r") as episode:
        if attr(episode, "schema_version") not in {
            "datahive_episode_v0",
            "datahive_episode_v1",
        }:
            raise ValueError(
                "schema_version must be datahive_episode_v0 or datahive_episode_v1"
            )
        if attr(episode, "is_biarm", False) or attr(episode, "uses_mobile_base", False):
            raise ValueError("the FR3 policy requires a single arm and a fixed base")
        space = attr(episode, "action_space", [])
        if "cartesian_position" not in space:
            raise ValueError(
                "action_space lacks cartesian_position: joint/velocity commands are not absolute Cartesian targets"
            )
        units = attr(episode, "units_and_frames", {})
        # DataHive's standard ee_pose convention is xyz in meters + xyzw in the base frame.
        pose_units = units.get("ee_pose", "position_m_quaternion_xyzw_in_base_frame")
        if pose_units != "position_m_quaternion_xyzw_in_base_frame":
            raise ValueError(
                "units_and_frames.ee_pose must declare position_m_quaternion_xyzw_in_base_frame"
            )
        for key in ("cartesian_position", "ee_state", "gripper_position"):
            allowed = {"m", "meters", "width_m"}
            if key == "ee_state" and state_gripper_unit == "opening_fraction":
                allowed = {"opening_fraction", "fraction", "1"}
            if key in units and units[key] not in allowed:
                raise ValueError(
                    f"units_and_frames.{key}: requires meters (no implicit deltas or normalization)"
                )
        state_stream = group(episode, "observations/robot_states", "proprioception")
        action_stream = group(episode, "actions", "commands")
        ts, ta = timestamps(state_stream), timestamps(action_stream)
        state_pose = poses(
            values(state_stream, "ee_pose", len(ts)),
            str(attr(episode, "robot_state_orientation_representation", "")),
            "ee_pose",
        )
        # Older DataHive writers stored a single actions/target vector.
        if "cartesian_position" in action_stream:
            target_pose = values(action_stream, "cartesian_position", len(ta))
            grip_name = (
                "gripper_position" if "gripper_position" in space else "gripper_binary"
            )
            commanded = values(action_stream, grip_name, len(ta))
        elif "target" in action_stream:
            representation = str(attr(episode, "orientation_representation", ""))
            width = {"quat": 7, "rotvec": 6, "matrix": 12}.get(representation)
            if width is None and representation.startswith("euler_"):
                width = 6
            grip_name = (
                "gripper_position" if "gripper_position" in space else "gripper_binary"
            )
            if space != ["cartesian_position", grip_name] or width is None:
                raise ValueError(
                    "legacy actions/target requires action_space=[cartesian_position, gripper_position|gripper_binary]"
                )
            target = values(action_stream, "target", len(ta))
            if target.shape[1] != width + 1:
                raise ValueError("legacy actions/target has incompatible dimensions")
            target_pose, commanded = target[:, :width], target[:, width:]
        else:
            raise ValueError("Cartesian actions not found")
        if set(space) != {"cartesian_position", grip_name}:
            raise ValueError(
                "action_space must contain only cartesian_position and a position/binary gripper command"
            )
        target_pose = poses(
            target_pose,
            str(attr(episode, "orientation_representation", "")),
            "cartesian_position",
        )
        measured_name = (
            "gripper_position" if "gripper_position" in state_stream else "ee_state"
        )
        measured = values(state_stream, measured_name, len(ts))
        if measured.shape[1] != 1 or commanded.shape[1] != 1:
            raise ValueError(
                "the policy requires one measured and one commanded gripper width"
            )
        if state_gripper_unit == "opening_fraction":
            if (
                gripper_open_width is None
                or gripper_open_width <= 0
                or np.any((measured < 0) | (measured > 1))
            ):
                raise ValueError(
                    "opening_fraction requires values in [0,1] and a positive gripper_open_width"
                )
            measured *= gripper_open_width
        elif state_gripper_unit != "width_m":
            raise ValueError("state_gripper_unit must be width_m or opening_fraction")
        if grip_name == "gripper_binary":
            if (
                binary_open_value not in (0, 1)
                or gripper_open_width is None
                or gripper_open_width <= 0
            ):
                raise ValueError(
                    "gripper_binary requires binary_open_value=0|1 and gripper_open_width in meters"
                )
            if not np.isin(commanded, [0, 1]).all():
                raise ValueError("gripper_binary must contain only 0/1")
            commanded = np.where(
                commanded == binary_open_value, gripper_open_width, 0.0
            )
        if np.any(measured < -1e-3) or np.any(commanded < -1e-3):
            raise ValueError("negative gripper widths")
        videos = {}
        cameras = {c["name"]: c for c in attr(episode, "cameras", [])}
        for feature, name in (
            ("front_image", front_camera),
            ("wrist_image", wrist_camera),
        ):
            key = f"observations/video_paths/{name}"
            raw = (
                episode[key][()]
                if key in episode
                else cameras.get(name, {}).get("file")
            )
            if raw is None:
                raise ValueError(f"missing video reference for camera {name!r}")
            filename = raw.decode() if isinstance(raw, bytes) else str(raw)
            videos[feature] = path.parent / Path(filename).name

    infos = {key: video_info(value) for key, value in videos.items()}
    if len({info[2:] for info in infos.values()}) != 1:
        raise ValueError("cameras must have the same resolution")
    origin = ts[0]
    for fps, count, _, _ in infos.values():
        if fps < target_fps - 0.01:
            raise ValueError("video frame rate is below 10 Hz")
        # Videos have no per-frame clock in DataHive; frame zero must coincide with state zero.
        if abs(count / fps - (ts[-1] - origin)) > max(
            2 / fps, 3 * np.median(np.diff(ts))
        ):
            raise ValueError(
                "video duration does not match the states; synchronize camera start times during collection"
            )
    if (
        np.median(np.diff(ts)) > 1 / target_fps + 1e-4
        or np.median(np.diff(ta)) > 1 / target_fps + 1e-4
    ):
        raise ValueError("states and commands must be recorded at least at 10 Hz")
    begin = max(ts[0], ta[0])
    end = min(
        ts[-1], ta[-1], *(origin + (info[1] - 1) / info[0] for info in infos.values())
    )
    grid = (
        begin
        + np.arange(max(0, int(np.floor((end - begin) * target_fps + 1e-7)) + 1))
        / target_fps
    )
    if len(grid) < 2:
        raise ValueError(
            "insufficient time overlap between states, commands and videos"
        )
    # Last known state/command: no future command is paired with an earlier observation.
    si = np.searchsorted(ts, grid + 1e-8, side="right") - 1
    ai = np.searchsorted(ta, grid + 1e-8, side="right") - 1
    state = np.concatenate([state_pose[si], measured[si]], axis=1).astype(np.float32)
    state[:, 3:7] = shared.state_quaternions(state[:, 3:7])
    actions = np.concatenate([target_pose[ai], commanded[ai]], axis=1).astype(
        np.float32
    )
    start = shared.motion_start(actions[:, :7], actions[:, 7:])
    indices = {
        key: np.rint((grid - origin) * info[0]).astype(int)[start:]
        for key, info in infos.items()
    }
    width, height = next(iter(infos.values()))[2:]
    return EpisodeData(
        state[start:],
        actions[start:],
        videos,
        indices,
        start,
        target_fps,
        width,
        height,
    )


def selected_frames(path: Path, indices: np.ndarray):
    """Stream only selected RGB frames; avoid loading an entire video into memory."""
    capture = cv2.VideoCapture(str(path))
    position = -1
    frame = None
    try:
        for index in indices:
            while position < index:
                ok, frame = capture.read()
                position += 1
                if not ok:
                    raise ValueError(f"video ended before frame {index}: {path}")
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
