from __future__ import annotations

import json
import math
from typing import Optional, Tuple

import h5py
import numpy as np

SUCCESS_OUTCOMES = ("success", "success_suboptimal")
INVALID_TAXONOMY = "invalid_taxonomy"
INVALID_SUCCESS = "invalid_success"
HOME_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])
MIN_IDLE_FRAMES = 7


def _text(value) -> str:
    """Decode an HDF5 string attribute (str, bytes or 0-d array) to str."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def episode_outcome_with_issue(h5: h5py.File) -> Tuple[Optional[str], Optional[str]]:
    """Return (outcome, issue); issue names an unreadable label (INVALID_TAXONOMY/_SUCCESS)."""
    annotations = h5.get("episode_annotations")
    if not isinstance(annotations, h5py.Group):
        return None, None
    groups = [group for group in annotations.values() if isinstance(group, h5py.Group)]
    if not groups:
        return None, None
    latest = max(groups, key=lambda group: _text(group.attrs.get("timestamp", "")).strip())

    taxonomy = latest.attrs.get("taxonomy")
    raw = "" if taxonomy is None else _text(taxonomy).strip()
    if raw:
        try:
            outcome = json.loads(raw).get("outcome")
        except (ValueError, AttributeError):
            return None, INVALID_TAXONOMY
        if isinstance(outcome, str) and outcome:
            return outcome, None
        return None, INVALID_TAXONOMY

    if "success" not in latest.attrs:
        return None, None
    try:
        score = float(np.asarray(latest.attrs["success"]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None, INVALID_SUCCESS
    if not math.isfinite(score):
        return None, INVALID_SUCCESS
    return ("success" if score >= 0.5 else "failure"), None


def episode_outcome(h5: h5py.File) -> Optional[str]:
    """Outcome label of the latest annotation, or None when absent or unreadable."""
    return episode_outcome_with_issue(h5)[0]


def gripper_actions(
    commanded: np.ndarray, measured: np.ndarray, max_width: float = 0.08
) -> np.ndarray:
    """Replace commands logged before the first real gripper goal with the measured state.

    Until fr3_teleop sent a goal it published a guessed fully open width, so a gripper that
    started closed was logged as commanded open. Real commands are grasp (0 m) or release
    (max_width), so the substitute is max_width while the hand reads fully open and 0
    otherwise, including while it holds an object. Accepts (T,) or (T, k) arrays, including
    T == 1, and returns float32 shaped (T, k).
    """
    commanded = np.array(commanded, dtype=np.float32)
    if commanded.ndim < 2:
        commanded = commanded.reshape(-1, 1)
    if len(commanded) == 0:
        return commanded
    changes = np.flatnonzero(np.abs(np.diff(commanded[:, 0])) > 1e-6)
    first = int(changes[0]) + 1 if len(changes) else len(commanded)
    measured = np.asarray(measured, dtype=np.float32)
    opened = measured.reshape(len(measured), -1)[:first, :1] > 0.9 * max_width
    commanded[:first] = np.where(opened, max_width, 0.0)
    return commanded


def motion_start(
    targets: np.ndarray,
    gripper: np.ndarray,
    min_idle: int = MIN_IDLE_FRAMES,
    tolerance: float = 1e-4,
) -> int:
    """First frame to keep so an episode does not open with a long stationary command.

    Mirrors OpenPI's DROID idle filter, which drops runs of at least ``min_idle`` frames
    because a policy executing 8-action chunks otherwise learns to stand still: when the
    commanded pose and gripper stay unchanged for ``min_idle`` or more leading frames, only
    the last of them is kept. Returns 0 for shorter idle starts and never-moving episodes.
    """
    command = np.concatenate(
        [
            np.asarray(targets, dtype=np.float64),
            np.asarray(gripper, dtype=np.float64).reshape(len(targets), -1),
        ],
        axis=1,
    )
    changed = np.flatnonzero(np.abs(np.diff(command, axis=0)).max(axis=1) >= tolerance)
    if len(changed) == 0:
        return 0
    first_change = int(changed[0]) + 1
    return first_change - 1 if first_change >= min_idle else 0


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of broadcastable [..., 4] xyzw quaternions."""
    x1, y1, z1, w1 = np.moveaxis(left, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def _rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """Shortest-rotation axis-angle vector of [..., 4] xyzw quaternions."""
    q = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    q = np.where(q[..., 3:4] < 0, -q, q)
    norm = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(q[..., 3:4], -1.0, 1.0))
    return np.where(norm > 1e-12, q[..., :3] / np.maximum(norm, 1e-12) * angle, 0.0)


def training_chunks(states: np.ndarray, targets: np.ndarray, horizon: int) -> np.ndarray:
    """Every frame's next ``horizon`` [xyz, xyzw, gripper] targets relative to that frame's state.

    Matches what OpenPI trains on: LeRobot repeats an episode's last frame past its end, and
    ``Fr3Inputs`` expresses each chunk relative to its first state (world-frame ``dxyz``, the
    rotation vector of ``q_target * q_state^-1``, absolute gripper). Returns (T, horizon, 7).
    """
    states = np.asarray(states, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    index = np.minimum(
        np.arange(len(targets))[:, None] + np.arange(horizon)[None, :], len(targets) - 1
    )
    chunks = targets[index]
    q = states[:, None, 3:7]
    inverse = q * np.array([-1.0, -1.0, -1.0, 1.0]) / np.sum(q * q, axis=-1, keepdims=True)
    delta = _quaternion_multiply(chunks[..., 3:7], inverse)
    return np.concatenate(
        (chunks[..., :3] - states[:, None, :3], _rotation_vector(delta), chunks[..., 7:8]), axis=-1
    )


def state_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Choose the sign of [qx, qy, qz, qw] quaternions presented to a policy as state.

    The recorder keeps the largest component positive, so the sign flips whenever two
    components swap order, e.g. while turning a gate valve with the hand pointing down.
    Keeping every quaternion on the side of the gripper-down home pose instead is
    continuous within 180 degrees of it. Accepts (4,) or (T, 4) and preserves the dtype.
    """
    quaternions = np.array(quaternions)
    flip = quaternions @ HOME_QUATERNION.astype(quaternions.dtype) < 0
    return np.where(np.expand_dims(flip, -1), -quaternions, quaternions)
