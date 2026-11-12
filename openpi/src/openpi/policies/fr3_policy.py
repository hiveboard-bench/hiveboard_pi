"""OpenPI transforms for the Hiveboard FR3 Cartesian dataset and runtime."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of [..., 4] xyzw quaternions."""
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


def anchored_actions(state: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Express absolute [xyz, xyzw, gripper] targets relative to the observation's pose.

    Like OpenPI's ``DeltaActions``, every action of the chunk is relative to the state the
    chunk was predicted from: world-frame ``dxyz`` and the rotation vector of
    ``q_target * q_state^-1``; the gripper width stays absolute. The deployed bridge
    inverts this against the pose it observed when requesting the chunk.
    """
    state = np.asarray(state, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    inverse = state[3:7] * np.array([-1.0, -1.0, -1.0, 1.0]) / np.dot(state[3:7], state[3:7])
    delta = _quaternion_multiply(
        targets[..., 3:7], np.broadcast_to(inverse, targets[..., 3:7].shape)
    )
    return np.concatenate(
        (targets[..., :3] - state[:3], _rotation_vector(delta), targets[..., 7:8]), axis=-1
    ).astype(np.float32)


def _image(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255, 0, 255).astype(np.uint8)
    if value.ndim != 3:
        raise ValueError(f"expected an HWC or CHW image, got {value.shape}")
    if value.shape[0] == 3:
        value = einops.rearrange(value, "c h w -> h w c")
    return value


@dataclasses.dataclass(frozen=True)
class Fr3Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front = _image(data["observation/front_image"])
        wrist = _image(data["observation/wrist_image"])
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = anchored_actions(inputs["state"], data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Fr3Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}
