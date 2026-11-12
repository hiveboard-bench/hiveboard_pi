"""Integration fixtures are written by DataHive's actual public EpisodeWriter API."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/openpi/data"))
from datahive_adapter import poses, read_episode
from build_datahive_manifest import build
from convert_hiveboard_to_lerobot import convert

datahive = pytest.importorskip(
    "datahive",
    reason="Set PYTHONPATH=/path/to/DataHive/src to test the actual collection API",
)
from datahive.episode import EpisodeWriter
from datahive.profile import RobotProfile
from datahive.schema import TrialAnnotation
from datahive.trials import upsert_row


@pytest.fixture
def make_episode(tmp_path):
    root = tmp_path / "samples"

    def create(number=0, *, binary=False, outcome="success", legacy=False):
        profile = RobotProfile(
            robot_name="synthetic_fr3",
            gripper_name="synthetic_gripper",
            platform_id="synthetic_rig",
            manipulator={
                "model": "synthetic_fr3",
                "joint_names": [f"j{i}" for i in range(7)],
            },
            end_effector={
                "type": "gripper",
                "actuated_dof": 1,
                "command_modality": "binary" if binary else "position",
            },
            low_level={"mode": "stock", "controller_type": "cartesian_impedance"},
            control_mode="cartesian_impedance",
            control_freq=100,
            is_biarm=False,
            uses_mobile_base=False,
            action_space=[
                "cartesian_position",
                "gripper_binary" if binary else "gripper_position",
            ],
            orientation_representation="quat",
            robot_state_orientation_representation="quat",
            cameras=[{"name": "external", "position": "front"}, {"name": "wrist"}],
            units_and_frames={"ee_pose": "position_m_quaternion_xyzw_in_base_frame"},
        )
        writer = EpisodeWriter(
            root,
            "synthetic_session",
            f"episode{number}",
            trial_id=str(number),
            task_ids=["valve_ball"],
            lab_id="synthetic_lab",
            profile=profile,
        )
        for i in range(101):
            t = i / 100
            writer.append_proprioception(
                timestamp=100 + t,
                ee_pose=[0.4 + t * 0.01, 0, 0.2, 1, 0, 0, 0],
                ee_state=[0.06],
                provenance={"ee_pose": "measured", "ee_state": "measured"},
            )
        for i in range(21):
            writer.append_command(
                timestamp=100 + i / 20,
                target={
                    "cartesian_position": [0.5 + i * 0.001, 0, 0.2, 1, 0, 0, 0],
                    "gripper_binary" if binary else "gripper_position": [
                        1 if binary else 0.07
                    ],
                },
                control_mode="cartesian_impedance",
            )
        for name in ("external", "wrist"):
            video = tmp_path / f"{number}-{name}.mp4"
            out = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32)
            )
            assert out.isOpened()
            for i in range(30):
                out.write(np.full((32, 32, 3), i * 5, dtype=np.uint8))
            out.release()
            writer.attach_video(name, video)
        path = writer.paths.h5
        writer.close()
        csv_path = path.parent.parent / "trials_synthetic_session.csv"
        upsert_row(
            csv_path,
            TrialAnnotation(
                trial_id=str(number),
                lab_id="synthetic_lab",
                platform_id="synthetic_rig",
                attachment_id="valve_ball",
                date="2026-11-12",
                outcome=outcome,
                strategy="prehensile",
                completion_time_s=1.0 if outcome == "success" else None,
                failure_cause=None if outcome == "success" else "slip",
            ),
        )
        if legacy:
            with h5py.File(path, "r+") as episode:
                episode.attrs["schema_version"] = "datahive_episode_v0"
                actions = episode["actions"]
                targets = np.concatenate(
                    [actions["cartesian_position"][:], actions["gripper_position"][:]],
                    axis=1,
                )
                del actions["cartesian_position"], actions["gripper_position"]
                actions.create_dataset("target", data=targets)
                episode.move("actions", "commands")
                episode.move("observations/robot_states", "proprioception")
        return root, path

    return create


OPTIONS = {"front_camera": "external", "wrist_camera": "wrist"}


def test_timestamp_alignment_and_numeric_convention(make_episode):
    _, path = make_episode()
    data = read_episode(path, **OPTIONS)
    assert data.fps == 10 and data.state.shape == data.actions.shape == (10, 8)
    np.testing.assert_allclose(data.state[:, 0], 0.4 + np.arange(10) * 0.001, atol=1e-6)
    np.testing.assert_allclose(
        data.actions[:, 0], 0.5 + np.arange(10) * 0.002, atol=1e-6
    )
    np.testing.assert_allclose(data.actions[:, 7], 0.07)
    np.testing.assert_array_equal(data.video_indices["front_image"], np.arange(10) * 3)
    np.testing.assert_allclose(data.state[:, 3:7], np.tile([1, 0, 0, 0], (10, 1)))


def test_binary_gripper_requires_explicit_meaning(make_episode):
    _, path = make_episode(binary=True)
    with pytest.raises(ValueError, match="binary_open_value"):
        read_episode(path, **OPTIONS)
    data = read_episode(path, **OPTIONS, binary_open_value=1, gripper_open_width=0.08)
    np.testing.assert_allclose(data.actions[:, 7], 0.08)
    inverted = read_episode(
        path, **OPTIONS, binary_open_value=0, gripper_open_width=0.08
    )
    np.testing.assert_array_equal(inverted.actions[:, 7], 0)


@pytest.mark.parametrize("width", [float("nan"), float("inf"), 0, -0.01])
def test_invalid_gripper_open_width_is_rejected(make_episode, width):
    _, path = make_episode(binary=True)
    with pytest.raises(ValueError, match="finite and positive"):
        read_episode(path, **OPTIONS, binary_open_value=1, gripper_open_width=width)


def test_fractional_gripper_state_uses_declared_open_width(make_episode):
    _, path = make_episode()
    with h5py.File(path, "r+") as episode:
        episode["observations/robot_states/ee_state"][:] = 0.5
        episode.attrs["units_and_frames"] = json.dumps({"ee_state": "opening_fraction"})
    data = read_episode(
        path, **OPTIONS, state_gripper_unit="opening_fraction", gripper_open_width=0.08
    )
    np.testing.assert_allclose(data.state[:, 7], 0.04)
    np.testing.assert_allclose(data.actions[:, 7], 0.07)


@pytest.mark.parametrize(
    "representation,orientation",
    [
        ("rotvec", [0, 0, np.pi / 2]),
        ("euler_xyz", [0, 0, np.pi / 2]),
        ("matrix", [0, -1, 0, 1, 0, 0, 0, 0, 1]),
    ],
)
def test_orientation_conversion_to_xyzw(representation, orientation):
    result = poses(np.array([[0.4, 0, 0.2, *orientation]]), representation, "synthetic")
    np.testing.assert_allclose(
        result[0, 3:], [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-8
    )


def test_legacy_flat_actions_and_groups(make_episode):
    _, path = make_episode(legacy=True)
    np.testing.assert_allclose(
        read_episode(path, **OPTIONS).actions[:, 0],
        0.5 + np.arange(10) * 0.002,
        atol=1e-6,
    )


def test_rejects_joint_targets_and_timestamp_gaps(make_episode):
    _, path = make_episode()
    with h5py.File(path, "r+") as episode:
        episode.attrs["action_space"] = json.dumps(["joint_position", "gripper_binary"])
    with pytest.raises(ValueError, match="lacks cartesian_position"):
        read_episode(path, **OPTIONS)
    with h5py.File(path, "r+") as episode:
        episode.attrs["action_space"] = json.dumps(
            ["cartesian_position", "gripper_position"]
        )
        ts = episode["observations/robot_states/timestamp"]
        ts[50:] += 1
    with pytest.raises(ValueError, match="timestamp gaps"):
        read_episode(path, **OPTIONS)


def test_missing_annotation_is_reported_without_editing_source(make_episode, tmp_path):
    root, path = make_episode()
    before = path.read_bytes()
    (path.parent.parent / "trials_synthetic_session.csv").unlink()
    manifest = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="no compatible episodes"):
        build(root, manifest, **OPTIONS)
    report = json.loads(manifest.with_suffix(".report.json").read_text())
    assert report["excluded"][0]["reason"] == "session annotation CSV not found"
    assert path.read_bytes() == before


def test_invalid_quaternion_and_relative_units_are_rejected(make_episode):
    _, path = make_episode()
    with h5py.File(path, "r+") as episode:
        pose = episode["observations/robot_states/ee_pose"]
        pose[0, 3:] = 0
    with pytest.raises(ValueError, match="invalid xyzw quaternion"):
        read_episode(path, **OPTIONS)
    with h5py.File(path, "r+") as episode:
        episode.attrs["units_and_frames"] = json.dumps(
            {"cartesian_position": "delta_tool_frame"}
        )
    with pytest.raises(ValueError, match="no implicit deltas"):
        read_episode(path, **OPTIONS)


def test_manifest_split_and_exclusion_report(make_episode, tmp_path):
    for number in range(3):
        root, _ = make_episode(number)
    make_episode(3, outcome="fail")
    manifest = tmp_path / "manifest.jsonl"
    report = build(root, manifest, **OPTIONS)
    assert report["selected"] == 3 and len(report["excluded"]) == 1
    assert all(
        value == {"episodes": 1, "frames": 10} for value in report["splits"].values()
    )
    before = manifest.read_bytes()
    assert build(root, manifest, **OPTIONS) == report
    assert manifest.read_bytes() == before
    make_episode(4)
    with pytest.raises(ValueError, match="existing manifest differs"):
        build(root, manifest, **OPTIONS)


def test_full_lerobot_conversion_and_reference_config(
    make_episode, tmp_path, monkeypatch
):
    for number in range(3):
        root, _ = make_episode(number)
    manifest = tmp_path / "manifest.jsonl"
    build(root, manifest, **OPTIONS)
    from lerobot.common.datasets import lerobot_dataset as module

    old_encode, old_stats = module.encode_video_frames, module.compute_episode_stats
    try:
        for split in ("train", "validation", "test"):
            convert(
                manifest,
                tmp_path / "lerobot",
                input_root=root,
                repo_id=f"local/fr3_datahive_{split}",
                split=split,
                source_format="datahive",
                **OPTIONS,
            )
            assert module.encode_video_frames is old_encode
            assert module.compute_episode_stats is old_stats
            with pytest.raises(FileExistsError):
                convert(
                    manifest,
                    tmp_path / "lerobot",
                    input_root=root,
                    repo_id=f"local/fr3_datahive_{split}",
                    split=split,
                    source_format="datahive",
                    **OPTIONS,
                )
            assert module.encode_video_frames is old_encode
            assert module.compute_episode_stats is old_stats
            folder = tmp_path / "lerobot" / f"local/fr3_datahive_{split}"
            info = json.loads((folder / "meta/info.json").read_text())
            assert (info["total_episodes"], info["total_frames"], info["fps"]) == (
                1,
                10,
                10,
            )
            for feature in ("front_image", "wrist_image"):
                video = next(folder.rglob(f"{feature}/episode_*.mp4"))
                capture = cv2.VideoCapture(str(video))
                assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
                capture.release()
            import pyarrow.parquet as pq

            table = pq.read_table(next(folder.rglob("*.parquet")))
            np.testing.assert_allclose(
                np.array(table["actions"].to_pylist())[:, 0],
                0.5 + np.arange(10) * 0.002,
                atol=1e-6,
            )
    finally:
        module.encode_video_frames, module.compute_episode_stats = old_encode, old_stats
    monkeypatch.setenv("MANIFEST", str(manifest))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lerobot"))
    monkeypatch.setattr(
        sys, "argv", ["verify_run.py", "--datasets", "--source", "datahive"]
    )
    spec = importlib.util.spec_from_file_location(
        "verify_run", ROOT / "scripts/verify_run.py"
    )
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    verify.main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_run.py", "--datasets", "--source", "datahive", "--training-ready"],
    )
    with pytest.raises(SystemExit, match="batch 32"):
        verify.main()
    monkeypatch.setattr(
        sys, "argv", ["verify_run.py", "--datasets", "--source", "datahive"]
    )
    source_file = (
        tmp_path / "lerobot/local/fr3_datahive_train/meta/hiveboard_source.json"
    )
    source = json.loads(source_file.read_text())
    source["reader_options"]["front_camera"] = "incorrect"
    source_file.write_text(json.dumps(source))
    with pytest.raises(
        SystemExit, match="different manifest or camera/gripper mapping"
    ):
        verify.main()
