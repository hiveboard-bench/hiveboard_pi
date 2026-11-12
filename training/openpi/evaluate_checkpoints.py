"""Offline checkpoint comparison on the held-out LeRobot splits.

Besides the first action of every chunk, it scores the ``--executed-horizon`` actions the
deployed bridge runs open loop, and with ``--wandb`` logs every checkpoint's metrics into
the training run under ``eval/<split>/...`` against ``eval/train_step``.
"""

import argparse, gc, json, os, re
from pathlib import Path
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies import policy_config
from openpi.policies.fr3_policy import anchored_actions
from openpi.training import config as train_config

GRIPPER_OPEN_THRESHOLD_M = 0.04


def target_chunk(dataset, idx, horizon):
    """Absolute target rows idx..idx+horizon-1, repeating the episode's last frame like training."""
    episode = int(dataset.hf_dataset[idx]["episode_index"])
    last = int(dataset.episode_data_index["to"][episode]) - 1
    rows = [min(idx + k, last) for k in range(horizon)]
    return np.stack([np.asarray(dataset.hf_dataset[row]["actions"]) for row in rows])


def chunk_errors(pred, target, executed):
    """Translation (m), rotation (rad), gripper errors and gripper-decision agreement per step."""
    err = pred[:executed, :7] - target[:executed, :7]
    return dict(
        chunk_trans=float(np.mean(np.linalg.norm(err[:, :3], axis=1))),
        chunk_rot=float(np.mean(np.linalg.norm(err[:, 3:6], axis=1))),
        chunk_grip=float(np.mean(np.abs(err[:, 6]))),
        chunk_end_trans=float(np.linalg.norm(err[-1, :3])),
        gripper_agreement=float(
            np.mean(
                (pred[:executed, 6] > GRIPPER_OPEN_THRESHOLD_M)
                == (target[:executed, 6] > GRIPPER_OPEN_THRESHOLD_M)
            )
        ),
    )


def evaluate(policy, dataset, stride, max_samples, task_filter=None, executed=8):
    sums = dict(
        se=0.0,
        ae=0.0,
        trans=0.0,
        rot=0.0,
        grip=0.0,
        chunk_trans=0.0,
        chunk_rot=0.0,
        chunk_grip=0.0,
        chunk_end_trans=0.0,
        gripper_agreement=0.0,
        n=0,
    )
    by_task = {}
    indices = list(range(0, len(dataset), stride))
    if task_filter:
        indices = [
            idx
            for idx in indices
            if dataset.meta.tasks[int(dataset[idx]["task_index"])] == task_filter
        ]
    if max_samples and len(indices) > max_samples:
        if task_filter:
            picks = np.linspace(0, len(indices) - 1, max_samples, dtype=int)
            indices = [indices[pick] for pick in picks]
        else:
            indices = np.linspace(0, len(dataset) - 1, max_samples, dtype=int).tolist()
    for idx in indices:
        s = dataset[idx]
        task = dataset.meta.tasks[int(s["task_index"])]
        obs = {
            "observation/front_image": s["front_image"].numpy(),
            "observation/wrist_image": s["wrist_image"].numpy(),
            "observation/state": s["state"].numpy(),
            "prompt": task,
        }
        with torch.inference_mode():
            pred = np.asarray(policy.infer(obs)["actions"])
        pred = pred.reshape(-1, pred.shape[-1])
        horizon = min(len(pred), executed)
        target = anchored_actions(s["state"].numpy(), target_chunk(dataset, idx, horizon))
        err = pred[0, :7] - target[0, :7]
        vals = dict(
            se=float(np.mean(err**2)),
            ae=float(np.mean(np.abs(err))),
            trans=float(np.linalg.norm(err[:3])),
            rot=float(np.linalg.norm(err[3:6])),
            grip=float(abs(err[6])),
            **chunk_errors(pred, target, horizon),
            n=1,
        )
        for k, v in vals.items():
            sums[k] += v
        b = by_task.setdefault(task, dict(se=0.0, ae=0.0, chunk_trans=0.0, chunk_rot=0.0, n=0))
        for key in ("se", "ae", "chunk_trans", "chunk_rot"):
            b[key] += vals[key]
        b["n"] += 1
    n = max(1, sums.pop("n"))
    out = {k: v / n for k, v in sums.items()}
    out["samples"] = n
    out["by_task"] = {
        k: {
            "mse": v["se"] / v["n"],
            "mae": v["ae"] / v["n"],
            "chunk_trans": v["chunk_trans"] / v["n"],
            "chunk_rot": v["chunk_rot"] / v["n"],
            "samples": v["n"],
        }
        for k, v in by_task.items()
    }
    return out


def wandb_metrics(step, split_results):
    """Flatten one checkpoint's results into eval/<split>/... keys plotted against eval/train_step."""
    row = {"eval/train_step": step}
    for split, result in split_results.items():
        for key in ("se", "trans", "rot", "grip", "chunk_rot", "chunk_grip", "gripper_agreement"):
            row[f"eval/{split}/{key}"] = result[key]
        row[f"eval/{split}/chunk_trans_cm"] = 100.0 * result["chunk_trans"]
        row[f"eval/{split}/chunk_end_trans_cm"] = 100.0 * result["chunk_end_trans"]
        for task, values in result["by_task"].items():
            name = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:48]
            row[f"eval/{split}/task/{name}/chunk_trans_cm"] = 100.0 * values["chunk_trans"]
            row[f"eval/{split}/task/{name}/mse"] = values["mse"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=600)
    ap.add_argument("--config-name", required=True)
    ap.add_argument(
        "--dataset-prefix",
        default="local/fr3_hiveboard",
        help="LeRobot repo prefix; validation/test are appended to it.",
    )
    ap.add_argument("--task-filter", default=None)
    ap.add_argument(
        "--steps",
        default="5000,10000,15000,20000,25000,29999",
        help="Comma-separated checkpoint step directories.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--executed-horizon",
        type=int,
        default=8,
        help="chunk actions scored per sample; match policy/open_loop_horizon of the bridge",
    )
    ap.add_argument(
        "--wandb",
        action="store_true",
        help="log metrics into the training run of --checkpoint-root",
    )
    args = ap.parse_args()
    cfg = train_config.get_config(args.config_name)
    run = None
    if args.wandb:
        import wandb

        run_id = (Path(args.checkpoint_root) / "wandb_id.txt").read_text().strip()
        run = wandb.init(id=run_id, resume="must", project=cfg.project_name)
        run.define_metric("eval/train_step")
        run.define_metric("eval/*", step_metric="eval/train_step")
    dsroot = os.environ["HF_LEROBOT_HOME"]
    datasets = {
        name: LeRobotDataset(
            args.dataset_prefix + "_" + name,
            root=f"{dsroot}/{args.dataset_prefix}_{name}",
        )
        for name in ("validation", "test")
    }
    results = {}
    for step_text in args.steps.split(","):
        step = int(step_text.strip())
        ck = Path(args.checkpoint_root) / str(step)
        if not ck.exists():
            print(f"skipping missing checkpoint {step}: {ck}", flush=True)
            continue
        print(f"evaluating {step}", flush=True)
        policy = policy_config.create_trained_policy(cfg, ck, pytorch_device=args.device)
        results[str(step)] = {
            split: evaluate(
                policy, ds, args.stride, args.max_samples, args.task_filter, args.executed_horizon
            )
            for split, ds in datasets.items()
        }
        if run is not None:
            run.log(wandb_metrics(step, results[str(step)]))
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
