#!/usr/bin/env python3
"""Validate the linear artifact against the exact deployment URDF geometry.

Unlike the lightweight fitter report, this audit constructs the configured
WujiHandModel and MANO model.  It therefore measures the real URDF-limit and
geometry-cycle gates used by ``Mano45RuntimeCodec``.  HDF5 inputs are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


try:  # Allows the source to be streamed after the production module for audits.
    Mano45LinearInverse
except NameError:  # pragma: no cover - normal installed execution path
    policy_dir = Path(__file__).resolve().parents[1]
    workspace_parent = policy_dir.parents[2]
    for path in (workspace_parent, policy_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from XPolicyLab.policy.VITRA.mano45_linear_inverse import (
        LinearInverseSafetyError,
        Mano45LinearInverse,
        sha256_file,
    )


TASKS = (
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "insert_block",
    "retrieve_gap",
    "stack_bowls",
)
SIDES = ("left", "right")


def parse_range(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            values.extend(range(start, end + 1))
        elif part:
            values.append(int(part))
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--spark-root", required=True, type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--test-ids", type=parse_range, default=parse_range("90-99"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    return parser.parse_args()


def episode_path(root: Path, task: str, episode_id: int) -> Path:
    name = f"episode_{episode_id:07d}.hdf5"
    for candidate in (
        root / task / name,
        root / task / "tianji_marvin_wuji" / "data" / name,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing {task}/{name}")


def summary(values: list[float], scale: float = 1.0) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64) * scale
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifact(args: argparse.Namespace) -> Any:
    embedded = globals().get("_EMBEDDED_ARTIFACT_JSON")
    if embedded is not None:
        text = str(embedded)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if args.artifact_sha256 and digest != args.artifact_sha256:
            raise RuntimeError("embedded artifact hash mismatch")
        return Mano45LinearInverse(
            json.loads(text),
            artifact_path=Path("<embedded-read-only-artifact>"),
            artifact_sha256=digest,
        )
    if args.artifact is None or args.artifact_sha256 is None:
        raise ValueError("--artifact and --artifact-sha256 are required")
    return Mano45LinearInverse.load(
        args.artifact, expected_sha256=args.artifact_sha256
    )


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1 or args.max_frames_per_episode < 0:
        raise ValueError("frame stride must be positive and max frames non-negative")
    artifact = load_artifact(args)

    requested = args.spark_root.expanduser().resolve()
    candidates = ((requested, requested / "Spark_0"), (requested.parent, requested))
    for repo_root, package_root in candidates:
        if (package_root / "tools/data_alignment/targets/wuji.py").is_file():
            break
    else:
        raise FileNotFoundError(f"invalid Spark-0 root: {requested}")
    sys.path[:0] = [str(package_root), str(repo_root)]
    from tools.data_alignment.mano_wuji_map import mano_joints21_to_add_mano_chains
    from tools.data_alignment.targets.mano import (
        ManoFitter,
        ManoModel,
        default_mano_asset,
        forward_hand_wrist_anchored,
    )
    from tools.data_alignment.targets.wuji import WujiHandModel
    from tools.data_alignment.transforms import canonicalize_hand_points
    from data_trasfomer.to_spark_world_frame import HAND_MOUNT_R, HAND_MOUNT_T

    mano_model = ManoModel(default_mano_asset())
    mano_fitter = ManoFitter(mano_model)
    hand_models = {
        side: WujiHandModel(side, variant="wujihand2") for side in SIDES
    }
    records: dict[str, dict[str, Any]] = {
        side: {
            "attempted": 0,
            "linear_accepted": 0,
            "geometry_accepted": 0,
            "urdf_clipped": 0,
            "fallback_reasons": Counter(),
            "geometry_rms": [],
            "q_abs": [],
            "link7_rotation_error": [],
            "link7_position_error": [],
        }
        for side in SIDES
    }

    for task in TASKS:
        for episode_id in args.test_ids:
            path = episode_path(args.dataset_root, task, episode_id)
            with h5py.File(path, "r") as handle:
                for side in SIDES:
                    pose = np.asarray(
                        handle[f"mano/action/{side}_ee_joint_states"], dtype=np.float64
                    )
                    betas = np.asarray(
                        handle[f"mano/action/{side}_hand_betas"], dtype=np.float64
                    )
                    q_true = np.asarray(
                        handle[f"action/{side}_ee_joint_states"], dtype=np.float64
                    )
                    root_pose = np.asarray(
                        handle[f"mano/action/{side}_ee_poses"], dtype=np.float64
                    )
                    link7_pose = np.asarray(
                        handle[f"action/{side}_ee_poses"], dtype=np.float64
                    )
                    previous_q = np.asarray(
                        handle[f"state/{side}_ee_joint_states"], dtype=np.float64
                    )
                    valid = np.asarray(
                        handle[f"mano/validity/action_{side}_ee_joint_states"],
                        dtype=bool,
                    ).all(axis=1)
                    indices = np.flatnonzero(
                        valid
                        & np.isfinite(pose).all(axis=1)
                        & np.isfinite(betas).all(axis=1)
                        & np.isfinite(q_true).all(axis=1)
                        & np.isfinite(root_pose).all(axis=1)
                        & np.isfinite(link7_pose).all(axis=1)
                    )[:: args.frame_stride]
                    if args.max_frames_per_episode:
                        indices = indices[: args.max_frames_per_episode]
                    hand = hand_models[side]
                    record = records[side]
                    for index in indices.tolist():
                        record["attempted"] += 1
                        try:
                            prediction = artifact.predict(
                                side=side,
                                hand_pose_rotvec=pose[index],
                                previous_q_stage_major=previous_q[index],
                                lower=hand.lower,
                                upper=hand.upper,
                            )
                        except LinearInverseSafetyError as exc:
                            record["fallback_reasons"][str(exc).split(" (")[0]] += 1
                            continue
                        record["linear_accepted"] += 1
                        record["urdf_clipped"] += int(prediction.clipped_to_urdf)
                        q = prediction.q_stage_major
                        joints21 = forward_hand_wrist_anchored(
                            mano_model,
                            is_right=(side == "right"),
                            global_orient=np.zeros((1, 3), dtype=np.float64),
                            hand_pose=pose[index].reshape(1, 45),
                            betas=betas[index].reshape(1, 10),
                            wrist=np.zeros((1, 3), dtype=np.float64),
                            left_convention="mirror_model",
                        )[0]
                        points, weights = mano_joints21_to_add_mano_chains(
                            joints21, model=hand
                        )
                        local, ok = canonicalize_hand_points(
                            points, joints21[0], weights, knuckle_index=1
                        )
                        if not ok:
                            record["fallback_reasons"]["canonicalization"] += 1
                            continue
                        geometry_valid, rms, _distance = hand.evaluate(
                            q, local, weights
                        )
                        record["geometry_rms"].append(float(rms))
                        if (
                            not geometry_valid
                            or not np.isfinite(rms)
                            or rms > artifact.geometry_limit_m(side)
                        ):
                            record["fallback_reasons"]["geometry_cycle"] += 1
                            continue
                        record["geometry_accepted"] += 1
                        record["q_abs"].extend(np.abs(q - q_true[index]).tolist())
                        points_q = np.asarray(
                            hand.forward_points(q, canonical=False), dtype=np.float64
                        )
                        points_q = (
                            points_q @ np.asarray(HAND_MOUNT_R[side]).T
                            + np.asarray(HAND_MOUNT_T[side])
                        )
                        mirror_x = np.diag([-1.0, 1.0, 1.0])
                        if side == "left":
                            points_q = points_q.copy()
                            points_q[..., 0] *= -1.0
                        palm = mano_fitter._palm_frame(
                            points_q[1, 1],
                            points_q[4, 1],
                            points_q[2, 1],
                            points_q[0, 0],
                        )
                        if palm is None:
                            raise RuntimeError(f"degenerate {side} palm frame")
                        root_in_link7 = palm @ np.asarray(mano_fitter.rest_palm).T
                        if side == "left":
                            root_in_link7 = mirror_x @ root_in_link7 @ mirror_x
                        root_quat = root_pose[index, [4, 5, 6, 3]]
                        true_quat = link7_pose[index, [4, 5, 6, 3]]
                        root_rotation = Rotation.from_quat(
                            root_quat / np.linalg.norm(root_quat)
                        ).as_matrix()
                        true_rotation = Rotation.from_quat(
                            true_quat / np.linalg.norm(true_quat)
                        ).as_matrix()
                        predicted_rotation = root_rotation @ root_in_link7.T
                        rotation_error = Rotation.from_matrix(
                            predicted_rotation @ true_rotation.T
                        ).magnitude()
                        record["link7_rotation_error"].append(float(rotation_error))
                        record["link7_position_error"].append(
                            float(
                                np.linalg.norm(
                                    root_pose[index, :3] - link7_pose[index, :3]
                                )
                            )
                        )

    report: dict[str, Any] = {
        "artifact_sha256": artifact.artifact_sha256,
        "episode_split": f"{min(args.test_ids)}-{max(args.test_ids)}",
        "frame_stride": args.frame_stride,
        "max_frames_per_episode": args.max_frames_per_episode,
        "urdf_sha256": {
            side: sha256(Path(hand_models[side].urdf_path).resolve()) for side in SIDES
        },
        "sides": {},
    }
    for side in SIDES:
        record = records[side]
        attempted = record["attempted"]
        report["sides"][side] = {
            "attempted": attempted,
            "linear_gate_accept_fraction": record["linear_accepted"] / attempted,
            "full_runtime_accept_fraction": record["geometry_accepted"] / attempted,
            "urdf_clipped": record["urdf_clipped"],
            "fallback_reasons": dict(record["fallback_reasons"]),
            "geometry_rms_mm": summary(record["geometry_rms"], scale=1000.0),
            "accepted_q_abs_rad": summary(record["q_abs"]),
            "link7_rotation_error_deg": summary(
                record["link7_rotation_error"], scale=180.0 / np.pi
            ),
            "link7_position_error_m": summary(record["link7_position_error"]),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
