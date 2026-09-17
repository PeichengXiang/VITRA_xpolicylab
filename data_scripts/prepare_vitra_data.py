#!/usr/bin/env python3
"""Prepare Spark0 for VITRA without copying or rewriting source episodes.

The resulting layout is::

    <data-root>/TeleData/<task>/episode_*.hdf5 -> <source episode>
    <data-root>/spark0_manifest.json

Every episode is linked individually so the prepared set is auditable and can
be validated without mutating ``/mnt/xspark-data/tjy/spark0_bench``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np


WUJI20_REQUIRED = {
    "state/left_ee_poses": 7,
    "state/right_ee_poses": 7,
    "state/left_ee_joint_states": 20,
    "state/right_ee_joint_states": 20,
    "action/left_ee_poses": 7,
    "action/right_ee_poses": 7,
    "action/left_ee_joint_states": 20,
    "action/right_ee_joint_states": 20,
}
MANO45_REQUIRED = {
    f"mano/{group}/{side}_{leaf}": dimension
    for group in ("state", "action")
    for side in ("left", "right")
    for leaf, dimension in (
        ("ee_poses", 7),
        ("ee_joint_states", 45),
        ("hand_betas", 10),
    )
}
MANO45_VALIDITY = tuple(
    f"mano/validity/{prefix}{side}_{leaf}"
    for prefix in ("", "action_")
    for side in ("left", "right")
    for leaf in ("ee_pose", "ee_joint_states", "hand_betas")
)


def discover(source_root: Path, robot_name: str) -> list[tuple[str, Path]]:
    episodes = []
    for suffix in ("*.hdf5", "*.h5"):
        for path in source_root.glob(f"*/{robot_name}/data/{suffix}"):
            task = path.parents[2].name
            episodes.append((task, path.resolve()))
    return sorted(set(episodes), key=lambda item: (item[0], str(item[1])))


def validate_source(path: Path, representation: str) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        required = WUJI20_REQUIRED if representation == "wuji20" else MANO45_REQUIRED
        extra_required = [] if representation == "wuji20" else list(MANO45_VALIDITY)
        missing = [
            key
            for key in [
                *required,
                *extra_required,
                "vision/cam_head/colors",
                "vision/cam_head/extrinsics",
                "vision/cam_head/intrinsics",
                "instruction",
            ]
            if key not in handle
        ]
        if missing:
            raise ValueError(f"{path}: missing required fields {missing}")
        state_prefix = "state" if representation == "wuji20" else "mano/state"
        source_length = int(handle[f"{state_prefix}/left_ee_poses"].shape[0])
        if source_length < 2:
            raise ValueError(f"{path}: expected at least two frames, got {source_length}")
        for key, dim in required.items():
            shape = handle[key].shape
            if not shape or shape[0] != source_length or shape[-1] != dim:
                raise ValueError(
                    f"{path}: {key} expected ({source_length},...,{dim}), got {shape}"
                )
        if representation == "mano45":
            for key in MANO45_VALIDITY:
                validity = handle[key]
                if key.endswith("ee_pose"):
                    validity_dim = 7
                elif key.endswith("ee_joint_states"):
                    validity_dim = 45
                elif key.endswith("hand_betas"):
                    validity_dim = 10
                else:
                    raise AssertionError(key)
                if validity.shape != (source_length, validity_dim) or validity.dtype.kind != "b":
                    raise ValueError(
                        f"{path}: {key} expected boolean ({source_length},{validity_dim}), "
                        f"got {validity.shape} {validity.dtype}"
                    )
            for side in ("left", "right"):
                for leaf in ("ee_poses", "ee_joint_states", "hand_betas"):
                    state = handle[f"mano/state/{side}_{leaf}"]
                    action = handle[f"mano/action/{side}_{leaf}"]
                    validity_leaf = "ee_pose" if leaf == "ee_poses" else leaf
                    state_validity = np.asarray(
                        handle[f"mano/validity/{side}_{validity_leaf}"], dtype=bool
                    )
                    action_validity = np.asarray(
                        handle[f"mano/validity/action_{side}_{validity_leaf}"], dtype=bool
                    )
                    if not state_validity.all() or not action_validity[:-1].all():
                        raise ValueError(f"{path}: retained MANO validity is false for {side}_{leaf}")
                    if action_validity[-1].any():
                        raise ValueError(f"{path}: terminal MANO action validity must be false")
                    state_values = np.asarray(state, dtype=np.float64)
                    action_values = np.asarray(action, dtype=np.float64)
                    if not np.isfinite(state_values).all() or not np.isfinite(
                        action_values[:-1]
                    ).all():
                        raise ValueError(f"{path}: retained MANO values are non-finite for {side}_{leaf}")
                    if np.isfinite(action_values[-1]).any():
                        raise ValueError(f"{path}: terminal MANO action must be all non-finite")
                    if not np.array_equal(action_values[:-1], state_values[1:]):
                        raise ValueError(
                            f"{path}: mano/action/{side}_{leaf}[t] is not exact "
                            "mano/state[t+1]"
                        )
        extrinsics_shape = handle["vision/cam_head/extrinsics"].shape
        if extrinsics_shape not in {
            (7,),
            (source_length, 7),
            (4, 4),
            (source_length, 4, 4),
        }:
            raise ValueError(
                f"{path}: camera extrinsics must be constant or per-frame pose7/4x4, "
                f"got {extrinsics_shape}"
            )
        intrinsics_shape = handle["vision/cam_head/intrinsics"].shape
        if intrinsics_shape not in {
            (4,),
            (source_length, 4),
            (3, 3),
            (source_length, 3, 3),
        }:
            raise ValueError(
                f"{path}: camera intrinsics must be constant or per-frame [fx,fy,cx,cy]/3x3, "
                f"got {intrinsics_shape}"
            )
        if handle["vision/cam_head/colors"].shape[0] != source_length:
            raise ValueError(f"{path}: image/state length mismatch")
        sample_length = source_length - 1 if representation == "mano45" else source_length
        return source_length, sample_length


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> dict:
    source_root = args.source_root.resolve()
    data_root = args.data_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    episodes = discover(source_root, args.robot_name)
    if not episodes:
        raise FileNotFoundError(
            f"No episodes matched <task>/{args.robot_name}/data below {source_root}"
        )
    if args.expert_data_num:
        selected = []
        per_task = Counter()
        for task, path in episodes:
            if per_task[task] < args.expert_data_num:
                selected.append((task, path))
                per_task[task] += 1
        episodes = selected
    task_counts = Counter(task for task, _ in episodes)
    if not args.allow_count_mismatch:
        if args.expected_tasks and len(task_counts) != args.expected_tasks:
            raise RuntimeError(f"Expected {args.expected_tasks} tasks, found {len(task_counts)}: {dict(task_counts)}")
        expected_episodes = (
            args.expert_data_num * args.expected_tasks
            if args.expert_data_num and args.expected_tasks
            else args.expected_episodes
        )
        if expected_episodes and len(episodes) != expected_episodes:
            raise RuntimeError(f"Expected {expected_episodes} episodes, found {len(episodes)}")

    teledata = data_root / "TeleData"
    teledata.mkdir(parents=True, exist_ok=True)
    desired_destinations = {
        (teledata / task / source.name).absolute() for task, source in episodes
    }
    existing_destinations = {
        path.absolute()
        for suffix in ("*.hdf5", "*.h5")
        for path in teledata.rglob(suffix)
    }
    stale_destinations = sorted(existing_destinations - desired_destinations)
    if stale_destinations:
        preview = "\n  ".join(str(path) for path in stale_destinations[:20])
        remainder = len(stale_destinations) - min(20, len(stale_destinations))
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise RuntimeError(
            "Prepared TeleData contains episodes outside the requested selection. "
            "Use a fresh --data-root (no files were removed):\n  " + preview + suffix
        )
    records = []
    total_frames = 0
    for task, source in episodes:
        source_length, sample_length = validate_source(source, args.representation)
        destination_dir = teledata / task
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if os.path.lexists(destination):
            if not destination.is_symlink():
                raise FileExistsError(f"Refusing to replace non-symlink destination: {destination}")
            if destination.resolve() != source:
                if not args.replace_wrong_symlinks:
                    raise FileExistsError(
                        f"Wrong symlink at {destination}; pass --replace-wrong-symlinks to repair it"
                    )
                destination.unlink()
                destination.symlink_to(source)
        else:
            destination.symlink_to(source)
        total_frames += sample_length
        records.append(
            {
                "task": task,
                "frames": sample_length,
                "source_frames": source_length,
                "source": str(source),
                "prepared": str(destination.relative_to(data_root)),
            }
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "data_root": str(data_root),
        "robot": args.robot_name,
        "representation": args.representation,
        "episode_count": len(records),
        "frame_count": total_frames,
        "task_counts": dict(sorted(task_counts.items())),
        "storage": "individual absolute symlinks; source HDF5 files are not modified",
        "semantics": (
            {
                "source_rows": "T source states; T-1 training transitions",
                "source_action": "mano/action[t] is already mano/state[t+1]; never shift twice",
                "vitra_native_state": "per hand: camera root6 + local MANO Euler45 + betas10",
                "vitra_native_action": "per hand: root step delta6 + next absolute local MANO Euler45",
                "mano_conversion": "stored rotvec45 -> per-joint matrix -> Euler xyz; neither side is mirrored again",
                "root": "stored MANO global root directly; no Wuji EE-to-wrist calibration",
                "camera_basis": "raw x-right/y-up/z-back to VITRA x-right/y-down/z-forward",
            }
            if args.representation == "mano45"
            else {
                "external_control": "absolute dual-arm EE pose plus absolute Wuji20 hand targets",
                "vitra_native_state": "per hand: camera-space absolute EEF xyz+Euler_xyz plus hand20 absolute",
                "vitra_native_action": "per hand: step EEF delta using dR=R_next@R_current.T plus next hand20 absolute",
                "camera_basis": "raw x-right/y-up/z-back to VITRA x-right/y-down/z-forward",
                "calibration": "see XPolicyLab/policy/VITRA/mapping_wuji20_mano45.json",
            }
        ),
        "episodes": records,
    }
    data_root.mkdir(parents=True, exist_ok=True)
    atomic_json(data_root / "spark0_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/xspark-data/tjy/spark0_bench"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--robot-name", default="tianji_marvin_wuji")
    parser.add_argument(
        "--representation", choices=("wuji20", "mano45"), default="wuji20"
    )
    parser.add_argument("--expected-tasks", type=int, default=6)
    parser.add_argument("--expected-episodes", type=int, default=600)
    parser.add_argument(
        "--expert-data-num",
        type=int,
        default=0,
        help="Optional maximum episodes selected per task; zero uses all episodes.",
    )
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--replace-wrong-symlinks", action="store_true")
    return parser


def main() -> None:
    manifest = prepare(build_parser().parse_args())
    print(
        f"Prepared {manifest['episode_count']} episodes, {manifest['frame_count']} frames, "
        f"{len(manifest['task_counts'])} tasks below {manifest['data_root']}"
    )


if __name__ == "__main__":
    main()
