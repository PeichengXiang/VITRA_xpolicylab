#!/usr/bin/env python3
"""Prepare raw EgoVLA-Humanoid-Sim HDF5 for VITRA inspire12 training.

Source files are never copied or rewritten. The prepared tree is::

    <data-root>/TeleData/<task>/episode_*.hdf5 -> <source episode>
    <data-root>/spark0_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

VERSION = "egovla_inspire12_robot_command_v3"
ACTION_CONTRACT_ID = "egovla_observed_ee_step_future_hand_command_v3"
DEFAULT_SOURCE = Path(
    "/personal/xiangpc/EgoVLA benchmark/XPolicyLab/data/EgoVLA/raw_remove_deprecated"
)
DEFAULT_DATA = Path(
    "/personal/xiangpc/0813_Xpolicylab_bench/VITRA/data/egovla_inspire12_robot_command_v3"
)
EXPECTED_TASKS = {
    "Close-Drawer": 50,
    "Flip-Mug": 100,
    "Insert-And-Unload-Cans": 900,
    "Insert-Cans": 100,
    "Open-Drawer": 100,
    "Open-Laptop": 100,
    "Pour-Balls": 102,
    "Push-Box": 100,
    "Sort-Cans": 101,
    "Stack-Can": 100,
    "Stack-Can-Into-Drawer": 50,
    "Unload-Cans": 100,
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def discover(root: Path) -> tuple[list[tuple[str, Path, str]], list[str]]:
    active: list[tuple[str, Path, str]] = []
    excluded: list[str] = []
    for path in sorted(root.rglob("episode_*.hdf5")) + sorted(root.rglob("episode_*.h5")):
        rel = path.relative_to(root)
        if any("deprecated" in part.casefold() for part in rel.parts):
            excluded.append(str(rel))
            continue
        active.append((rel.parts[0], path.resolve(), str(rel)))
    unique = {(task, str(path)): (task, path, rel) for task, path, rel in active}
    return sorted(unique.values(), key=lambda item: (item[0], item[2])), sorted(set(excluded))


def validate_source(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        required = (
            "action",
            "observations/qpos",
            "observations/images/main",
        )
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        for side in ("left", "right"):
            current_keys = (
                f"observations/{side}_ee_pose",
                f"observations/{side}_curr_ee_pose",
            )
            current_key = next((key for key in current_keys if key in handle), None)
            if current_key is None:
                raise ValueError(f"{path}: missing current EE pose; tried {current_keys}")
        frames = int(handle["observations/qpos"].shape[0])
        if frames < 2 or handle["observations/qpos"].shape != (frames, 50):
            raise ValueError(f"{path}: qpos must be (T>=2, 50), got {handle['observations/qpos'].shape}")
        if handle["action"].shape != (frames, 50):
            raise ValueError(f"{path}: action shape {handle['action'].shape}")
        for side in ("left", "right"):
            current_key = next(
                key
                for key in (
                    f"observations/{side}_ee_pose",
                    f"observations/{side}_curr_ee_pose",
                )
                if key in handle
            )
            if handle[current_key].shape != (frames, 7):
                raise ValueError(f"{path}: {current_key} shape {handle[current_key].shape}")
        if handle["observations/images/main"].shape != (frames, 384, 384, 3):
            raise ValueError(
                f"{path}: images/main must be (T,384,384,3), got {handle['observations/images/main'].shape}"
            )
        return frames


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    data_root = args.data_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    episodes, excluded = discover(source_root)
    if not episodes:
        raise FileNotFoundError(f"No EgoVLA episodes under {source_root}")

    tele = data_root / "TeleData"
    tele.mkdir(parents=True, exist_ok=True)
    records = []
    frame_count = 0
    for task, source, rel in episodes:
        frames = validate_source(source)
        destination = tele / task / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        os.symlink(source, destination)
        records.append(
            {
                "task": task,
                "relative_path": rel,
                "source": str(source),
                "link": str(destination),
                "frames": frames,
            }
        )
        frame_count += frames

    task_counts = Counter(task for task, _, _ in episodes)
    if dict(task_counts) != EXPECTED_TASKS:
        raise ValueError(f"Unexpected task counts: {dict(task_counts)} != {EXPECTED_TASKS}")
    if len(episodes) != sum(EXPECTED_TASKS.values()):
        raise ValueError(f"Unexpected episode count: {len(episodes)}")

    manifest = {
        "schema_version": 1,
        "converter": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "data_root": str(data_root),
        "representation": "inspire12",
        "codec_id": "inspire12_mano45_xyz_sparse_v1",
        "action_contract_id": ACTION_CONTRACT_ID,
        "action_contract": {
            "wrist": "observed camera-space EEF[t] to observed EEF[t+1] step delta",
            "hand": "action[t] direct future Inspire12 execution command",
            "terminal": "masked because observed EEF[t+1] is unavailable",
        },
        "robot_name": "ego_h1_inspire",
        "episode_count": len(episodes),
        "frame_count": frame_count,
        "task_counts": dict(sorted(task_counts.items())),
        "deprecated_exclusion": {
            "rule": "exclude any path component containing deprecated (case-insensitive)",
            "count": len(excluded),
            "paths": excluded,
        },
        "episodes": records,
    }
    write_json(data_root / "spark0_manifest.json", manifest)
    print(
        f"[EgoVLA inspire12] prepared episodes={len(episodes)} frames={frame_count} "
        f"tasks={len(task_counts)} -> {data_root}"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
