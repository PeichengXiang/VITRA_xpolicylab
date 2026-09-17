#!/usr/bin/env python3
"""Prepare flat egovla_humanoid_sim_processed HDF5 for VITRA MANO training.

The source already contains MANO and Wuji arrays.  We only build external-link
wrappers, normalize the camera frame attribute for RoboDatasetCore, and create
the TeleData symlink tree.  The source files are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

VERSION = "egovla_humanoid_external_link_v1"
DEFAULT_SOURCE = Path("/personal/zijian/Spark_0/hand_data/hdf5/egovla_humanoid_sim_processed")
DEFAULT_DATA = Path("/personal/xiangpc/0813_Xpolicylab_bench/VITRA/data/egovla_humanoid_sim_mano_20260902")
DEFAULT_WRAPPERS = Path("/personal/xiangpc/0813_Xpolicylab_bench/VITRA/data/egovla_humanoid_sim_mano_20260902_source_wrappers")
SIDES = ("left", "right")
LEAVES = (("ee_poses", 7, "ee_pose"), ("ee_joint_states", 45, "ee_joint_states"), ("hand_betas", 10, "hand_betas"))


def text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def copy_attrs(src: Any, dst: Any) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def discover(root: Path) -> tuple[list[tuple[str, Path, str]], list[str]]:
    active: list[tuple[str, Path, str]] = []
    excluded: list[str] = []
    for suffix in ("*.hdf5", "*.h5"):
        for p in sorted(root.glob(f"*/{suffix}")):
            rel = p.relative_to(root)
            if any("deprecated" in part.casefold() for part in rel.parts):
                excluded.append(str(rel))
            elif p.name.startswith("episode_"):
                active.append((rel.parts[0], p.resolve(), str(rel)))
    unique = {(task, str(path)): (task, path, rel) for task, path, rel in active}
    return sorted(unique.values(), key=lambda x: (x[0], x[2])), sorted(set(excluded))


def historical_deprecated(root: Path) -> list[str]:
    found: set[str] = set()
    pattern = re.compile(r'''[^"'\s,]+[Dd]eprecated[^"'\s,]*\.hdf5''')
    for p in root.rglob("*.jsonl"):
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for value in pattern.findall(raw):
            value = value.replace("\\/", "/")
            # The shard logs mention the same file through both a raw and a
            # processed absolute root.  Keep the task-relative suffix so each
            # deprecated episode is counted once.
            canonical = value
            for task in ("Insert-Cans", "Sort-Cans"):
                marker = task + "/"
                if marker in value:
                    canonical = value[value.index(marker) :]
                    break
            found.add(canonical)
    return sorted(found)


def require_shape(handle: h5py.File, key: str, shape: tuple[int, ...]) -> None:
    if key not in handle or tuple(handle[key].shape) != shape:
        actual = None if key not in handle else tuple(handle[key].shape)
        raise ValueError(f"{handle.filename}: {key} expected {shape}, got {actual}")


def validate(path: Path) -> tuple[int, int, dict[str, str]]:
    with h5py.File(path, "r") as h:
        n = int(h["mano/state/left_ee_poses"].shape[0])
        if n < 2:
            raise ValueError(f"{path}: MANO trajectory has only {n} rows")
        for side in SIDES:
            for leaf, dim, vleaf in LEAVES:
                sk = f"mano/state/{side}_{leaf}"
                ak = f"mano/action/{side}_{leaf}"
                svk = f"mano/validity/{side}_{vleaf}"
                avk = f"mano/validity/action_{side}_{vleaf}"
                for key in (sk, ak, svk, avk):
                    require_shape(h, key, (n, dim))
                state = np.asarray(h[sk], dtype=np.float64)
                action = np.asarray(h[ak], dtype=np.float64)
                sv = np.asarray(h[svk], dtype=bool)
                av = np.asarray(h[avk], dtype=bool)
                if not sv[:-1].all() or not av[:-1].all():
                    raise ValueError(f"{path}: invalid transition validity in {side}_{leaf}")
                if not np.isfinite(state).all() or not np.isfinite(action[:-1]).all():
                    raise ValueError(f"{path}: non-finite transition in {side}_{leaf}")
                if not np.array_equal(action[:-1], state[1:]):
                    raise ValueError(f"{path}: {ak}[:-1] does not equal {sk}[1:]")
                # The terminal action points to a source frame truncated by the
                # producer.  It is intentionally retained but not sampled.
                if not av[-1].all() or not np.isfinite(action[-1]).all():
                    raise ValueError(f"{path}: unexpected terminal MANO action contract")
            for leaf, dim in (("ee_poses", 7), ("ee_joint_states", 20)):
                for prefix in ("state", "action"):
                    require_shape(h, f"{prefix}/{side}_{leaf}", (n, dim))
        for key in ("vision/cam_head/colors", "vision/cam_head/extrinsics", "vision/cam_head/intrinsics", "instruction"):
            if key not in h:
                raise ValueError(f"{path}: missing {key}")
        if int(h["vision/cam_head/colors"].shape[0]) != n:
            raise ValueError(f"{path}: image/state length mismatch")
        frame = text(h["vision/cam_head/extrinsics"].attrs.get("frame", ""))
        if frame not in ("camera_to_world", "camera_to_env"):
            raise ValueError(f"{path}: unsupported camera frame {frame!r}")
        if "additional_info/ee_frame" in h:
            ee = text(h["additional_info/ee_frame"][()])
        else:
            ee = text(h["additional_info"].attrs.get("ee_frame", ""))
        return n, n - 1, {"source_camera_frame": frame, "source_ee_frame": ee}


def write_wrapper(source: Path, destination: Path) -> dict[str, bool]:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    copied_pose = False
    rewritten = False
    source_name = str(source)
    with h5py.File(source, "r") as src, h5py.File(tmp, "w") as out:
        copy_attrs(src, out)
        out.attrs["vitra_wrapper"] = VERSION
        out.attrs["vitra_source"] = source_name
        for name in src.keys():
            if name == "action":
                group = out.create_group("action")
                copy_attrs(src["action"], group)
                for child in src["action"].keys():
                    group[child] = h5py.ExternalLink(source_name, f"action/{child}")
                for side in SIDES:
                    key = f"{side}_ee_poses"
                    if key not in src["action"]:
                        values = np.asarray(src[f"state/{key}"], dtype=np.float64)
                        group.create_dataset(key, data=np.concatenate([values[1:], values[-1:]]))
                        copied_pose = True
            elif name != "vision":
                out[name] = h5py.ExternalLink(source_name, name)
            else:
                vision = out.create_group("vision")
                copy_attrs(src["vision"], vision)
                for camera in src["vision"].keys():
                    if camera != "cam_head":
                        vision[camera] = h5py.ExternalLink(source_name, f"vision/{camera}")
                        continue
                    src_cam = src["vision/cam_head"]
                    dst_cam = vision.create_group("cam_head")
                    copy_attrs(src_cam, dst_cam)
                    for dataset in src_cam.keys():
                        if dataset != "extrinsics":
                            dst_cam[dataset] = h5py.ExternalLink(source_name, f"vision/cam_head/{dataset}")
                    ex = src_cam["extrinsics"]
                    copied = dst_cam.create_dataset("extrinsics", data=ex[()])
                    copy_attrs(ex, copied)
                    old = text(ex.attrs.get("frame", ""))
                    copied.attrs["frame"] = "camera_to_env"
                    if old and old != "camera_to_env":
                        copied.attrs["original_frame"] = old
                        rewritten = True
        if text(out["vision/cam_head/extrinsics"].attrs.get("frame", "")) != "camera_to_env":
            raise ValueError(f"{source}: camera frame rewrite failed")
    os.replace(tmp, destination)
    return {"camera_frame_rewritten": rewritten, "action_ee_poses_copied": copied_pose}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--wrapper-root", type=Path, default=DEFAULT_WRAPPERS)
    parser.add_argument("--expected-deprecated", type=int, default=100)
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()
    source = args.source_root.resolve()
    data = args.data_root.resolve()
    wrappers = args.wrapper_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if data.name in {"data", "data_mano"} or data == source:
        raise SystemExit(f"Refusing unsafe data root: {data}")
    if (data.exists() and any(data.iterdir())) or (wrappers.exists() and any(wrappers.iterdir())):
        raise SystemExit("Refusing non-empty output roots; choose a fresh data/wrapper path")
    episodes, current_excluded = discover(source)
    historical = historical_deprecated(source)
    excluded = sorted(set(current_excluded) | set(historical))
    if args.expected_deprecated >= 0 and len(excluded) != args.expected_deprecated:
        raise RuntimeError(f"Expected {args.expected_deprecated} Deprecated paths, found {len(excluded)} (current={len(current_excluded)}, historical={len(historical)})")
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise RuntimeError("No active flat episode_*.hdf5 files found")
    data.mkdir(parents=True)
    wrappers.mkdir(parents=True)
    teledata = data / "TeleData"
    teledata.mkdir()
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    transitions = 0
    rewrites = 0
    for i, (task, src, rel) in enumerate(episodes, 1):
        source_frames, ntrans, details = validate(src)
        inside = Path(rel).relative_to(task)
        wrapper = wrappers / task / inside
        flags = write_wrapper(src, wrapper)
        dest = teledata / task / inside
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(wrapper)
        counts[task] += 1
        transitions += ntrans
        rewrites += int(flags["camera_frame_rewritten"])
        records.append({"task": task, "source": str(src), "wrapper": str(wrapper), "prepared": str(dest.relative_to(data)), "source_frames": source_frames, "training_transitions": ntrans, **details, **flags})
        if i == 1 or i == len(episodes) or i % 100 == 0:
            print(f"prepared {i}/{len(episodes)}: {task}/{inside}", flush=True)
    manifest = {
        "schema_version": 1,
        "converter": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "data_root": str(data),
        "wrapper_root": str(wrappers),
        "representation": "mano45",
        "robot_name": "tianji_marvin_wuji",
        "episode_count": len(records),
        "frame_count": transitions,
        "source_frame_count": sum(x["source_frames"] for x in records),
        "task_counts": dict(sorted(counts.items())),
        "deprecated_exclusion": {"rule": "exclude any path component containing deprecated (case-insensitive)", "count": len(excluded), "current_tree_count": len(current_excluded), "historical_log_count": len(historical), "paths": excluded},
        "source_not_modified": True,
        "storage": "TeleData symlinks to wrappers; wrappers external-link large source arrays",
        "camera_contract": {"source": "camera_to_world", "loader": "camera_to_env", "wrapper_action": "copy extrinsics values and relabel frame; no numeric transform"},
        "terminal_action_contract": "action[:-1]==state[1:]; final valid finite action targets truncated next source frame and is ignored by RoboDatasetCore (T-1 transitions)",
        "embodiment_warning": "source metadata identifies egovla_inspire_ee/Fourier GR1 end-link; launcher uses tianji_marvin_wuji; verify arm-frame compatibility before physical execution",
        "camera_rewrites": rewrites,
        "episodes": records,
    }
    write_json(data / "spark0_manifest.json", manifest)
    write_json(data / "conversion_report.json", {"converter": VERSION, "source_root": str(source), "episode_count": len(records), "task_counts": dict(sorted(counts.items())), "deprecated_exclusion": manifest["deprecated_exclusion"], "source_not_modified": True, "camera_rewrites": rewrites, "representation": "mano45", "training_transitions": transitions})
    print(f"Prepared {len(records)} episodes / {transitions} MANO transitions across {len(counts)} tasks; excluded {len(excluded)} Deprecated paths.", flush=True)


if __name__ == "__main__":
    main()
