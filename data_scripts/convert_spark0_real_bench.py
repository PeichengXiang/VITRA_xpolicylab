#!/usr/bin/env python3
"""Convert spark0_real bench dumps into a VITRA training tree without mutating source.

The official prepare step only understands
``<task>/<robot>/data/episode_*.hdf5`` and requires ``camera_to_env`` plus
``action/{left,right}_ee_poses``. Accepted source layouts are:

- ``<task>/episode_*.hdf5`` (spark0_real/bench)
- ``<task>/<robot_cfg>/*.hdf5`` (spark0_real_bench_v5_processed)

This tool writes a new wrapper HDF5 per episode. Large arrays stay in the
immutable source via external links. Only the camera extrinsics (attribute
rename) and synthesized next-state EE-pose actions are copied locally.
The wrappers are then fed through the existing prepare / statistics /
validate pipeline into a fresh ``--data-root``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


WRAPPER_VERSION = "spark0_real_bench_external_link_v1"
DEFAULT_SOURCE = Path(
    "/personal/zijian/Spark_0/data/v2_processed/spark0_real_bench_v5_processed"
)
DEFAULT_ROBOT = "tianji_marvin_wuji"
# metadata is copied locally so add_mano can write metadata/mano_json on the
# wrapper. Linking that group would let r+ follow ExternalLink into the source.
LINKED_GROUPS = (
    "state",
    "additional_info",
    "mano",
    "tactile",
    "validity",
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def discover_real_bench(source_root: Path) -> list[tuple[str, Path]]:
    episodes: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for suffix in ("*.hdf5", "*.h5"):
        for path in source_root.glob(f"*/{suffix}"):
            if not path.name.startswith("episode_"):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            episodes.append((path.parent.name, resolved))
        for path in source_root.glob(f"*/*/{suffix}"):
            if path.parent.name in {"data", "tianji_marvin_wuji"}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            episodes.append((path.parents[1].name, resolved))
    return sorted(episodes, key=lambda item: (item[0], str(item[1])))


def copy_group_local(src_group: h5py.Group, dst_group: h5py.Group) -> None:
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            copy_group_local(obj, dst_group.create_group(name))
            continue
        copied = dst_group.create_dataset(name, data=obj[()])
        for key, value in obj.attrs.items():
            copied.attrs[key] = value


def sanitize_joint_confidence(src: h5py.File) -> np.ndarray | None:
    if "aligned/joint_confidence" not in src:
        return None
    confidence = np.asarray(src["aligned/joint_confidence"], dtype=np.float64)
    needs_sanitize = (not np.isfinite(confidence).all()) or np.any(
        (confidence < 0.0) | (confidence > 1.0)
    )
    if not needs_sanitize:
        return None
    if "aligned/joints_world" in src:
        joints = np.asarray(src["aligned/joints_world"], dtype=np.float64)
        if joints.shape[:-1] == confidence.shape:
            return np.isfinite(joints).all(axis=-1).astype(np.float64)
    return np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)


def next_state_hold(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 1:
        raise ValueError("cannot synthesize action from an empty trajectory")
    if values.shape[0] == 1:
        return np.asarray(values, dtype=np.float64)
    return np.concatenate([values[1:], values[-1:]], axis=0)


def write_wrapper(source: Path, destination: Path) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()

    source_str = str(source)
    created = {
        "action_ee_poses": False,
        "camera_frame_rewritten": False,
        "metadata_copied": False,
        "aligned_confidence_sanitized": False,
    }
    with h5py.File(source, "r") as src, h5py.File(temporary, "w") as out:
        for key, value in src.attrs.items():
            out.attrs[key] = value
        out.attrs["vitra_wrapper"] = WRAPPER_VERSION
        out.attrs["vitra_source"] = source_str

        for name in src.keys():
            if name in LINKED_GROUPS:
                out[name] = h5py.ExternalLink(source_str, name)
            elif name == "metadata":
                copy_group_local(src["metadata"], out.create_group("metadata"))
                created["metadata_copied"] = True
            elif name == "aligned":
                out.create_group("aligned")
                sanitized = sanitize_joint_confidence(src)
                for dataset in src["aligned"].keys():
                    if dataset == "joint_confidence" and sanitized is not None:
                        out.create_dataset("aligned/joint_confidence", data=sanitized)
                        created["aligned_confidence_sanitized"] = True
                        continue
                    out[f"aligned/{dataset}"] = h5py.ExternalLink(
                        source_str, f"aligned/{dataset}"
                    )
            elif name == "action":
                out.create_group("action")
                for dataset in src["action"].keys():
                    out[f"action/{dataset}"] = h5py.ExternalLink(
                        source_str, f"action/{dataset}"
                    )
                for side in ("left", "right"):
                    key = f"{side}_ee_poses"
                    if key in src["action"]:
                        continue
                    poses = np.asarray(src[f"state/{key}"], dtype=np.float64)
                    out.create_dataset(f"action/{key}", data=next_state_hold(poses))
                    created["action_ee_poses"] = True
            elif name == "vision":
                out.create_group("vision")
                if "cam_head" not in src["vision"]:
                    raise ValueError(f"{source}: missing vision/cam_head")
                for camera in src["vision"].keys():
                    if camera != "cam_head":
                        out[f"vision/{camera}"] = h5py.ExternalLink(
                            source_str, f"vision/{camera}"
                        )
                        continue
                    out.create_group("vision/cam_head")
                    for dataset in src["vision/cam_head"].keys():
                        if dataset == "extrinsics":
                            continue
                        out[f"vision/cam_head/{dataset}"] = h5py.ExternalLink(
                            source_str, f"vision/cam_head/{dataset}"
                        )
                    extrinsics = src["vision/cam_head/extrinsics"]
                    copied = out.create_dataset(
                        "vision/cam_head/extrinsics", data=extrinsics[()]
                    )
                    original_frame = extrinsics.attrs.get("frame", "")
                    if isinstance(original_frame, (bytes, bytearray, np.bytes_)):
                        original_frame = bytes(original_frame).decode("utf-8", "replace")
                    for key, value in extrinsics.attrs.items():
                        copied.attrs[key] = (
                            "camera_to_env" if key == "frame" else value
                        )
                    copied.attrs["frame"] = "camera_to_env"
                    if original_frame and original_frame != "camera_to_env":
                        copied.attrs["original_frame"] = original_frame
                        created["camera_frame_rewritten"] = True
            else:
                out[name] = h5py.ExternalLink(source_str, name)

        if "action/left_ee_poses" not in out:
            raise ValueError(f"{source}: failed to materialize action EE poses")
        if out["vision/cam_head/extrinsics"].attrs.get("frame") != "camera_to_env":
            raise ValueError(f"{source}: failed to rewrite camera frame")

    os.replace(temporary, destination)
    return created


def convert_wrappers(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    episodes = discover_real_bench(args.source_root)
    if not episodes:
        raise FileNotFoundError(
            f"No <task>/episode_*.hdf5 or <task>/*/*.hdf5 files under {args.source_root}"
        )
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    written: list[tuple[str, Path, Path]] = []
    for index, (task, source) in enumerate(episodes, start=1):
        destination = (
            args.wrapper_root / task / args.robot_name / "data" / source.name
        )
        write_wrapper(source, destination)
        written.append((task, source, destination.resolve()))
        if index == 1 or index == len(episodes) or index % 50 == 0:
            print(f"wrappers: {index}/{len(episodes)} {task}/{source.name}")
    return written


def run_pipeline(args: argparse.Namespace) -> None:
    scripts = Path(__file__).resolve().parent
    python = sys.executable
    prepare = [
        python,
        str(scripts / "prepare_vitra_data.py"),
        "--source-root",
        str(args.wrapper_root),
        "--data-root",
        str(args.data_root),
        "--robot-name",
        args.robot_name,
        "--representation",
        args.representation,
        "--allow-count-mismatch",
        "--replace-wrong-symlinks",
    ]
    if args.max_episodes:
        # Wrappers were already truncated; keep prepare from applying the
        # official 6-task / 600-episode defaults.
        pass
    subprocess.run(prepare, check=True)
    if not args.skip_statistics:
        subprocess.run(
            [
                python,
                str(scripts / "calculate_vitra_statistics.py"),
                "--data-root",
                str(args.data_root),
                "--representation",
                args.representation,
            ],
            check=True,
        )
    if not args.skip_validate:
        validate = [
            python,
            str(scripts / "validate_vitra_data.py"),
            "--data-root",
            str(args.data_root),
            "--representation",
            args.representation,
            "--output",
            str(args.data_root / "validation_report.json"),
        ]
        if args.decode_images:
            validate.append("--decode-images")
        if args.max_episodes:
            validate.extend(["--max-episodes", str(args.max_episodes)])
        subprocess.run(validate, check=True)


def build_parser() -> argparse.ArgumentParser:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--wrapper-root",
        type=Path,
        default=workspace / "data_spark0_real_v5_source",
        help="Official-layout wrappers. Source HDF5 files are never written.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=workspace / "data_spark0_real_v5",
        help="VITRA training root (TeleData + manifest + statistics).",
    )
    parser.add_argument("--robot-name", default=DEFAULT_ROBOT)
    parser.add_argument(
        "--representation", choices=("wuji20", "mano45"), default="mano45"
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--skip-statistics", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--wrappers-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.source_root = args.source_root.resolve()
    args.wrapper_root = args.wrapper_root.resolve()
    args.data_root = args.data_root.resolve()
    protected_names = {"data", "data_mano", "data_spark0_real", "data_spark0_real_source"}
    if args.data_root.name in protected_names:
        raise SystemExit(
            f"Refusing to overwrite the existing sim data root: {args.data_root}"
        )
    if args.source_root == args.wrapper_root or args.source_root == args.data_root:
        raise SystemExit("wrapper/data roots must be distinct from the immutable source")

    written = convert_wrappers(args)
    report = {
        "schema": WRAPPER_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root),
        "wrapper_root": str(args.wrapper_root),
        "data_root": str(args.data_root),
        "representation": args.representation,
        "episode_count": len(written),
        "task_counts": dict(sorted(Counter(task for task, _, _ in written).items())),
        "source_unmodified": True,
        "wrapper_policy": {
            "storage": "external links to immutable source HDF5",
            "copied_locally": [
                "vision/cam_head/extrinsics (frame rewritten camera_to_world -> camera_to_env)",
                "action/{left,right}_ee_poses when missing (state[t+1], last frame held)",
                "metadata (local copy so add_mano cannot write through to source)",
                "aligned/joint_confidence when source values are NaN or outside [0, 1]",
            ],
        },
        "episodes": [
            {
                "task": task,
                "source": str(source),
                "wrapper": str(wrapper),
            }
            for task, source, wrapper in written
        ],
    }
    args.wrapper_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.wrapper_root / "conversion_report.json", report)
    print(
        f"Wrote {len(written)} wrappers under {args.wrapper_root} "
        f"from {args.source_root} (source not modified)"
    )
    if args.wrappers_only:
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
