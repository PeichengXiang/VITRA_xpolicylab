#!/usr/bin/env python3
"""Validate prepared Spark0 episodes and the VITRA-native semantics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def add_import_paths(xpolicylab_root: Path) -> None:
    root = xpolicylab_root.resolve()
    sys.path.insert(0, str(root.parent))
    sys.path.insert(0, str(root / "policy" / "VITRA" / "VITRA"))


def quaternion_matrices(pose: np.ndarray) -> np.ndarray:
    quat = np.asarray(pose, dtype=np.float64)[..., 3:7]
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("zero quaternion")
    xyzw = (quat / norms)[..., [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(pose.shape[:-1] + (3, 3))


def alignment_errors(handle: h5py.File, side: str, representation: str) -> dict:
    prefix = "" if representation == "wuji20" else "mano/"
    action_pose = np.asarray(
        handle[f"{prefix}action/{side}_ee_poses"][:-1], dtype=np.float64
    )
    next_pose = np.asarray(
        handle[f"{prefix}state/{side}_ee_poses"][1:], dtype=np.float64
    )
    action_hand = np.asarray(
        handle[f"{prefix}action/{side}_ee_joint_states"][:-1], dtype=np.float64
    )
    next_hand = np.asarray(
        handle[f"{prefix}state/{side}_ee_joint_states"][1:], dtype=np.float64
    )
    position = float(np.max(np.abs(action_pose[:, :3] - next_pose[:, :3])))
    hand = float(np.max(np.abs(action_hand - next_hand)))
    relative = quaternion_matrices(action_pose) @ np.swapaxes(quaternion_matrices(next_pose), -1, -2)
    angle = float(np.max(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()))
    result = {"position_max_abs": position, "rotation_max_rad": angle, "hand_max_abs": hand}
    if representation == "mano45":
        action_betas = np.asarray(
            handle[f"mano/action/{side}_hand_betas"][:-1], dtype=np.float64
        )
        next_betas = np.asarray(
            handle[f"mano/state/{side}_hand_betas"][1:], dtype=np.float64
        )
        result["betas_max_abs"] = float(np.max(np.abs(action_betas - next_betas)))
    return result


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def validate(args: argparse.Namespace) -> dict:
    add_import_paths(args.xpolicylab_root)
    from vitra.datasets.spark0_dataset import (
        MANO_ACTION_DUAL_DIM,
        MANO_STATE_DUAL_DIM,
        NATIVE_DUAL_DIM,
        REPRESENTATION_MANO45,
        RoboDatasetCore,
        load_wuji_mapping,
        mano_action_to_human,
        mano_state_to_human,
        native_action_to_human,
        native_state_to_human,
        raw_camera_pose_to_vitra_matrix,
        validate_episode_file,
    )

    if args.representation == REPRESENTATION_MANO45 and args.mapping is not None:
        raise ValueError("--mapping is only valid for --representation wuji20")
    mapping = (
        load_wuji_mapping(str(args.mapping.resolve()) if args.mapping else None)
        if args.representation == "wuji20"
        else None
    )
    teledata = args.data_root.resolve() / "TeleData"
    files = sorted({*teledata.rglob("*.hdf5"), *teledata.rglob("*.h5")})
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No episodes below {teledata}")

    errors = []
    warnings = []
    episodes = []
    total_frames = 0
    for index, path in enumerate(files):
        try:
            metadata = validate_episode_file(
                path,
                decode_images=args.decode_images,
                representation=args.representation,
            )
            with h5py.File(path, "r") as handle:
                side_errors = {
                    side: alignment_errors(handle, side, args.representation)
                    for side in ("left", "right")
                }
                for side, metrics in side_errors.items():
                    if max(metrics.values()) > args.alignment_tolerance:
                        errors.append(
                            f"{path} {side}: action[t] != state[t+1] within "
                            f"{args.alignment_tolerance}: {metrics}"
                        )
                if args.representation == REPRESENTATION_MANO45:
                    for side in ("left", "right"):
                        for leaf in ("ee_poses", "ee_joint_states", "hand_betas"):
                            terminal = np.asarray(
                                handle[f"mano/action/{side}_{leaf}"][-1], dtype=np.float64
                            )
                            validity_leaf = "ee_pose" if leaf == "ee_poses" else leaf
                            terminal_valid = bool(
                                np.asarray(
                                    handle[f"mano/validity/action_{side}_{validity_leaf}"][-1],
                                    dtype=bool,
                                ).any()
                            )
                            if np.isfinite(terminal).any() or terminal_valid:
                                errors.append(
                                    f"{path} {side}_{leaf}: terminal MANO action is not all-invalid"
                                )
                camera_dataset = handle["vision/cam_head/extrinsics"]
                source_length = metadata.get("source_length", metadata["length"])
                camera_value = (
                    camera_dataset[0]
                    if camera_dataset.shape
                    in {(source_length, 7), (source_length, 4, 4)}
                    else camera_dataset[()]
                )
                camera = raw_camera_pose_to_vitra_matrix(camera_value)
                determinant = float(np.linalg.det(camera[:3, :3]))
                if not np.isclose(determinant, 1.0, atol=1e-5):
                    errors.append(f"{path}: converted camera rotation determinant {determinant}")
            metadata["alignment"] = side_errors
            episodes.append(metadata)
            total_frames += metadata["length"]
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    # Instantiate the actual lazy loader and exercise episode boundaries.  The
    # complete set is used unless --max-episodes requested a prefix audit.
    if not errors and not args.max_episodes:
        try:
            dataset = RoboDatasetCore(
                root_dir=str(teledata),
                statistics_path=None,
                action_past_window_size=0,
                action_future_window_size=15,
                load_images=args.decode_images,
                mapping_path=str(args.mapping.resolve()) if args.mapping else None,
                representation=args.representation,
            )
            boundary_indices = {0, len(dataset) - 1}
            running = 0
            for length in dataset.episode_lengths:
                running += length
                boundary_indices.add(running - 1)
                if running < len(dataset):
                    boundary_indices.add(running)
            for sample_index in sorted(boundary_indices):
                sample = dataset[sample_index]
                expected_state_dim = (
                    MANO_STATE_DUAL_DIM
                    if args.representation == REPRESENTATION_MANO45
                    else NATIVE_DUAL_DIM
                )
                expected_action_dim = (
                    MANO_ACTION_DUAL_DIM
                    if args.representation == REPRESENTATION_MANO45
                    else NATIVE_DUAL_DIM
                )
                if sample["current_state"].shape != (expected_state_dim,):
                    errors.append(f"sample {sample_index}: native state shape {sample['current_state'].shape}")
                if sample["action_list"].shape != (16, expected_action_dim):
                    errors.append(f"sample {sample_index}: native action shape {sample['action_list'].shape}")
                episode_id = np.searchsorted(
                    dataset._cumulative_lengths, sample_index, side="right"
                )
                if sample["frame_index"] == dataset.episode_lengths[episode_id] - 1:
                    if args.representation == REPRESENTATION_MANO45:
                        if not sample["action_mask"][0].all() or sample["action_mask"][1:].any():
                            errors.append(
                                f"sample {sample_index}: final MANO transition/mask is invalid"
                            )
                    elif sample["action_mask"].any():
                        errors.append(
                            f"sample {sample_index}: terminal action padding is not fully masked"
                        )
                if args.representation == REPRESENTATION_MANO45:
                    state_h, state_m = mano_state_to_human(
                        sample["current_state"],
                        sample["current_state_mask"],
                        return_mask=True,
                    )
                    action_h, action_m = mano_action_to_human(
                        sample["action_list"], sample["action_mask"], return_mask=True
                    )
                else:
                    state_h, state_m = native_state_to_human(
                        sample["current_state"], mapping, sample["current_state_mask"], return_mask=True
                    )
                    action_h, action_m = native_action_to_human(
                        sample["action_list"], mapping, sample["action_mask"], return_mask=True
                    )
                if state_h.shape != (212,) or state_m.sum() != 102:
                    errors.append(f"sample {sample_index}: state injection/mask is invalid")
                expected_rows = int(sample["action_mask"][:, 0].sum()) * 102
                if action_h.shape != (16, 192) or int(action_m.sum()) != expected_rows:
                    errors.append(f"sample {sample_index}: action injection/mask is invalid")
        except Exception as exc:
            errors.append(f"loader audit: {type(exc).__name__}: {exc}")

    report = {
        "ok": not errors,
        "data_root": str(args.data_root.resolve()),
        "episode_count_checked": len(episodes),
        "frame_count_checked": total_frames,
        "representation": args.representation,
        "mapping": mapping["name"] if mapping is not None else None,
        "errors": errors,
        "warnings": warnings,
        "episodes": episodes,
    }
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))
    if errors:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    default_xpl = Path(__file__).resolve().parents[1] / "XPolicyLab"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--xpolicylab-root", type=Path, default=default_xpl)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument(
        "--representation", choices=("wuji20", "mano45"), default="wuji20"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--alignment-tolerance", type=float, default=1e-4)
    return parser


if __name__ == "__main__":
    validate(build_parser().parse_args())
