#!/usr/bin/env python3
"""Fit the production MANO45 -> Wuji20 linear inverse artifact.

The split is by episode, never by frame.  Source HDF5 files are opened read
only.  Defaults reserve episodes 80--89 exclusively for safety calibration and
90--99 exclusively for the final held-out report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import base64
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA_VERSION = 1
ARTIFACT_KIND = "xpolicylab_vitra_mano45_to_wujihand2_linear_inverse"
INPUT_REPRESENTATION = "mano_local_rotvec45"
MODEL_INPUT_ENCODING = "mano_local_euler_xyz45"
OUTPUT_REPRESENTATION = "wujihand2_q20_stage_major"
SIDES = ("left", "right")
MANO_JOINT_ORDER = (
    "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
)
WUJI_STAGE_MAJOR_ORDER = (
    "index_mcp_flex", "middle_mcp_flex", "pinky_mcp_flex",
    "ring_mcp_flex", "thumb_cmc_flex",
    "index_mcp_abd", "middle_mcp_abd", "pinky_mcp_abd",
    "ring_mcp_abd", "thumb_cmc_abd",
    "index_pip", "middle_pip", "pinky_pip", "ring_pip", "thumb_mcp",
    "index_dip", "middle_dip", "pinky_dip", "ring_dip", "thumb_ip",
)
WUJI_STAGE_MAJOR_LOWER = (
    -1.047, -1.047, -1.047, -1.047, -1.187,
    -0.698, -0.698, -0.698, -0.698, -1.484,
    -1.047, -1.047, -1.047, -1.047, -1.047,
    -1.047, -1.047, -1.047, -1.047, -1.047,
)
WUJI_STAGE_MAJOR_UPPER = (
    1.57, 1.57, 1.57, 1.57, 1.291,
    0.698, 0.698, 0.698, 0.698, 0.698,
    2.094, 2.094, 2.094, 2.094, 1.57,
    1.57, 1.57, 1.57, 1.57, 1.57,
)
WUJI_URDF_SHA256 = {
    "left": "8eafbe7a3bf8129628a53abf5e7c3f4019dc99497fcdcafac52eca9e6b939156",
    "right": "35e0018a74dd717c25f34999260aa8620309adb2deedd99f95b670222411a0b4",
}


def encode_array(array: object, *, dtype: str) -> dict[str, object]:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.dtype(dtype)))
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "data_b64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
    }


DEFAULT_TASKS = (
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "insert_block",
    "retrieve_gap",
    "stack_bowls",
)


def parse_range(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid descending range {part!r}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"invalid episode IDs {value!r}")
    return tuple(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, help="artifact JSON path, or '-' for stdout")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--train-ids", type=parse_range, default=parse_range("0-79"))
    parser.add_argument(
        "--calibration-ids", type=parse_range, default=parse_range("80-89")
    )
    parser.add_argument("--test-ids", type=parse_range, default=parse_range("90-99"))
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--calibration-quantile", type=float, default=0.999)
    parser.add_argument("--threshold-margin", type=float, default=1.20)
    parser.add_argument("--minimum-ood-rms", type=float, default=3.0)
    parser.add_argument("--minimum-ood-abs", type=float, default=8.0)
    parser.add_argument("--minimum-cycle-rms", type=float, default=0.50)
    parser.add_argument("--minimum-step-delta-rad", type=float, default=0.10)
    parser.add_argument("--step-threshold-margin", type=float, default=1.50)
    parser.add_argument("--max-limit-excess-rad", type=float, default=0.05)
    parser.add_argument("--max-geometry-rms-m", type=float, default=0.015)
    return parser.parse_args()


def episode_path(root: Path, task: str, episode_id: int) -> Path:
    name = f"episode_{episode_id:07d}.hdf5"
    candidates = (
        root / task / name,
        root / task / "tianji_marvin_wuji" / "data" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing {task}/{name}; tried " + ", ".join(str(path) for path in candidates)
    )


@dataclass
class Samples:
    euler: dict[str, list[np.ndarray]]
    q: dict[str, list[np.ndarray]]
    previous_q: dict[str, list[np.ndarray]]

    @classmethod
    def empty(cls) -> "Samples":
        return cls(
            euler={side: [] for side in SIDES},
            q={side: [] for side in SIDES},
            previous_q={side: [] for side in SIDES},
        )

    def concatenate(self, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.euler[side]:
            raise RuntimeError(f"no valid samples for side={side}")
        return (
            np.concatenate(self.euler[side]),
            np.concatenate(self.q[side]),
            np.concatenate(self.previous_q[side]),
        )


def load_split(
    root: Path,
    tasks: tuple[str, ...],
    episode_ids: tuple[int, ...],
    fingerprint: hashlib._Hash,
) -> Samples:
    samples = Samples.empty()
    for task in tasks:
        for episode_id in episode_ids:
            path = episode_path(root, task, episode_id)
            with h5py.File(path, "r") as handle:
                for side in SIDES:
                    rotvec = np.asarray(
                        handle[f"mano/action/{side}_ee_joint_states"], dtype=np.float64
                    )
                    q = np.asarray(
                        handle[f"action/{side}_ee_joint_states"], dtype=np.float64
                    )
                    previous_q = np.asarray(
                        handle[f"state/{side}_ee_joint_states"], dtype=np.float64
                    )
                    validity_path = f"mano/validity/action_{side}_ee_joint_states"
                    validity = (
                        np.asarray(handle[validity_path], dtype=bool).all(axis=1)
                        if validity_path in handle
                        else np.ones(len(rotvec), dtype=bool)
                    )
                    valid = (
                        validity
                        & np.isfinite(rotvec).all(axis=1)
                        & np.isfinite(q).all(axis=1)
                        & np.isfinite(previous_q).all(axis=1)
                    )
                    rotvec = rotvec[valid]
                    q = q[valid]
                    previous_q = previous_q[valid]
                    euler = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_euler(
                        "xyz", degrees=False
                    ).reshape(rotvec.shape)
                    samples.euler[side].append(euler)
                    samples.q[side].append(q)
                    samples.previous_q[side].append(previous_q)
                    fingerprint.update(
                        f"{task}\0{episode_id}\0{side}\0{len(q)}\0".encode("utf-8")
                    )
                    fingerprint.update(np.ascontiguousarray(rotvec).tobytes())
                    fingerprint.update(np.ascontiguousarray(q).tobytes())
                    fingerprint.update(np.ascontiguousarray(previous_q).tobytes())
    return samples


@dataclass(frozen=True)
class Ridge:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.center) / self.scale
        return np.concatenate([np.ones((len(z), 1)), z], axis=1) @ self.weights


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> Ridge:
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - center) / scale
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(gram + penalty, design.T @ y)
    return Ridge(center=center, scale=scale, weights=weights)


def wrap(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def scalar_metrics(values: Iterable[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.ridge) or args.ridge < 0:
        raise ValueError("--ridge must be finite and non-negative")
    if not 0.5 < args.calibration_quantile < 1.0:
        raise ValueError("--calibration-quantile must be in (0.5,1.0)")
    if args.threshold_margin < 1.0 or args.step_threshold_margin < 1.0:
        raise ValueError("threshold margins must be >= 1")
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks:
        raise ValueError("--tasks must not be empty")
    split_sets = [set(args.train_ids), set(args.calibration_ids), set(args.test_ids)]
    if any(split_sets[first] & split_sets[second] for first in range(3) for second in range(first + 1, 3)):
        raise ValueError("train/calibration/test episode IDs must be disjoint")

    fingerprint = hashlib.sha256()
    train = load_split(args.dataset_root, tasks, args.train_ids, fingerprint)
    calibration = load_split(
        args.dataset_root, tasks, args.calibration_ids, fingerprint
    )
    test = load_split(args.dataset_root, tasks, args.test_ids, fingerprint)
    side_payloads: dict[str, object] = {}
    heldout: dict[str, object] = {}
    frame_counts: dict[str, dict[str, int]] = {}

    for side in SIDES:
        train_euler, train_q, _train_previous = train.concatenate(side)
        calibration_euler, calibration_q, calibration_previous = calibration.concatenate(
            side
        )
        test_euler, test_q, test_previous = test.concatenate(side)
        reverse = fit_ridge(train_euler, train_q, args.ridge)
        forward = fit_ridge(train_q, train_euler, args.ridge)

        calibration_z = (calibration_euler - reverse.center) / reverse.scale
        calibration_ood_rms = np.sqrt(np.mean(calibration_z * calibration_z, axis=1))
        calibration_ood_abs = np.max(np.abs(calibration_z), axis=1)
        lower = np.asarray(WUJI_STAGE_MAJOR_LOWER, dtype=np.float64)
        upper = np.asarray(WUJI_STAGE_MAJOR_UPPER, dtype=np.float64)
        calibration_pred = np.clip(
            reverse.predict(calibration_euler), lower, upper
        )
        calibration_cycle = wrap(forward.predict(calibration_pred) - calibration_euler)
        calibration_cycle_rms = np.sqrt(
            np.mean((calibration_cycle / reverse.scale) ** 2, axis=1)
        )
        calibration_step = np.abs(calibration_q - calibration_previous)

        quantile = args.calibration_quantile
        max_ood_rms = max(
            args.minimum_ood_rms,
            float(np.quantile(calibration_ood_rms, quantile)) * args.threshold_margin,
        )
        max_ood_abs = max(
            args.minimum_ood_abs,
            float(np.quantile(calibration_ood_abs, quantile)) * args.threshold_margin,
        )
        max_cycle_rms = max(
            args.minimum_cycle_rms,
            float(np.quantile(calibration_cycle_rms, quantile))
            * args.threshold_margin,
        )
        max_step_delta = np.maximum(
            args.minimum_step_delta_rad,
            np.quantile(calibration_step, quantile, axis=0)
            * args.step_threshold_margin,
        )

        test_pred_raw = reverse.predict(test_euler)
        test_limit_excess = np.maximum(lower - test_pred_raw, 0.0) + np.maximum(
            test_pred_raw - upper, 0.0
        )
        test_max_limit_excess = np.max(test_limit_excess, axis=1)
        test_pred = np.clip(test_pred_raw, lower, upper)
        test_abs = np.abs(test_pred - test_q)
        test_z = (test_euler - reverse.center) / reverse.scale
        test_ood_rms = np.sqrt(np.mean(test_z * test_z, axis=1))
        test_ood_abs = np.max(np.abs(test_z), axis=1)
        test_cycle = wrap(forward.predict(test_pred) - test_euler)
        test_cycle_rms = np.sqrt(np.mean((test_cycle / reverse.scale) ** 2, axis=1))
        test_step_ratio = np.max(
            np.abs(test_pred - test_previous) / max_step_delta, axis=1
        )
        learned_accepted = (
            (test_ood_rms <= max_ood_rms)
            & (test_ood_abs <= max_ood_abs)
            & (test_cycle_rms <= max_cycle_rms)
            & (test_step_ratio <= 1.0)
        )
        runtime_pre_geometry_accepted = learned_accepted & (
            test_max_limit_excess <= args.max_limit_excess_rad
        )

        side_payloads[side] = {
            "reverse_center": encode_array(reverse.center, dtype="float64"),
            "reverse_scale": encode_array(reverse.scale, dtype="float64"),
            "reverse_weights": encode_array(reverse.weights, dtype="float32"),
            "forward_center": encode_array(forward.center, dtype="float64"),
            "forward_scale": encode_array(forward.scale, dtype="float64"),
            "forward_weights": encode_array(forward.weights, dtype="float32"),
            "max_ood_rms": max_ood_rms,
            "max_ood_abs": max_ood_abs,
            "max_cycle_rms": max_cycle_rms,
            "max_step_delta_rad": encode_array(max_step_delta, dtype="float64"),
            "max_limit_excess_rad": args.max_limit_excess_rad,
            "max_geometry_rms_m": args.max_geometry_rms_m,
        }
        heldout[side] = {
            "frames": int(len(test_q)),
            "q_abs_rad": scalar_metrics(test_abs),
            "per_frame_q_mae_rad": scalar_metrics(np.mean(test_abs, axis=1)),
            "ood_rms": scalar_metrics(test_ood_rms),
            "ood_abs": scalar_metrics(test_ood_abs),
            "cycle_rms": scalar_metrics(test_cycle_rms),
            "step_delta_ratio": scalar_metrics(test_step_ratio),
            "urdf_limit_excess_rad": scalar_metrics(test_max_limit_excess),
            "urdf_clip_fraction": float(np.mean(test_max_limit_excess > 0.0)),
            "learned_gate_accept_fraction": float(np.mean(learned_accepted)),
            "runtime_pre_geometry_gate_accepted_frames": int(
                np.sum(runtime_pre_geometry_accepted)
            ),
            "runtime_pre_geometry_gate_accept_fraction": float(
                np.mean(runtime_pre_geometry_accepted)
            ),
            # The remaining exact hand.evaluate geometry gate is audited by
            # validate_mano45_linear_inverse_runtime.py with the pinned URDF.
            "accepted_q_abs_rad": scalar_metrics(
                test_abs[runtime_pre_geometry_accepted]
            )
            if np.any(runtime_pre_geometry_accepted)
            else None,
        }
        frame_counts[side] = {
            "train": int(len(train_q)),
            "calibration": int(len(calibration_q)),
            "test": int(len(test_q)),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "input_representation": INPUT_REPRESENTATION,
        "model_input_encoding": MODEL_INPUT_ENCODING,
        "output_representation": OUTPUT_REPRESENTATION,
        "units": "radians",
        "mano_joint_order": list(MANO_JOINT_ORDER),
        "mano_components": ["x", "y", "z"],
        "wuji_stage_major_order": list(WUJI_STAGE_MAJOR_ORDER),
        "deployment_urdf_contract": {
            "lower_stage_major_rad": list(WUJI_STAGE_MAJOR_LOWER),
            "upper_stage_major_rad": list(WUJI_STAGE_MAJOR_UPPER),
            "sha256": WUJI_URDF_SHA256,
        },
        "training_provenance": {
            "dataset": "spark0_bench_mano",
            "tasks": list(tasks),
            "episode_split": {
                "train": f"{min(args.train_ids)}-{max(args.train_ids)}",
                "calibration": f"{min(args.calibration_ids)}-{max(args.calibration_ids)}",
                "test": f"{min(args.test_ids)}-{max(args.test_ids)}",
            },
            "episode_ids": {
                "train": list(args.train_ids),
                "calibration": list(args.calibration_ids),
                "test": list(args.test_ids),
            },
            "frame_counts": frame_counts,
            "dataset_pair_sha256": fingerprint.hexdigest(),
            "ridge": args.ridge,
            "calibration_quantile": args.calibration_quantile,
            "threshold_margin": args.threshold_margin,
            "step_threshold_margin": args.step_threshold_margin,
        },
        "sides": side_payloads,
        "heldout_metrics": heldout,
    }
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(f"wrote {output}", file=sys.stderr)
        print(f"sha256 {digest}", file=sys.stderr)
    print(json.dumps(heldout, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
