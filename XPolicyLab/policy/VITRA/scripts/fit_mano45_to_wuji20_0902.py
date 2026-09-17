#!/usr/bin/env python3
"""Fit a hash-bound MANO45 -> Wuji20 decoder for the 0902 VITRA run.

This fitter is intentionally run-specific.  It reads the episode paths from
the supplied 0902 manifest, verifies the saved config/statistics relationship,
uses the native MANO rotvec -> Euler-xyz conversion, and fits a side-specific
affine ridge map to the already paired Wuji20 labels.  No clipping, padding,
truncation (apart from the manifest's explicit terminal action exclusion), or
implicit joint reordering is performed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


SIDES = ("left", "right")
MANO_DIM = 45
WUJI_DIM = 20
MANO_ORDER = (
    "index_mcp", "index_pip", "index_dip", "middle_mcp", "middle_pip", "middle_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "ring_mcp", "ring_pip", "ring_dip",
    "thumb_cmc", "thumb_mcp", "thumb_ip",
)
WUJI_ORDER = (
    "index_mcp_flex", "middle_mcp_flex", "pinky_mcp_flex", "ring_mcp_flex", "thumb_cmc_flex",
    "index_mcp_abd", "middle_mcp_abd", "pinky_mcp_abd", "ring_mcp_abd", "thumb_cmc_abd",
    "index_pip", "middle_pip", "pinky_pip", "ring_pip", "thumb_mcp",
    "index_dip", "middle_dip", "pinky_dip", "ring_dip", "thumb_ip",
)
EXPECTED_STATS_SHA = "8c0170503168fba85850a0842c35081f00e9f9084b06d678381a73742b107003"
EXPECTED_MANIFEST_SHA = "ba6aa23212581ba1b426853fe4ac13129bb13e9d4c9cf754be4b868f89f26ee2"
EXPECTED_WEIGHTS_SHA = "6833a0b06137a5b330193ad3b5cde6834bdf00ea3e13752a89a19656ad270803"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def enc(a: Any, dtype: str) -> dict[str, Any]:
    x = np.ascontiguousarray(np.asarray(a, dtype=np.dtype(dtype)))
    return {"dtype": x.dtype.str, "shape": list(x.shape),
            "data_b64": base64.b64encode(x.tobytes()).decode("ascii")}


def stats(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"count": int(x.size), "mean": float(np.mean(x)),
            "p50": float(np.quantile(x, .5)), "p95": float(np.quantile(x, .95)),
            "p99": float(np.quantile(x, .99)), "max": float(np.max(x))}


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - center) / scale
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    gram = design.T @ design
    gram[1:, 1:] += ridge * np.eye(z.shape[1])
    weights = np.linalg.solve(gram, design.T @ y)
    return center, scale, weights


def predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    center, scale, weights = model
    z = (x - center) / scale
    return np.concatenate([np.ones((len(z), 1)), z], axis=1) @ weights


def load_rows(episodes: list[dict[str, Any]], *, root_alias: tuple[str, str]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {s: [[], [], []] for s in SIDES}
    for ep in episodes:
        source = Path(str(ep["source"]))
        if not source.exists() and str(source).startswith(root_alias[0]):
            source = Path(root_alias[1] + str(source)[len(root_alias[0]):])
        if not source.is_file():
            raise FileNotFoundError(source)
        with h5py.File(source, "r") as h:
            for side in SIDES:
                rv_d = h[f"mano/action/{side}_ee_joint_states"]
                q_d = h[f"action/{side}_ee_joint_states"]
                prev_d = h[f"state/{side}_ee_joint_states"]
                if rv_d.shape[1:] != (MANO_DIM,) or q_d.shape[1:] != (WUJI_DIM,) or prev_d.shape[1:] != (WUJI_DIM,):
                    raise ValueError(f"{source} {side} has unexpected shapes {rv_d.shape} {q_d.shape} {prev_d.shape}")
                order = str(rv_d.attrs.get("order", ""))
                unit = str(rv_d.attrs.get("unit", ""))
                if order != ",".join(MANO_ORDER) or unit != "axis_angle_radians":
                    raise ValueError(f"{source} {side} MANO contract is order={order!r}, unit={unit!r}")
                # The manifest explicitly says the final source row has no next action.
                rv = np.asarray(rv_d[:-1], dtype=np.float64)
                q = np.asarray(q_d[:-1], dtype=np.float64)
                prev = np.asarray(prev_d[:-1], dtype=np.float64)
                valid = np.isfinite(rv).all(1) & np.isfinite(q).all(1) & np.isfinite(prev).all(1)
                rv, q, prev = rv[valid], q[valid], prev[valid]
                euler = Rotation.from_rotvec(rv.reshape(-1, 3)).as_euler("xyz", degrees=False).reshape(-1, MANO_DIM)
                out[side][0].append(euler)
                out[side][1].append(q)
                out[side][2].append(prev)
    return {s: tuple(np.concatenate(v) for v in out[s]) for s in SIDES}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--spark-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ridge", type=float, default=1e-5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint_dir.resolve()
    config_path = args.config.resolve()
    stats_path = args.statistics.resolve()
    manifest_path = args.manifest.resolve()
    if not checkpoint.is_dir() or not (checkpoint / "weights.pt").is_file():
        raise FileNotFoundError(f"checkpoint directory/weights missing: {checkpoint}")
    config = json.loads(config_path.read_text())
    statistics = json.loads(stats_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if config.get("data_representation") != "mano45" or config.get("train_dataset", {}).get("representation") != "mano45":
        raise ValueError("saved config is not the 0902 mano45 run")
    if config.get("statistics_sha256", "").lower() != EXPECTED_STATS_SHA:
        raise ValueError("saved config statistics_sha256 does not match the required 0902 SHA")
    if sha256(stats_path) != EXPECTED_STATS_SHA:
        raise ValueError("statistics SHA mismatch")
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA:
        raise ValueError("manifest SHA mismatch")
    weights_path = checkpoint / "weights.pt"
    weights_sha = sha256(weights_path)
    if weights_sha != EXPECTED_WEIGHTS_SHA:
        raise ValueError(f"weights SHA mismatch: {weights_sha}")
    if statistics.get("representation") != "mano45" or statistics.get("state_dimension_per_hand") != 61 or statistics.get("action_dimension_per_hand") != 51:
        raise ValueError("statistics dimensions/representation are not 0902 MANO45 61/51")
    if not bool(config.get("train_dataset", {}).get("normalization")):
        raise ValueError("0902 run does not declare normalization=true")
    if manifest.get("representation") != "mano45" or int(manifest.get("episode_count", -1)) != len(manifest.get("episodes", [])):
        raise ValueError("manifest representation/episode count mismatch")
    if not np.isfinite(args.ridge) or args.ridge < 0:
        raise ValueError("ridge must be finite and non-negative")

    # Use the same source checkout and current URDF that the runtime adapter uses.
    spark = args.spark_root.resolve()
    sys.path[:0] = [str(spark / "Spark_data/src"), str(spark)]
    from spark_data.alignment.targets.wuji import WujiHandModel
    hands = {s: WujiHandModel(s, variant="wujihand2") for s in SIDES}
    urdf_paths = {s: Path(hands[s].urdf_path).resolve() for s in SIDES}
    urdf_sha = {s: sha256(urdf_paths[s]) for s in SIDES}
    lower = {s: np.asarray(hands[s].lower, dtype=np.float64) for s in SIDES}
    upper = {s: np.asarray(hands[s].upper, dtype=np.float64) for s in SIDES}
    source_rel = (
        "egovla_scripts/inspire_to_wuji.py",
        "egovla_scripts/wuji_to_inspire.py",
        "Spark_data/src/spark_data/alignment/sources/inspire_hand.py",
        "Spark_data/src/spark_data/alignment/targets/wuji.py",
        "Spark_data/src/spark_data/alignment/sources/egovla.py",
        "Spark_data/src/spark_data/alignment/mano_wuji_map.py",
    )
    source_hashes = {rel: sha256(spark / rel) for rel in source_rel}

    episodes = manifest["episodes"]
    split = {
        "train": [e for i, e in enumerate(episodes) if i % 10 not in (8, 9)],
        "calibration": [e for i, e in enumerate(episodes) if i % 10 == 8],
        "test": [e for i, e in enumerate(episodes) if i % 10 == 9],
    }
    rows = {name: load_rows(value, root_alias=("/mnt/xspark-data/", "/personal/")) for name, value in split.items()}
    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sides_payload: dict[str, Any] = {}
    heldout: dict[str, Any] = {}
    frame_counts: dict[str, dict[str, int]] = {}
    for side in SIDES:
        x_train, y_train, prev_train = rows["train"][side]
        x_cal, y_cal, prev_cal = rows["calibration"][side]
        x_test, y_test, prev_test = rows["test"][side]
        model = fit_ridge(x_train, y_train, args.ridge)
        models[side] = model
        c, sc, w = model
        cal_pred = predict(model, x_cal)
        cal_z = (x_cal - c) / sc
        cal_ood_rms = np.sqrt(np.mean(cal_z * cal_z, axis=1))
        cal_ood_abs = np.max(np.abs(cal_z), axis=1)
        cal_step = np.abs(y_cal - prev_cal)
        fwd = fit_ridge(y_train, x_train, args.ridge)
        cal_cycle = (predict(fwd, cal_pred) - x_cal + np.pi) % (2 * np.pi) - np.pi
        cal_cycle_rms = np.sqrt(np.mean((cal_cycle / sc) ** 2, axis=1))
        quantile = .999
        max_ood_rms = max(3.0, float(np.quantile(cal_ood_rms, quantile) * 1.2))
        max_ood_abs = max(8.0, float(np.quantile(cal_ood_abs, quantile) * 1.2))
        max_cycle_rms = max(.5, float(np.quantile(cal_cycle_rms, quantile) * 1.2))
        max_step_delta = np.maximum(.1, np.quantile(cal_step, quantile, axis=0) * 1.5)
        pred = predict(model, x_test)
        excess = np.maximum(lower[side] - pred, 0) + np.maximum(pred - upper[side], 0)
        z = (x_test - c) / sc
        cycle = (predict(fwd, pred) - x_test + np.pi) % (2 * np.pi) - np.pi
        cycle_rms = np.sqrt(np.mean((cycle / sc) ** 2, axis=1))
        step_ratio = np.max(np.abs(pred - prev_test) / max_step_delta, axis=1)
        learned = (np.sqrt(np.mean(z*z, axis=1)) <= max_ood_rms) & (np.max(np.abs(z), axis=1) <= max_ood_abs) & (cycle_rms <= max_cycle_rms) & (step_ratio <= 1)
        accepted = learned & (np.max(excess, axis=1) <= 0.0)
        geom = []
        for q_p, q_t in zip(pred, y_test):
            pp = np.asarray(hands[side].forward_points(q_p, canonical=False), dtype=np.float64)
            pt = np.asarray(hands[side].forward_points(q_t, canonical=False), dtype=np.float64)
            geom.append(float(np.sqrt(np.mean((pp - pt) ** 2))))
        geom = np.asarray(geom)
        geom_acc = geom[accepted]
        sides_payload[side] = {
            "reverse_center": enc(c, "float64"), "reverse_scale": enc(sc, "float64"),
            "reverse_weights": enc(w, "float32"),
            "forward_center": enc(fwd[0], "float64"), "forward_scale": enc(fwd[1], "float64"),
            "forward_weights": enc(fwd[2], "float32"),
            "max_ood_rms": max_ood_rms, "max_ood_abs": max_ood_abs,
            "max_cycle_rms": max_cycle_rms, "max_step_delta_rad": enc(max_step_delta, "float64"),
            "max_limit_excess_rad": 0.0, "max_geometry_rms_m": .08,
        }
        heldout[side] = {
            "frames": int(len(y_test)), "q_abs_rad": stats(np.abs(pred-y_test)),
            "q_row_rms_rad": stats(np.sqrt(np.mean((pred-y_test)**2, axis=1))),
            "ood_rms": stats(np.sqrt(np.mean(z*z, axis=1))), "ood_abs": stats(np.max(np.abs(z), axis=1)),
            "cycle_rms": stats(cycle_rms), "step_delta_ratio": stats(step_ratio),
            "urdf_limit_excess_rad": stats(np.max(excess, axis=1)),
            "urdf_limit_violation_fraction": float(np.mean(np.max(excess, axis=1) > 0)),
            "learned_gate_accept_fraction": float(np.mean(learned)),
            "no_clip_accept_fraction": float(np.mean(accepted)),
            "geometry_rms_m": stats(geom),
            "geometry_rms_m_no_clip_accepted": stats(geom_acc) if len(geom_acc) else None,
        }
        frame_counts[side] = {name: int(rows[name][side][1].shape[0]) for name in rows}

    payload = {
        "schema_version": 1,
        "artifact_kind": "xpolicylab_vitra_mano45_to_wujihand2_linear_inverse",
        "artifact_name": "mano45_to_wuji20_linear_0902_v1",
        "input_representation": "mano_local_rotvec45",
        "model_input_encoding": "mano_local_euler_xyz45",
        "output_representation": "wujihand2_q20_stage_major",
        "units": "radians",
        "mano_joint_order": list(MANO_ORDER), "mano_components": ["x", "y", "z"],
        "wuji_stage_major_order": list(WUJI_ORDER), "clip_to_urdf": False,
        "deployment_urdf_contract": {
            "lower_stage_major_rad": lower["left"].tolist(),
            "upper_stage_major_rad": upper["left"].tolist(), "sha256": urdf_sha,
        },
        "training_provenance": {
            "dataset": "egovla_humanoid_sim_mano_20260902",
            "checkpoint_dir": str(checkpoint), "weights_path": str(weights_path), "weights_sha256": weights_sha,
            "config_path": str(config_path), "config_sha256": sha256(config_path),
            "statistics_path": str(stats_path), "statistics_sha256": EXPECTED_STATS_SHA,
            "manifest_path": str(manifest_path), "manifest_sha256": EXPECTED_MANIFEST_SHA,
            "representation": "mano45", "normalization": True,
            "normalizer_contract": {
                "state_per_hand": 61, "action_per_hand": 51,
                "action_hand_slice": [6, 51], "state_hand_slice": [6, 61],
                "runtime": "normalizer.unnormalize_action before human_action_to_mano; mapping receives raw Euler radians",
            },
            "mano_rotvec_to_euler": "scipy Rotation.from_rotvec per joint -> as_euler('xyz', degrees=False)",
            "mano_joint_order": list(MANO_ORDER), "wuji_stage_major_order": list(WUJI_ORDER),
            "source_hashes": source_hashes, "spark_root": str(spark),
            "frame_contract": "source HDF5 action rows; q20 target already paired by same-row manifest conversion",
            "temporal_contract": {
                "source_ik_stride": 2, "terminal_action_exclusion": 1,
                "model_chunk": int(config.get("fwd_pred_next_n", 16)), "execute_action_chunk": 16,
                "action_chunk_mapping": "native action is unnormalized before decoding; no second shift",
            },
            "ridge": args.ridge, "episode_split": {k: len(v) for k, v in split.items()}, "frame_counts": frame_counts,
            "no_silent_transform": {"reorder": False, "clip": False, "padding": False, "truncation": "manifest terminal row only"},
        },
        "heldout_metrics": heldout,
        "calibration_contract": {"heldout_requirement": "geometry RMS <= 0.08 m", "passed": all(heldout[s]["geometry_rms_m"]["max"] <= .08 for s in SIDES)},
        "sides": sides_payload,
    }
    if not payload["calibration_contract"]["passed"]:
        raise RuntimeError("held-out geometry RMS exceeds 0.08 m")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    args.output.write_text(text)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(text.encode()).hexdigest(), "heldout": heldout}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
