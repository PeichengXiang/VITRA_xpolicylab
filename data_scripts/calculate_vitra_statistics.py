#!/usr/bin/env python3
"""Calculate representation-specific Spark0 Gaussian statistics for VITRA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader


def add_import_paths(xpolicylab_root: Path) -> None:
    xpolicylab_root = xpolicylab_root.resolve()
    sys.path.insert(0, str(xpolicylab_root.parent))
    sys.path.insert(0, str(xpolicylab_root / "policy" / "VITRA" / "VITRA"))


@dataclass
class RunningMoments:
    dimension: int

    def __post_init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.dimension)
        if not len(values):
            return
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        delta = batch_mean - self.mean
        combined = self.count + batch_count
        self.mean += delta * (batch_count / combined)
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / combined
        self.count = combined

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("No valid samples reached statistics accumulator")
        return self.mean, np.sqrt(self.m2 / self.count)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def calculate(args: argparse.Namespace) -> dict:
    if args.std_floor <= 0:
        raise ValueError(f"--std-floor must be positive, got {args.std_floor}")
    add_import_paths(args.xpolicylab_root)
    if args.representation == "inspire12":
        from vitra.datasets.egovla_inspire_dataset import (
            EGOVLA_ACTION_CONTRACT_ID,
            EgoVLAInspireDatasetCore,
            representation_dimensions,
        )

        dataset_core_cls = EgoVLAInspireDatasetCore
        if args.mapping is None:
            args.mapping = (
                args.xpolicylab_root.resolve()
                / "policy"
                / "VITRA"
                / "mapping_inspire12_mano45.json"
            )
    else:
        from vitra.datasets.spark0_dataset import (
            REPRESENTATION_MANO45,
            SUPPORTED_REPRESENTATIONS,
            RoboDatasetCore,
            representation_dimensions,
        )

        if args.representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f"--representation must be one of {SUPPORTED_REPRESENTATIONS} or 'inspire12', "
                f"got {args.representation!r}"
            )
        if args.representation == REPRESENTATION_MANO45 and args.mapping is not None:
            raise ValueError("--mapping is only valid for --representation wuji20 or inspire12")
        dataset_core_cls = RoboDatasetCore
    state_hand_dim, action_hand_dim = representation_dimensions(args.representation)

    teledata = args.data_root.resolve() / "TeleData"
    dataset = dataset_core_cls(
        root_dir=str(teledata),
        statistics_path=None,
        action_past_window_size=0,
        action_future_window_size=0,
        image_past_window_size=0,
        image_future_window_size=0,
        load_images=False,
        mapping_path=str(args.mapping.resolve()) if args.mapping else None,
        representation=args.representation,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
    )
    state_moments = [RunningMoments(state_hand_dim), RunningMoments(state_hand_dim)]
    action_moments = [RunningMoments(action_hand_dim), RunningMoments(action_hand_dim)]

    processed = 0
    for batch_index, batch in enumerate(loader):
        state = batch["current_state"].numpy()
        state_mask = batch["current_state_mask"].numpy().astype(bool)
        action = batch["action_list"].numpy()[:, 0]
        action_mask = batch["action_mask"].numpy()[:, 0].astype(bool)
        for hand in range(2):
            state_span = slice(hand * state_hand_dim, (hand + 1) * state_hand_dim)
            action_span = slice(hand * action_hand_dim, (hand + 1) * action_hand_dim)
            state_moments[hand].update(state[state_mask[:, hand], state_span])
            action_moments[hand].update(action[action_mask[:, hand], action_span])
        processed += len(state)
        if args.progress_every and (batch_index + 1) % args.progress_every == 0:
            print(f"statistics: {processed}/{len(dataset)} frames")

    sides = ("left", "right")
    result = {
        "dataset_name": "robo_dataset_angle_statistics.json",
        "representation": args.representation,
        "representation_semantics": (
            "per-hand root6 + local MANO Euler45 + betas10 state; "
            "root step-delta6 + next absolute local MANO Euler45 action"
            if args.representation == "mano45"
            else "per-hand EEF6 + Inspire12 state; observed EEF step-delta6 + direct future Inspire12 execution command action[t]; normalize before sparse MANO injection"
            if args.representation == "inspire12"
            else "per-hand EEF6 + Wuji20; normalize before sparse MANO injection"
        ),
        "state_dimension_per_hand": state_hand_dim,
        "action_dimension_per_hand": action_hand_dim,
        "num_traj": len(dataset),
        "num_samples": len(dataset),
        "num_episodes": len(dataset.episode_paths),
    }
    if args.representation == "inspire12":
        result["action_contract_id"] = EGOVLA_ACTION_CONTRACT_ID
    for hand, side in enumerate(sides):
        state_mean, state_std = state_moments[hand].result()
        action_mean, action_std = action_moments[hand].result()
        near_constant_state = np.flatnonzero(state_std < args.std_floor).tolist()
        near_constant_action = np.flatnonzero(action_std < args.std_floor).tolist()
        state_std = np.maximum(state_std, args.std_floor)
        action_std = np.maximum(action_std, args.std_floor)
        result[f"state_{side}"] = {"mean": state_mean.tolist(), "std": state_std.tolist()}
        result[f"action_{side}"] = {"mean": action_mean.tolist(), "std": action_std.tolist()}
        result[f"near_constant_state_dims_{side}"] = near_constant_state
        result[f"near_constant_action_dims_{side}"] = near_constant_action
        result[f"valid_state_samples_{side}"] = state_moments[hand].count
        result[f"valid_action_samples_{side}"] = action_moments[hand].count

    output = args.output.resolve() if args.output else args.data_root.resolve() / "teledata_statistics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(f"Wrote native VITRA statistics to {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    default_xpl = Path(__file__).resolve().parents[1] / "XPolicyLab"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--xpolicylab-root", type=Path, default=default_xpl)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument(
        "--representation",
        choices=("wuji20", "mano45", "inspire12"),
        default="wuji20",
        help="Explicit source representation; mano45 never uses a sparse table map.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--std-floor", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    calculate(build_parser().parse_args())
