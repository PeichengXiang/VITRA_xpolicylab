#!/usr/bin/env python3
"""Materialize a run-specific VITRA config without persisting credentials."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path


EXPECTED_MAX_STEPS = 80_000
EXPECTED_SAVE_STEPS = 10_000
EXPECTED_CHUNK_SIZE = 16
EXPECTED_STATE_DIM = 212
EXPECTED_ACTION_DIM = 192
EXPECTED_PER_DEVICE_BATCH_SIZE = 8
EXPECTED_GLOBAL_BATCH_SIZE = 64
EXPECTED_USE_BF16 = True
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
CHECKPOINT_NAME = re.compile(r"^epoch=(?P<epoch>[0-9]+)-step=(?P<step>[0-9]+)\.ckpt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--representation", required=True, choices=("wuji20", "mano45", "inspire12"))
    parser.add_argument("--bench-name", required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument(
        "--run-date",
        required=True,
        help="ISO calendar date (YYYY-MM-DD) embedded in the checkpoint run directory.",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Explicit complete epoch=E-step=S.ckpt directory to resume from.",
    )
    return parser.parse_args()


def apply_per_device_batch(config: dict) -> int:
    raw = os.environ.get("VITRA_PER_DEVICE_BATCH")
    per_device = int(raw) if raw else int(config["batch_size"])
    if per_device not in (2, 4, 8):
        raise ValueError(f"per-device batch must be 2, 4, or 8, got {per_device}")
    if EXPECTED_GLOBAL_BATCH_SIZE % per_device != 0:
        raise ValueError(
            f"global batch {EXPECTED_GLOBAL_BATCH_SIZE} is not divisible by {per_device}"
        )
    config["batch_size"] = per_device
    return per_device


def validate_static_config(config: dict) -> None:
    apply_per_device_batch(config)
    checks = {
        "trainer.max_steps": config["trainer"]["max_steps"] == EXPECTED_MAX_STEPS,
        "save_steps": config["save_steps"] == EXPECTED_SAVE_STEPS,
        "epoch_save_interval": config["epoch_save_interval"] == 0,
        "fwd_pred_next_n": config["fwd_pred_next_n"] == EXPECTED_CHUNK_SIZE,
        "state_encoder.state_dim": config["state_encoder"]["state_dim"] == EXPECTED_STATE_DIM,
        "action_model.action_dim": config["action_model"]["action_dim"] == EXPECTED_ACTION_DIM,
        "batch_size": config["batch_size"] in (2, 4, 8),
        "total_batch_size": config["total_batch_size"] == EXPECTED_GLOBAL_BATCH_SIZE,
        "use_bf16": config["use_bf16"] is EXPECTED_USE_BF16,
        "train_dataset.action_type": config["train_dataset"]["action_type"] == "angle",
        "train_dataset.use_rel": config["train_dataset"]["use_rel"] is False,
        "train_dataset.rel_mode": config["train_dataset"]["rel_mode"] == "step",
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Training invariants changed in the base config: {', '.join(failures)}")


def require_safe_name(label: str, value: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label} must match {SAFE_NAME.pattern!r}, got {value!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 < args.seed < 2**32 - 1:
        raise ValueError(f"seed must be in the range 1..{2**32 - 2}, got {args.seed}")
    bench_name = require_safe_name("bench_name", args.bench_name)
    ckpt_name = require_safe_name("ckpt_name", args.ckpt_name)
    try:
        run_date = dt.date.fromisoformat(args.run_date).isoformat()
    except ValueError as exc:
        raise ValueError(f"run-date must be a valid YYYY-MM-DD date, got {args.run_date!r}") from exc
    if run_date != args.run_date:
        raise ValueError(f"run-date must use zero-padded ISO form, got {args.run_date!r}")
    workspace_root = args.workspace_root.resolve()
    output_root = args.output_root.resolve()

    with args.base_config.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    validate_static_config(config)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Prepared data root does not exist: {data_root}")
    manifest_path = data_root / "spark0_manifest.json"
    statistics_path = data_root / "teledata_statistics.json"
    for required_data_file in (manifest_path, statistics_path):
        if not required_data_file.is_file() or required_data_file.stat().st_size == 0:
            raise FileNotFoundError(f"Prepared data file is missing: {required_data_file}")
    pretrain_root = workspace_root / "pretrain_model" / "VITRA-VLA-3B"
    paligemma_root = workspace_root / "pretrain_model" / "paligemma2-3b-mix-224-local"
    config.update(
        {
            "task_name": f"{run_date}-{bench_name}-{ckpt_name}",
            "run_date": run_date,
            "seed": args.seed,
            "output_root": str(output_root),
            "log_root": str(output_root / "logs"),
            "cache_root": str(workspace_root / "pretrain_model" / "cache"),
            "pretrain_path": str(pretrain_root / "vitra-vla-3b.pt"),
            "statistics_path": str(statistics_path),
            "statistics_sha256": sha256_file(statistics_path),
            "data_manifest_path": str(manifest_path),
            "data_manifest_sha256": sha256_file(manifest_path),
            "data_representation": args.representation,
            "wandb_project": os.environ.get("WANDB_PROJECT", config.get("wandb_project")),
            "wandb_entity": os.environ.get("WANDB_ENTITY", config.get("wandb_entity")),
            "hf_cache_dir": str(workspace_root / "pretrain_model" / "huggingface_cache"),
        }
    )
    if args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint.resolve()
        checkpoint_match = CHECKPOINT_NAME.fullmatch(resume_checkpoint.name)
        if checkpoint_match is None or resume_checkpoint.parent.name != "checkpoints":
            raise ValueError(f"Resume path is not an epoch=E-step=S.ckpt directory: {resume_checkpoint}")
        checkpoint_epoch = int(checkpoint_match.group("epoch"))
        checkpoint_step = int(checkpoint_match.group("step"))
        expected_run_name = (
            f"{run_date}-{bench_name}-{ckpt_name}_"
            f"TB{EXPECTED_GLOBAL_BATCH_SIZE}_B{config['batch_size']}_bf16{EXPECTED_USE_BF16}"
        )
        checkpoint_run_dir = resume_checkpoint.parent.parent
        if checkpoint_run_dir.parent != output_root or checkpoint_run_dir.name != expected_run_name:
            raise ValueError(
                "Resume checkpoint does not belong to the requested dated run: "
                f"found={checkpoint_run_dir}, expected={output_root / expected_run_name}"
            )
        meta_path = resume_checkpoint / "meta.json"
        weights_path = resume_checkpoint / "weights.pt"
        optimizer_path = resume_checkpoint / "optimizer.pt"
        if not all(path.is_file() and path.stat().st_size > 0 for path in (meta_path, weights_path, optimizer_path)):
            raise ValueError(f"Resume checkpoint is incomplete: {resume_checkpoint}")
        with meta_path.open(encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
        if (
            meta.get("complete") is not True
            or meta.get("epoch") != checkpoint_epoch
            or meta.get("global_step") != checkpoint_step
            or checkpoint_step % EXPECTED_SAVE_STEPS != 0
            or not 0 < checkpoint_step < EXPECTED_MAX_STEPS
        ):
            raise ValueError(f"Resume checkpoint metadata is invalid: {meta_path}: {meta}")
        config["resume"] = True
        config["model_load_path"] = str(resume_checkpoint)
    else:
        config["resume"] = False
        config["model_load_path"] = None
    config["train_dataset"]["data_root_dir"] = str(data_root)
    config["train_dataset"]["representation"] = args.representation
    config["vlm"]["pretrained_model_name_or_path"] = str(paligemma_root)
    config["vlm"]["initialize_from_config"] = True

    # Credentials are deliberately absent: train.py reads WANDB_API_KEY directly
    # from its environment and Hugging Face uses its normal token resolution.
    forbidden_keys = {"wandb_api_key", "WANDB_API_KEY", "hf_token", "HF_TOKEN"}
    if forbidden_keys.intersection(config):
        raise ValueError("A credential key was found in the generated config")

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_config.with_suffix(f"{args.output_config.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(config, output_file, indent=2)
        output_file.write("\n")
    temporary_path.replace(args.output_config)


if __name__ == "__main__":
    main()
