#!/usr/bin/env python3
"""Verify the eight complete VITRA checkpoints and write a SHA-256 manifest."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


EXPECTED_STEPS = tuple(range(10_000, 80_001, 10_000))
CHECKPOINT_NAME = re.compile(r"^epoch=(?P<epoch>[0-9]+)-step=(?P<step>[0-9]+)\.ckpt$")
FILES_TO_HASH = ("weights.pt", "optimizer.pt", "meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to <checkpoint-root>/checkpoint_manifest.sha256.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_file(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a state-dict-like mapping")
    summary = {
        "mapping_entries": len(payload),
        "top_level_type": type(payload).__name__,
    }
    del payload
    gc.collect()
    return summary


def discover_checkpoints(root: Path) -> dict[int, tuple[int, Path]]:
    checkpoint_dirs = sorted(path for path in root.rglob("*.ckpt") if path.is_dir())
    unexpected = []
    by_step: dict[int, tuple[int, Path]] = {}
    for checkpoint_dir in checkpoint_dirs:
        match = CHECKPOINT_NAME.fullmatch(checkpoint_dir.name)
        if match is None:
            unexpected.append(checkpoint_dir)
            continue
        epoch = int(match.group("epoch"))
        step = int(match.group("step"))
        if step in by_step:
            raise ValueError(f"Duplicate checkpoint step {step}: {by_step[step][1]} and {checkpoint_dir}")
        by_step[step] = (epoch, checkpoint_dir)
    if unexpected:
        names = ", ".join(str(path.relative_to(root)) for path in unexpected)
        raise ValueError(f"Unexpected checkpoint directories (including epoch-end saves): {names}")
    if tuple(sorted(by_step)) != EXPECTED_STEPS:
        raise ValueError(
            f"Expected exactly steps {EXPECTED_STEPS}, found {tuple(sorted(by_step))}"
        )
    return by_step


def main() -> None:
    args = parse_args()
    root = args.checkpoint_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    manifest_path = (args.manifest or root / "checkpoint_manifest.sha256.json").resolve()
    checkpoints = discover_checkpoints(root)
    expected_weight_paths = {
        checkpoint_dir / "weights.pt" for _, checkpoint_dir in checkpoints.values()
    }
    discovered_weight_paths = {path for path in root.rglob("weights.pt") if path.is_file()}
    if discovered_weight_paths != expected_weight_paths:
        unexpected = sorted(discovered_weight_paths - expected_weight_paths)
        missing = sorted(expected_weight_paths - discovered_weight_paths)
        raise ValueError(
            "Checkpoint root must contain exactly the eight expected weights.pt files; "
            f"unexpected={unexpected}, missing={missing}"
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_root": str(root),
        "expected_steps": list(EXPECTED_STEPS),
        "checkpoints": [],
    }
    for step in EXPECTED_STEPS:
        directory_epoch, checkpoint_dir = checkpoints[step]
        meta_path = checkpoint_dir / "meta.json"
        with meta_path.open(encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
        if meta.get("global_step") != step or meta.get("epoch") != directory_epoch:
            raise ValueError(f"Metadata mismatch in {meta_path}: {meta}")
        if meta.get("complete") is not True:
            raise ValueError(f"Checkpoint is not marked complete in {meta_path}")

        files: dict[str, Any] = {}
        for filename in FILES_TO_HASH:
            checkpoint_file = checkpoint_dir / filename
            if not checkpoint_file.is_file() or checkpoint_file.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty checkpoint file: {checkpoint_file}")
            file_record: dict[str, Any] = {
                "size_bytes": checkpoint_file.stat().st_size,
                "sha256": sha256_file(checkpoint_file),
            }
            if filename in {"weights.pt", "optimizer.pt"}:
                file_record["load"] = load_checkpoint_file(checkpoint_file)
            files[filename] = file_record
        manifest["checkpoints"].append(
            {
                "step": step,
                "epoch": directory_epoch,
                "directory": str(checkpoint_dir.relative_to(root)),
                "files": files,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")
    temporary_path.replace(manifest_path)
    print(f"Verified 8 complete checkpoints at steps {EXPECTED_STEPS}.")
    print(f"SHA-256 manifest: {manifest_path}")


if __name__ == "__main__":
    main()
