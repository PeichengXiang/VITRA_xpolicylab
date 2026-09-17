#!/usr/bin/env python3
"""Augment the Spark-0 LeRobot v3 dataset with fitted MANO trajectories.

The source ``spark0_bench_mano`` tree contains the original Spark HDF5 data,
plus a ``mano`` group.  Its existing ``sim_6tasks_lerobot`` export predates
that group and therefore contains only the 54-D Wuji state/action.  This tool
preserves that LeRobot dataset and its videos, then appends the MANO leaves as
independent float64 fixed-size-list columns.

The LeRobot row contract is T-1: row i stores state[i] and next-state
action[i].  The source MANO action is already shifted, so it must not be
shifted a second time.  The valid terminal state is retained separately in
``meta/mano_terminal_states.parquet``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FORMAT_NAME = "spark0_mano_lerobot_v30_v1"
MANO_ASSET_SHA256 = "800db7acb35f4b283f63c14f12e1aafb410744920886ee7dc401a0bbe27df8a3"
QUANTILES = OrderedDict((name, value) for name, value in (
    ("q01", 0.01),
    ("q10", 0.10),
    ("q50", 0.50),
    ("q90", 0.90),
    ("q99", 0.99),
))

JOINT_ORDER = (
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
)
AXES = ("x", "y", "z")
POSE_NAMES = ("x", "y", "z", "qw", "qx", "qy", "qz")
JOINT_NAMES = tuple(f"{joint}_{axis}" for joint in JOINT_ORDER for axis in AXES)
BETA_NAMES = tuple(f"beta_{index}" for index in range(10))


def _feature_specs() -> OrderedDict[str, dict[str, Any]]:
    specs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    leaves = (
        ("ee_poses", 7, POSE_NAMES),
        ("ee_joint_states", 45, JOINT_NAMES),
        ("hand_betas", 10, BETA_NAMES),
    )
    for target in ("observation", "action"):
        for side in ("left", "right"):
            for leaf, dim, names in leaves:
                column = f"{target}.mano.{side}_{leaf}"
                hdf5_group = "state" if target == "observation" else "action"
                specs[column] = {
                    "dim": dim,
                    "names": names,
                    "hdf5_path": f"mano/{hdf5_group}/{side}_{leaf}",
                    "validity_path": (
                        f"mano/validity/{side}_{leaf[:-1] if leaf == 'ee_poses' else leaf}"
                        if target == "observation"
                        else f"mano/validity/action_{side}_{leaf[:-1] if leaf == 'ee_poses' else leaf}"
                    ),
                }
    return specs


FEATURE_SPECS = _feature_specs()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_link_tree(source: Path, destination: Path, mode: str) -> dict[str, int]:
    counts = {"files": 0, "hardlinks": 0, "copies": 0}
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        if not source_path.is_file():
            raise ValueError(f"Unsupported video tree entry: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            try:
                os.link(source_path, destination_path)
                counts["hardlinks"] += 1
            except OSError:
                shutil.copy2(source_path, destination_path)
                counts["copies"] += 1
        elif mode == "copy":
            shutil.copy2(source_path, destination_path)
            counts["copies"] += 1
        else:
            raise ValueError(f"Unsupported video mode: {mode}")
        if destination_path.stat().st_size != source_path.stat().st_size:
            raise IOError(f"Video size mismatch after copy/link: {relative}")
        counts["files"] += 1
    return counts


def _fixed_list(array: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    array = np.ascontiguousarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"Expected (*,{dim}) array, got {array.shape}")
    values = pa.array(array.reshape(-1), type=pa.float64())
    return pa.FixedSizeListArray.from_arrays(values, dim)


def _episode_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"Invalid stats input: shape={array.shape}, finite={np.isfinite(array).all()}")
    result = {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "count": np.asarray([array.shape[0]], dtype=np.int64),
    }
    for name, quantile in QUANTILES.items():
        result[name] = np.quantile(array, quantile, axis=0)
    return result


def _aggregate_stats(items: list[dict[str, np.ndarray]]) -> dict[str, list[Any]]:
    counts = np.stack([item["count"] for item in items]).astype(np.float64)
    total_count = np.sum(counts, axis=0)
    means = np.stack([item["mean"] for item in items])
    variances = np.stack([item["std"] ** 2 for item in items])
    expanded_counts = counts
    while expanded_counts.ndim < means.ndim:
        expanded_counts = np.expand_dims(expanded_counts, axis=-1)
    mean = np.sum(means * expanded_counts, axis=0) / total_count
    variance = np.sum((variances + (means - mean) ** 2) * expanded_counts, axis=0) / total_count
    result: dict[str, np.ndarray] = {
        "min": np.min(np.stack([item["min"] for item in items]), axis=0),
        "max": np.max(np.stack([item["max"] for item in items]), axis=0),
        "mean": mean,
        "std": np.sqrt(variance),
        "count": total_count.astype(np.int64),
    }
    for name in QUANTILES:
        values = np.stack([item[name] for item in items])
        result[name] = np.sum(values * expanded_counts, axis=0) / total_count
    return {name: value.tolist() for name, value in result.items()}


def _read_mano_episode(source_path: Path, expected_rows: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    terminals: dict[str, np.ndarray] = {}
    with h5py.File(source_path, "r") as handle:
        for column, spec in FEATURE_SPECS.items():
            hdf5_path = spec["hdf5_path"]
            if hdf5_path not in handle:
                raise KeyError(f"{source_path}: missing {hdf5_path}")
            full = np.asarray(handle[hdf5_path], dtype=np.float64)
            expected_shape = (expected_rows + 1, spec["dim"])
            if full.shape != expected_shape:
                raise ValueError(f"{source_path}: {hdf5_path} shape {full.shape}, expected {expected_shape}")
            retained = full[:-1]
            if not np.isfinite(retained).all():
                raise ValueError(f"{source_path}: retained values are non-finite in {hdf5_path}")
            validity_path = spec["validity_path"]
            if validity_path not in handle:
                raise KeyError(f"{source_path}: missing {validity_path}")
            validity = np.asarray(handle[validity_path], dtype=bool)
            if validity.shape != expected_shape:
                raise ValueError(
                    f"{source_path}: {validity_path} shape {validity.shape}, expected {expected_shape}"
                )
            if not validity[:-1].all():
                raise ValueError(f"{source_path}: retained validity is false in {validity_path}")
            if column.startswith("action."):
                if np.isfinite(full[-1]).any() or validity[-1].any():
                    raise ValueError(f"{source_path}: action terminal is not fully invalid in {hdf5_path}")
            else:
                if not np.isfinite(full[-1]).all() or not validity[-1].all():
                    raise ValueError(f"{source_path}: observation terminal is invalid in {hdf5_path}")
                terminals[column] = full[-1].copy()
            arrays[column] = retained

        for side in ("left", "right"):
            for leaf in ("ee_poses", "ee_joint_states", "hand_betas"):
                observation = arrays[f"observation.mano.{side}_{leaf}"]
                action = arrays[f"action.mano.{side}_{leaf}"]
                source_state = np.asarray(handle[f"mano/state/{side}_{leaf}"], dtype=np.float64)
                if not np.array_equal(action, source_state[1:]):
                    max_abs = float(np.max(np.abs(action - source_state[1:])))
                    raise ValueError(
                        f"{source_path}: MANO action is not state[t+1] for {side}_{leaf}; max_abs={max_abs}"
                    )
    return arrays, terminals


def _update_huggingface_metadata(table: pa.Table) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    raw = metadata.get(b"huggingface")
    huggingface = json.loads(raw.decode("utf-8")) if raw else {"info": {"features": {}}}
    features = huggingface.setdefault("info", {}).setdefault("features", {})
    for column, spec in FEATURE_SPECS.items():
        features[column] = {
            "feature": {"dtype": "float64", "_type": "Value"},
            "length": spec["dim"],
            "_type": "List",
        }
    huggingface.pop("fingerprint", None)
    metadata[b"huggingface"] = json.dumps(huggingface, separators=(",", ":")).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def _feature_info_entry(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "dtype": "float64",
        "shape": [spec["dim"]],
        "names": [list(spec["names"])],
    }


def _build_mano_metadata(source_root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": FORMAT_NAME,
        "source_root": str(source_root.resolve()),
        "parameterization": {
            "ee_poses": {
                "shape_per_hand": [7],
                "order": list(POSE_NAMES),
                "position_frame": "env/world frame",
                "position_anchor": "numerically anchored to the source Wuji Link7 EE position",
                "position_warning": (
                    "The source HDF5 labels xyz as a physical MANO wrist, but the stored values "
                    "equal the source Wuji Link7 EE xyz. Treat Link7 as the verified anchor until "
                    "a separate Link7-to-physical-wrist calibration is supplied."
                ),
                "orientation": "MANO global_orient quaternion in wxyz order; not Wuji palm-to-EE orientation",
            },
            "ee_joint_states": {
                "shape_per_hand": [45],
                "representation": "15 local axis-angle rotation vectors",
                "unit": "radians",
                "joint_order": list(JOINT_ORDER),
                "component_order_per_joint": list(AXES),
                "warning": "These are rotation vectors, not Euler xyz. Convert rotvec -> matrix -> Euler xyz for VITRA.",
            },
            "hand_betas": {
                "shape_per_hand": [10],
                "order": list(BETA_NAMES),
            },
            "left_hand_convention": "mirror_model_true_left_hand",
            "translation": "MANO transl is not stored; ee_poses xyz stores the fitted wrist position",
        },
        "temporal_contract": {
            "source_hdf5_frames": 151410,
            "lerobot_rows": 150810,
            "episodes": 600,
            "row_i_observation": "mano/state/*[i]",
            "row_i_action": "mano/action/*[i] == mano/state/*[i+1]",
            "terminal_action": "one all-NaN/invalid row per source episode is dropped",
            "terminal_observation": "stored in meta/mano_terminal_states.parquet",
        },
        "quality_scope": {
            "retained_rows": "all MANO state/action leaves are finite and source validity masks are true",
            "important": "Finite/valid does not prove that MANO retarget residual is below a geometric threshold.",
        },
        "mano_asset_sha256": MANO_ASSET_SHA256,
        "storage_dtype": "float64 (preserves source MANO array precision)",
        "statistics_contract": (
            "mean/std are exact population aggregates; q01/q10/q50/q90/q99 follow LeRobot's "
            "frame-count-weighted aggregation of per-episode quantiles"
        ),
    }


def _append_episode_stats(
    base_table: pa.Table,
    stats_by_feature: dict[str, list[dict[str, np.ndarray]]],
) -> pa.Table:
    result = base_table
    for feature in FEATURE_SPECS:
        feature_stats = stats_by_feature[feature]
        if len(feature_stats) != base_table.num_rows:
            raise ValueError(f"Episode stats count mismatch for {feature}")
        for stat_name in ("min", "max", "mean", "std", "count", *QUANTILES.keys()):
            values = [item[stat_name].tolist() for item in feature_stats]
            value_type = pa.int64() if stat_name == "count" else pa.float64()
            result = result.append_column(
                f"stats/{feature}/{stat_name}",
                pa.array(values, type=pa.list_(value_type)),
            )
    return result


def _write_checksums(root: Path) -> None:
    checksum_path = root / "meta" / "CHECKSUMS.sha256"
    files = [path for path in root.rglob("*") if path.is_file() and path != checksum_path]
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in sorted(files)]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_published_dataset(root: Path, expected_episodes: int, expected_rows: int) -> None:
    data_path = root / "data/chunk-000/file-000.parquet"
    parquet = pq.ParquetFile(data_path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"Published row count {parquet.metadata.num_rows}, expected {expected_rows}")
    if parquet.metadata.num_row_groups != expected_episodes:
        raise ValueError(
            f"Published row-group count {parquet.metadata.num_row_groups}, expected {expected_episodes}"
        )
    names = set(parquet.schema_arrow.names)
    missing = set(FEATURE_SPECS) - names
    if missing:
        raise ValueError(f"Published parquet misses MANO columns: {sorted(missing)}")
    for feature, spec in FEATURE_SPECS.items():
        field = parquet.schema_arrow.field(feature)
        if not pa.types.is_fixed_size_list(field.type) or field.type.list_size != spec["dim"]:
            raise ValueError(f"Published field type mismatch for {feature}: {field.type}")
        if field.type.value_type != pa.float64():
            raise ValueError(f"Published field dtype mismatch for {feature}: {field.type}")

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != expected_episodes or info["total_frames"] != expected_rows:
        raise ValueError("info.json totals do not match the parquet")
    for feature, spec in FEATURE_SPECS.items():
        entry = info["features"].get(feature)
        if entry is None or entry["dtype"] != "float64" or entry["shape"] != [spec["dim"]]:
            raise ValueError(f"info.json feature mismatch for {feature}: {entry}")

    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet")
    if episodes.num_rows != expected_episodes:
        raise ValueError("Episode metadata row count mismatch")
    for feature in FEATURE_SPECS:
        for stat_name in ("min", "max", "mean", "std", "count", *QUANTILES.keys()):
            if f"stats/{feature}/{stat_name}" not in episodes.column_names:
                raise ValueError(f"Missing per-episode stats for {feature}/{stat_name}")

    terminal = pq.read_table(root / "meta/mano_terminal_states.parquet")
    if terminal.num_rows != expected_episodes:
        raise ValueError("Terminal-state metadata row count mismatch")
    for column in terminal.column_names:
        if column.startswith("terminal.") and terminal[column].null_count:
            raise ValueError(f"Null values in terminal metadata column {column}")

    retained = pq.read_table(data_path, columns=list(FEATURE_SPECS))
    for feature in FEATURE_SPECS:
        values = np.asarray(retained[feature].combine_chunks().values).reshape(expected_rows, -1)
        if values.shape[1] != FEATURE_SPECS[feature]["dim"] or not np.isfinite(values).all():
            raise ValueError(f"Non-finite or malformed published feature {feature}")


def _preflight(source_root: Path, base_root: Path, max_episodes: int | None) -> None:
    manifest = json.loads((base_root / "meta/shard_manifest.json").read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    parquet = pq.ParquetFile(base_root / "data/chunk-000/file-000.parquet")
    if parquet.metadata.num_row_groups < len(episodes):
        raise ValueError("Base parquet has fewer row groups than the manifest")
    total = 0
    for offset, item in enumerate(episodes):
        episode_index = int(item["local_episode_index"])
        if episode_index != offset:
            raise ValueError(f"Manifest episode index discontinuity at {offset}: {episode_index}")
        table = parquet.read_row_group(episode_index, columns=["episode_index", "frame_index"])
        source_path = source_root / item["source_relative"]
        _read_mano_episode(source_path, table.num_rows)
        total += table.num_rows
        if (offset + 1) % 25 == 0 or offset + 1 == len(episodes):
            print(f"preflight {offset + 1}/{len(episodes)} episodes, {total} rows", flush=True)


def convert(source_root: Path, base_root: Path, output_root: Path, video_mode: str) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    for required in (
        base_root / "data/chunk-000/file-000.parquet",
        base_root / "meta/episodes/chunk-000/file-000.parquet",
        base_root / "meta/info.json",
        base_root / "meta/stats.json",
        base_root / "meta/modality.json",
        base_root / "meta/shard_manifest.json",
        base_root / "meta/tasks.parquet",
        base_root / "videos",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex[:10]}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        (staging / "data/chunk-000").mkdir(parents=True)
        (staging / "meta/episodes/chunk-000").mkdir(parents=True)
        shutil.copy2(base_root / "meta/tasks.parquet", staging / "meta/tasks.parquet")
        video_counts = _copy_or_link_tree(base_root / "videos", staging / "videos", video_mode)
        print(f"videos ready: {video_counts}", flush=True)

        manifest = json.loads((base_root / "meta/shard_manifest.json").read_text(encoding="utf-8"))
        manifest_episodes = manifest.get("episodes", [])
        base_data_path = base_root / "data/chunk-000/file-000.parquet"
        base_parquet = pq.ParquetFile(base_data_path)
        expected_episodes = int(manifest.get("global_source_count", len(manifest_episodes)))
        if len(manifest_episodes) != expected_episodes:
            raise ValueError("Shard manifest episode count mismatch")
        if base_parquet.metadata.num_row_groups != expected_episodes:
            raise ValueError("Base parquet must have exactly one row group per episode")

        stats_by_feature: dict[str, list[dict[str, np.ndarray]]] = {
            feature: [] for feature in FEATURE_SPECS
        }
        terminal_records: dict[str, list[Any]] = {
            "episode_index": [],
            "source_relative": [],
            "source_frame_index": [],
        }
        for feature in FEATURE_SPECS:
            if feature.startswith("observation."):
                terminal_records[f"terminal.{feature}"] = []

        output_data_path = staging / "data/chunk-000/file-000.parquet"
        writer: pq.ParquetWriter | None = None
        expected_rows = 0
        try:
            for episode_offset, item in enumerate(manifest_episodes):
                episode_index = int(item["local_episode_index"])
                if episode_index != episode_offset:
                    raise ValueError(
                        f"Manifest episode index discontinuity: expected {episode_offset}, got {episode_index}"
                    )
                table = base_parquet.read_row_group(episode_index)
                episode_values = table["episode_index"].to_numpy(zero_copy_only=False)
                frame_values = table["frame_index"].to_numpy(zero_copy_only=False)
                if not np.all(episode_values == episode_index):
                    raise ValueError(f"Base row group {episode_index} has wrong episode_index")
                if not np.array_equal(frame_values, np.arange(table.num_rows, dtype=frame_values.dtype)):
                    raise ValueError(f"Base row group {episode_index} frame_index is not contiguous")

                source_relative = str(item["source_relative"])
                source_path = source_root / source_relative
                arrays, terminals = _read_mano_episode(source_path, table.num_rows)
                for feature, spec in FEATURE_SPECS.items():
                    table = table.append_column(feature, _fixed_list(arrays[feature], spec["dim"]))
                    stats_by_feature[feature].append(_episode_stats(arrays[feature]))
                table = _update_huggingface_metadata(table)
                if writer is None:
                    writer = pq.ParquetWriter(output_data_path, table.schema, compression="snappy")
                elif table.schema != writer.schema:
                    raise ValueError(f"Arrow schema drift at episode {episode_index}")
                writer.write_table(table, row_group_size=table.num_rows)

                terminal_records["episode_index"].append(episode_index)
                terminal_records["source_relative"].append(source_relative)
                terminal_records["source_frame_index"].append(table.num_rows)
                for feature, value in terminals.items():
                    terminal_records[f"terminal.{feature}"].append(value.tolist())

                expected_rows += table.num_rows
                if (episode_index + 1) % 25 == 0 or episode_index + 1 == expected_episodes:
                    print(
                        f"converted {episode_index + 1}/{expected_episodes} episodes, {expected_rows} rows",
                        flush=True,
                    )
        finally:
            if writer is not None:
                writer.close()

        base_info = json.loads((base_root / "meta/info.json").read_text(encoding="utf-8"))
        if base_info["total_episodes"] != expected_episodes or base_info["total_frames"] != expected_rows:
            raise ValueError("Base info totals do not match converted totals")
        base_info["robot_type"] = "tianji_marvin_wujihand2_mano"
        # This augmented parquet is about 349 MB; keep the metadata target honest.
        base_info["data_files_size_in_mb"] = 400
        for feature, spec in FEATURE_SPECS.items():
            base_info["features"][feature] = _feature_info_entry(spec)
        _json_dump(staging / "meta/info.json", base_info)

        global_stats = json.loads((base_root / "meta/stats.json").read_text(encoding="utf-8"))
        for feature in FEATURE_SPECS:
            global_stats[feature] = _aggregate_stats(stats_by_feature[feature])
        _json_dump(staging / "meta/stats.json", global_stats)

        episode_table = pq.read_table(base_root / "meta/episodes/chunk-000/file-000.parquet")
        if episode_table.num_rows != expected_episodes:
            raise ValueError("Base episode metadata count mismatch")
        episode_table = _append_episode_stats(episode_table, stats_by_feature)
        pq.write_table(
            episode_table,
            staging / "meta/episodes/chunk-000/file-000.parquet",
            compression="snappy",
            row_group_size=10,
        )

        terminal_arrays: dict[str, pa.Array] = {}
        terminal_arrays["episode_index"] = pa.array(terminal_records["episode_index"], type=pa.int64())
        terminal_arrays["source_relative"] = pa.array(terminal_records["source_relative"], type=pa.string())
        terminal_arrays["source_frame_index"] = pa.array(
            terminal_records["source_frame_index"], type=pa.int64()
        )
        for feature, spec in FEATURE_SPECS.items():
            if feature.startswith("observation."):
                values = np.asarray(terminal_records[f"terminal.{feature}"], dtype=np.float64)
                terminal_arrays[f"terminal.{feature}"] = _fixed_list(values, spec["dim"])
        pq.write_table(
            pa.table(terminal_arrays),
            staging / "meta/mano_terminal_states.parquet",
            compression="snappy",
        )

        modality = json.loads((base_root / "meta/modality.json").read_text(encoding="utf-8"))
        modality["mano"] = {
            "storage": "separate_float64_columns",
            "state": {
                side: {
                    leaf: {"original_key": f"observation.mano.{side}_{leaf}", "absolute": True}
                    for leaf in ("ee_poses", "ee_joint_states", "hand_betas")
                }
                for side in ("left", "right")
            },
            "action": {
                side: {
                    leaf: {"original_key": f"action.mano.{side}_{leaf}", "absolute": True}
                    for leaf in ("ee_poses", "ee_joint_states", "hand_betas")
                }
                for side in ("left", "right")
            },
            "parameterization_metadata": "meta/mano.json",
            "terminal_states": "meta/mano_terminal_states.parquet",
        }
        _json_dump(staging / "meta/modality.json", modality)
        _json_dump(staging / "meta/mano.json", _build_mano_metadata(source_root))

        manifest["format"] = FORMAT_NAME
        manifest["augmentation"] = {
            "base_lerobot_root": str(base_root.resolve()),
            "source_root": str(source_root.resolve()),
            "mano_columns": list(FEATURE_SPECS),
            "dtype": "float64",
        }
        _json_dump(staging / "meta/shard_manifest.json", manifest)

        conversion_manifest = {
            "format": FORMAT_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root.resolve()),
            "base_lerobot_root": str(base_root.resolve()),
            "episodes": expected_episodes,
            "rows": expected_rows,
            "source_hdf5_frames": expected_rows + expected_episodes,
            "video_storage": video_counts,
            "video_policy": "Reuse the existing 3-camera LeRobot streams; third-person camera is not exported.",
            "mano_columns": list(FEATURE_SPECS),
            "mano_dtype": "float64",
            "terminal_state_file": "meta/mano_terminal_states.parquet",
        }
        _json_dump(staging / "meta/conversion_manifest.json", conversion_manifest)

        readme = f"""# Spark-0 MANO LeRobot v3.0

This dataset augments the original 600-episode Spark-0 LeRobot export with the fitted MANO arrays from `{source_root}`.

- Rows: **{expected_rows:,}** (`T-1` per episode)
- Episodes: **{expected_episodes}**
- FPS: **{base_info['fps']}**
- Robot label: **tianji_marvin_wujihand2_mano**.
- Existing Wuji fields and camera videos are unchanged.
- MANO leaves are separate `float64` columns under `observation.mano.*` and `action.mano.*`.
- The source MANO action is already `state[t+1]`; it was not shifted again.
- The valid final observation of each episode is stored in `meta/mano_terminal_states.parquet`.
- See `meta/mano.json` before converting the 45-D rotation vectors to VITRA Euler-xyz.

The MANO xyz values are verified to equal the source Wuji **Link7 EE** xyz; they are not an independently calibrated physical-wrist translation. `valid` means finite and source validity-mask true, not that the geometric retarget residual passed a threshold.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")

        print("validating published staging dataset", flush=True)
        _validate_published_dataset(staging, expected_episodes, expected_rows)
        print("computing SHA256 manifest", flush=True)
        _write_checksums(staging)
        os.rename(staging, output_root)
        print(f"published: {output_root}", flush=True)
    except Exception:
        failure = staging / "FAILED.txt"
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"conversion failed; staging retained at {staging}", file=sys.stderr, flush=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-lerobot-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--video-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if not args.preflight_only and args.max_episodes is not None:
        parser.error("--max-episodes is only supported with --preflight-only")
    if not args.preflight_only and args.output_root is None:
        parser.error("--output-root is required unless --preflight-only is used")
    return args


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    base_root = (
        args.base_lerobot_root.resolve()
        if args.base_lerobot_root is not None
        else source_root / "sim_6tasks_lerobot"
    )
    if args.preflight_only:
        _preflight(source_root, base_root, args.max_episodes)
        return
    convert(source_root, base_root, args.output_root.resolve(), args.video_mode)


if __name__ == "__main__":
    main()
