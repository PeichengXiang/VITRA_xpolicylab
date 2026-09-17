#!/usr/bin/env python3
"""Minimal native VITRA MANO45 first-batch check; intentionally does no conversion."""
from pathlib import Path
import json, sys
MODEL_ROOT = Path("/personal/xiangpc/0813_Xpolicylab_bench/VITRA")
DATA_ROOT = MODEL_ROOT / "data/spark0_bench_7task_0908_mano"
XPL_ROOT = MODEL_ROOT / "XPolicyLab"
sys.path.insert(0, str(XPL_ROOT.parent))
sys.path.insert(0, str(XPL_ROOT / "policy/VITRA/VITRA"))
print("VITRA_CHECK phase=metadata", flush=True)
manifest = json.loads((DATA_ROOT / "spark0_manifest.json").read_text())
assert manifest["representation"] == "mano45" and manifest["episode_count"] == 700
stats = json.loads((DATA_ROOT / "teledata_statistics.json").read_text())
assert stats["representation"] == "mano45" and stats["num_episodes"] == 700
assert (stats["state_dimension_per_hand"], stats["action_dimension_per_hand"]) == (61, 51)
print("VITRA_CHECK phase=native_import", flush=True)
import h5py
import numpy as np
from vitra.datasets.spark0_dataset import RoboDatasetCore, REPRESENTATION_MANO45
print("VITRA_CHECK phase=dataset_constructor", flush=True)
dataset = RoboDatasetCore(root_dir=str(DATA_ROOT / "TeleData"), statistics_path=str(DATA_ROOT / "teledata_statistics.json"), action_past_window_size=0, action_future_window_size=16, image_past_window_size=0, image_future_window_size=0, load_images=False, representation=REPRESENTATION_MANO45)
assert len(dataset.episode_paths) == 700
assert len(dataset) == manifest["frame_count"]
print("VITRA_CHECK phase=first_sample", flush=True)
sample = dataset[0]
assert sample["representation"] == REPRESENTATION_MANO45
assert sample["current_state"].shape == (122,)
assert sample["action_list"].shape == (17, 102)
assert sample["action_mask"].shape == (17, 2)
assert np.isfinite(sample["current_state"]).all() and np.isfinite(sample["action_list"]).all()
with h5py.File(sample["episode_path"], "r") as h5:
    # raw_action_mano_fast writes provenance on the mano group, not the file.
    provenance = h5["mano"].attrs
    for key, expected in (("action_source", "raw HDF5 action/* at same timestep t"), ("state_source", "raw HDF5 state/* at same timestep t")):
        actual = provenance[key]
        if isinstance(actual, bytes):
            actual = actual.decode("utf-8")
        assert actual == expected, (key, actual)
    assert int(provenance["action_temporal_shift"]) == 0
    assert provenance["next_state_used"] in (False, 0)
print(json.dumps({"status":"ok", "episodes":len(dataset.episode_paths), "frames":len(dataset), "representation":"mano45", "state_shape":list(sample["current_state"].shape), "action_shape":list(sample["action_list"].shape), "frame_index":int(sample["frame_index"]), "episode_path":sample["episode_path"]}, ensure_ascii=False))
