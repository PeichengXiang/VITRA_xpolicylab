"""Table-index Inspire12 <-> MANO45 must be exactly invertible."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR.parents[2]))
sys.path.insert(0, str(POLICY_DIR / "VITRA"))

from vitra.datasets.egovla_inspire_dataset import (  # noqa: E402
    INSPIRE2MANO_CODEC_ID,
    INSPIRE_HAND_DIM,
    NATIVE_DUAL_DIM,
    _INSPIRE2MANO_DST,
    _INSPIRE2MANO_SIGNS,
    human_action_to_inspire,
    inspire12_to_mano45,
    inspire_action_to_human,
    load_inspire_mapping,
    mano45_to_inspire12,
    unpack_inspire12,
)


def test_default_mapping_is_locked_table() -> None:
    spec = load_inspire_mapping()
    assert spec["codec_id"] == INSPIRE2MANO_CODEC_ID
    src = [int(entry["source_index"]) for entry in spec["joint_mapping"]]
    dst = [int(entry["mano_pose_index"]) for entry in spec["joint_mapping"]]
    signs = [int(entry["sign"]) for entry in spec["joint_mapping"]]
    assert src == list(range(INSPIRE_HAND_DIM))
    assert tuple(dst) == _INSPIRE2MANO_DST
    assert tuple(signs) == _INSPIRE2MANO_SIGNS


def test_table_index_roundtrip_keeps_inspire12() -> None:
    rng = np.random.default_rng(0)
    q12 = rng.normal(size=(8, INSPIRE_HAND_DIM)).astype(np.float32)
    pose = inspire12_to_mano45(q12)
    recovered = mano45_to_inspire12(pose)
    np.testing.assert_allclose(recovered, q12, atol=0, rtol=0)
    assert pose.shape[-1] == 45
    unused = np.ones(45, dtype=bool)
    unused[list(_INSPIRE2MANO_DST)] = False
    np.testing.assert_array_equal(pose[..., unused], 0)


def test_native_action_roundtrip_keeps_36d() -> None:
    rng = np.random.default_rng(1)
    native = rng.normal(size=(4, NATIVE_DUAL_DIM)).astype(np.float32)
    human, mask = inspire_action_to_human(native, return_mask=True)
    assert human.shape == (4, 192)
    assert mask.shape == (4, 192)
    recovered = human_action_to_inspire(human)
    np.testing.assert_allclose(recovered, native, atol=0, rtol=0)


def test_h1_unpack_uses_official_interleaved_slots() -> None:
    q50 = np.arange(50, dtype=np.float32)
    left = unpack_inspire12(q50, "left")
    right = unpack_inspire12(q50, "right")
    np.testing.assert_array_equal(
        left, np.asarray([26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        right, np.asarray([31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49], dtype=np.float32)
    )


if __name__ == "__main__":
    test_default_mapping_is_locked_table()
    test_table_index_roundtrip_keeps_inspire12()
    test_native_action_roundtrip_keeps_36d()
    test_h1_unpack_uses_official_interleaved_slots()
    print("all inspire12 table tests passed")
