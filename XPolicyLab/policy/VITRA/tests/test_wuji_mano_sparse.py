"""Table-index Wuji20 <-> MANO45 must match wuji2_mano45_xyz_sparse_v1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR.parents[2]))
sys.path.insert(0, str(POLICY_DIR / "VITRA"))

from vitra.datasets.spark0_dataset import (  # noqa: E402
    WUJI2MANO_CODEC_ID,
    _WUJI2MANO_ISAAC_DST,
    _WUJI2MANO_ISAAC_SIGNS,
    human_action_to_native,
    load_wuji_mapping,
    native_action_to_human,
)


def test_default_mapping_is_locked_wenwei_table() -> None:
    spec = load_wuji_mapping()
    assert spec["codec_id"] == WUJI2MANO_CODEC_ID
    src = [int(entry["source_index"]) for entry in spec["joint_mapping"]]
    dst = [int(entry["mano_pose_index"]) for entry in spec["joint_mapping"]]
    signs = [int(entry["sign"]) for entry in spec["joint_mapping"]]
    assert src == list(range(20))
    assert tuple(dst) == _WUJI2MANO_ISAAC_DST
    assert tuple(signs) == _WUJI2MANO_ISAAC_SIGNS


def test_table_index_roundtrip_keeps_wuji20() -> None:
    rng = np.random.default_rng(0)
    native = rng.normal(size=(4, 52)).astype(np.float32)
    human, mask = native_action_to_human(native, return_mask=True)
    assert human.shape == (4, 192)
    assert mask.shape == (4, 192)
    recovered = human_action_to_native(human)
    np.testing.assert_allclose(recovered, native, atol=0, rtol=0)


def test_table_matches_wuji2mano2wuji_if_installed() -> None:
    mapping_root = Path("/personal/wenwei/ego_pro/wuji2mano2wuji")
    if not mapping_root.is_dir():
        return
    sys.path.insert(0, str(mapping_root))
    from wuji2mano2wuji import mano2wuji, wuji2mano

    rng = np.random.default_rng(1)
    q20 = rng.normal(size=(7, 20)).astype(np.float32)
    native = np.zeros((7, 52), dtype=np.float32)
    native[:, 6:26] = q20
    native[:, 32:52] = q20
    human = native_action_to_human(native)
    for hand_base in (0, 51):
        pose45 = human[:, hand_base + 6 : hand_base + 51]
        expected = wuji2mano(q20, side="right", order="isaac")
        np.testing.assert_allclose(pose45, expected, atol=0, rtol=0)
        np.testing.assert_allclose(
            mano2wuji(pose45, side="right", order="isaac"), q20, atol=0, rtol=0
        )
