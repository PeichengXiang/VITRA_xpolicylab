"""Spark0 dual-arm dataset adapter for VITRA.

Two explicitly selected source representations are supported. ``wuji20`` keeps
the original robot adapter contract.  Each hand has 26 values::

    state  = absolute camera-space EEF [xyz, Euler_xyz] + absolute hand20
    action = step EEF delta [dxyz, Euler_xyz(dR)] + next absolute hand20

The dual-hand Wuji vector is 52-dimensional.  It is normalized in this native
space and only then sparsely injected into VITRA's human 192/212 layout
with the locked ``wuji2_mano45_xyz_sparse_v1`` table-index map (Isaac order).

``mano45`` consumes the fitted MANO leaves already stored in the source HDF5.
The stored 45-D hand values are 15 *local rotation vectors*, never Euler
angles.  They are converted joint-by-joint through rotation matrices.  Its
official VITRA-compatible native layout is::

    state  = absolute camera-space MANO root [xyz, Euler_xyz]
             + absolute local MANO Euler_xyz45 + betas10       # 61 / hand
    action = step MANO-root delta [dxyz, Euler_xyz(dR)]
             + next absolute local MANO Euler_xyz45            # 51 / hand

The source ``mano/action/*[t]`` is already ``mano/state/*[t+1]`` and is read at
the same row (it is not shifted again).  State61/action51 are normalized per
hand first.  Padding then copies the first 51 values of each state hand (thus
dropping betas exactly like :func:`human_dataset.pad_state_human`) and pads
state/action to 212/192.

Spark0 stores camera poses as ``camera_to_env`` in computer-graphics axes
``x-right, y-up, z-back``.  VITRA uses ``x-right, y-down, z-forward``.  The
conversion therefore right-multiplies the camera pose by ``diag(1,-1,-1,1)``.
Images are decoded only through :func:`XPolicyLab.utils.process_data.decode_image_bit`.
"""

from __future__ import annotations

import bisect
import json
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from XPolicyLab.utils.process_data import decode_image_bit
from vitra.datasets.dataset_utils import ActionFeature, StateFeature, calculate_fov
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics


REPRESENTATION_WUJI20 = "wuji20"
REPRESENTATION_MANO45 = "mano45"
SUPPORTED_REPRESENTATIONS = (REPRESENTATION_WUJI20, REPRESENTATION_MANO45)

NATIVE_HAND_DIM = 26
NATIVE_DUAL_DIM = 52
EEF_DIM = 6
HAND_DIM = 20
VITRA_HAND_DIM = 51
MANO_POSE_DIM = 45
MANO_BETA_DIM = 10
MANO_STATE_HAND_DIM = EEF_DIM + MANO_POSE_DIM + MANO_BETA_DIM
MANO_ACTION_HAND_DIM = EEF_DIM + MANO_POSE_DIM
MANO_STATE_DUAL_DIM = 2 * MANO_STATE_HAND_DIM
MANO_ACTION_DUAL_DIM = 2 * MANO_ACTION_HAND_DIM
MODEL_ACTION_DIM = ActionFeature.ALL_FEATURES[1]
MODEL_STATE_DIM = StateFeature.ALL_FEATURES[1]

CAMERA_RAW_TO_VITRA = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[3] / "mapping_wuji20_mano45.json"

SIDES = ("left", "right")
COMMON_REQUIRED_DATASETS = (
    "vision/cam_head/colors",
    "vision/cam_head/extrinsics",
    "vision/cam_head/intrinsics",
    "instruction",
)
WUJI20_REQUIRED_DATASETS = (
    "state/left_ee_poses",
    "state/right_ee_poses",
    "state/left_ee_joint_states",
    "state/right_ee_joint_states",
    "action/left_ee_poses",
    "action/right_ee_poses",
    "action/left_ee_joint_states",
    "action/right_ee_joint_states",
)
MANO45_REQUIRED_DATASETS = tuple(
    f"mano/{group}/{side}_{leaf}"
    for group in ("state", "action")
    for side in SIDES
    for leaf in ("ee_poses", "ee_joint_states", "hand_betas")
) + tuple(
    f"mano/validity/{prefix}{side}_{leaf}"
    for prefix in ("", "action_")
    for side in SIDES
    for leaf in ("ee_pose", "ee_joint_states", "hand_betas")
)
# Backward-compatible export used by older validation code.
REQUIRED_DATASETS = WUJI20_REQUIRED_DATASETS + COMMON_REQUIRED_DATASETS


def validate_representation(representation: str) -> str:
    """Return a canonical adapter representation or fail before data loading."""

    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {SUPPORTED_REPRESENTATIONS}, got {representation!r}"
        )
    return representation


def representation_dimensions(representation: str) -> Tuple[int, int]:
    """Return per-hand ``(state_dim, action_dim)`` for a representation."""

    representation = validate_representation(representation)
    if representation == REPRESENTATION_WUJI20:
        return NATIVE_HAND_DIM, NATIVE_HAND_DIM
    return MANO_STATE_HAND_DIM, MANO_ACTION_HAND_DIM


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def pose7_to_matrix(pose: Any) -> np.ndarray:
    """Convert ``[..., x,y,z,qw,qx,qy,qz]`` to homogeneous matrices."""

    pose = _as_float_array(pose)
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected pose last dimension 7, got {pose.shape}")
    quat_wxyz = pose[..., 3:7]
    norms = np.linalg.norm(quat_wxyz, axis=-1)
    if np.any(~np.isfinite(pose)) or np.any(norms < 1e-8):
        raise ValueError("Pose contains non-finite values or a zero quaternion")
    quat_wxyz = quat_wxyz / norms[..., None]
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    matrices = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float64)
    matrices[..., :3, :3] = Rotation.from_quat(quat_xyzw.reshape(-1, 4)).as_matrix().reshape(
        pose.shape[:-1] + (3, 3)
    )
    matrices[..., :3, 3] = pose[..., :3]
    matrices[..., 3, 3] = 1.0
    return matrices


def matrix_to_pose7(matrix: Any) -> np.ndarray:
    """Convert homogeneous matrices to ``[..., x,y,z,qw,qx,qy,qz]``."""

    matrix = _as_float_array(matrix)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected homogeneous matrix shape (...,4,4), got {matrix.shape}")
    quat_xyzw = Rotation.from_matrix(matrix[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(
        matrix.shape[:-2] + (4,)
    )
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
    return np.concatenate([matrix[..., :3, 3], quat_wxyz], axis=-1).astype(np.float32)


def matrix_to_pose6(matrix: Any) -> np.ndarray:
    """Convert homogeneous matrices to ``[..., xyz, Euler_xyz]``."""

    matrix = _as_float_array(matrix)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected homogeneous matrix shape (...,4,4), got {matrix.shape}")
    euler = Rotation.from_matrix(matrix[..., :3, :3].reshape(-1, 3, 3)).as_euler(
        "xyz", degrees=False
    ).reshape(matrix.shape[:-2] + (3,))
    return np.concatenate([matrix[..., :3, 3], euler], axis=-1).astype(np.float32)


def pose6_to_matrix(pose: Any) -> np.ndarray:
    """Convert ``[..., xyz, Euler_xyz]`` to homogeneous matrices."""

    pose = _as_float_array(pose)
    if pose.shape[-1] != 6:
        raise ValueError(f"Expected pose last dimension 6, got {pose.shape}")
    matrices = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float64)
    matrices[..., :3, :3] = Rotation.from_euler(
        "xyz", pose[..., 3:6].reshape(-1, 3), degrees=False
    ).as_matrix().reshape(pose.shape[:-1] + (3, 3))
    matrices[..., :3, 3] = pose[..., :3]
    matrices[..., 3, 3] = 1.0
    return matrices


def mano_rotvec45_to_euler45(rotvec45: Any) -> np.ndarray:
    """Convert 15 local MANO rotation vectors to VITRA Euler-xyz values.

    Both source sides already contain their final fitted MANO parameters.  In
    particular, the left side must *not* be mirrored or sign-flipped again.
    Converting through matrices makes the axis-angle/Euler boundary explicit.
    """

    rotvec45 = _as_float_array(rotvec45)
    if rotvec45.shape[-1] != MANO_POSE_DIM:
        raise ValueError(
            f"Expected MANO local rotvec last dimension {MANO_POSE_DIM}, got {rotvec45.shape}"
        )
    if not np.all(np.isfinite(rotvec45)):
        raise ValueError("MANO local rotation vectors contain non-finite values")
    leading_shape = rotvec45.shape[:-1]
    rotations = Rotation.from_rotvec(rotvec45.reshape(-1, 3)).as_matrix()
    euler = Rotation.from_matrix(rotations).as_euler("xyz", degrees=False)
    return euler.reshape(leading_shape + (MANO_POSE_DIM,)).astype(np.float32)


def mano_euler45_to_rotvec45(euler45: Any) -> np.ndarray:
    """Invert :func:`mano_rotvec45_to_euler45` joint by joint.

    Runtime MANO45 actions are normalized/stored as 15 local Euler-xyz
    rotations, while the fitted MANO toolchain consumes 15 local axis-angle
    vectors.  Converting each joint through a rotation matrix avoids treating
    the two 45-D arrays as interchangeable.
    """

    euler45 = _as_float_array(euler45)
    if euler45.shape[-1] != MANO_POSE_DIM:
        raise ValueError(
            f"Expected MANO local Euler last dimension {MANO_POSE_DIM}, got {euler45.shape}"
        )
    if not np.all(np.isfinite(euler45)):
        raise ValueError("MANO local Euler rotations contain non-finite values")
    leading_shape = euler45.shape[:-1]
    rotations = Rotation.from_euler("xyz", euler45.reshape(-1, 3), degrees=False).as_matrix()
    rotvec = Rotation.from_matrix(rotations).as_rotvec()
    return rotvec.reshape(leading_shape + (MANO_POSE_DIM,)).astype(np.float32)


def mano_state_to_human(
    state: Any,
    hand_mask: Optional[Any] = None,
    *,
    return_mask: bool = False,
):
    """Pad normalized MANO state122 to state212, dropping each beta10 span."""

    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != MANO_STATE_DUAL_DIM:
        raise ValueError(
            f"Expected MANO state last dimension {MANO_STATE_DUAL_DIM}, got {state.shape}"
        )
    if hand_mask is None:
        active = np.ones(state.shape[:-1] + (2,), dtype=bool)
    else:
        active = np.asarray(hand_mask, dtype=bool)
        if active.shape != state.shape[:-1] + (2,):
            raise ValueError(
                f"hand_mask shape must be {state.shape[:-1] + (2,)}, got {active.shape}"
            )

    padded = np.zeros(state.shape[:-1] + (MODEL_STATE_DIM,), dtype=np.float32)
    mask = np.zeros(state.shape[:-1] + (MODEL_STATE_DIM,), dtype=bool)
    for hand_index in range(2):
        source = hand_index * MANO_STATE_HAND_DIM
        target = hand_index * MANO_ACTION_HAND_DIM
        values = state[..., source : source + MANO_ACTION_HAND_DIM]
        padded[..., target : target + MANO_ACTION_HAND_DIM] = (
            values * active[..., hand_index, None]
        )
        mask[..., target : target + MANO_ACTION_HAND_DIM] = active[..., hand_index, None]
    return (padded, mask) if return_mask else padded


def mano_action_to_human(
    action: Any,
    hand_mask: Optional[Any] = None,
    *,
    return_mask: bool = False,
):
    """Pad normalized MANO action102 to VITRA action192."""

    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != MANO_ACTION_DUAL_DIM:
        raise ValueError(
            f"Expected MANO action last dimension {MANO_ACTION_DUAL_DIM}, got {action.shape}"
        )
    if hand_mask is None:
        active = np.ones(action.shape[:-1] + (2,), dtype=bool)
    else:
        active = np.asarray(hand_mask, dtype=bool)
        if active.shape != action.shape[:-1] + (2,):
            raise ValueError(
                f"hand_mask shape must be {action.shape[:-1] + (2,)}, got {active.shape}"
            )

    padded = np.zeros(action.shape[:-1] + (MODEL_ACTION_DIM,), dtype=np.float32)
    mask = np.zeros(action.shape[:-1] + (MODEL_ACTION_DIM,), dtype=bool)
    for hand_index in range(2):
        span = slice(hand_index * MANO_ACTION_HAND_DIM, (hand_index + 1) * MANO_ACTION_HAND_DIM)
        padded[..., span] = action[..., span] * active[..., hand_index, None]
        mask[..., span] = active[..., hand_index, None]
    return (padded, mask) if return_mask else padded


def human_action_to_mano(human_action: Any) -> np.ndarray:
    """Extract normalized MANO action102 from VITRA action192.

    MANO45 training pads the two contiguous 51-D hand spans into the first
    102 model features.  This is an exact slice, not the legacy sparse
    Wuji20/MANO projection.
    """

    human = np.asarray(human_action, dtype=np.float32)
    if human.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(f"Expected human action last dimension {MODEL_ACTION_DIM}, got {human.shape}")
    return np.array(human[..., :MANO_ACTION_DUAL_DIM], dtype=np.float32, copy=True)


def _statistics_representation(statistics_path: str) -> Optional[str]:
    with open(statistics_path, "r", encoding="utf-8") as handle:
        value = json.load(handle).get("representation")
    if value in SUPPORTED_REPRESENTATIONS:
        return value
    # Statistics generated by the first Wuji20 integration used a descriptive
    # string.  Keep that file backward-compatible, while never accepting it for
    # MANO45.
    if isinstance(value, str) and "Wuji20" in value:
        return REPRESENTATION_WUJI20
    return None


def _validate_statistics_contract(
    statistics_path: str,
    representation: str,
    statistics: Mapping[str, np.ndarray],
) -> None:
    declared = _statistics_representation(statistics_path)
    if representation == REPRESENTATION_MANO45 and declared != REPRESENTATION_MANO45:
        raise ValueError(
            "MANO45 requires statistics explicitly generated with "
            f"representation='mano45'; {statistics_path} declares {declared!r}"
        )
    if declared is not None and declared != representation:
        raise ValueError(
            f"Statistics representation {declared!r} does not match requested {representation!r}: "
            f"{statistics_path}"
        )

    state_dim, action_dim = representation_dimensions(representation)
    expected = {
        "state_left_mean": state_dim,
        "state_left_std": state_dim,
        "state_right_mean": state_dim,
        "state_right_std": state_dim,
        "action_left_mean": action_dim,
        "action_left_std": action_dim,
        "action_right_mean": action_dim,
        "action_right_std": action_dim,
    }
    if representation == REPRESENTATION_MANO45:
        with open(statistics_path, "r", encoding="utf-8") as handle:
            raw_statistics = json.load(handle)
        missing = []
        for key in expected:
            prefix, field = key.rsplit("_", 1)
            nested = raw_statistics.get(prefix)
            if key not in raw_statistics and not (
                isinstance(nested, Mapping) and field in nested
            ):
                missing.append(key)
        if missing:
            raise ValueError(
                "MANO45 statistics must explicitly contain both-hand state/action "
                f"moments; missing {missing}: {statistics_path}"
            )
    wrong = {
        key: np.asarray(statistics[key]).shape
        for key, dimension in expected.items()
        if np.asarray(statistics[key]).shape != (dimension,)
    }
    if wrong:
        raise ValueError(
            f"Statistics dimensions do not match representation {representation!r}: {wrong}"
        )


def raw_camera_pose_to_vitra_matrix(camera_pose: Any) -> np.ndarray:
    """Return ``T_env_camera_vitra`` from Spark0 ``camera_to_env`` pose/matrix."""

    value = _as_float_array(camera_pose)
    if value.shape[-2:] == (4, 4):
        matrix = value
    elif value.shape[-1] == 7:
        matrix = pose7_to_matrix(value)
    else:
        raise ValueError(f"Camera extrinsics must be pose7 or 4x4, got {value.shape}")
    return matrix @ CAMERA_RAW_TO_VITRA


def intrinsics_to_matrix(intrinsics: Any) -> np.ndarray:
    """Convert Spark0 ``[fx,fy,cx,cy]`` or a 3x3 matrix to a 3x3 matrix."""

    value = _as_float_array(intrinsics)
    if value.shape == (3, 3):
        return value.astype(np.float32)
    value = value.reshape(-1)
    if value.size != 4:
        raise ValueError(f"Intrinsics must contain [fx,fy,cx,cy] or be 3x3, got {value.shape}")
    fx, fy, cx, cy = value
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


@lru_cache(maxsize=8)
def _load_mapping_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    _validate_mapping(mapping)
    return mapping


WUJI2MANO_CODEC_ID = "wuji2_mano45_xyz_sparse_v1"
# Isaac-order table from wuji2mano2wuji. Inverse is the same (src, dst, sign).
_WUJI2MANO_ISAAC_DST = (
    2, 11, 20, 29, 36, 1, 10, 19, 28, 37, 5, 14, 23, 32, 40, 8, 17, 26, 35, 43
)
_WUJI2MANO_ISAAC_SIGNS = (
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1
)


def _validate_mapping(mapping: Mapping[str, Any]) -> None:
    entries = mapping.get("joint_mapping", [])
    if len(entries) != HAND_DIM:
        raise ValueError(f"Wuji mapping must contain exactly {HAND_DIM} entries")
    src = [int(entry["source_index"]) for entry in entries]
    dst = [int(entry["mano_pose_index"]) for entry in entries]
    signs = [float(entry["sign"]) for entry in entries]
    if sorted(src) != list(range(HAND_DIM)):
        raise ValueError("Wuji mapping source indices must be a permutation of 0..19")
    if len(set(dst)) != HAND_DIM or any(index < 0 or index >= 45 for index in dst):
        raise ValueError("Wuji mapping MANO destinations must be 20 unique indices in 0..44")
    if any(sign not in (-1.0, 1.0) for sign in signs):
        raise ValueError("Wuji mapping signs must be +1 or -1")

    injection = np.zeros((45, HAND_DIM), dtype=np.float64)
    for source, target, sign in zip(src, dst, signs):
        injection[target, source] = sign
    if not np.array_equal(injection.T @ injection, np.eye(HAND_DIM)):
        raise ValueError("Signed Wuji-to-MANO injection must satisfy E.T @ E == I")

    codec_id = mapping.get("codec_id")
    if codec_id not in (None, WUJI2MANO_CODEC_ID):
        raise ValueError(f"Unsupported Wuji/MANO codec_id={codec_id!r}")
    if codec_id == WUJI2MANO_CODEC_ID:
        expected_src = list(range(HAND_DIM))
        if src != expected_src:
            raise ValueError(
                f"{WUJI2MANO_CODEC_ID} requires Isaac source_index 0..19 in order, got {src}"
            )
        if tuple(dst) != _WUJI2MANO_ISAAC_DST or tuple(int(s) for s in signs) != _WUJI2MANO_ISAAC_SIGNS:
            raise ValueError(
                f"{WUJI2MANO_CODEC_ID} destination/sign table does not match the locked index map"
            )

    calibration = mapping.get("calibration", {})
    for side in SIDES:
        key = f"{side}_ee_to_vitra_wrist"
        transform = np.asarray(calibration.get(key), dtype=np.float64)
        if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError(f"Invalid calibration matrix {key}")


def _mapping_index_tables(
    mapping: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (source20, mano45, signs20) for table-index encode/decode."""

    entries = mapping["joint_mapping"]
    source = np.asarray([int(entry["source_index"]) for entry in entries], dtype=np.int64)
    destination = np.asarray(
        [int(entry["mano_pose_index"]) for entry in entries], dtype=np.int64
    )
    signs = np.asarray([float(entry["sign"]) for entry in entries], dtype=np.float32)
    return source, destination, signs


def load_wuji_mapping(mapping: Optional[Any] = None) -> Mapping[str, Any]:
    """Load and validate the signed Wuji20-to-MANO45 mapping."""

    if mapping is None:
        return _load_mapping_file(str(DEFAULT_MAPPING_PATH))
    if isinstance(mapping, (str, os.PathLike)):
        return _load_mapping_file(str(Path(mapping).resolve()))
    if not isinstance(mapping, Mapping):
        raise TypeError(f"mapping must be a path or mapping object, got {type(mapping).__name__}")
    _validate_mapping(mapping)
    return mapping


def _map_native_to_human(
    native: Any,
    output_dim: int,
    mapping: Optional[Any] = None,
    hand_mask: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    native = np.asarray(native, dtype=np.float32)
    if native.shape[-1] != NATIVE_DUAL_DIM:
        raise ValueError(f"Expected native last dimension {NATIVE_DUAL_DIM}, got {native.shape}")
    spec = load_wuji_mapping(mapping)
    source, destination, signs = _mapping_index_tables(spec)
    if hand_mask is None:
        active = np.ones(native.shape[:-1] + (2,), dtype=bool)
    else:
        active = np.asarray(hand_mask, dtype=bool)
        if active.shape != native.shape[:-1] + (2,):
            raise ValueError(
                f"hand_mask shape must be {native.shape[:-1] + (2,)}, got {active.shape}"
            )

    human = np.zeros(native.shape[:-1] + (output_dim,), dtype=np.float32)
    mask = np.zeros(native.shape[:-1] + (output_dim,), dtype=bool)
    for hand_index, side in enumerate(SIDES):
        native_base = hand_index * NATIVE_HAND_DIM
        human_base = hand_index * VITRA_HAND_DIM
        side_active = active[..., hand_index]
        human[..., human_base : human_base + EEF_DIM] = native[
            ..., native_base : native_base + EEF_DIM
        ]
        human[..., human_base + EEF_DIM + destination] = (
            signs * native[..., native_base + EEF_DIM + source]
        )
        # This matches VITRA's official XHand behavior: an available robot hand
        # activates its complete 51-D semantic span, including injected zeros.
        mask[..., human_base : human_base + VITRA_HAND_DIM] = side_active[..., None]
        human[..., human_base : human_base + VITRA_HAND_DIM] *= side_active[..., None]
    return human, mask


def native_state_to_human(
    native_state: Any,
    mapping: Optional[Any] = None,
    hand_mask: Optional[Any] = None,
    *,
    return_mask: bool = False,
):
    """Inject native state52 into VITRA state212 after native normalization."""

    state, mask = _map_native_to_human(native_state, MODEL_STATE_DIM, mapping, hand_mask)
    return (state, mask) if return_mask else state


def native_action_to_human(
    native_action: Any,
    mapping: Optional[Any] = None,
    hand_mask: Optional[Any] = None,
    *,
    return_mask: bool = False,
):
    """Inject native action52 into VITRA action192 after native normalization."""

    action, mask = _map_native_to_human(native_action, MODEL_ACTION_DIM, mapping, hand_mask)
    return (action, mask) if return_mask else action


def human_action_to_native(human_action: Any, mapping: Optional[Any] = None) -> np.ndarray:
    """Project VITRA action192 back to native action52 (linear map only)."""

    human = np.asarray(human_action, dtype=np.float32)
    if human.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(f"Expected human action last dimension {MODEL_ACTION_DIM}, got {human.shape}")
    spec = load_wuji_mapping(mapping)
    source, destination, signs = _mapping_index_tables(spec)
    native = np.zeros(human.shape[:-1] + (NATIVE_DUAL_DIM,), dtype=np.float32)
    for hand_index in range(2):
        native_base = hand_index * NATIVE_HAND_DIM
        human_base = hand_index * VITRA_HAND_DIM
        native[..., native_base : native_base + EEF_DIM] = human[
            ..., human_base : human_base + EEF_DIM
        ]
        native[..., native_base + EEF_DIM + source] = (
            signs * human[..., human_base + EEF_DIM + destination]
        )
    return native


def _latest_vector(value: Any, expected_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (expected_dim,):
        return array.astype(np.float64)
    if array.ndim >= 2 and array.shape[-1] == expected_dim:
        return array.reshape(-1, expected_dim)[-1].astype(np.float64)
    raise ValueError(f"{name} must end in dimension {expected_dim}, got {array.shape}")


def _latest_pose_or_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (4, 4):
        return array.astype(np.float64)
    if array.ndim >= 3 and array.shape[-2:] == (4, 4):
        return array.reshape(-1, 4, 4)[-1].astype(np.float64)
    return _latest_vector(array, 7, name)


def _decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size:
            value = value.reshape(-1)[-1]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def format_vitra_instruction(value: Any) -> str:
    """Ensure VITRA's explicit two-hand prompt format."""

    text = _decode_text(value).strip()
    if "Left hand:" in text and "Right hand:" in text:
        return text
    if not text:
        text = "follow the task instruction"
    return f"Left hand: {text} Right hand: {text}"


def _camera_dict(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    vision = observation.get("vision")
    if not isinstance(vision, Mapping):
        raise KeyError("Runtime observation must contain obs['vision']")
    for key in (
        "cam_head",
        "head",
        "camera_head",
        "head_camera",
        "cam_high",
        "top_camera",
    ):
        if key in vision and isinstance(vision[key], Mapping):
            return vision[key]
    raise KeyError("Runtime observation must contain vision/cam_head")


def _runtime_image(camera: Mapping[str, Any]) -> np.ndarray:
    for key in ("color", "colors", "rgb", "image"):
        if key in camera:
            image = np.asarray(camera[key])
            if image.ndim == 4:
                image = image[-1]
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    "Runtime cam_head image must already be decoded RGB HWC; "
                    f"got {key} shape {image.shape}"
                )
            return image.astype(np.uint8, copy=False)
    raise KeyError("Runtime cam_head must contain one of color/colors/rgb/image")


def runtime_observation_to_native(
    observation: Mapping[str, Any], mapping_path: Optional[Any] = None
) -> dict:
    """Convert one decoded XPolicyLab observation to native VITRA inputs.

    This function does not normalize or inject dimensions.  The returned
    ``context`` is required by :func:`native_action_chunk_to_xpolicy`.
    """

    spec = load_wuji_mapping(mapping_path)
    state = observation.get("state")
    if not isinstance(state, Mapping):
        raise KeyError("Runtime observation must contain obs['state']")
    camera = _camera_dict(observation)
    image = _runtime_image(camera)

    extrinsics_key = next(
        (
            key
            for key in ("extrinsics", "extrinsic", "extrinsics_matrix", "extrinsic_matrix")
            if key in camera
        ),
        None,
    )
    if extrinsics_key is None:
        raise KeyError(
            "Runtime cam_head requires camera_to_env extrinsics under "
            "extrinsics/extrinsic(s)_matrix"
        )
    camera_raw = _latest_pose_or_matrix(
        camera[extrinsics_key], f"cam_head/{extrinsics_key}"
    )
    camera_vitra_to_env = raw_camera_pose_to_vitra_matrix(camera_raw)

    intrinsics_key = next(
        (
            key
            for key in ("intrinsics", "intrinsic", "intrinsics_matrix", "intrinsic_matrix")
            if key in camera
        ),
        None,
    )
    if intrinsics_key is None:
        raise KeyError(
            "Runtime cam_head requires intrinsics under intrinsics/intrinsic(s)_matrix"
        )
    intrinsics_value = np.asarray(camera[intrinsics_key])
    if intrinsics_value.ndim >= 2 and intrinsics_value.shape[-2:] != (3, 3):
        intrinsics_value = intrinsics_value.reshape(-1, intrinsics_value.shape[-1])[-1]
    elif intrinsics_value.ndim >= 3 and intrinsics_value.shape[-2:] == (3, 3):
        intrinsics_value = intrinsics_value.reshape(-1, 3, 3)[-1]
    intrinsics = intrinsics_to_matrix(intrinsics_value)

    native_parts = []
    current_wrist = {}
    calibrations = {}
    for side in SIDES:
        ee_pose = _latest_vector(state[f"{side}_ee_pose"], 7, f"state/{side}_ee_pose")
        hand = _latest_vector(
            state[f"{side}_ee_joint_state"], HAND_DIM, f"state/{side}_ee_joint_state"
        )
        ee_to_wrist = np.asarray(
            spec["calibration"][f"{side}_ee_to_vitra_wrist"], dtype=np.float64
        )
        wrist_camera = np.linalg.inv(camera_vitra_to_env) @ pose7_to_matrix(ee_pose) @ ee_to_wrist
        native_parts.append(np.concatenate([matrix_to_pose6(wrist_camera), hand]).astype(np.float32))
        current_wrist[side] = wrist_camera
        calibrations[side] = ee_to_wrist

    instruction = format_vitra_instruction(
        observation.get(
            "instruction",
            observation.get("task_instruction", observation.get("instructions", "")),
        )
    )
    fov = calculate_fov(image.shape[0], image.shape[1], intrinsics).astype(np.float32)
    context = {
        "camera_vitra_to_env": camera_vitra_to_env,
        "current_wrist_in_vitra_camera": current_wrist,
        "ee_to_vitra_wrist": calibrations,
    }
    return {
        "image": image,
        "instruction": instruction,
        "native_state": np.concatenate(native_parts).astype(np.float32),
        "context": context,
        "fov": fov,
        "intrinsics": intrinsics,
    }


def native_action_chunk_to_xpolicy(native_action: Any, context: Mapping[str, Any]) -> list[dict]:
    """Integrate native step deltas and emit absolute XPolicyLab EE actions."""

    actions = np.asarray(native_action, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != NATIVE_DUAL_DIM:
        raise ValueError(f"Expected native actions [T,{NATIVE_DUAL_DIM}], got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Native action chunk contains non-finite values")

    camera_vitra_to_env = np.asarray(context["camera_vitra_to_env"], dtype=np.float64)
    current = {
        side: np.asarray(context["current_wrist_in_vitra_camera"][side], dtype=np.float64).copy()
        for side in SIDES
    }
    ee_to_wrist = {
        side: np.asarray(context["ee_to_vitra_wrist"][side], dtype=np.float64)
        for side in SIDES
    }

    output = []
    for row in actions:
        command = {}
        for hand_index, side in enumerate(SIDES):
            base = hand_index * NATIVE_HAND_DIM
            current[side][:3, 3] += row[base : base + 3]
            delta_rotation = Rotation.from_euler(
                "xyz", row[base + 3 : base + 6], degrees=False
            ).as_matrix()
            # Official VITRA step convention: dR = R_next @ R_current.T.
            current[side][:3, :3] = delta_rotation @ current[side][:3, :3]
            ee_env = camera_vitra_to_env @ current[side] @ np.linalg.inv(ee_to_wrist[side])
            command[f"{side}_ee_pose"] = matrix_to_pose7(ee_env).astype(np.float32)
            command[f"{side}_ee_joint_state"] = row[
                base + EEF_DIM : base + NATIVE_HAND_DIM
            ].copy()
        output.append(command)
    return output


def runtime_observation_to_mano(
    observation: Mapping[str, Any],
    *,
    codec: Any,
    previous_fit: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Convert runtime Link7 + Wuji20 state into MANO state122.

    ``codec.wuji_to_mano`` is the configured ``add_mano`` kinematic forward.
    In particular, this path never loads or applies the legacy sparse
    ``mapping_wuji20_mano45.json`` projection.
    """

    if codec is None or not callable(getattr(codec, "wuji_to_mano", None)):
        raise TypeError("MANO45 runtime requires a configured Mano45RuntimeCodec")
    state = observation.get("state")
    if not isinstance(state, Mapping):
        raise KeyError("Runtime observation must contain obs['state']")
    camera = _camera_dict(observation)
    image = _runtime_image(camera)

    extrinsics_key = next(
        (
            key
            for key in ("extrinsics", "extrinsic", "extrinsics_matrix", "extrinsic_matrix")
            if key in camera
        ),
        None,
    )
    if extrinsics_key is None:
        raise KeyError(
            "Runtime cam_head requires camera_to_env extrinsics under "
            "extrinsics/extrinsic(s)_matrix"
        )
    camera_raw = _latest_pose_or_matrix(
        camera[extrinsics_key], f"cam_head/{extrinsics_key}"
    )
    camera_vitra_to_env = raw_camera_pose_to_vitra_matrix(camera_raw)

    intrinsics_key = next(
        (
            key
            for key in ("intrinsics", "intrinsic", "intrinsics_matrix", "intrinsic_matrix")
            if key in camera
        ),
        None,
    )
    if intrinsics_key is None:
        raise KeyError(
            "Runtime cam_head requires intrinsics under intrinsics/intrinsic(s)_matrix"
        )
    intrinsics_value = np.asarray(camera[intrinsics_key])
    if intrinsics_value.ndim >= 3 and intrinsics_value.shape[-2:] == (3, 3):
        intrinsics_value = intrinsics_value.reshape(-1, 3, 3)[-1]
    elif intrinsics_value.ndim >= 2 and intrinsics_value.shape[-1] == 4:
        intrinsics_value = intrinsics_value.reshape(-1, 4)[-1]
    intrinsics = intrinsics_to_matrix(intrinsics_value)

    previous_fit = previous_fit or {}
    state_parts: list[np.ndarray] = []
    current_root: dict[str, np.ndarray] = {}
    current_q: dict[str, np.ndarray] = {}
    observed_link7_pose7: dict[str, np.ndarray] = {}
    observed_link7_in_env: dict[str, np.ndarray] = {}
    betas: dict[str, np.ndarray] = {}
    fit_state: dict[str, np.ndarray] = {}
    fit_rms_m: dict[str, float] = {}
    camera_env_to_vitra = np.linalg.inv(camera_vitra_to_env)
    for side in SIDES:
        link7_pose = _latest_vector(state[f"{side}_ee_pose"], 7, f"state/{side}_ee_pose")
        q_stage_major = _latest_vector(
            state[f"{side}_ee_joint_state"], HAND_DIM, f"state/{side}_ee_joint_state"
        )
        link7_in_env = pose7_to_matrix(link7_pose)
        fit = codec.wuji_to_mano(
            side=side,
            q_stage_major=q_stage_major,
            link7_in_env=link7_in_env,
            previous_hand_pose=previous_fit.get(side),
        )
        root_env = np.asarray(fit.root_in_env, dtype=np.float64)
        hand_euler = mano_rotvec45_to_euler45(fit.hand_pose_rotvec)
        beta = np.asarray(fit.betas, dtype=np.float32).reshape(MANO_BETA_DIM)
        root_camera = camera_env_to_vitra @ root_env
        state_parts.append(
            np.concatenate([matrix_to_pose6(root_camera), hand_euler, beta]).astype(np.float32)
        )
        current_root[side] = root_camera
        current_q[side] = q_stage_major.astype(np.float32)
        observed_link7_pose7[side] = link7_pose.astype(np.float32, copy=True)
        observed_link7_in_env[side] = link7_in_env.copy()
        betas[side] = beta
        fit_state[side] = np.asarray(fit.hand_pose_rotvec, dtype=np.float32).reshape(MANO_POSE_DIM)
        fit_rms_m[side] = float(fit.rms_m)

    instruction = format_vitra_instruction(
        observation.get(
            "instruction",
            observation.get("task_instruction", observation.get("instructions", "")),
        )
    )
    context = {
        "representation": REPRESENTATION_MANO45,
        "camera_vitra_to_env": camera_vitra_to_env,
        "current_mano_root_in_vitra_camera": current_root,
        "current_wuji_q_stage_major": current_q,
        "observed_link7_pose7": observed_link7_pose7,
        "observed_link7_in_env": observed_link7_in_env,
        "mano_betas": betas,
        "wuji_to_mano_fit_rms_m": fit_rms_m,
    }
    native_state = np.concatenate(state_parts).astype(np.float32)
    if native_state.shape != (MANO_STATE_DUAL_DIM,):
        raise ValueError(
            f"MANO runtime state must be {MANO_STATE_DUAL_DIM}-D, got {native_state.shape}"
        )
    return {
        "image": image,
        "instruction": instruction,
        "native_state": native_state,
        "context": context,
        "fov": calculate_fov(image.shape[0], image.shape[1], intrinsics).astype(np.float32),
        "intrinsics": intrinsics,
        "fit_state": fit_state,
    }


def mano_action_chunk_to_xpolicy(
    mano_action: Any,
    context: Mapping[str, Any],
    *,
    codec: Any,
) -> list[dict]:
    """Decode MANO action102 into absolute Link7 + stage-major Wuji q20.

    The first IK frame starts at the current observed q.  Each later action in
    the same chunk starts at the previous predicted q, matching the temporal
    warm-start used by the offline retargeter.  The external Wuji model clips
    to URDF limits.  Rows commit atomically for both hands; a numerical fit
    failure degrades the remainder of the chunk to the last complete command.
    """

    if codec is None or not callable(getattr(codec, "mano_to_wuji", None)):
        raise TypeError("MANO45 runtime requires a configured Mano45RuntimeCodec")
    actions = np.asarray(mano_action, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != MANO_ACTION_DUAL_DIM:
        raise ValueError(
            f"Expected MANO actions [T,{MANO_ACTION_DUAL_DIM}], got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("MANO action chunk contains non-finite values")
    if context.get("representation") != REPRESENTATION_MANO45:
        raise ValueError("MANO action decoder received a non-MANO45 context")

    # Import only on the MANO runtime path.  Catching this exact operational
    # failure type preserves fail-closed behavior without hiding malformed
    # context, incompatible codec APIs, or other programming/contract errors.
    from XPolicyLab.policy.VITRA.mano45_runtime import ManoRuntimeFitError

    camera_vitra_to_env = np.asarray(context["camera_vitra_to_env"], dtype=np.float64)
    current = {
        side: np.asarray(
            context["current_mano_root_in_vitra_camera"][side], dtype=np.float64
        ).copy()
        for side in SIDES
    }
    previous_q = {
        side: np.asarray(
            context["current_wuji_q_stage_major"][side], dtype=np.float64
        ).reshape(HAND_DIM)
        for side in SIDES
    }
    betas = {
        side: np.asarray(context["mano_betas"][side], dtype=np.float64).reshape(MANO_BETA_DIM)
        for side in SIDES
    }

    observed_link7 = {
        side: np.asarray(
            context["observed_link7_in_env"][side], dtype=np.float64
        ).copy()
        for side in SIDES
    }
    for side in SIDES:
        if observed_link7[side].shape != (4, 4):
            raise ValueError(
                f"Observed {side} Link7 matrix must be 4x4, got {observed_link7[side].shape}"
            )
        if not np.all(np.isfinite(observed_link7[side])):
            raise ValueError(f"Observed {side} Link7 matrix contains non-finite values")

    def _copy_command(command: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return {
            key: np.array(value, dtype=np.float32, copy=True)
            for key, value in command.items()
        }

    # A row is committed only after both hands decode.  Before the first
    # successful row, the only safe complete command is the observed Link7/q.
    safe_command = {
        **{
            f"{side}_ee_pose": matrix_to_pose7(observed_link7[side]).astype(np.float32)
            for side in SIDES
        },
        **{
            f"{side}_ee_joint_state": previous_q[side].astype(np.float32, copy=True)
            for side in SIDES
        },
    }

    output: list[dict[str, np.ndarray]] = []
    for row_index, row in enumerate(actions):
        candidate_roots: dict[str, np.ndarray] = {}
        candidate_decodes: dict[str, Any] = {}
        failed: Optional[tuple[str, ManoRuntimeFitError]] = None
        for hand_index, side in enumerate(SIDES):
            base = hand_index * MANO_ACTION_HAND_DIM
            candidate_root = current[side].copy()
            candidate_root[:3, 3] += row[base : base + 3]
            delta_rotation = Rotation.from_euler(
                "xyz", row[base + 3 : base + 6], degrees=False
            ).as_matrix()
            # VITRA step convention: dR = R_next @ R_current.T.
            candidate_root[:3, :3] = delta_rotation @ candidate_root[:3, :3]
            local_rotvec = mano_euler45_to_rotvec45(
                row[base + EEF_DIM : base + MANO_ACTION_HAND_DIM]
            )
            root_env = camera_vitra_to_env @ candidate_root
            try:
                decoded = codec.mano_to_wuji(
                    side=side,
                    root_in_env=root_env,
                    hand_pose_rotvec=local_rotvec,
                    betas=betas[side],
                    initial_q_stage_major=previous_q[side],
                )
                q_stage_major = np.asarray(
                    decoded.q_stage_major, dtype=np.float32
                ).reshape(HAND_DIM)
                link7_in_env = np.asarray(decoded.link7_in_env, dtype=np.float64)
                if link7_in_env.shape != (4, 4):
                    raise ValueError(
                        f"{side} codec Link7 result must be 4x4, got {link7_in_env.shape}"
                    )
                rms_m = float(decoded.rms_m)
                tolerance_m = float(getattr(codec, "tolerance_m", 0.03))
                if (
                    not np.all(np.isfinite(q_stage_major))
                    or not np.all(np.isfinite(link7_in_env))
                    or not np.isfinite(rms_m)
                    or rms_m > tolerance_m
                ):
                    raise ManoRuntimeFitError(
                        f"{side} MANO inverse returned an unsafe numerical result "
                        f"(rms={rms_m!r}, limit={tolerance_m:.6f})"
                    )
            except ManoRuntimeFitError as exc:
                failed = (side, exc)
                break
            candidate_roots[side] = candidate_root
            candidate_decodes[side] = (decoded, q_stage_major, link7_in_env)

        if failed is not None:
            failed_side, exc = failed
            print(
                "[VITRA] MANO45 degraded chunk: holding both hands from failed row",
                f"row={row_index} side={failed_side} reason={exc}",
                flush=True,
            )
            output.extend(
                _copy_command(safe_command)
                for _ in range(actions.shape[0] - row_index)
            )
            break

        command: dict[str, np.ndarray] = {}
        for side in SIDES:
            _decoded, q_stage_major, link7_in_env = candidate_decodes[side]
            command[f"{side}_ee_pose"] = matrix_to_pose7(link7_in_env).astype(np.float32)
            command[f"{side}_ee_joint_state"] = q_stage_major
            current[side] = candidate_roots[side]
            previous_q[side] = q_stage_major.astype(np.float64)
        output.append(command)
        safe_command = _copy_command(command)
    return output


def _dataset_frame(dataset: h5py.Dataset, index: int, episode_length: int) -> Any:
    if dataset.ndim >= 1 and dataset.shape[0] == episode_length:
        return dataset[index]
    return dataset[()]


def _episode_length(
    handle: h5py.File, representation: str = REPRESENTATION_WUJI20
) -> int:
    """Return the source HDF5 frame count (before MANO terminal exclusion)."""

    representation = validate_representation(representation)
    prefix = "state" if representation == REPRESENTATION_WUJI20 else "mano/state"
    return int(handle[f"{prefix}/left_ee_poses"].shape[0])


def _mano_validity_path(group: str, side: str, leaf: str) -> str:
    if group not in ("state", "action"):
        raise ValueError(f"Unknown MANO group {group!r}")
    prefix = "" if group == "state" else "action_"
    validity_leaf = "ee_pose" if leaf == "ee_poses" else leaf
    return f"mano/validity/{prefix}{side}_{validity_leaf}"


def _mano_row_is_valid(handle: h5py.File, group: str, side: str, index: int) -> bool:
    return all(
        bool(np.asarray(handle[_mano_validity_path(group, side, leaf)][index], dtype=bool).all())
        for leaf in ("ee_poses", "ee_joint_states", "hand_betas")
    )


def validate_episode_file(
    path: Any,
    *,
    decode_images: bool = False,
    representation: str = REPRESENTATION_WUJI20,
) -> dict:
    """Validate the Spark0 fields needed by one explicit representation."""

    representation = validate_representation(representation)
    path = Path(path)
    with h5py.File(path, "r") as handle:
        representation_required = (
            WUJI20_REQUIRED_DATASETS
            if representation == REPRESENTATION_WUJI20
            else MANO45_REQUIRED_DATASETS
        )
        missing = [
            key for key in (*representation_required, *COMMON_REQUIRED_DATASETS) if key not in handle
        ]
        if missing:
            raise ValueError(
                f"{path}: missing required {representation} datasets: {missing}"
            )
        source_length = _episode_length(handle, representation)
        if source_length < 2:
            raise ValueError(f"{path}: episode needs at least two states, got {source_length}")
        if representation == REPRESENTATION_WUJI20:
            expected_dims = {
                f"{group}/{side}_{leaf}": dimension
                for group in ("state", "action")
                for side in SIDES
                for leaf, dimension in (("ee_poses", 7), ("ee_joint_states", HAND_DIM))
            }
        else:
            expected_dims = {
                f"mano/{group}/{side}_{leaf}": dimension
                for group in ("state", "action")
                for side in SIDES
                for leaf, dimension in (
                    ("ee_poses", 7),
                    ("ee_joint_states", MANO_POSE_DIM),
                    ("hand_betas", MANO_BETA_DIM),
                )
            }
        for key, expected_dim in expected_dims.items():
            dataset = handle[key]
            if dataset.shape[0] != source_length or dataset.shape[-1] != expected_dim:
                raise ValueError(
                    f"{path}: {key} expected ({source_length},...,{expected_dim}), "
                    f"got {dataset.shape}"
                )
        if representation == REPRESENTATION_MANO45:
            for key in MANO45_REQUIRED_DATASETS:
                if not key.startswith("mano/validity/"):
                    continue
                dataset = handle[key]
                if key.endswith("ee_pose"):
                    validity_dim = 7
                elif key.endswith("ee_joint_states"):
                    validity_dim = MANO_POSE_DIM
                elif key.endswith("hand_betas"):
                    validity_dim = MANO_BETA_DIM
                else:
                    raise AssertionError(key)
                if dataset.shape != (source_length, validity_dim):
                    raise ValueError(
                        f"{path}: {key} expected ({source_length},{validity_dim}), "
                        f"got {dataset.shape}"
                    )
                if dataset.dtype.kind != "b":
                    raise ValueError(f"{path}: {key} must be boolean, got {dataset.dtype}")

        extrinsics_shape = handle["vision/cam_head/extrinsics"].shape
        if extrinsics_shape not in {
            (7,),
            (source_length, 7),
            (4, 4),
            (source_length, 4, 4),
        }:
            raise ValueError(
                f"{path}: camera extrinsics must be constant or per-frame pose7/4x4, "
                f"got {extrinsics_shape}"
            )
        intrinsics_shape = handle["vision/cam_head/intrinsics"].shape
        if intrinsics_shape not in {
            (4,),
            (source_length, 4),
            (3, 3),
            (source_length, 3, 3),
        }:
            raise ValueError(
                f"{path}: camera intrinsics must be constant or per-frame [fx,fy,cx,cy]/3x3, "
                f"got {intrinsics_shape}"
            )
        if handle["vision/cam_head/colors"].shape[0] != source_length:
            raise ValueError(f"{path}: cam_head/colors length differs from state length")

        camera = handle["vision/cam_head/extrinsics"]
        frame_attr = camera.attrs.get("frame", "")
        if isinstance(frame_attr, bytes):
            frame_attr = frame_attr.decode("utf-8", errors="replace")
        if frame_attr and str(frame_attr) != "camera_to_env":
            raise ValueError(f"{path}: expected camera_to_env extrinsics, got {frame_attr!r}")

        image_shape = None
        if decode_images:
            image = np.asarray(decode_image_bit(handle["vision/cam_head/colors"][0]))
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{path}: decoded cam_head frame is not RGB HWC: {image.shape}")
            image_shape = list(image.shape)

        # MANO training has one valid sample per transition.  The all-invalid
        # terminal action row is deliberately excluded, matching LeRobot T-1.
        sample_length = (
            source_length - 1 if representation == REPRESENTATION_MANO45 else source_length
        )
        return {
            "path": str(path),
            "length": sample_length,
            "source_length": source_length,
            "task": path.parent.name,
            "image_shape": image_shape,
            "ee_pose_frame": _decode_text(handle.attrs.get("ee_pose_frame", "")),
            "representation": representation,
        }


class RoboDatasetCore:
    """Worker-safe lazy HDF5 dataset for guarded Spark0 representations."""

    def __init__(
        self,
        root_dir: str,
        statistics_path: Optional[str] = None,
        action_past_window_size: int = 0,
        action_future_window_size: int = 16,
        image_past_window_size: int = 0,
        image_future_window_size: int = 0,
        load_images: bool = True,
        mapping_path: Optional[str] = None,
        max_open_files: int = 8,
        representation: str = REPRESENTATION_WUJI20,
    ):
        self.root = str(Path(root_dir).resolve())
        self.representation = validate_representation(representation)
        self.state_hand_dim, self.action_hand_dim = representation_dimensions(
            self.representation
        )
        self.state_dual_dim = 2 * self.state_hand_dim
        self.action_dual_dim = 2 * self.action_hand_dim
        self.action_past_window_size = int(action_past_window_size)
        self.action_future_window_size = int(action_future_window_size)
        self.image_past_window_size = int(image_past_window_size)
        self.image_future_window_size = int(image_future_window_size)
        self.load_images = bool(load_images)
        if self.representation == REPRESENTATION_WUJI20:
            self.mapping_path = (
                str(Path(mapping_path).resolve()) if mapping_path else str(DEFAULT_MAPPING_PATH)
            )
            self.mapping = load_wuji_mapping(self.mapping_path)
        else:
            if mapping_path is not None:
                raise ValueError(
                    "mapping_path is a Wuji20 sparse-injection setting and must not be "
                    "provided for representation='mano45'"
                )
            self.mapping_path = None
            self.mapping = None
        self.max_open_files = max(1, int(max_open_files))
        if min(
            self.action_past_window_size,
            self.action_future_window_size,
            self.image_past_window_size,
            self.image_future_window_size,
        ) < 0:
            raise ValueError("Window sizes must be non-negative")

        root = Path(self.root)
        files = sorted({*root.rglob("*.hdf5"), *root.rglob("*.h5")})
        if not files:
            raise FileNotFoundError(f"No .hdf5/.h5 episodes found below {root}")
        self.episode_paths = [str(path.resolve()) for path in files]
        self.episode_lengths = []
        self.source_episode_lengths = []
        for path in self.episode_paths:
            metadata = validate_episode_file(
                path, decode_images=False, representation=self.representation
            )
            self.episode_lengths.append(int(metadata["length"]))
            self.source_episode_lengths.append(int(metadata["source_length"]))
        self._cumulative_lengths = np.cumsum(self.episode_lengths, dtype=np.int64).tolist()
        self.num_valid_frames = int(self._cumulative_lengths[-1])

        if statistics_path is not None:
            self.data_statistics = read_dataset_statistics(statistics_path)
            _validate_statistics_contract(
                statistics_path,
                self.representation,
                self.data_statistics,
            )
        else:
            self.data_statistics = None
        self.global_data_statistics = None
        self._handle_pid: Optional[int] = None
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()

    def __len__(self) -> int:
        return self.num_valid_frames

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle_pid"] = None
        state["_handles"] = OrderedDict()
        return state

    def _close_handles(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = OrderedDict()

    def __del__(self):
        self._close_handles()

    def _handle(self, path: str) -> h5py.File:
        pid = os.getpid()
        if self._handle_pid != pid:
            self._close_handles()
            self._handle_pid = pid
        if path in self._handles:
            handle = self._handles.pop(path)
            self._handles[path] = handle
            return handle
        # One read-only handle per DataLoader process is sufficient; requesting
        # SWMR here would reject otherwise valid files created without SWMR.
        handle = h5py.File(path, "r")
        self._handles[path] = handle
        if len(self._handles) > self.max_open_files:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle

    def _locate(self, index: int) -> Tuple[int, int]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Dataset index must be int, got {type(index).__name__}")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._cumulative_lengths, int(index))
        start = 0 if episode_index == 0 else self._cumulative_lengths[episode_index - 1]
        return episode_index, int(index - start)

    def set_global_data_statistics(self, global_data_statistics: dict) -> None:
        normalizer = GaussianNormalizer(global_data_statistics)
        if normalizer.state_mean.shape != (self.state_dual_dim,):
            raise ValueError(
                f"Global state statistics shape {normalizer.state_mean.shape} does not match "
                f"{self.representation} state dimension {self.state_dual_dim}"
            )
        if normalizer.action_mean.shape != (self.action_dual_dim,):
            raise ValueError(
                f"Global action statistics shape {normalizer.action_mean.shape} does not match "
                f"{self.representation} action dimension {self.action_dual_dim}"
            )
        if (
            not np.all(np.isfinite(normalizer.state_mean))
            or not np.all(np.isfinite(normalizer.state_std))
            or not np.all(np.isfinite(normalizer.action_mean))
            or not np.all(np.isfinite(normalizer.action_std))
            or np.any(normalizer.state_std <= 0)
            or np.any(normalizer.action_std <= 0)
        ):
            raise ValueError("Global normalization statistics must be finite with positive std")
        self.global_data_statistics = global_data_statistics
        self.gaussian_normalizer = normalizer

    def _camera_pose(self, handle: h5py.File, index: int, length: int) -> np.ndarray:
        return raw_camera_pose_to_vitra_matrix(
            _dataset_frame(handle["vision/cam_head/extrinsics"], index, length)
        )

    def _eef_in_anchor_camera(
        self, handle: h5py.File, group: str, side: str, index: int, anchor_camera: np.ndarray
    ) -> np.ndarray:
        if self.representation != REPRESENTATION_WUJI20:
            raise RuntimeError("Wuji EE-to-wrist calibration cannot be used for MANO45")
        ee_env = pose7_to_matrix(handle[f"{group}/{side}_ee_poses"][index])
        ee_to_wrist = np.asarray(
            self.mapping["calibration"][f"{side}_ee_to_vitra_wrist"], dtype=np.float64
        )
        return np.linalg.inv(anchor_camera) @ ee_env @ ee_to_wrist

    def _mano_root_in_anchor_camera(
        self, handle: h5py.File, group: str, side: str, index: int, anchor_camera: np.ndarray
    ) -> np.ndarray:
        """Read the fitted MANO global root directly; no Wuji wrist calibration."""

        if self.representation != REPRESENTATION_MANO45:
            raise RuntimeError("MANO root helper requires representation='mano45'")
        root_env = pose7_to_matrix(handle[f"mano/{group}/{side}_ee_poses"][index])
        return np.linalg.inv(anchor_camera) @ root_env

    def _mano_state_part(
        self, handle: h5py.File, side: str, index: int, anchor_camera: np.ndarray
    ) -> np.ndarray:
        if not _mano_row_is_valid(handle, "state", side, index):
            raise ValueError(
                f"{self.episode_paths}: invalid MANO state row {index} for {side}"
            )
        root = self._mano_root_in_anchor_camera(handle, "state", side, index, anchor_camera)
        local_euler = mano_rotvec45_to_euler45(
            handle[f"mano/state/{side}_ee_joint_states"][index]
        )
        betas = np.asarray(
            handle[f"mano/state/{side}_hand_betas"][index], dtype=np.float32
        )
        if betas.shape != (MANO_BETA_DIM,) or not np.all(np.isfinite(betas)):
            raise ValueError(f"Invalid MANO betas for {side} at row {index}: {betas.shape}")
        return np.concatenate([matrix_to_pose6(root), local_euler, betas]).astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        episode_index, frame = self._locate(index)
        handle = self._handle(self.episode_paths[episode_index])
        sample_length = self.episode_lengths[episode_index]
        source_length = self.source_episode_lengths[episode_index]
        anchor_camera = self._camera_pose(handle, frame, source_length)

        state_parts = []
        for side in SIDES:
            if self.representation == REPRESENTATION_WUJI20:
                wrist = self._eef_in_anchor_camera(
                    handle, "state", side, frame, anchor_camera
                )
                hand = np.asarray(
                    handle[f"state/{side}_ee_joint_states"][frame], dtype=np.float32
                )
                state_parts.append(
                    np.concatenate([matrix_to_pose6(wrist), hand]).astype(np.float32)
                )
            else:
                state_parts.append(
                    self._mano_state_part(handle, side, frame, anchor_camera)
                )
        current_state = np.concatenate(state_parts).astype(np.float32)
        if current_state.shape != (self.state_dual_dim,) or not np.all(np.isfinite(current_state)):
            raise ValueError(
                f"Invalid {self.representation} current state at {self.episode_paths[episode_index]} "
                f"row {frame}: shape={current_state.shape}"
            )

        action_indices = np.arange(
            frame - self.action_past_window_size,
            frame + self.action_future_window_size + 1,
            dtype=np.int64,
        )
        action_list = np.zeros(
            (len(action_indices), self.action_dual_dim), dtype=np.float32
        )
        action_mask = np.zeros((len(action_indices), 2), dtype=bool)
        for row, action_index in enumerate(action_indices.tolist()):
            # In both source contracts, action[t] is the already-shifted target
            # state[t+1].  Read the same action row and never shift it again.
            if action_index < 0 or action_index >= source_length - 1:
                continue
            for hand_index, side in enumerate(SIDES):
                if self.representation == REPRESENTATION_WUJI20:
                    current_wrist = self._eef_in_anchor_camera(
                        handle, "state", side, action_index, anchor_camera
                    )
                    target_wrist = self._eef_in_anchor_camera(
                        handle, "action", side, action_index, anchor_camera
                    )
                    hand_target = np.asarray(
                        handle[f"action/{side}_ee_joint_states"][action_index],
                        dtype=np.float32,
                    )
                else:
                    if not _mano_row_is_valid(handle, "state", side, action_index):
                        raise ValueError(
                            f"Invalid MANO state transition start for {side} at row {action_index}"
                        )
                    if not _mano_row_is_valid(handle, "action", side, action_index):
                        raise ValueError(
                            f"Invalid MANO action target for {side} at row {action_index}"
                        )
                    current_wrist = self._mano_root_in_anchor_camera(
                        handle, "state", side, action_index, anchor_camera
                    )
                    target_wrist = self._mano_root_in_anchor_camera(
                        handle, "action", side, action_index, anchor_camera
                    )
                    hand_target = mano_rotvec45_to_euler45(
                        handle[f"mano/action/{side}_ee_joint_states"][action_index]
                    )
                delta_position = target_wrist[:3, 3] - current_wrist[:3, 3]
                delta_rotation = target_wrist[:3, :3] @ current_wrist[:3, :3].T
                delta_euler = Rotation.from_matrix(delta_rotation).as_euler("xyz", degrees=False)
                base = hand_index * self.action_hand_dim
                action_part = np.concatenate(
                    [delta_position, delta_euler, hand_target]
                ).astype(np.float32)
                if action_part.shape != (self.action_hand_dim,) or not np.all(
                    np.isfinite(action_part)
                ):
                    raise ValueError(
                        f"Invalid {self.representation} action for {side} at row {action_index}"
                    )
                action_list[row, base : base + self.action_hand_dim] = action_part
                action_mask[row, hand_index] = True

        intrinsics = intrinsics_to_matrix(
            _dataset_frame(handle["vision/cam_head/intrinsics"], frame, source_length)
        )
        if self.load_images:
            image_indices = np.arange(
                frame - self.image_past_window_size,
                frame + self.image_future_window_size + 1,
                dtype=np.int64,
            )
            image_mask = (image_indices >= 0) & (image_indices < source_length)
            clipped = np.clip(image_indices, 0, source_length - 1)
            images = [
                np.asarray(decode_image_bit(handle["vision/cam_head/colors"][int(i)]))
                for i in clipped
            ]
            image_list = np.stack(images, axis=0).astype(np.uint8, copy=False)
            height, width = image_list[-1].shape[:2]
        else:
            image_list = None
            image_mask = None
            height = float(2.0 * intrinsics[1, 2])
            width = float(2.0 * intrinsics[0, 2])

        instruction_value = _dataset_frame(handle["instruction"], frame, source_length)
        sample = {
            "instruction": format_vitra_instruction(instruction_value),
            "action_list": action_list,
            "action_mask": action_mask,
            "current_state": current_state,
            "current_state_mask": np.ones(2, dtype=bool),
            "fov": calculate_fov(height, width, intrinsics).astype(np.float32),
            "intrinsics": intrinsics.astype(np.float32),
            "episode_path": self.episode_paths[episode_index],
            "frame_index": frame,
            "representation": self.representation,
            "episode_sample_length": sample_length,
            "episode_source_length": source_length,
        }
        if image_list is not None:
            sample["image_list"] = image_list
            sample["image_mask"] = image_mask
        return sample

    def transform_trajectory(self, sample_dict: dict, normalization: bool = True) -> dict:
        """Normalize in native space, then map/pad to VITRA's model layout."""

        action = np.asarray(sample_dict["action_list"], dtype=np.float32)
        state = np.asarray(sample_dict["current_state"], dtype=np.float32)
        if normalization:
            if not hasattr(self, "gaussian_normalizer"):
                raise RuntimeError(
                    "Normalization requested before set_global_data_statistics(); "
                    "ensure teledata_statistics.json exists"
                )
            action = self.gaussian_normalizer.normalize_action(action).astype(np.float32)
            state = self.gaussian_normalizer.normalize_state(state).astype(np.float32)

        if self.representation == REPRESENTATION_WUJI20:
            human_action, human_action_mask = native_action_to_human(
                action,
                self.mapping,
                sample_dict["action_mask"],
                return_mask=True,
            )
            human_state, human_state_mask = native_state_to_human(
                state,
                self.mapping,
                sample_dict["current_state_mask"],
                return_mask=True,
            )
        else:
            human_action, human_action_mask = mano_action_to_human(
                action,
                sample_dict["action_mask"],
                return_mask=True,
            )
            human_state, human_state_mask = mano_state_to_human(
                state,
                sample_dict["current_state_mask"],
                return_mask=True,
            )
        sample_dict["action_list"] = torch.from_numpy(human_action)
        sample_dict["action_mask"] = torch.from_numpy(human_action_mask)
        sample_dict["current_state"] = torch.from_numpy(human_state)
        sample_dict["current_state_mask"] = torch.from_numpy(human_state_mask)
        return sample_dict
