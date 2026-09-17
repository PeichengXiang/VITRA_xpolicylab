"""EgoVLA H1-Inspire raw dataset with table-index Inspire12 <-> MANO45.

``inspire12`` keeps the robot adapter contract. Each hand has 18 values::

    state  = absolute camera-space EEF [xyz, Euler_xyz] + absolute Inspire12
    action = observed step EEF delta [dxyz, Euler_xyz(dR)]
             + next observed Inspire12

The dual-hand vector is 36-dimensional. It is normalized in this native space
and only then sparsely injected into VITRA's human 192/212 layout with the
locked ``inspire12_mano45_xyz_sparse_v1`` table-index map.

Native 12-D order is the official EgoVLA H1 interleaved Inspire layout
(index/middle/pinky/ring proximal+intermediate, then thumb yaw/pitch/PIP/IP).
All 12 simulator channels are mapped, including intermediate/distal values,
so ``mano2inspire(inspire2mano(q)) == q`` exactly.

Raw EgoVLA HDF5 has no camera K/E. Camera slots use the official
EgoVLA_Release world-fixed ``main_camera`` (same constants as
``spark_data.alignment.egovla_camera``). Images are already 384x384 RGB.
"""

from __future__ import annotations

import bisect
import json
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from vitra.datasets.dataset_utils import calculate_fov
from vitra.datasets.spark0_dataset import (
    EEF_DIM,
    MODEL_ACTION_DIM,
    MODEL_STATE_DIM,
    VITRA_HAND_DIM,
    format_vitra_instruction,
    intrinsics_to_matrix,
    matrix_to_pose6,
    matrix_to_pose7,
    pose7_to_matrix,
    raw_camera_pose_to_vitra_matrix,
)
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics


REPRESENTATION_INSPIRE12 = "inspire12"
SUPPORTED_REPRESENTATIONS = (REPRESENTATION_INSPIRE12,)
INSPIRE_HAND_DIM = 12
NATIVE_HAND_DIM = EEF_DIM + INSPIRE_HAND_DIM
NATIVE_DUAL_DIM = 2 * NATIVE_HAND_DIM
SIDES = ("left", "right")
DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[3] / "mapping_inspire12_mano45.json"
INSPIRE2MANO_CODEC_ID = "inspire12_mano45_xyz_sparse_v1"
_INSPIRE2MANO_DST = (2, 5, 11, 14, 20, 23, 29, 32, 37, 36, 40, 43)
_INSPIRE2MANO_SIGNS = (1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1)

# Official H1+Inspire 50-D layout from egovla_xpolicy.action.
EGO_H1_LEFT_HAND_INDICES = (26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48)
EGO_H1_RIGHT_HAND_INDICES = (31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49)
HAND_QPOS_INDICES = {
    "left": EGO_H1_LEFT_HAND_INDICES,
    "right": EGO_H1_RIGHT_HAND_INDICES,
}

TASK_INSTRUCTIONS = {
    "Close-Drawer": "Close the opened drawer",
    "Flip-Mug": "Flip the mug",
    "Insert-And-Unload-Cans": "Insert and unload the cans",
    "Insert-Cans": "Insert the cans",
    "Open-Drawer": "Open the drawer",
    "Open-Laptop": "Open the laptop",
    "Pour-Balls": "Pour the balls",
    "Push-Box": "Push the box",
    "Sort-Cans": "Sort the cans",
    "Stack-Can": "Stack the can",
    "Stack-Can-Into-Drawer": "Stack the can into the drawer",
    "Unload-Cans": "Unload the cans",
}

EGOVLA_IMAGE_HW = (384, 384)
QPOS_DIM = 50
REQUIRED_DATASETS = (
    "action",
    "observations/qpos",
    "observations/left_target_ee_pose",
    "observations/right_target_ee_pose",
    "observations/images/main",
)
CURRENT_EE_KEYS = {
    "left": ("observations/left_ee_pose", "observations/left_curr_ee_pose"),
    "right": ("observations/right_ee_pose", "observations/right_curr_ee_pose"),
}


def current_ee_key(handle: h5py.File, side: str) -> str:
    for key in CURRENT_EE_KEYS[side]:
        if key in handle:
            return key
    raise KeyError(f"missing current EE pose for {side}; tried {CURRENT_EE_KEYS[side]}")

# Locked EgoVLA_Release main_camera. See spark_data.alignment.egovla_camera.
ISAAC_MAIN_CAM_XYZ = np.array([0.09, 0.0, 1.7], dtype=np.float64)
ISAAC_MAIN_CAM_WXYZ = np.array(
    [0.66446, 0.24184, -0.24184, -0.664464], dtype=np.float64
)
EGOVLA_GT_CAM_WXYZ = np.array(
    [0.9063077870366499, 0.0, 0.42261826174069944, 0.0], dtype=np.float64
)
CAM_AXIS_TRANSFORM = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
GRAPHICS_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
K_1280_720 = np.array([488.6662, 488.6662, 640.0, 360.0], dtype=np.float64)
SRC_WH = (1280, 720)
DST_WH = (384, 384)


def validate_representation(representation: str) -> str:
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {SUPPORTED_REPRESENTATIONS}, got {representation!r}"
        )
    return representation


def representation_dimensions(representation: str) -> Tuple[int, int]:
    validate_representation(representation)
    return NATIVE_HAND_DIM, NATIVE_HAND_DIM


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(matrix)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def official_head_intrinsics() -> np.ndarray:
    fx, fy, cx, cy = K_1280_720
    src_w, src_h = SRC_WH
    dst_w, dst_h = DST_WH
    return np.array(
        [fx * dst_w / src_w, fy * dst_h / src_h, cx * dst_w / src_w, cy * dst_h / src_h],
        dtype=np.float64,
    )


def official_head_pose7() -> np.ndarray:
    isaac_rot = _quat_wxyz_to_matrix(ISAAC_MAIN_CAM_WXYZ)
    rotation_t = _quat_wxyz_to_matrix(EGOVLA_GT_CAM_WXYZ)
    if not np.allclose(
        (rotation_t @ np.linalg.inv(isaac_rot)) @ isaac_rot, rotation_t, atol=1e-6
    ):
        raise RuntimeError("EgoVLA official camera frame-change identity failed")
    rotation_spark = rotation_t @ np.linalg.inv(CAM_AXIS_TRANSFORM) @ GRAPHICS_TO_OPENCV
    pose = np.empty(7, dtype=np.float64)
    pose[:3] = ISAAC_MAIN_CAM_XYZ
    pose[3:] = _matrix_to_quat_wxyz(rotation_spark)
    return pose


OFFICIAL_CAMERA_VITRA = raw_camera_pose_to_vitra_matrix(official_head_pose7())
OFFICIAL_INTRINSICS = official_head_intrinsics()


def unpack_inspire12(q50: Any, side: str) -> np.ndarray:
    """Extract H1-interleaved Inspire12 from a 50-D qpos/action vector."""

    value = np.asarray(q50, dtype=np.float32)
    indices = HAND_QPOS_INDICES[side]
    if value.shape[-1] != QPOS_DIM:
        raise ValueError(f"Expected last dimension {QPOS_DIM}, got {value.shape}")
    return value[..., list(indices)]


def pack_inspire12_into_q50(q12: Any, side: str, target: Optional[np.ndarray] = None) -> np.ndarray:
    """Scatter H1-interleaved Inspire12 into a 50-D articulation vector."""

    hand = np.asarray(q12, dtype=np.float32)
    if hand.shape[-1] != INSPIRE_HAND_DIM:
        raise ValueError(f"Expected Inspire last dimension {INSPIRE_HAND_DIM}, got {hand.shape}")
    if target is None:
        out = np.zeros(hand.shape[:-1] + (QPOS_DIM,), dtype=np.float32)
    else:
        out = np.asarray(target, dtype=np.float32)
        if out.shape != hand.shape[:-1] + (QPOS_DIM,):
            raise ValueError(f"target shape {out.shape} does not match {hand.shape[:-1] + (QPOS_DIM,)}")
    out[..., list(HAND_QPOS_INDICES[side])] = hand
    return out


@lru_cache(maxsize=8)
def _load_mapping_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    _validate_mapping(mapping)
    return mapping


def _validate_mapping(mapping: Mapping[str, Any]) -> None:
    entries = mapping.get("joint_mapping", [])
    if len(entries) != INSPIRE_HAND_DIM:
        raise ValueError(f"Inspire mapping must contain exactly {INSPIRE_HAND_DIM} entries")
    src = [int(entry["source_index"]) for entry in entries]
    dst = [int(entry["mano_pose_index"]) for entry in entries]
    signs = [float(entry["sign"]) for entry in entries]
    if src != list(range(INSPIRE_HAND_DIM)):
        raise ValueError("Inspire mapping source indices must be 0..11 in order")
    if len(set(dst)) != INSPIRE_HAND_DIM or any(index < 0 or index >= 45 for index in dst):
        raise ValueError("Inspire mapping MANO destinations must be 12 unique indices in 0..44")
    if any(sign not in (-1.0, 1.0) for sign in signs):
        raise ValueError("Inspire mapping signs must be +1 or -1")

    injection = np.zeros((45, INSPIRE_HAND_DIM), dtype=np.float64)
    for source, target, sign in zip(src, dst, signs):
        injection[target, source] = sign
    if not np.array_equal(injection.T @ injection, np.eye(INSPIRE_HAND_DIM)):
        raise ValueError("Signed Inspire-to-MANO injection must satisfy E.T @ E == I")

    codec_id = mapping.get("codec_id")
    if codec_id not in (None, INSPIRE2MANO_CODEC_ID):
        raise ValueError(f"Unsupported Inspire/MANO codec_id={codec_id!r}")
    if codec_id == INSPIRE2MANO_CODEC_ID:
        if tuple(dst) != _INSPIRE2MANO_DST or tuple(int(s) for s in signs) != _INSPIRE2MANO_SIGNS:
            raise ValueError(
                f"{INSPIRE2MANO_CODEC_ID} destination/sign table does not match the locked index map"
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
    entries = mapping["joint_mapping"]
    source = np.asarray([int(entry["source_index"]) for entry in entries], dtype=np.int64)
    destination = np.asarray(
        [int(entry["mano_pose_index"]) for entry in entries], dtype=np.int64
    )
    signs = np.asarray([float(entry["sign"]) for entry in entries], dtype=np.float32)
    return source, destination, signs


def load_inspire_mapping(mapping: Optional[Any] = None) -> Mapping[str, Any]:
    """Load and validate the signed Inspire12-to-MANO45 mapping."""

    if mapping is None:
        return _load_mapping_file(str(DEFAULT_MAPPING_PATH))
    if isinstance(mapping, (str, os.PathLike)):
        return _load_mapping_file(str(Path(mapping).resolve()))
    if not isinstance(mapping, Mapping):
        raise TypeError(f"mapping must be a path or mapping object, got {type(mapping).__name__}")
    _validate_mapping(mapping)
    return mapping


def inspire12_to_mano45(q12: Any, mapping: Optional[Any] = None) -> np.ndarray:
    """Sparse table encode: pose45[dst] = sign * q12[src]."""

    hand = np.asarray(q12, dtype=np.float32)
    if hand.shape[-1] != INSPIRE_HAND_DIM:
        raise ValueError(f"Expected Inspire last dimension {INSPIRE_HAND_DIM}, got {hand.shape}")
    spec = load_inspire_mapping(mapping)
    source, destination, signs = _mapping_index_tables(spec)
    pose = np.zeros(hand.shape[:-1] + (45,), dtype=np.float32)
    pose[..., destination] = signs * hand[..., source]
    return pose


def mano45_to_inspire12(pose45: Any, mapping: Optional[Any] = None) -> np.ndarray:
    """Sparse table decode: q12[src] = sign * pose45[dst]."""

    pose = np.asarray(pose45, dtype=np.float32)
    if pose.shape[-1] != 45:
        raise ValueError(f"Expected MANO last dimension 45, got {pose.shape}")
    spec = load_inspire_mapping(mapping)
    source, destination, signs = _mapping_index_tables(spec)
    hand = np.zeros(pose.shape[:-1] + (INSPIRE_HAND_DIM,), dtype=np.float32)
    hand[..., source] = signs * pose[..., destination]
    return hand


def _map_native_to_human(
    native: Any,
    output_dim: int,
    mapping: Optional[Any] = None,
    hand_mask: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    native = np.asarray(native, dtype=np.float32)
    if native.shape[-1] != NATIVE_DUAL_DIM:
        raise ValueError(f"Expected native last dimension {NATIVE_DUAL_DIM}, got {native.shape}")
    spec = load_inspire_mapping(mapping)
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
    for hand_index in range(2):
        native_base = hand_index * NATIVE_HAND_DIM
        human_base = hand_index * VITRA_HAND_DIM
        side_active = active[..., hand_index]
        human[..., human_base : human_base + EEF_DIM] = native[
            ..., native_base : native_base + EEF_DIM
        ]
        human[..., human_base + EEF_DIM + destination] = (
            signs * native[..., native_base + EEF_DIM + source]
        )
        mask[..., human_base : human_base + VITRA_HAND_DIM] = side_active[..., None]
        human[..., human_base : human_base + VITRA_HAND_DIM] *= side_active[..., None]
    return human, mask


def inspire_state_to_human(
    native_state: Any,
    mapping: Optional[Any] = None,
    hand_mask: Optional[Any] = None,
    *,
    return_mask: bool = False,
    mapping_path: Optional[Any] = None,
):
    """Inject native state36 into VITRA state212 after native normalization."""

    state, mask = _map_native_to_human(
        native_state, MODEL_STATE_DIM, mapping if mapping is not None else mapping_path, hand_mask
    )
    return (state, mask) if return_mask else state


def inspire_action_to_human(
    native_action: Any,
    mapping: Optional[Any] = None,
    hand_mask: Optional[Any] = None,
    *,
    return_mask: bool = False,
    mapping_path: Optional[Any] = None,
):
    """Inject native action36 into VITRA action192 after native normalization."""

    action, mask = _map_native_to_human(
        native_action, MODEL_ACTION_DIM, mapping if mapping is not None else mapping_path, hand_mask
    )
    return (action, mask) if return_mask else action


def human_action_to_inspire(
    human_action: Any,
    mapping: Optional[Any] = None,
    *,
    mapping_path: Optional[Any] = None,
) -> np.ndarray:
    """Project VITRA action192 back to native action36 (linear map only)."""

    human = np.asarray(human_action, dtype=np.float32)
    if human.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(f"Expected human action last dimension {MODEL_ACTION_DIM}, got {human.shape}")
    spec = load_inspire_mapping(mapping if mapping is not None else mapping_path)
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


def _camera_dict(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    vision = observation.get("vision")
    if isinstance(vision, Mapping):
        for key in ("cam_head", "head", "camera_head", "main"):
            camera = vision.get(key)
            if isinstance(camera, Mapping):
                return camera
    raise KeyError("Runtime observation must contain obs['vision'][cam_head]")


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


def _eef_in_camera(
    ee_pose: Any,
    camera_vitra_to_env: np.ndarray,
    ee_to_wrist: np.ndarray,
) -> np.ndarray:
    return np.linalg.inv(camera_vitra_to_env) @ pose7_to_matrix(ee_pose) @ ee_to_wrist


def _step_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta_position = target[:3, 3] - current[:3, 3]
    delta_rotation = target[:3, :3] @ current[:3, :3].T
    delta_euler = Rotation.from_matrix(delta_rotation).as_euler("xyz", degrees=False)
    return np.concatenate([delta_position, delta_euler]).astype(np.float32)


def _statistics_representation(statistics_path: str) -> Optional[str]:
    with open(statistics_path, "r", encoding="utf-8") as handle:
        value = json.load(handle).get("representation")
    if value in SUPPORTED_REPRESENTATIONS:
        return value
    return None


def _validate_statistics_contract(
    statistics_path: str,
    representation: str,
    statistics: Mapping[str, np.ndarray],
) -> None:
    declared = _statistics_representation(statistics_path)
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
    wrong = {
        key: np.asarray(statistics[key]).shape
        for key, dimension in expected.items()
        if np.asarray(statistics[key]).shape != (dimension,)
    }
    if wrong:
        raise ValueError(
            f"Statistics dimensions do not match representation {representation!r}: {wrong}"
        )


def task_instruction_from_path(path: str | Path) -> str:
    task = Path(path).resolve().parent.name
    return TASK_INSTRUCTIONS.get(task, task.replace("-", " ").lower())


def validate_egovla_episode(path: Any) -> dict:
    """Validate one raw EgoVLA-Humanoid-Sim episode for inspire12 training."""

    path = str(Path(path).resolve())
    with h5py.File(path, "r") as handle:
        missing = [key for key in REQUIRED_DATASETS if key not in handle]
        if missing:
            raise ValueError(f"{path}: missing required EgoVLA datasets: {missing}")
        qpos = handle["observations/qpos"]
        action = handle["action"]
        image = handle["observations/images/main"]
        if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
            raise ValueError(f"{path}: qpos must be (T,50), got {qpos.shape}")
        if action.shape != qpos.shape:
            raise ValueError(f"{path}: action shape {action.shape} != qpos {qpos.shape}")
        source_length = int(qpos.shape[0])
        if source_length < 2:
            raise ValueError(f"{path}: expected at least two frames, got {source_length}")
        if image.shape != (source_length, *EGOVLA_IMAGE_HW, 3):
            raise ValueError(f"{path}: images/main must be (T,384,384,3), got {image.shape}")
        for side in SIDES:
            current_key = current_ee_key(handle, side)
            target_key = f"observations/{side}_target_ee_pose"
            if handle[current_key].shape != (source_length, 7):
                raise ValueError(f"{path}: {current_key} must be (T,7), got {handle[current_key].shape}")
            if handle[target_key].shape != (source_length, 7):
                raise ValueError(f"{path}: {target_key} must be (T,7), got {handle[target_key].shape}")
        return {
            "path": path,
            "length": source_length,
            "source_length": source_length,
            "representation": REPRESENTATION_INSPIRE12,
        }


def runtime_observation_to_inspire(
    observation: Mapping[str, Any],
    mapping: Optional[Any] = None,
    *,
    mapping_path: Optional[Any] = None,
) -> dict:
    """Convert one decoded XPolicyLab observation to native Inspire36 inputs."""

    spec = load_inspire_mapping(mapping if mapping is not None else mapping_path)
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
        camera_vitra_to_env = OFFICIAL_CAMERA_VITRA
    else:
        camera_vitra_to_env = raw_camera_pose_to_vitra_matrix(
            _latest_pose_or_matrix(camera[extrinsics_key], f"cam_head/{extrinsics_key}")
        )

    intrinsics_key = next(
        (
            key
            for key in ("intrinsics", "intrinsic", "intrinsics_matrix", "intrinsic_matrix")
            if key in camera
        ),
        None,
    )
    if intrinsics_key is None:
        intrinsics = intrinsics_to_matrix(OFFICIAL_INTRINSICS)
    else:
        intrinsics_value = np.asarray(camera[intrinsics_key])
        if intrinsics_value.ndim >= 3 and intrinsics_value.shape[-2:] == (3, 3):
            intrinsics_value = intrinsics_value.reshape(-1, 3, 3)[-1]
        elif intrinsics_value.ndim >= 2 and intrinsics_value.shape[-2:] != (3, 3):
            intrinsics_value = intrinsics_value.reshape(-1, intrinsics_value.shape[-1])[-1]
        intrinsics = intrinsics_to_matrix(intrinsics_value)

    native_parts = []
    current_wrist = {}
    calibrations = {}
    for side in SIDES:
        ee_pose = _latest_vector(state[f"{side}_ee_pose"], 7, f"state/{side}_ee_pose")
        hand = _latest_vector(
            state[f"{side}_ee_joint_state"], INSPIRE_HAND_DIM, f"state/{side}_ee_joint_state"
        )
        ee_to_wrist = np.asarray(
            spec["calibration"][f"{side}_ee_to_vitra_wrist"], dtype=np.float64
        )
        wrist_camera = _eef_in_camera(ee_pose, camera_vitra_to_env, ee_to_wrist)
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
        "representation": REPRESENTATION_INSPIRE12,
    }
    return {
        "image": image,
        "instruction": instruction,
        "native_state": np.concatenate(native_parts).astype(np.float32),
        "context": context,
        "fov": fov,
        "intrinsics": intrinsics,
    }


def inspire_action_chunk_to_xpolicy(
    native_action: Any,
    context: Mapping[str, Any],
    *,
    mapping_path: Optional[Any] = None,
) -> list[dict]:
    """Integrate native step deltas and emit absolute Inspire12 EE actions."""

    del mapping_path
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
            current[side][:3, :3] = delta_rotation @ current[side][:3, :3]
            ee_env = camera_vitra_to_env @ current[side] @ np.linalg.inv(ee_to_wrist[side])
            command[f"{side}_ee_pose"] = matrix_to_pose7(ee_env).astype(np.float32)
            command[f"{side}_ee_joint_state"] = row[
                base + EEF_DIM : base + NATIVE_HAND_DIM
            ].copy()
        output.append(command)
    return output


class EgoVLAInspireDatasetCore:
    """Worker-safe lazy HDF5 dataset for raw EgoVLA inspire12 episodes."""

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
        representation: str = REPRESENTATION_INSPIRE12,
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
        self.mapping_path = (
            str(Path(mapping_path).resolve()) if mapping_path else str(DEFAULT_MAPPING_PATH)
        )
        self.mapping = load_inspire_mapping(self.mapping_path)
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
        self.episode_instructions = []
        for path in self.episode_paths:
            metadata = validate_egovla_episode(path)
            self.episode_lengths.append(int(metadata["length"]))
            self.source_episode_lengths.append(int(metadata["source_length"]))
            self.episode_instructions.append(task_instruction_from_path(path))
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

    def _wrist_in_camera(self, ee_pose: Any) -> np.ndarray:
        ee_to_wrist = np.eye(4, dtype=np.float64)
        return _eef_in_camera(ee_pose, OFFICIAL_CAMERA_VITRA, ee_to_wrist)

    def __getitem__(self, index: int) -> dict:
        episode_index, frame = self._locate(index)
        handle = self._handle(self.episode_paths[episode_index])
        sample_length = self.episode_lengths[episode_index]
        source_length = self.source_episode_lengths[episode_index]
        qpos = handle["observations/qpos"]
        state_parts = []
        current_ee_keys = {side: current_ee_key(handle, side) for side in SIDES}
        for side in SIDES:
            wrist = self._wrist_in_camera(handle[current_ee_keys[side]][frame])
            hand = unpack_inspire12(qpos[frame], side)
            state_parts.append(np.concatenate([matrix_to_pose6(wrist), hand]).astype(np.float32))
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
        action_list = np.zeros((len(action_indices), self.action_dual_dim), dtype=np.float32)
        action_mask = np.zeros((len(action_indices), 2), dtype=bool)
        for row, action_index in enumerate(action_indices.tolist()):
            # VITRA's ``step`` contract is one realized transition per row:
            # observed state[t] -> observed state[t+1].  EgoVLA's
            # ``*_target_ee_pose[t]`` and ``action[t]`` are controller targets,
            # not the realized next state, so they must not be used as step
            # labels.  The terminal row has no t+1 observation and remains
            # masked, as do ordinary out-of-window rows.
            if action_index < 0 or action_index >= source_length - 1:
                continue
            for hand_index, side in enumerate(SIDES):
                current_wrist = self._wrist_in_camera(
                    handle[current_ee_keys[side]][action_index]
                )
                target_wrist = self._wrist_in_camera(
                    handle[current_ee_keys[side]][action_index + 1]
                )
                hand_target = unpack_inspire12(qpos[action_index + 1], side)
                action_part = np.concatenate(
                    [_step_delta(current_wrist, target_wrist), hand_target]
                ).astype(np.float32)
                if action_part.shape != (self.action_hand_dim,) or not np.all(
                    np.isfinite(action_part)
                ):
                    raise ValueError(
                        f"Invalid {self.representation} action for {side} at row {action_index}"
                    )
                base = hand_index * self.action_hand_dim
                action_list[row, base : base + self.action_hand_dim] = action_part
                action_mask[row, hand_index] = True

        intrinsics = intrinsics_to_matrix(OFFICIAL_INTRINSICS)
        if self.load_images:
            image_indices = np.arange(
                frame - self.image_past_window_size,
                frame + self.image_future_window_size + 1,
                dtype=np.int64,
            )
            image_mask = (image_indices >= 0) & (image_indices < source_length)
            clipped = np.clip(image_indices, 0, source_length - 1)
            images = [
                np.asarray(handle["observations/images/main"][int(i)]) for i in clipped
            ]
            image_list = np.stack(images, axis=0).astype(np.uint8, copy=False)
            if image_list.shape[-3:] != (*EGOVLA_IMAGE_HW, 3):
                raise ValueError(
                    f"{self.episode_paths[episode_index]}: decoded image must be "
                    f"{EGOVLA_IMAGE_HW + (3,)}, got {image_list.shape}"
                )
            height, width = image_list[-1].shape[:2]
        else:
            image_list = None
            image_mask = None
            height, width = EGOVLA_IMAGE_HW

        sample = {
            "instruction": format_vitra_instruction(self.episode_instructions[episode_index]),
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
        """Normalize in native Inspire36 space, then inject into 192/212."""

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

        human_action, human_action_mask = inspire_action_to_human(
            action,
            self.mapping,
            sample_dict["action_mask"],
            return_mask=True,
        )
        human_state, human_state_mask = inspire_state_to_human(
            state,
            self.mapping,
            sample_dict["current_state_mask"],
            return_mask=True,
        )
        sample_dict["action_list"] = torch.from_numpy(human_action)
        sample_dict["action_mask"] = torch.from_numpy(human_action_mask)
        sample_dict["current_state"] = torch.from_numpy(human_state)
        sample_dict["current_state_mask"] = torch.from_numpy(human_state_mask)
        return sample_dict
