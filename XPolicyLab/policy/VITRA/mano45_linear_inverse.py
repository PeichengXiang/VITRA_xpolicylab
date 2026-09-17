"""Hash-locked linear MANO45 -> Wuji Hand2 action decoder.

The MANO45 training labels are deterministically paired with native Wuji q20
targets.  This module loads a small, side-specific ridge artifact fitted on
those pairs and applies it directly at runtime.  It deliberately does *not*
run the geometric IK afterwards: the MANO/Wuji point objective has multiple
q-equivalent minima and was measured to pull an accurate supervised estimate
away from the recorded robot configuration.

Safety is fail-closed at this decoder boundary.  It rejects out-of-distribution
MANO poses, large reverse/forward-cycle residuals, implausible per-step q
changes, and material URDF-limit violations.  The runtime first retries a
rejected prediction through the historical bounded geometric inverse; only a
failure of that fallback reaches the existing atomic dual-hand hold behavior.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA_VERSION = 1
ARTIFACT_KIND = "xpolicylab_vitra_mano45_to_wujihand2_linear_inverse"
INPUT_REPRESENTATION = "mano_local_rotvec45"
MODEL_INPUT_ENCODING = "mano_local_euler_xyz45"
OUTPUT_REPRESENTATION = "wujihand2_q20_stage_major"
SIDES = ("left", "right")
MANO_JOINT_ORDER = (
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
)
WUJI_STAGE_MAJOR_ORDER = (
    "index_mcp_flex",
    "middle_mcp_flex",
    "pinky_mcp_flex",
    "ring_mcp_flex",
    "thumb_cmc_flex",
    "index_mcp_abd",
    "middle_mcp_abd",
    "pinky_mcp_abd",
    "ring_mcp_abd",
    "thumb_cmc_abd",
    "index_pip",
    "middle_pip",
    "pinky_pip",
    "ring_pip",
    "thumb_mcp",
    "index_dip",
    "middle_dip",
    "pinky_dip",
    "ring_dip",
    "thumb_ip",
)


class LinearInverseContractError(RuntimeError):
    """The artifact is missing, corrupt, or uses a different representation."""


class LinearInverseSafetyError(RuntimeError):
    """A finite prediction failed a calibrated runtime safety gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_array(array: Any, *, dtype: str) -> dict[str, Any]:
    """Encode one fixed-shape ndarray for deterministic JSON artifacts."""

    value = np.ascontiguousarray(np.asarray(array, dtype=np.dtype(dtype)))
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "data_b64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(
    payload: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    if not isinstance(payload, Mapping):
        raise LinearInverseContractError(f"{name} must be an encoded array object")
    if tuple(payload.get("shape", ())) != shape:
        raise LinearInverseContractError(
            f"{name} has shape={payload.get('shape')!r}, expected={list(shape)!r}"
        )
    try:
        encoded_dtype = np.dtype(str(payload["dtype"]))
        raw = base64.b64decode(str(payload["data_b64"]), validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise LinearInverseContractError(f"{name} has invalid encoded data") from exc
    if encoded_dtype.kind != dtype.kind or encoded_dtype.itemsize != dtype.itemsize:
        raise LinearInverseContractError(
            f"{name} has dtype={encoded_dtype}, expected {dtype}"
        )
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise LinearInverseContractError(
            f"{name} has {len(raw)} decoded bytes, expected {expected_bytes}"
        )
    value = np.frombuffer(raw, dtype=encoded_dtype).astype(dtype, copy=False).reshape(shape)
    if not np.all(np.isfinite(value)):
        raise LinearInverseContractError(f"{name} contains non-finite values")
    return np.array(value, dtype=dtype, copy=True, order="C")


def _positive_scalar(payload: Mapping[str, Any], key: str, side: str) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise LinearInverseContractError(
            f"artifact side={side} has invalid {key!r}"
        ) from exc
    if not np.isfinite(value) or value <= 0:
        raise LinearInverseContractError(
            f"artifact side={side} requires finite positive {key}, got {value!r}"
        )
    return value


def _nonnegative_scalar(payload: Mapping[str, Any], key: str, side: str) -> float:
    """Read a finite non-negative safety tolerance from a side payload."""

    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise LinearInverseContractError(
            f"artifact side={side} has invalid {key!r}"
        ) from exc
    if not np.isfinite(value) or value < 0:
        raise LinearInverseContractError(
            f"artifact side={side} requires finite non-negative {key}, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class _SideModel:
    reverse_center: np.ndarray
    reverse_scale: np.ndarray
    reverse_weights: np.ndarray
    forward_center: np.ndarray
    forward_scale: np.ndarray
    forward_weights: np.ndarray
    max_ood_rms: float
    max_ood_abs: float
    max_cycle_rms: float
    max_limit_excess_rad: float
    max_geometry_rms_m: float
    max_step_delta_rad: np.ndarray


@dataclass(frozen=True)
class LinearInverseResult:
    q_stage_major: np.ndarray
    mano_euler45: np.ndarray
    ood_rms: float
    ood_abs: float
    cycle_rms: float
    step_delta_ratio: float
    limit_excess_rad: float
    clipped_to_urdf: bool


class Mano45LinearInverse:
    """Validated side-specific affine inverse plus calibrated safety gates."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        artifact_path: Path,
        artifact_sha256: str,
    ) -> None:
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise LinearInverseContractError(
                f"unsupported linear inverse schema_version={payload.get('schema_version')!r}"
            )
        exact_contract = {
            "artifact_kind": ARTIFACT_KIND,
            "input_representation": INPUT_REPRESENTATION,
            "model_input_encoding": MODEL_INPUT_ENCODING,
            "output_representation": OUTPUT_REPRESENTATION,
            "units": "radians",
        }
        for key, expected in exact_contract.items():
            actual = payload.get(key)
            if actual != expected:
                raise LinearInverseContractError(
                    f"artifact {key}={actual!r}, expected {expected!r}"
                )
        if tuple(payload.get("mano_joint_order", ())) != MANO_JOINT_ORDER:
            raise LinearInverseContractError("artifact MANO joint order is incompatible")
        if tuple(payload.get("mano_components", ())) != ("x", "y", "z"):
            raise LinearInverseContractError("artifact MANO component order must be xyz")
        if tuple(payload.get("wuji_stage_major_order", ())) != WUJI_STAGE_MAJOR_ORDER:
            raise LinearInverseContractError("artifact Wuji q20 order is incompatible")

        urdf_contract = payload.get("deployment_urdf_contract")
        if not isinstance(urdf_contract, Mapping):
            raise LinearInverseContractError(
                "artifact is missing the deployment Wuji URDF contract"
            )
        try:
            urdf_lower = np.asarray(
                urdf_contract["lower_stage_major_rad"], dtype=np.float64
            )
            urdf_upper = np.asarray(
                urdf_contract["upper_stage_major_rad"], dtype=np.float64
            )
            urdf_sha256 = {
                side: str(urdf_contract["sha256"][side]).lower() for side in SIDES
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise LinearInverseContractError(
                "artifact has an invalid deployment Wuji URDF contract"
            ) from exc
        if (
            urdf_lower.shape != (20,)
            or urdf_upper.shape != (20,)
            or not np.all(np.isfinite(urdf_lower))
            or not np.all(np.isfinite(urdf_upper))
            or np.any(urdf_lower >= urdf_upper)
        ):
            raise LinearInverseContractError(
                "artifact deployment Wuji URDF bounds are invalid"
            )
        if any(
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for digest in urdf_sha256.values()
        ):
            raise LinearInverseContractError(
                "artifact deployment Wuji URDF SHA-256 values are invalid"
            )

        side_payloads = payload.get("sides")
        if not isinstance(side_payloads, Mapping) or set(side_payloads) != set(SIDES):
            raise LinearInverseContractError("artifact must contain exactly left/right side models")
        models: dict[str, _SideModel] = {}
        for side in SIDES:
            item = side_payloads[side]
            if not isinstance(item, Mapping):
                raise LinearInverseContractError(f"artifact side={side} must be an object")
            reverse_center = _decode_array(
                item.get("reverse_center"),
                name=f"{side}.reverse_center",
                shape=(45,),
                dtype=np.dtype("float64"),
            )
            reverse_scale = _decode_array(
                item.get("reverse_scale"),
                name=f"{side}.reverse_scale",
                shape=(45,),
                dtype=np.dtype("float64"),
            )
            reverse_weights = _decode_array(
                item.get("reverse_weights"),
                name=f"{side}.reverse_weights",
                shape=(46, 20),
                dtype=np.dtype("float32"),
            ).astype(np.float64)
            forward_center = _decode_array(
                item.get("forward_center"),
                name=f"{side}.forward_center",
                shape=(20,),
                dtype=np.dtype("float64"),
            )
            forward_scale = _decode_array(
                item.get("forward_scale"),
                name=f"{side}.forward_scale",
                shape=(20,),
                dtype=np.dtype("float64"),
            )
            forward_weights = _decode_array(
                item.get("forward_weights"),
                name=f"{side}.forward_weights",
                shape=(21, 45),
                dtype=np.dtype("float32"),
            ).astype(np.float64)
            step_limit = _decode_array(
                item.get("max_step_delta_rad"),
                name=f"{side}.max_step_delta_rad",
                shape=(20,),
                dtype=np.dtype("float64"),
            )
            if np.any(reverse_scale <= 0) or np.any(forward_scale <= 0):
                raise LinearInverseContractError(
                    f"artifact side={side} normalization scales must be positive"
                )
            if np.any(step_limit <= 0):
                raise LinearInverseContractError(
                    f"artifact side={side} velocity limits must be positive"
                )
            models[side] = _SideModel(
                reverse_center=reverse_center,
                reverse_scale=reverse_scale,
                reverse_weights=reverse_weights,
                forward_center=forward_center,
                forward_scale=forward_scale,
                forward_weights=forward_weights,
                max_ood_rms=_positive_scalar(item, "max_ood_rms", side),
                max_ood_abs=_positive_scalar(item, "max_ood_abs", side),
                max_cycle_rms=_positive_scalar(item, "max_cycle_rms", side),
                max_limit_excess_rad=_nonnegative_scalar(
                    item, "max_limit_excess_rad", side
                ),
                max_geometry_rms_m=_positive_scalar(
                    item, "max_geometry_rms_m", side
                ),
                max_step_delta_rad=step_limit,
            )

        self.models = models
        self.artifact_path = artifact_path
        self.artifact_sha256 = artifact_sha256
        self.training_provenance = payload.get("training_provenance", {})
        self.heldout_metrics = payload.get("heldout_metrics", {})
        # Historical artifacts clipped only after proving the prediction was
        # within their calibrated excess tolerance.  A run-specific artifact
        # may explicitly disable that behavior; it then rejects any limit
        # excess and returns the affine prediction unchanged.
        clip_to_urdf = payload.get("clip_to_urdf", True)
        if not isinstance(clip_to_urdf, bool):
            raise LinearInverseContractError("artifact clip_to_urdf must be boolean")
        self.clip_to_urdf = clip_to_urdf
        self.urdf_lower = urdf_lower
        self.urdf_upper = urdf_upper
        self.urdf_sha256 = urdf_sha256

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> "Mano45LinearInverse":
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.is_file():
            raise LinearInverseContractError(
                f"MANO45 linear inverse artifact is missing: {artifact_path}"
            )
        expected = str(expected_sha256).lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise LinearInverseContractError(
                "MANO45 linear inverse expected_sha256 must be a 64-character hex digest"
            )
        actual = sha256_file(artifact_path)
        if actual != expected:
            raise LinearInverseContractError(
                "MANO45 linear inverse artifact hash mismatch: "
                f"path={artifact_path} actual={actual} expected={expected}"
            )
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LinearInverseContractError(
                f"could not read MANO45 linear inverse artifact {artifact_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise LinearInverseContractError("MANO45 linear inverse artifact must be a JSON object")
        return cls(payload, artifact_path=artifact_path, artifact_sha256=actual)

    @staticmethod
    def _side(side: str) -> str:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        return side

    @staticmethod
    def _vector(value: Any, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite {size}-D, got {array.shape}")
        return array

    @staticmethod
    def _wrap(values: np.ndarray) -> np.ndarray:
        return (values + np.pi) % (2.0 * np.pi) - np.pi

    def predict(
        self,
        *,
        side: str,
        hand_pose_rotvec: Any,
        previous_q_stage_major: Any,
        lower: Any,
        upper: Any,
    ) -> LinearInverseResult:
        side = self._side(side)
        model = self.models[side]
        rotvec = self._vector(hand_pose_rotvec, 45, f"{side} MANO hand pose")
        previous_q = self._vector(
            previous_q_stage_major, 20, f"{side} previous Wuji q"
        )
        lower_q = self._vector(lower, 20, f"{side} Wuji lower limits")
        upper_q = self._vector(upper, 20, f"{side} Wuji upper limits")
        if np.any(lower_q >= upper_q):
            raise LinearInverseContractError(f"{side} Wuji URDF limits are invalid")

        euler = Rotation.from_rotvec(rotvec.reshape(15, 3)).as_euler(
            "xyz", degrees=False
        ).reshape(45)
        z = (euler - model.reverse_center) / model.reverse_scale
        ood_rms = float(np.sqrt(np.mean(z * z)))
        ood_abs = float(np.max(np.abs(z)))
        if ood_rms > model.max_ood_rms or ood_abs > model.max_ood_abs:
            raise LinearInverseSafetyError(
                f"{side} MANO pose is outside the calibrated inverse domain "
                f"(ood_rms={ood_rms:.4f}/{model.max_ood_rms:.4f}, "
                f"ood_abs={ood_abs:.4f}/{model.max_ood_abs:.4f})"
            )

        q_raw = np.concatenate(([1.0], z)) @ model.reverse_weights
        limit_excess = np.maximum(lower_q - q_raw, 0.0) + np.maximum(
            q_raw - upper_q, 0.0
        )
        max_limit_excess = float(np.max(limit_excess))
        if max_limit_excess > model.max_limit_excess_rad:
            raise LinearInverseSafetyError(
                f"{side} linear inverse materially exceeds Wuji URDF limits "
                f"({max_limit_excess:.6f} rad > {model.max_limit_excess_rad:.6f} rad)"
            )
        if self.clip_to_urdf:
            q = np.clip(q_raw, lower_q, upper_q)
            clipped = bool(np.any(q != q_raw))
        else:
            # The 0902 calibration contract forbids a hidden projection from
            # MANO to a different Wuji command.  A prediction outside the
            # deployed URDF is rejected so the caller can use its explicit
            # geometric fallback/hold path.
            if max_limit_excess > 0.0:
                raise LinearInverseSafetyError(
                    f"{side} linear inverse exceeds Wuji URDF limits without clipping "
                    f"({max_limit_excess:.6f} rad)"
                )
            q = q_raw
            clipped = False

        step_delta = np.abs(q - previous_q)
        step_ratio = float(np.max(step_delta / model.max_step_delta_rad))
        if step_ratio > 1.0:
            worst = int(np.argmax(step_delta / model.max_step_delta_rad))
            raise LinearInverseSafetyError(
                f"{side} linear inverse violates calibrated per-step velocity "
                f"at q[{worst}] ({step_delta[worst]:.6f} rad > "
                f"{model.max_step_delta_rad[worst]:.6f} rad)"
            )

        q_z = (q - model.forward_center) / model.forward_scale
        cycle_euler = np.concatenate(([1.0], q_z)) @ model.forward_weights
        cycle_error = self._wrap(cycle_euler - euler) / model.reverse_scale
        cycle_rms = float(np.sqrt(np.mean(cycle_error * cycle_error)))
        if cycle_rms > model.max_cycle_rms:
            raise LinearInverseSafetyError(
                f"{side} MANO/Wuji learned cycle is inconsistent "
                f"({cycle_rms:.4f} > {model.max_cycle_rms:.4f})"
            )

        return LinearInverseResult(
            q_stage_major=q.astype(np.float64, copy=False),
            mano_euler45=euler,
            ood_rms=ood_rms,
            ood_abs=ood_abs,
            cycle_rms=cycle_rms,
            step_delta_ratio=step_ratio,
            limit_excess_rad=max_limit_excess,
            clipped_to_urdf=clipped,
        )

    def geometry_limit_m(self, side: str) -> float:
        return self.models[self._side(side)].max_geometry_rms_m

    def validate_urdf_contract(
        self,
        *,
        side: str,
        lower: Any,
        upper: Any,
        urdf_sha256: str,
    ) -> None:
        """Require the deployed hand bounds and bytes used by calibration."""

        side = self._side(side)
        deployed_lower = self._vector(lower, 20, f"{side} Wuji lower limits")
        deployed_upper = self._vector(upper, 20, f"{side} Wuji upper limits")
        if not np.allclose(
            deployed_lower, self.urdf_lower, rtol=0.0, atol=1e-8
        ) or not np.allclose(
            deployed_upper, self.urdf_upper, rtol=0.0, atol=1e-8
        ):
            raise LinearInverseContractError(
                f"{side} deployed Wuji URDF limits differ from the linear artifact"
            )
        actual_sha = str(urdf_sha256).lower()
        if actual_sha != self.urdf_sha256[side]:
            raise LinearInverseContractError(
                f"{side} deployed Wuji URDF hash differs from the linear artifact: "
                f"actual={actual_sha} expected={self.urdf_sha256[side]}"
            )

    def provenance_summary(self) -> str:
        split = self.training_provenance.get("episode_split", {})
        return (
            f"artifact={self.artifact_path} sha256={self.artifact_sha256} "
            f"representation={INPUT_REPRESENTATION}->{OUTPUT_REPRESENTATION} "
            f"train={split.get('train', '?')} calibration={split.get('calibration', '?')} "
            f"test={split.get('test', '?')}"
        )
