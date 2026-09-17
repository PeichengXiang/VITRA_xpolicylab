"""Hash-bound 0902 Wuji20 -> Inspire6 calibration.

This module is deliberately small and policy-local.  It is not a replacement
for the official Spark geometric solver: the H1 adapter tries that solver
first, then uses this explicitly configured calibration only when the solver
reports a finite fit outside its 80 mm quality gate.  The calibration is an
affine map trained on the same 0902 manifest and emits six actuated Inspire
angles in the documented Spark order.  URDF packing and the H1 order
permutation remain explicit in :mod:`h1_inspire_retarget`.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SIDES = ("left", "right")
Q20_DIM = 20
Q6_DIM = 6
Q20_ORDER = (
    "index_mcp_flex", "middle_mcp_flex", "pinky_mcp_flex", "ring_mcp_flex", "thumb_cmc_flex",
    "index_mcp_abd", "middle_mcp_abd", "pinky_mcp_abd", "ring_mcp_abd", "thumb_cmc_abd",
    "index_pip", "middle_pip", "pinky_pip", "ring_pip", "thumb_mcp",
    "index_dip", "middle_dip", "pinky_dip", "ring_dip", "thumb_ip",
)
Q6_ORDER = (
    "thumb_proximal_yaw_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
    "thumb_proximal_pitch_joint",
)
ARTIFACT_KIND = "xpolicylab_vitra_wuji20_to_inspire6_linear_calibration"
SCHEMA_VERSION = 1


class H1LinearInverseContractError(ValueError):
    """The sidecar is missing, corrupt, or bound to another run."""


class H1LinearInverseSafetyError(ValueError):
    """A finite calibration prediction is outside its explicit H1 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise H1LinearInverseContractError(f"{name} must be an encoded array")
    if tuple(value.get("shape", ())) != shape:
        raise H1LinearInverseContractError(
            f"{name} shape={value.get('shape')!r}, expected={list(shape)!r}"
        )
    try:
        dtype = np.dtype(str(value["dtype"]))
        raw = base64.b64decode(str(value["data_b64"]), validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise H1LinearInverseContractError(f"{name} has invalid encoded bytes") from exc
    if dtype.kind != "f" or dtype.itemsize != 8:
        raise H1LinearInverseContractError(f"{name} must be float64, got {dtype}")
    if len(raw) != int(np.prod(shape)) * 8:
        raise H1LinearInverseContractError(f"{name} byte length is inconsistent")
    result = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise H1LinearInverseContractError(f"{name} contains non-finite values")
    return result


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise H1LinearInverseContractError(f"{name} must be finite {size}-D")
    return result


@dataclass(frozen=True)
class H1LinearPrediction:
    q6: np.ndarray
    input_z_rms: float
    input_z_max: float
    limit_excess_rad: float


@dataclass(frozen=True)
class _SideModel:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    max_input_z_rms: float
    max_input_z_abs: float


class Wuji20ToInspire6Linear:
    """Validated, side-specific affine map with no runtime projection."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        artifact_path: Path,
        artifact_sha256: str,
        expected_provenance: Mapping[str, Any] | None = None,
        source_hashes: Mapping[str, str] | None = None,
        urdf_hashes: Mapping[str, str] | None = None,
    ) -> None:
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise H1LinearInverseContractError("unsupported H1 calibration schema")
        expected = {
            "artifact_kind": ARTIFACT_KIND,
            "input_representation": "wujihand2_q20_stage_major",
            "output_representation": "inspire6_actuated_spark_order",
            "units": "radians",
        }
        for key, wanted in expected.items():
            if payload.get(key) != wanted:
                raise H1LinearInverseContractError(
                    f"artifact {key}={payload.get(key)!r}, expected {wanted!r}"
                )
        if tuple(payload.get("q20_order", ())) != Q20_ORDER:
            raise H1LinearInverseContractError("H1 calibration q20 order is incompatible")
        if tuple(payload.get("q6_order", ())) != Q6_ORDER:
            raise H1LinearInverseContractError("H1 calibration q6 order is incompatible")
        if payload.get("clip_to_limits") is not False or payload.get("padding") is not False or payload.get("truncation") is not False:
            raise H1LinearInverseContractError("H1 calibration permits a forbidden clip/pad/truncate")
        if "explicit" not in str(payload.get("reorder", "")).lower():
            raise H1LinearInverseContractError("H1 calibration must record an explicit order conversion")

        provenance = payload.get("training_provenance")
        if not isinstance(provenance, Mapping):
            raise H1LinearInverseContractError("H1 calibration is missing training provenance")
        if expected_provenance is not None:
            for key, wanted in expected_provenance.items():
                actual = provenance.get(key)
                if str(actual) != str(wanted):
                    raise H1LinearInverseContractError(
                        f"H1 calibration provenance {key}={actual!r} does not match {wanted!r}"
                    )
        payload_sources = provenance.get("source_hashes")
        if not isinstance(payload_sources, Mapping):
            raise H1LinearInverseContractError("H1 calibration is missing source hashes")
        if source_hashes is not None:
            for key, wanted in source_hashes.items():
                if str(payload_sources.get(key, "")).lower() != str(wanted).lower():
                    raise H1LinearInverseContractError(
                        f"H1 calibration source hash mismatch for {key}"
                    )

        urdf_contract = payload.get("deployment_urdf_contract")
        if not isinstance(urdf_contract, Mapping):
            raise H1LinearInverseContractError("H1 calibration is missing URDF contract")
        payload_urdf = urdf_contract.get("sha256")
        if not isinstance(payload_urdf, Mapping):
            raise H1LinearInverseContractError("H1 calibration is missing URDF hashes")
        if urdf_hashes is not None:
            for side, wanted in urdf_hashes.items():
                if str(payload_urdf.get(side, "")).lower() != str(wanted).lower():
                    raise H1LinearInverseContractError(
                        f"H1 calibration URDF hash mismatch for {side}"
                    )

        side_values = payload.get("sides")
        if not isinstance(side_values, Mapping) or set(side_values) != set(SIDES):
            raise H1LinearInverseContractError("H1 calibration must contain left/right models")
        models: dict[str, _SideModel] = {}
        for side in SIDES:
            item = side_values[side]
            if not isinstance(item, Mapping):
                raise H1LinearInverseContractError(f"H1 calibration side {side} is malformed")
            center = _decode(item.get("center"), name=f"{side}.center", shape=(Q20_DIM,))
            scale = _decode(item.get("scale"), name=f"{side}.scale", shape=(Q20_DIM,))
            weights = _decode(item.get("weights"), name=f"{side}.weights", shape=(Q20_DIM + 1, Q6_DIM))
            lower = _vector(item.get("h1_lower_rad"), Q6_DIM, f"{side}.h1_lower_rad")
            upper = _vector(item.get("h1_upper_rad"), Q6_DIM, f"{side}.h1_upper_rad")
            if np.any(scale <= 0) or np.any(lower >= upper):
                raise H1LinearInverseContractError(f"H1 calibration side {side} has invalid scales/bounds")
            if item.get("clip_to_limits") is not False:
                raise H1LinearInverseContractError(f"H1 calibration side {side} enables clipping")
            models[side] = _SideModel(
                center=center,
                scale=scale,
                weights=weights,
                lower=lower,
                upper=upper,
                max_input_z_rms=float("inf"),
                max_input_z_abs=float("inf"),
            )
        self.models = models
        self.artifact_path = artifact_path
        self.artifact_sha256 = artifact_sha256
        self.training_provenance = dict(provenance)
        self.heldout_metrics = payload.get("heldout_metrics", {})

        # A calibration sidecar is a deployment contract, not merely a set of
        # weights.  Refuse artifacts whose producer did not record a passing
        # held-out geometry check, or whose recorded worst-case RMS is already
        # outside the unchanged runtime quality gate.  This check is made at
        # load time so no caller can accidentally turn an unverified matrix
        # into a production fallback.
        calibration_contract = payload.get("calibration_contract")
        if not isinstance(calibration_contract, Mapping) or calibration_contract.get("passed") is not True:
            raise H1LinearInverseContractError(
                "H1 calibration is missing a passing held-out calibration contract"
            )
        try:
            requirement_m = float(calibration_contract["geometry_rms_requirement_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise H1LinearInverseContractError(
                "H1 calibration held-out geometry requirement is missing"
            ) from exc
        if not np.isfinite(requirement_m) or requirement_m > 0.08:
            raise H1LinearInverseContractError(
                "H1 calibration geometry requirement must be <= 0.080000 m"
            )
        for side in SIDES:
            metrics = self.heldout_metrics.get(side)
            if not isinstance(metrics, Mapping):
                raise H1LinearInverseContractError(
                    f"H1 calibration held-out metrics missing for {side}"
                )
            geometry = metrics.get("geometry_rms_m")
            try:
                worst_rms = float(geometry["max"])
            except (KeyError, TypeError, ValueError) as exc:
                raise H1LinearInverseContractError(
                    f"H1 calibration held-out RMS missing for {side}"
                ) from exc
            if not np.isfinite(worst_rms) or worst_rms > 0.08:
                raise H1LinearInverseContractError(
                    f"H1 calibration held-out RMS for {side} exceeds 0.080000 m"
                )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_provenance: Mapping[str, Any] | None = None,
        source_hashes: Mapping[str, str] | None = None,
        urdf_hashes: Mapping[str, str] | None = None,
    ) -> "Wuji20ToInspire6Linear":
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.is_file():
            raise H1LinearInverseContractError(f"H1 calibration is missing: {artifact_path}")
        expected = str(expected_sha256).lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise H1LinearInverseContractError("H1 calibration SHA-256 is malformed")
        actual = sha256_file(artifact_path)
        if actual != expected:
            raise H1LinearInverseContractError(
                f"H1 calibration hash mismatch: actual={actual}, expected={expected}"
            )
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise H1LinearInverseContractError(f"cannot parse H1 calibration {artifact_path}") from exc
        if not isinstance(payload, Mapping):
            raise H1LinearInverseContractError("H1 calibration root must be an object")
        return cls(
            payload,
            artifact_path=artifact_path,
            artifact_sha256=actual,
            expected_provenance=expected_provenance,
            source_hashes=source_hashes,
            urdf_hashes=urdf_hashes,
        )

    def predict(
        self,
        side: str,
        q20: Any,
        *,
        enforce_audited_limits: bool = True,
        clip_to_limits: bool = False,
    ) -> H1LinearPrediction:
        if side not in SIDES:
            raise H1LinearInverseContractError(f"unknown side {side!r}")
        q = np.asarray(q20, dtype=np.float64).reshape(-1)
        if q.shape != (Q20_DIM,) or not np.all(np.isfinite(q)):
            raise H1LinearInverseSafetyError(f"{side} Wuji20 input must be finite 20-D")
        model = self.models[side]
        z = (q - model.center) / model.scale
        q6 = np.concatenate(([1.0], z)) @ model.weights
        if not np.all(np.isfinite(q6)):
            raise H1LinearInverseSafetyError(f"{side} H1 calibration returned non-finite q6")
        excess = np.maximum(model.lower - q6, 0.0) + np.maximum(q6 - model.upper, 0.0)
        max_excess = float(np.max(excess))
        if max_excess > 0.0:
            # Keep the legacy strict behavior unless explicitly disabled by the
            # caller.  New callers can set `enforce_audited_limits=False` and
            # optionally `clip_to_limits=True` for debug/debugged workloads.
            if enforce_audited_limits:
                raise H1LinearInverseSafetyError(
                    f"{side} H1 calibration exceeds audited H1 limits by {max_excess:.6f} rad"
                )
            if clip_to_limits:
                q6 = np.minimum(np.maximum(q6, model.lower), model.upper)
                excess = np.maximum(model.lower - q6, 0.0) + np.maximum(q6 - model.upper, 0.0)
                max_excess = float(np.max(excess))
        return H1LinearPrediction(
            q6=q6,
            input_z_rms=float(np.sqrt(np.mean(z * z))),
            input_z_max=float(np.max(np.abs(z))),
            limit_excess_rad=max_excess,
        )

    def provenance_summary(self) -> str:
        return (
            f"path={self.artifact_path} sha256={self.artifact_sha256} "
            f"kind={ARTIFACT_KIND} units=radians no_clip=true"
        )


__all__ = [
    "H1LinearInverseContractError",
    "H1LinearInverseSafetyError",
    "H1LinearPrediction",
    "Q20_ORDER",
    "Q6_ORDER",
    "Wuji20ToInspire6Linear",
]
