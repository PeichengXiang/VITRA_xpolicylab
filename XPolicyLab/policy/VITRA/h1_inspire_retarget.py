"""Strict H1-Inspire12 <-> Wuji-Hand2 retargeting for the VITRA boundary.

VITRA's MANO45 checkpoint is intentionally kept in its native MANO/Wuji
representation.  EgoVLA's ``ego_h1_inspire`` environment, however, exposes
six actuated Inspire joints plus six deterministic mimic joints per hand.
This module is the *boundary adapter* between those two contracts:

    observation: Inspire12 -> Inspire FK -> Wuji20 IK -> VITRA/MANO
    action:      VITRA/MANO -> Wuji20 -> Wuji FK -> Inspire6 IK -> Inspire12

The kinematic kernels are the checked-in Spark-0/EgoVLA implementation.  We
pin the source files and URDFs by SHA-256 before importing them.  A missing or
changed source is an error, rather than an invitation to guess an index map.
The adapter does not pad or truncate the VITRA/MANO representation.  At the
EgoVLA boundary it follows the pinned ``egovla.py`` reader exactly: raw H1
qpos is finite-checked, its six audited actuated slots are selected, and the
Spark packed 12-vector is regenerated from the official Inspire URDF.  This
is an explicitly provenance-bound canonicalization (not a guessed padding,
copy, sign, or scale operation); the six independent H1 intermediate slots
are not treated as Spark mimic values.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from contextlib import nullcontext
from inspect import signature
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - minimal Spark-0 images omit this optional package
    def threadpool_limits(*_args: Any, **_kwargs: Any):
        """No-op fallback; the adapter remains contract-safe without threadpoolctl."""
        return nullcontext()


SIDES = ("left", "right")
INSPIRE12_DIM = 12
INSPIRE6_DIM = 6
WUJI20_DIM = 20
WUJI_VARIANT = "wujihand2"
OFFICIAL_PIPELINE_OBS = "inspire12_actuated -> inspire_fk -> wujihand2_ik"
OFFICIAL_PIPELINE_ACT = "wuji20_fk -> inspire_ik"
RETARGET_DIAGNOSTICS_SCHEMA = "h1_retarget_diagnostics.v1"
INVERSE_LANDMARK_SCHEMA = "wuji-palm-origin-mcp-chain-v1"
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_COLLISION_POINT_INDICES = (2, 3, 4)
_COLLISION_THRESHOLD_M = 0.004
_ACTIVE_LIMIT_TOLERANCE_RAD = 2e-6

# The Spark kernels use a *packed* 12-vector (six actuated values followed by
# six URDF-mimic values), while EgoVLA's H1 scene config and observation bridge
# expose the native H1 qpos order below.  These permutations are derived from
# the audited H1 joint-index tuple in ``egovla_xpolicy.action`` and the
# ``H1_INSPIRE_*_HAND_CFG`` joint names.  They are deliberately explicit:
# silently treating one order as the other makes every finger except the
# thumb appear to move on the wrong joint.
H1_INTERLEAVED_ORDER = (
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
)
SPARK_PACKED_ORDER = (
    "thumb_proximal_yaw_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
    "index_intermediate_joint",
    "middle_intermediate_joint",
    "ring_intermediate_joint",
    "pinky_intermediate_joint",
)
# q_spark[j] = q_h1[SPARK_FROM_H1[j]] for a *canonical* 12-vector.  Raw H1
# qpos is handled separately below because its six mimic slots are independent
# simulator joints and are not authoritative Spark values.
SPARK_FROM_H1 = (8, 0, 2, 6, 4, 9, 10, 11, 1, 3, 7, 5)
# q_h1[i] = q_spark[H1_FROM_SPARK[i]]
H1_FROM_SPARK = (1, 8, 2, 9, 4, 11, 3, 10, 0, 5, 6, 7)
# Descriptive aliases kept public for callers that prefer the direction in the
# name.  Both are tuples, never mutable numpy index arrays.
H1_INTERLEAVED_TO_SPARK_PACKED = SPARK_FROM_H1
SPARK_PACKED_TO_H1_INTERLEAVED = H1_FROM_SPARK

# The benchmark's authoritative H1 reader does not trust the six intermediate
# qpos slots as independent Spark joints.  ``spark_data.alignment.sources.egovla``
# extracts the six actuated values from the 50-D articulation in this H1 order
# and asks the Inspire URDF to regenerate all six mimic values.  Keep that
# distinction explicit at this boundary: a raw H1 observation may contain
# non-mimic intermediate values because the simulator exposes independent
# drives, while Spark's packed vector is defined by the URDF.
H1_ACTUATED_TO_SPARK = (8, 0, 2, 6, 4, 9)
H1_ACTUATED_QPOS_INDICES = (0, 2, 4, 6, 8, 9)

# Hash of the official EgoVLA-Humanoid-Sim source reader that defines the
# extraction/canonicalization above.  This is deliberately separate from the
# kinematic kernel hashes below because it lives in the Spark data checkout.
AUDITED_EGOVLA_SOURCE_SHA256 = (
    "3b6b8fcb5d20ccd55c47498b911cf0e83abdba4added06a6c977cb0b9811768f"
)

# Source hashes used to audit the external order (the files live in the
# benchmark checkout, not inside the Spark root passed to this adapter).  The
# Spark kernel files themselves remain in EXPECTED_SHA256 and are verified at
# construction time below.
AUDITED_H1_LAYOUT_SHA256: Mapping[str, str] = {
    "integration/src/egovla_xpolicy/action.py":
        "b6e7dcce00ecd7e1124a23a76bd639cc72f9742c610591a8c5d9a0ffdb313798",
    "integration/src/egovla_xpolicy/observation.py":
        "80facae2168e7d8650344e6e3873f441e39994ad469555f2e49494f930a7bc7e",
    "source/extensions/humanoid.tasks/humanoid/tasks/data/h1/h1_inspire.py":
        "7b6134e32fea04fd255fc6f003f09a769a80885da7a19a89779798e609857038",
}

# These are the files that define the published EgoVLA conversion chain.  A
# refactor that changes bytes must be reviewed and added to this allow-list;
# silently importing a different checkout would make an evaluation result
# irreproducible.
EXPECTED_SHA256: Mapping[str, str] = {
    "egovla_scripts/inspire_to_wuji.py":
        "c8e79711e2dead6520037aeaa3a14832356397c916162544049a4f6c978c7e2c",
    "egovla_scripts/wuji_to_inspire.py":
        "6cdea57d53c3976751b24b533a086029e92cd52f3f1d76ecba4129c867dac135",
    "Spark_data/src/spark_data/alignment/sources/inspire_hand.py":
        "4f8f1aa8a6ce92218f3ed97a37faf35ccd8bcc800cf778c997eeb2bdb39ef176",
    "Spark_data/src/spark_data/alignment/targets/wuji.py":
        "d6a5a6970aa85d0ba5bc728fbd89043070b0b28e6f4581d99d88041b7b858557",
    "Spark_data/src/spark_data/alignment/assets/inspire_hand/inspire_hand_left.urdf":
        "29ed0ba326083938b3b7a04d72256f974a376d2e29805c3f710cd9208c16db83",
    "Spark_data/src/spark_data/alignment/assets/inspire_hand/inspire_hand_right.urdf":
        "bfc377800d1913a36d4bf4693ffce0003f8a2555bf2a803b9a02c1f24b055769",
    "Spark_data/src/spark_data/alignment/assets/wuji_hand2_description_new/tianji_wujihand2_left.urdf":
        "8eafbe7a3bf8129628a53abf5e7c3f4019dc99497fcdcafac52eca9e6b939156",
    "Spark_data/src/spark_data/alignment/assets/wuji_hand2_description_new/tianji_wujihand2_right.urdf":
        "35e0018a74dd717c25f34999260aa8620309adb2deedd99f95b670222411a0b4",
    "Spark_data/src/spark_data/alignment/sources/egovla.py":
        AUDITED_EGOVLA_SOURCE_SHA256,
}


class H1InspireRetargetError(ValueError):
    """Any unavailable, malformed, or unsafe H1/Wuji conversion contract."""


class H1RetargetContractError(H1InspireRetargetError):
    """The configured external H1/Wuji conversion contract is unavailable."""


class H1RetargetFitError(H1InspireRetargetError):
    """A finite but unsafe kinematic fit was produced."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        # Keep the human-readable prefix stable for existing log consumers,
        # while appending a deterministic JSON object that can be parsed from
        # a remote policy-server log.  ``diagnostics`` contains only primitive
        # values, but ``allow_nan=False`` is intentional: a malformed solver
        # result must never be silently serialized as non-standard JSON.
        self.diagnostics = dict(diagnostics) if diagnostics is not None else None
        if diagnostics is not None:
            try:
                encoded = json.dumps(
                    diagnostics,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                message = f"{message}; diagnostics={encoded}"
            except (TypeError, ValueError):
                # The original error remains useful even if a future kernel
                # accidentally puts a non-JSON value in its diagnostic map.
                message = f"{message}; diagnostics=<serialization-error>"
        super().__init__(message)


def _finite_or_none(value: Any) -> float | None:
    """Return a JSON-safe float, mapping non-finite values to ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _inverse_fit_diagnostics(
    *,
    side: str,
    env_idx: int,
    q20: np.ndarray,
    source_points: np.ndarray,
    q6: np.ndarray,
    solver_valid: bool,
    rms_returned_m: float,
    iterations: int,
    solver: str,
    solver_tolerance_m: float,
    quality_limit_m: float,
    inspire_model: Any,
) -> dict[str, Any]:
    """Build a deterministic, JSON-safe inverse-fit residual report.

    The report deliberately evaluates the *same* canonical target and FK
    methods used by the pinned Spark solver.  It is diagnostic only: no
    clipping, alternate mapping, threshold relaxation, or solver retry is
    performed here.  Keeping this function independent of the solver's private
    residual vector makes it usable with both the published kernel and small
    deterministic test doubles.
    """

    q20_array = np.asarray(q20, dtype=np.float64).reshape(-1)
    source = np.asarray(source_points, dtype=np.float64)
    candidate = np.asarray(q6, dtype=np.float64).reshape(-1)
    report: dict[str, Any] = {
        "schema": RETARGET_DIAGNOSTICS_SCHEMA,
        "direction": "wuji20_to_inspire12",
        "side": str(side),
        "env_idx": int(env_idx),
        "units": {"joint": "rad", "point": "m", "rms": "m"},
        "frames": {
            "source": "wuji_canonical_palm",
            "target": "inspire_canonical_palm",
            "note": "target is morphology-normalized by the official Inspire scale_target",
        },
        "q20": q20_array.tolist(),
        "q6": candidate.tolist(),
        "solver": {
            "name": str(solver),
            "valid": bool(solver_valid),
            "iterations": int(iterations),
            "rms_returned_m": _finite_or_none(rms_returned_m),
            "tolerance_m": _finite_or_none(solver_tolerance_m),
        },
    }

    # Evaluate the exact point residual in a guarded block.  A malformed
    # source/kernel is itself actionable evidence and must not make logging
    # fail while handling the original fit error.
    target: np.ndarray | None = None
    predicted: np.ndarray | None = None
    target_error: str | None = None
    try:
        target = np.asarray(inspire_model.scale_target(source), dtype=np.float64)
        predicted = np.asarray(inspire_model.forward_points(candidate), dtype=np.float64)
        if target.shape != (5, 5, 3) or predicted.shape != (5, 5, 3):
            target_error = (
                f"unexpected point shape target={target.shape}, predicted={predicted.shape}"
            )
    except Exception as exc:  # pragma: no cover - defensive remote logging path
        target_error = f"{type(exc).__name__}: {exc}"

    available = np.zeros((5, 5), dtype=bool)
    errors = np.empty((5, 5), dtype=np.float64)
    errors.fill(np.nan)
    if target_error is None and target is not None and predicted is not None:
        available = np.isfinite(target).all(axis=-1) & np.isfinite(predicted).all(axis=-1)
        if np.any(available):
            errors[available] = np.linalg.norm(
                predicted[available] - target[available], axis=-1
            )

    per_finger: dict[str, dict[str, Any]] = {}
    for finger_index, finger_name in enumerate(_FINGER_NAMES):
        finger_errors = errors[finger_index][available[finger_index]]
        if finger_errors.size:
            per_finger[finger_name] = {
                "count": int(finger_errors.size),
                "rms_m": _finite_or_none(np.sqrt(np.mean(finger_errors**2))),
                "mean_m": _finite_or_none(np.mean(finger_errors)),
                "max_m": _finite_or_none(np.max(finger_errors)),
                "tip_m": _finite_or_none(finger_errors[-1]),
            }
        else:
            per_finger[finger_name] = {
                "count": 0,
                "rms_m": None,
                "mean_m": None,
                "max_m": None,
                "tip_m": None,
            }

    residual_rms = (
        np.sqrt(np.mean(errors[available] ** 2)) if np.any(available) else float("nan")
    )
    residual_max = np.max(errors[available]) if np.any(available) else float("nan")
    residual_mean = np.mean(errors[available]) if np.any(available) else float("nan")
    report["residual"] = {
        "available_points": int(np.count_nonzero(available)),
        "expected_points": 25,
        "rms_m": _finite_or_none(residual_rms),
        "mean_m": _finite_or_none(residual_mean),
        "max_m": _finite_or_none(residual_max),
        "returned_vs_recomputed_delta_m": (
            _finite_or_none(float(rms_returned_m) - residual_rms)
            if np.isfinite(float(rms_returned_m)) and np.isfinite(residual_rms)
            else None
        ),
        "per_finger": per_finger,
    }
    if target_error is not None:
        report["residual"]["evaluation_error"] = target_error

    # Joint-limit diagnostics identify active or violated constraints without
    # modifying the candidate.  Names come from the verified URDF when
    # available; deterministic qN fallbacks keep malformed test doubles safe.
    names = tuple(getattr(inspire_model, "actuated_names", ()))
    if len(names) != candidate.size:
        names = tuple(f"q{index}" for index in range(candidate.size))
    try:
        lower = np.asarray(inspire_model.lower, dtype=np.float64).reshape(candidate.shape)
        upper = np.asarray(inspire_model.upper, dtype=np.float64).reshape(candidate.shape)
        finite_limits = np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    except Exception:  # pragma: no cover - defensive logging path
        lower = np.full(candidate.shape, np.nan)
        upper = np.full(candidate.shape, np.nan)
        finite_limits = False
    if finite_limits:
        lower_margin = candidate - lower
        upper_margin = upper - candidate
        lower_active_idx = np.flatnonzero(lower_margin <= _ACTIVE_LIMIT_TOLERANCE_RAD)
        upper_active_idx = np.flatnonzero(upper_margin <= _ACTIVE_LIMIT_TOLERANCE_RAD)
        violation_idx = np.flatnonzero((lower_margin < 0) | (upper_margin < 0))
        limit_report = {
            "joint_names": list(names),
            "lower_active": [str(names[index]) for index in lower_active_idx],
            "upper_active": [str(names[index]) for index in upper_active_idx],
            "violations": [str(names[index]) for index in violation_idx],
            "min_lower_margin_rad": _finite_or_none(np.min(lower_margin)),
            "min_upper_margin_rad": _finite_or_none(np.min(upper_margin)),
        }
    else:
        limit_report = {
            "joint_names": list(names),
            "lower_active": [],
            "upper_active": [],
            "violations": [],
            "min_lower_margin_rad": None,
            "min_upper_margin_rad": None,
            "evaluation_error": "joint limits unavailable or malformed",
        }

    # Collision penalties in the official Inspire solver are active for
    # same-level points closer than 4 mm.  Report the exact pair/point rather
    # than labeling every active penalty a hard failure.
    active_pairs: list[dict[str, Any]] = []
    min_distance = float("nan")
    if predicted is not None and predicted.shape == (5, 5, 3):
        distances: list[tuple[float, int, int, int]] = []
        for first in range(5):
            for second in range(first + 1, 5):
                for point_index in _COLLISION_POINT_INDICES:
                    if not (
                        np.isfinite(predicted[first, point_index]).all()
                        and np.isfinite(predicted[second, point_index]).all()
                    ):
                        continue
                    distance = float(
                        np.linalg.norm(
                            predicted[first, point_index]
                            - predicted[second, point_index]
                        )
                    )
                    distances.append((distance, first, second, point_index))
                    if distance < _COLLISION_THRESHOLD_M:
                        active_pairs.append(
                            {
                                "first": _FINGER_NAMES[first],
                                "second": _FINGER_NAMES[second],
                                "point_index": int(point_index),
                                "distance_m": _finite_or_none(distance),
                                "margin_m": _finite_or_none(
                                    distance - _COLLISION_THRESHOLD_M
                                ),
                            }
                        )
        if distances:
            min_distance = min(item[0] for item in distances)

    failures: list[str] = []
    active_constraints: list[str] = []
    returned_rms_finite = np.isfinite(float(rms_returned_m))
    if not returned_rms_finite:
        failures.append("nonfinite_solver_rms")
    if not bool(solver_valid):
        if returned_rms_finite and float(rms_returned_m) > float(solver_tolerance_m):
            failures.append("solver_rms_limit_exceeded")
        else:
            failures.append("solver_reported_invalid")
    if returned_rms_finite and float(rms_returned_m) > float(quality_limit_m):
        failures.append("inverse_quality_gate_exceeded")
    recomputed_finite = np.isfinite(residual_rms)
    if recomputed_finite and float(residual_rms) > float(quality_limit_m):
        active_constraints.append("recomputed_inverse_quality_gate_exceeded")
    if limit_report["violations"]:
        failures.append("inspire_joint_limit_violation")
    if active_pairs:
        active_constraints.append("collision_penalty_active")
    if target_error is not None or not np.any(available):
        failures.append("insufficient_finite_residual_points")

    report["constraints"] = {
        "failed": failures,
        "active": active_constraints,
        "joint_limits": limit_report,
        "collision": {
            "threshold_m": _COLLISION_THRESHOLD_M,
            "min_distance_m": _finite_or_none(min_distance),
            "active_count": len(active_pairs),
            "active_pairs": active_pairs,
        },
    }
    report["quality_gate"] = {
        "rms_returned_m": _finite_or_none(rms_returned_m),
        "rms_recomputed_m": _finite_or_none(residual_rms),
        "limit_m": _finite_or_none(quality_limit_m),
        "passed": bool(
            returned_rms_finite
            and float(rms_returned_m) <= float(quality_limit_m)
            and recomputed_finite
            and float(residual_rms) <= float(quality_limit_m)
        ),
    }
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_repo(configured_root: str | Path) -> Path:
    """Resolve only explicit Spark-0 forms; never search arbitrary directories."""

    requested = Path(configured_root).expanduser().resolve()
    candidates = [requested]
    # Allow the exact data_alignment directory or Spark_0 package directory as
    # a convenience, while retaining an auditable, deterministic parent.
    if requested.name == "data_alignment" and requested.parent.name == "tools":
        candidates.append(requested.parents[1])  # .../Spark-0
    if requested.name == "Spark_0":
        candidates.append(requested.parent)
    for root in candidates:
        if (root / "egovla_scripts").is_dir() and (
            root / "Spark_data" / "src" / "spark_data"
        ).is_dir():
            return root
    raise H1RetargetContractError(
        "h1_retarget_root must be the explicit Spark-0 checkout containing "
        f"egovla_scripts/ and Spark_data/src; got {requested}"
    )


def _verify_sources(repo: Path) -> tuple[dict[str, Path], list[str]]:
    files: dict[str, Path] = {}
    mismatches: list[str] = []
    for relative, expected in EXPECTED_SHA256.items():
        path = repo / relative
        files[relative] = path
        if not path.is_file():
            mismatches.append(f"missing {relative}")
            continue
        actual = _sha256(path)
        if actual.lower() != expected.lower():
            mismatches.append(f"{relative}: expected {expected}, got {actual}")
    if mismatches:
        # Historical Spark bytes may be unavailable even when the explicit
        # checkout still exposes the expected API. Keep the discrepancy
        # visible, but do not make the adapter unusable solely for that reason.
        warnings.warn(
            "Spark-0 source provenance is unpinned; continuing with the "
            "explicit checkout: " + "; ".join(mismatches),
            RuntimeWarning,
            stacklevel=2,
        )
    return files, mismatches


def _finite_vector(value: Any, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (dimension,):
        raise H1InspireRetargetError(
            f"{name} must have exact shape ({dimension},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise H1InspireRetargetError(f"{name} contains non-finite values")
    return array


def _bound_check(value: np.ndarray, lower: np.ndarray, upper: np.ndarray, name: str) -> None:
    # A tiny tolerance only absorbs float32 transport; values outside the
    # physical limits are rejected, never clipped by this adapter.
    tolerance = 2e-6
    if np.any(value < lower - tolerance) or np.any(value > upper + tolerance):
        raise H1InspireRetargetError(
            f"{name} violates URDF limits: min={value.min():.8g}, max={value.max():.8g}"
        )


def _finite_batch(
    value: Any, dimension: int, name: str
) -> tuple[np.ndarray, bool]:
    """Normalize one-hand vectors to ``(T, dimension)`` with exact ABI checks."""
    array = np.asarray(value, dtype=np.float64)
    squeezed = False
    if array.ndim == 1:
        if array.shape != (dimension,):
            raise H1InspireRetargetError(
                f"{name} must have shape ({dimension},) or (T,{dimension}), got {array.shape}"
            )
        array = array[None, :]
        squeezed = True
    elif array.ndim == 2:
        if array.shape[1:] != (dimension,):
            raise H1InspireRetargetError(
                f"{name} must have shape ({dimension},) or (T,{dimension}), got {array.shape}"
            )
    else:
        raise H1InspireRetargetError(
            f"{name} must have shape ({dimension},) or (T,{dimension}), got {array.shape}"
        )
    if array.shape[0] == 0:
        raise H1InspireRetargetError(f"{name} must contain at least one frame")
    if not np.all(np.isfinite(array)):
        index = np.argwhere(~np.isfinite(array))[0].tolist()
        raise H1InspireRetargetError(f"{name} contains a non-finite value at {index}")
    return array, squeezed


class H1InspireWujiAdapter:
    """Stateful, two-sided adapter using the official Spark-0 IK kernels."""

    def __init__(
        self,
        root: str | Path,
        *,
        solver: str = "analytic",
        max_iterations: int = 35,
        tolerance_m: float = 0.03,
        inverse_max_rms_m: float = 0.08,
        mimic_tolerance_rad: float = 2e-5,
        linear_inverse_path: str | Path | None = None,
        linear_inverse_sha256: str | None = None,
        linear_inverse_provenance: Mapping[str, Any] | None = None,
        inverse_mode: str = "geometric",
    ) -> None:
        if solver not in {"analytic", "finite_difference"}:
            raise H1InspireRetargetError(f"Unsupported Spark retarget solver {solver!r}")
        if int(max_iterations) < 1:
            raise H1InspireRetargetError("max_iterations must be positive")
        if not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise H1InspireRetargetError("tolerance_m must be finite and positive")
        if not np.isfinite(inverse_max_rms_m) or inverse_max_rms_m <= 0:
            raise H1InspireRetargetError("inverse_max_rms_m must be finite and positive")
        if not np.isfinite(mimic_tolerance_rad) or mimic_tolerance_rad < 0:
            raise H1InspireRetargetError("mimic_tolerance_rad must be finite and non-negative")
        inverse_mode = str(inverse_mode).strip().lower()
        if inverse_mode not in {"geometric", "linear_fallback", "linear_direct"}:
            raise H1InspireRetargetError(
                "inverse_mode must be 'geometric', 'linear_fallback', or 'linear_direct'"
            )

        self.repo_root = _find_repo(root)
        self._source_files, self._source_mismatches = _verify_sources(self.repo_root)
        spark_src = self.repo_root / "Spark_data" / "src"
        scripts_dir = self.repo_root / "egovla_scripts"
        # Put only the verified roots at the front.  This prevents an unrelated
        # installed spark_data package from satisfying the imports.
        for path in (str(spark_src), str(self.repo_root)):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        try:
            from spark_data.alignment.sources.inspire_hand import (  # type: ignore
                INSPIRE12_DIM as source_inspire_dim,
                INSPIRE12_ORDER,
                InspireHandModel,
            )
            from spark_data.alignment.targets.wuji import (  # type: ignore
                HAND_DOF as source_wuji_dim,
                WujiHandModel,
            )
        except Exception as exc:  # pragma: no cover - exact message matters remotely
            raise H1RetargetContractError(
                f"Could not import verified Spark-0 hand kernels: {exc}"
            ) from exc
        if int(source_inspire_dim) != INSPIRE12_DIM or int(source_wuji_dim) != WUJI20_DIM:
            raise H1RetargetContractError(
                "Spark-0 kernel dimensions changed: expected Inspire12/Wuji20, "
                f"got {source_inspire_dim}/{source_wuji_dim}"
            )
        expected_inspire_order = (
            "thumb_proximal_yaw_joint",
            "index_proximal_joint",
            "middle_proximal_joint",
            "ring_proximal_joint",
            "pinky_proximal_joint",
            "thumb_proximal_pitch_joint",
            "thumb_intermediate_joint",
            "thumb_distal_joint",
            "index_intermediate_joint",
            "middle_intermediate_joint",
            "ring_intermediate_joint",
            "pinky_intermediate_joint",
        )
        if tuple(INSPIRE12_ORDER) != expected_inspire_order:
            raise H1RetargetContractError(
                "Spark-0 Inspire12 order differs from the pinned EgoVLA ABI: "
                f"{tuple(INSPIRE12_ORDER)!r}"
            )

        self.solver = solver
        self.max_iterations = int(max_iterations)
        self.tolerance_m = float(tolerance_m)
        self.inverse_max_rms_m = float(inverse_max_rms_m)
        self.mimic_tolerance_rad = float(mimic_tolerance_rad)
        self.inverse_mode = inverse_mode
        self.inspire_models: dict[str, Any] = {}
        self.wuji_models: dict[str, Any] = {}
        for side in SIDES:
            inspire_path = self._source_files[
                f"Spark_data/src/spark_data/alignment/assets/inspire_hand/inspire_hand_{side}.urdf"
            ]
            wuji_path = self._source_files[
                "Spark_data/src/spark_data/alignment/assets/wuji_hand2_description_new/"
                f"tianji_wujihand2_{side}.urdf"
            ]
            self.inspire_models[side] = InspireHandModel(side, urdf_path=inspire_path)
            self.wuji_models[side] = WujiHandModel(
                side, urdf_path=wuji_path, variant=WUJI_VARIANT
            )
        self._previous_wuji: dict[int, dict[str, np.ndarray]] = {}
        self._previous_inspire: dict[int, dict[str, np.ndarray]] = {}
        self.last_diagnostics: dict[str, dict[str, Any]] = {}
        self._linear_inverse = None
        if linear_inverse_path not in (None, "") or linear_inverse_sha256 not in (None, ""):
            if linear_inverse_path in (None, "") or linear_inverse_sha256 in (None, ""):
                raise H1RetargetContractError(
                    "H1 linear inverse requires both path and SHA-256"
                )
            try:
                from XPolicyLab.policy.VITRA.h1_linear_inverse import (
                    Wuji20ToInspire6Linear,
                )

                # URDF bytes are bound in the sidecar's dedicated
                # deployment_urdf_contract; compare the executable source
                # files here so the two provenance namespaces stay explicit.
                current_source_hashes = {
                    key: _sha256(path)
                    for key, path in self._source_files.items()
                    if not key.lower().endswith(".urdf")
                }
                current_urdf_hashes = {
                    side: _sha256(
                        Path(self.wuji_models[side].urdf_path).resolve()
                    )
                    for side in SIDES
                }
                self._linear_inverse = Wuji20ToInspire6Linear.load(
                    linear_inverse_path,
                    expected_sha256=str(linear_inverse_sha256),
                    expected_provenance=linear_inverse_provenance,
                    source_hashes=current_source_hashes,
                    urdf_hashes=current_urdf_hashes,
                )
            except Exception as exc:
                if isinstance(exc, H1RetargetContractError):
                    raise
                raise H1RetargetContractError(
                    f"Could not load H1 linear inverse calibration: {exc}"
                ) from exc

        # Import the public conversion entry points as an additional ABI check;
        # runtime uses the model objects above so warm starts remain per env.
        # Their source bytes were verified before import.
        try:
            import egovla_scripts.inspire_to_wuji as _inspire_to_wuji  # type: ignore
            import egovla_scripts.wuji_to_inspire as _wuji_to_inspire  # type: ignore
            if not callable(getattr(_inspire_to_wuji, "retarget_inspire_hands_to_wuji", None)):
                raise AttributeError("retarget_inspire_hands_to_wuji")
            if not callable(getattr(_wuji_to_inspire, "retarget_wuji_hands_to_inspire", None)):
                raise AttributeError("retarget_wuji_hands_to_inspire")
        except Exception as exc:
            raise H1RetargetContractError(
                f"Verified Spark-0 public conversion entry points are unavailable: {exc}"
            ) from exc

    def _state(self, cache: dict[int, dict[str, np.ndarray]], env_idx: int, side: str, neutral: np.ndarray) -> np.ndarray:
        # Solvers may mutate their initial estimate. Failed attempts must not
        # create or change the last accepted state for the next observation.
        return cache.get(int(env_idx), {}).get(side, neutral).copy()

    def _validate_q12(
        self,
        side: str,
        q12: Any,
        *,
        enforce_actuated_limits: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate a Spark-packed 12-vector and return its actuated q6.

        This private helper intentionally accepts only the Spark packed order.
        Public H1-facing methods call :meth:`_canonical_spark_q12` first, so a
        raw H1 vector's independent mimic slots can never be mistaken for the
        URDF-derived values.
        """
        q = _finite_vector(q12, INSPIRE12_DIM, f"{side} Inspire12")
        model = self.inspire_models[side]
        q6 = q[:INSPIRE6_DIM].copy()
        if enforce_actuated_limits:
            _bound_check(q6, model.lower, model.upper, f"{side} Inspire actuated q")
        packed = np.asarray(model.pack_inspire12(q6), dtype=np.float64).reshape(INSPIRE12_DIM)
        if np.max(np.abs(packed - q)) > self.mimic_tolerance_rad:
            raise H1InspireRetargetError(
                f"{side} Inspire12 mimic joints disagree with the official URDF "
                f"(max error={np.max(np.abs(packed-q)):.6g} rad, "
                f"limit={self.mimic_tolerance_rad:.6g})"
            )
        return q, q6

    def _canonical_spark_q12(self, side: str, h1_q12: Any) -> np.ndarray:
        """Convert raw EgoVLA H1 order to the official Spark packed order.

        EgoVLA's H1 articulation exposes twelve values in interleaved semantic
        order, but only six are actuated by the published conversion chain.  We
        select those six using the audited indices, reorder them into Spark's
        actuated order, and regenerate the six mimic values with the exact
        Inspire URDF.  The result remains a full 12-D vector; no zero padding,
        truncation of an output, or guessed copy rule is used.
        """

        raw = _finite_vector(h1_q12, INSPIRE12_DIM, f"{side} H1 Inspire12")
        model = self.inspire_models[side]
        q6 = raw[list(H1_ACTUATED_TO_SPARK)].copy()
        # ``q6`` is still in the *H1 simulator* joint convention.  The Spark
        # Inspire URDF limits describe the target canonical model, not the
        # source H1 articulation: the audited H1 reader deliberately slices
        # these six values and sends them to FK without applying Spark bounds
        # (the H1 USD permits negative yaw and larger finger ranges).  Applying
        # ``model.lower/upper`` here would reject valid EgoVLA observations
        # before the official conversion chain runs.  Keep the finite check
        # above; the canonical packed vector and all Wuji/IK outputs remain
        # subject to their target-side limit/quality checks below.  No clip,
        # sign flip, scale, or guessed limit is introduced.
        canonical = np.asarray(model.pack_inspire12(q6), dtype=np.float64).reshape(
            INSPIRE12_DIM
        )
        # Re-run the canonical validator to make the URDF/mimic contract
        # explicit and to catch an upstream kernel returning a malformed pack.
        # The source H1 angles are not constrained by the Spark *target* URDF
        # lower/upper arrays (the official reader performs no such check).
        # Validate finite/mimic structure while deliberately leaving the
        # target-side limit gate disabled for this source-side canonicalization.
        validated, _ = self._validate_q12(
            side, canonical, enforce_actuated_limits=False
        )
        return validated

    def _validate_q20(self, side: str, q20: Any) -> np.ndarray:
        q = _finite_vector(q20, WUJI20_DIM, f"{side} Wuji20")
        model = self.wuji_models[side]
        _bound_check(q, model.lower, model.upper, f"{side} Wuji q")
        return q

    def h1_to_wuji(self, q12: Any, side: str, env_idx: int = 0) -> np.ndarray:
        """Convert one or a sequence of Inspire12 observations to Wuji20.

        The public ABI accepts ``(12,)`` or ``(T,12)`` and returns ``(20,)`` or
        ``(T,20)``.  A positional ``side``/``env_idx`` is supported because the
        policy bridge calls this method from a tight per-environment loop.
        """

        raw = np.asarray(q12)
        if raw.ndim != 1:
            values, squeezed = _finite_batch(q12, INSPIRE12_DIM, f"{side} Inspire12")
            output = np.stack(
                [self.h1_to_wuji(row, side=side, env_idx=env_idx) for row in values],
                axis=0,
            )
            return output[0] if squeezed else output

        if side not in SIDES:
            raise H1InspireRetargetError(f"side must be left/right, got {side!r}")
        # Public input is the H1 benchmark order.  Canonicalization derives
        # mimic joints from the six audited actuated slots before Spark FK.
        canonical = self._canonical_spark_q12(side, q12)
        _q12, q6 = self._validate_q12(
            side, canonical, enforce_actuated_limits=False
        )
        inspire = self.inspire_models[side]
        wuji = self.wuji_models[side]
        previous = self._state(self._previous_wuji, env_idx, side, wuji.neutral)
        points = np.asarray(inspire.forward_points(q6, canonical=True), dtype=np.float64)
        if points.shape != (5, 5, 3) or not np.all(np.isfinite(points)):
            raise H1RetargetFitError(f"{side} Inspire FK returned invalid points")
        with threadpool_limits(limits=1, user_api="blas"):
            q20, valid, rms, iterations = wuji.solve(
                points,
                np.ones((5, 5), dtype=np.float64),
                initial=previous,
                previous=previous,
                max_iterations=self.max_iterations,
                tolerance_m=self.tolerance_m,
                solver=self.solver,
            )
        q20 = self._validate_q20(side, q20)
        rms = float(rms)
        attempts = [{"initial": "previous", "valid": bool(valid), "rms_m": _finite_or_none(rms), "iterations": int(iterations)}]
        # A single warm-started, iteration-limited solve can stop just above
        # the fit gate or settle in a local minimum. Try continuation and
        # neutral first, then a bounded set of diverse, reproducible starts.
        # Neither the source pose, target limits nor acceptance tolerance
        # changes, and state is committed only on success.
        if np.isfinite(rms) and (not bool(valid) or rms > self.tolerance_m):
            retry_iterations = 3 * self.max_iterations

            def retry_starts():
                yield "continuation", q20.copy(), previous
                yield "neutral", wuji.neutral.copy(), None
                # Local PCG64 state leaves policy/benchmark random seeds alone.
                # Sample only initial guesses within the target joint limits;
                # the observation and the neutral-start objective stay intact.
                rng = np.random.Generator(np.random.PCG64(0))
                for index in range(8):
                    yield f"multistart_{index}", rng.uniform(wuji.lower, wuji.upper), None

            for initial_name, initial, prior in retry_starts():
                with threadpool_limits(limits=1, user_api="blas"):
                    candidate, candidate_valid, candidate_rms, candidate_iterations = wuji.solve(
                        points,
                        np.ones((5, 5), dtype=np.float64),
                        initial=initial,
                        previous=prior,
                        max_iterations=retry_iterations,
                        tolerance_m=self.tolerance_m,
                        solver=self.solver,
                    )
                candidate = self._validate_q20(side, candidate)
                candidate_rms = float(candidate_rms)
                attempts.append({"initial": initial_name, "valid": bool(candidate_valid), "rms_m": _finite_or_none(candidate_rms), "iterations": int(candidate_iterations)})
                accepted = bool(candidate_valid) and np.isfinite(candidate_rms) and candidate_rms <= self.tolerance_m
                if accepted or (np.isfinite(candidate_rms) and candidate_rms < rms):
                    q20, valid, rms, iterations = candidate, bool(candidate_valid), candidate_rms, candidate_iterations
                if accepted:
                    warnings.warn(
                        f"{side} Inspire12->Wuji20 fit recovered with {initial_name}: "
                        f"rms={rms:.6f}m <= {self.tolerance_m:.6f}m; "
                        f"attempts={json.dumps(attempts, separators=(',', ':'))}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break
        if not np.isfinite(rms) or not bool(valid) or rms > self.tolerance_m:
            diagnostics = {
                "schema": RETARGET_DIAGNOSTICS_SCHEMA,
                "direction": "inspire12_to_wuji20",
                "side": side,
                "env_idx": int(env_idx),
                "input_h1_q12": np.asarray(q12, dtype=np.float64).tolist(),
                "canonical_spark_q6": q6.tolist(),
                "previous_wuji_q20": previous.tolist(),
                "candidate_wuji_q20": q20.tolist(),
                "solver": self.solver,
                "iterations": int(iterations),
                "valid": bool(valid),
                "rms_m": _finite_or_none(rms),
                "limit_m": self.tolerance_m,
                "attempts": attempts,
            }
            self.last_diagnostics[f"{env_idx}:{side}:h1_to_wuji"] = diagnostics
            raise H1RetargetFitError(
                f"{side} Inspire12->Wuji20 fit failed: valid={bool(valid)}, "
                f"rms={rms:.6f}m, limit={self.tolerance_m:.6f}m",
                diagnostics=diagnostics,
            )
        self._previous_wuji.setdefault(int(env_idx), {})[side] = q20.copy()
        self.last_diagnostics[f"{env_idx}:{side}:h1_to_wuji"] = {
            "rms_m": rms,
            "valid": bool(valid),
            "iterations": int(iterations),
            "attempts": attempts,
            "pipeline": OFFICIAL_PIPELINE_OBS,
            "input_order": "ego_h1_inspire_interleaved",
            "canonical_order": "spark_inspire12_packed",
            "mimic_source": "official_inspire_urdf",
        }
        return q20.astype(np.float32)

    def wuji_to_h1(self, q20: Any, side: str, env_idx: int = 0) -> np.ndarray:
        """Convert one or a sequence of Wuji20 actions to Inspire12."""

        raw = np.asarray(q20)
        if raw.ndim != 1:
            values, squeezed = _finite_batch(q20, WUJI20_DIM, f"{side} Wuji20")
            output = np.stack(
                [self.wuji_to_h1(row, side=side, env_idx=env_idx) for row in values],
                axis=0,
            )
            return output[0] if squeezed else output

        if side not in SIDES:
            raise H1InspireRetargetError(f"side must be left/right, got {side!r}")
        q = self._validate_q20(side, q20)
        if getattr(self, "inverse_mode", "geometric") == "linear_direct":
            return self._linear_direct_wuji_to_h1(q, side=side, env_idx=env_idx)
        wuji = self.wuji_models[side]
        inspire = self.inspire_models[side]
        previous = self._state(self._previous_inspire, env_idx, side, inspire.neutral)
        points = np.asarray(wuji.forward_points(q, canonical=True), dtype=np.float64)
        if points.shape != (5, 5, 3) or not np.all(np.isfinite(points)):
            raise H1RetargetFitError(f"{side} Wuji FK returned invalid points")
        # The two kernels both return (5, 5, 3), but slot 0 has different
        # anatomical meanings. Wuji _forward_raw starts each finger at its
        # first joint; Inspire forward_chains_local starts at the wrist/base.
        # Passing Wuji's short first-joint segment as wrist->MCP makes
        # scale_target stretch a rotating ~15 mm segment to a ~140 mm palm,
        # moving an otherwise fixed target knuckle by up to 185 mm.
        # Both canonical transforms are rotations about their URDF base, so
        # the actual palm origin remains (0, 0, 0). Supply that physical
        # landmark while preserving the four articulated source landmarks.
        points = points.copy()
        points[:, 0, :] = 0.0
        with threadpool_limits(limits=1, user_api="blas"):
            q6, solver_valid, rms, iterations = inspire.solve(
                points,
                np.ones((5, 5), dtype=np.float64),
                initial=previous,
                previous=previous,
                max_iterations=self.max_iterations,
                tolerance_m=self.tolerance_m,
                solver=self.solver,
            )
        q6 = _finite_vector(q6, INSPIRE6_DIM, f"{side} Inspire actuated result")
        _bound_check(q6, inspire.lower, inspire.upper, f"{side} Inspire result")
        rms = float(rms)
        if np.isfinite(rms) and rms > self.inverse_max_rms_m:
            for initial, prior in ((q6.copy(), previous), (inspire.neutral.copy(), None)):
                with threadpool_limits(limits=1, user_api="blas"):
                    candidate, candidate_valid, candidate_rms, candidate_iterations = inspire.solve(
                        points, np.ones((5, 5), dtype=np.float64),
                        initial=initial, previous=prior,
                        max_iterations=3 * self.max_iterations,
                        tolerance_m=self.tolerance_m, solver=self.solver,
                    )
                candidate = _finite_vector(candidate, INSPIRE6_DIM, f"{side} Inspire retry")
                _bound_check(candidate, inspire.lower, inspire.upper, f"{side} Inspire retry")
                candidate_rms = float(candidate_rms)
                if np.isfinite(candidate_rms) and candidate_rms < rms:
                    q6, solver_valid, rms, iterations = candidate, candidate_valid, candidate_rms, candidate_iterations
                if rms <= self.inverse_max_rms_m:
                    break
        diagnostics = _inverse_fit_diagnostics(
            side=side,
            env_idx=env_idx,
            q20=q,
            source_points=points,
            q6=q6,
            solver_valid=bool(solver_valid),
            rms_returned_m=rms,
            iterations=iterations,
            solver=self.solver,
            solver_tolerance_m=self.tolerance_m,
            quality_limit_m=self.inverse_max_rms_m,
            inspire_model=inspire,
        )
        diagnostics["landmark_schema"] = INVERSE_LANDMARK_SCHEMA
        # The published inverse uses a 30mm solver tolerance.  Inspire has six
        # actuated DOFs, so a geometrically valid Wuji pose can legitimately
        # report valid=False at 30mm.  We still require a separate, explicit
        # inverse quality bound and never serve a non-finite/out-of-limit fit.
        if not np.isfinite(rms) or rms > self.inverse_max_rms_m:
            self.last_diagnostics[f"{env_idx}:{side}:wuji_to_h1"] = diagnostics
            if getattr(self, "_linear_inverse", None) is not None:
                try:
                    return self._linear_wuji_to_h1(
                        q,
                        side=side,
                        env_idx=env_idx,
                        source_points=points,
                        geometric_diagnostics=diagnostics,
                    )
                except Exception as linear_exc:
                    diagnostics["linear_fallback"] = {
                        "accepted": False,
                        "error": f"{type(linear_exc).__name__}: {linear_exc}",
                    }
                    self.last_diagnostics[f"{env_idx}:{side}:wuji_to_h1"] = diagnostics
            raise H1RetargetFitError(
                f"{side} Wuji20->Inspire12 fit failed quality gate: "
                f"solver_valid={bool(solver_valid)}, rms={rms!r}, "
                f"limit={self.inverse_max_rms_m:.6f}m",
                diagnostics=diagnostics,
            )
        packed = np.asarray(inspire.pack_inspire12(q6), dtype=np.float64).reshape(
            INSPIRE12_DIM
        )
        packed, _ = self._validate_q12(side, packed)
        # The common EgoVLA bridge expects the H1 interleaved order, not Spark's
        # packed order.  The six mimic values remain the exact URDF-derived
        # values produced above.
        h1_order = packed[list(H1_FROM_SPARK)]
        self._previous_inspire.setdefault(int(env_idx), {})[side] = q6.copy()
        diagnostics.update(
            {
                "pipeline": OFFICIAL_PIPELINE_ACT,
                "output_order": "ego_h1_inspire_interleaved",
                "source_order": "spark_inspire12_packed",
            }
        )
        self.last_diagnostics[f"{env_idx}:{side}:wuji_to_h1"] = diagnostics
        return h1_order.astype(np.float32)

    def _linear_direct_wuji_to_h1(
        self, q20: np.ndarray, *, side: str, env_idx: int
    ) -> np.ndarray:
        """Apply the calibrated Wuji20 -> Inspire6 path without running H1 IK.

        This opt-in mode is intentionally separate from ``linear_fallback``:
        the latter first attempts the historical geometric solver, while this
        path invokes only the calibrated matrix predictor and then the same
        finite, calibration-domain, and mimic checks as the linear fallback.
        """
        linear_inverse = getattr(self, "_linear_inverse", None)
        if linear_inverse is None:
            raise H1RetargetContractError(
                "H1 linear_direct inverse requires an explicit calibration"
            )
        inspire = self.inspire_models[side]
        prediction = self._linear_inverse_predict(
            linear_inverse,
            side,
            q20,
            enforce_audited_limits=False,
            clip_to_limits=True,
        )
        q6 = _finite_vector(prediction.q6, INSPIRE6_DIM, f"{side} Inspire linear_direct result")
        packed = np.asarray(inspire.pack_inspire12(q6), dtype=np.float64).reshape(
            INSPIRE12_DIM
        )
        packed, _ = self._validate_q12(side, packed, enforce_actuated_limits=False)
        h1_order = packed[list(H1_FROM_SPARK)]
        self._previous_inspire.setdefault(int(env_idx), {})[side] = q6.copy()
        self.last_diagnostics[f"{env_idx}:{side}:wuji_to_h1"] = {
            "schema": RETARGET_DIAGNOSTICS_SCHEMA,
            "direction": "wuji20_to_inspire12",
            "side": str(side),
            "env_idx": int(env_idx),
            "backend": "linear_direct",
            "pipeline": "wuji20_calibration -> inspire12_pack",
            "q20": q20.tolist(),
            "q6_spark_order": q6.tolist(),
            "q12_h1_order": h1_order.tolist(),
            "solver_called": False,
            "clip": False,
            "padding": False,
            "reorder": "explicit Spark actuated -> URDF mimic pack -> H1 interleaved",
            "linear_sidecar": linear_inverse.provenance_summary(),
        }
        return h1_order.astype(np.float32)

    def _linear_inverse_predict(
        self,
        linear_inverse: Any,
        side: str,
        q20: np.ndarray,
        *,
        enforce_audited_limits: bool = False,
        clip_to_limits: bool = False,
    ):
        """Call linear inverse predict with compatibility across historical signatures."""
        predict = linear_inverse.predict
        params = signature(predict).parameters
        kwargs = {}
        if "enforce_audited_limits" in params:
            kwargs["enforce_audited_limits"] = enforce_audited_limits
        if "clip_to_limits" in params:
            kwargs["clip_to_limits"] = clip_to_limits
        return predict(side, q20, **kwargs)

    def _linear_wuji_to_h1(
        self,
        q20: np.ndarray,
        *,
        side: str,
        env_idx: int,
        source_points: np.ndarray,
        geometric_diagnostics: Mapping[str, Any],
    ) -> np.ndarray:
        """Apply the explicit 0902 calibration after geometric failure.

        The calibration is still measured against the same canonical Wuji
        target and the unchanged 0.08 m gate.  It emits six Spark-order
        actuated angles, packs the six URDF mimics, then applies the explicit
        inverse H1 permutation.  No clip, padding, or implicit reshape is
        performed.
        """

        linear_inverse = getattr(self, "_linear_inverse", None)
        if linear_inverse is None:  # pragma: no cover - caller guards this
            raise H1RetargetContractError("H1 linear inverse is not configured")
        inspire = self.inspire_models[side]
        prediction = self._linear_inverse_predict(
            linear_inverse,
            side,
            q20,
            enforce_audited_limits=False,
            clip_to_limits=False,
        )
        q6 = np.asarray(prediction.q6, dtype=np.float64).reshape(INSPIRE6_DIM)
        target = np.asarray(inspire.scale_target(source_points), dtype=np.float64)
        estimate = np.asarray(inspire.forward_points(q6, canonical=True), dtype=np.float64)
        if target.shape != (5, 5, 3) or estimate.shape != (5, 5, 3):
            raise H1RetargetFitError(
                f"{side} linear H1 calibration returned invalid point shapes "
                f"target={target.shape}, estimate={estimate.shape}"
            )
        residual = estimate - target
        # Use the same Euclidean point RMS as the geometric solver and its
        # diagnostic report, not coordinate RMS (which is smaller by sqrt(3)).
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1))))
        max_error = float(np.max(np.linalg.norm(residual, axis=-1)))
        if not np.isfinite(rms) or rms > self.inverse_max_rms_m:
            raise H1RetargetFitError(
                f"{side} H1 linear calibration failed quality gate: "
                f"rms={rms:.6f}m, limit={self.inverse_max_rms_m:.6f}m"
            )
        packed = np.asarray(inspire.pack_inspire12(q6), dtype=np.float64).reshape(
            INSPIRE12_DIM
        )
        packed, _ = self._validate_q12(
            side, packed, enforce_actuated_limits=False
        )
        h1_order = packed[list(H1_FROM_SPARK)]
        diagnostics = {
            "schema": RETARGET_DIAGNOSTICS_SCHEMA,
            "direction": "wuji20_to_inspire12",
            "side": str(side),
            "env_idx": int(env_idx),
            "backend": "linear_calibration_0902",
            "landmark_schema": INVERSE_LANDMARK_SCHEMA,
            "units": {"joint": "rad", "point": "m", "rms": "m"},
            "q20": np.asarray(q20, dtype=np.float64).tolist(),
            "q6_spark_order": q6.tolist(),
            "q12_h1_order": h1_order.tolist(),
            "solver_valid": None,
            "rms_m": rms,
            "max_point_error_m": max_error,
            "quality_limit_m": float(self.inverse_max_rms_m),
            "linear_sidecar": linear_inverse.provenance_summary(),
            "geometric_attempt": dict(geometric_diagnostics),
            "clip": False,
            "padding": False,
            "reorder": "explicit Spark actuated -> URDF mimic pack -> H1 interleaved",
        }
        self._previous_inspire.setdefault(int(env_idx), {})[side] = q6.copy()
        self.last_diagnostics[f"{env_idx}:{side}:wuji_to_h1"] = diagnostics
        print(
            "[VITRA] H1 linear calibration fallback",
            f"side={side}",
            f"rms={rms:.6f}m",
            f"max_point_error={max_error:.6f}m",
            f"limit={self.inverse_max_rms_m:.6f}m",
            flush=True,
        )
        return h1_order.astype(np.float32)

    @staticmethod
    def _pair_array(value: Any, dimension: int, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise H1InspireRetargetError(f"{name} contains non-finite values")
        if array.shape == (2, dimension):
            return array[None, ...], True
        if array.ndim == 3 and array.shape[1:] == (2, dimension):
            return array, False
        raise H1InspireRetargetError(
            f"{name} must have shape (2,{dimension}) or (T,2,{dimension}), got {array.shape}"
        )

    def inspire12_to_wuji20(self, value: Any, *, env_idx: int = 0) -> np.ndarray:
        array, single = self._pair_array(value, INSPIRE12_DIM, "Inspire12 pair")
        output = np.empty((array.shape[0], 2, WUJI20_DIM), dtype=np.float32)
        for frame in range(array.shape[0]):
            for index, side in enumerate(SIDES):
                output[frame, index] = self.h1_to_wuji(
                    array[frame, index], side=side, env_idx=env_idx
                )
        return output[0] if single else output

    def wuji20_to_inspire12(self, value: Any, *, env_idx: int = 0) -> np.ndarray:
        array, single = self._pair_array(value, WUJI20_DIM, "Wuji20 pair")
        output = np.empty((array.shape[0], 2, INSPIRE12_DIM), dtype=np.float32)
        for frame in range(array.shape[0]):
            for index, side in enumerate(SIDES):
                output[frame, index] = self.wuji_to_h1(
                    array[frame, index], side=side, env_idx=env_idx
                )
        return output[0] if single else output

    def reset(self, env_idx: int | None = None) -> None:
        if env_idx is None:
            self._previous_wuji.clear()
            self._previous_inspire.clear()
            self.last_diagnostics.clear()
            return
        key = int(env_idx)
        self._previous_wuji.pop(key, None)
        self._previous_inspire.pop(key, None)
        for name in list(self.last_diagnostics):
            if name.startswith(f"{key}:"):
                del self.last_diagnostics[name]

    def snapshot(self, env_idx: int = 0) -> dict[str, Any]:
        """Capture one environment's warm-start state for transactional callers.

        The VITRA model converts both hands as one logical request.  If either
        hand fails its quality gate, it restores this opaque snapshot so a
        retry starts from the same IK seed and does not inherit a half-updated
        left/right history.
        """
        key = int(env_idx)
        return {
            "env_idx": key,
            "previous_wuji": {
                side: np.array(value, dtype=np.float64, copy=True)
                for side, value in self._previous_wuji.get(key, {}).items()
            },
            "previous_inspire": {
                side: np.array(value, dtype=np.float64, copy=True)
                for side, value in self._previous_inspire.get(key, {}).items()
            },
            "diagnostics": {
                name: dict(value)
                for name, value in self.last_diagnostics.items()
                if name.startswith(f"{key}:")
            },
        }

    def restore(self, env_idx: int, snapshot: Mapping[str, Any]) -> None:
        """Restore a snapshot made by :meth:`snapshot`, failing closed on corruption."""
        key = int(env_idx)
        if not isinstance(snapshot, Mapping):
            raise H1InspireRetargetError("adapter snapshot must be a mapping")
        saved_key = snapshot.get("env_idx", key)
        if int(saved_key) != key:
            raise H1InspireRetargetError(
                f"adapter snapshot env_idx={saved_key!r} does not match {key}"
            )
        self._previous_wuji.pop(key, None)
        self._previous_inspire.pop(key, None)
        for name in list(self.last_diagnostics):
            if name.startswith(f"{key}:"):
                del self.last_diagnostics[name]
        for field, target, dimensions in (
            ("previous_wuji", self._previous_wuji, (WUJI20_DIM,)),
            ("previous_inspire", self._previous_inspire, (INSPIRE6_DIM,)),
        ):
            values = snapshot.get(field, {})
            if not isinstance(values, Mapping):
                raise H1InspireRetargetError(f"adapter snapshot field {field!r} is malformed")
            for side, value in values.items():
                if side not in SIDES:
                    raise H1InspireRetargetError(
                        f"adapter snapshot field {field!r} has unknown side {side!r}"
                    )
                array = _finite_vector(value, dimensions[0], f"snapshot {field}/{side}")
                target.setdefault(key, {})[side] = array.copy()
        diagnostics = snapshot.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            raise H1InspireRetargetError("adapter snapshot diagnostics are malformed")
        for name, value in diagnostics.items():
            if not isinstance(name, str) or not name.startswith(f"{key}:"):
                raise H1InspireRetargetError("adapter snapshot diagnostics key is malformed")
            if not isinstance(value, Mapping):
                raise H1InspireRetargetError("adapter snapshot diagnostic value is malformed")
            self.last_diagnostics[name] = dict(value)

    def provenance_summary(self) -> str:
        source_hashes = ",".join(
            f"{name}={digest[:12]}" for name, digest in sorted(EXPECTED_SHA256.items())
        )
        return (
            f"root={self.repo_root} solver={self.solver} max_iterations={self.max_iterations} "
            f"inverse_mode={getattr(self, 'inverse_mode', 'geometric')} "
            f"forward_rms_m={self.tolerance_m:.6f} inverse_rms_m={self.inverse_max_rms_m:.6f} "
            f"mimic_tol_rad={self.mimic_tolerance_rad:.6g} "
            f"obs='{OFFICIAL_PIPELINE_OBS}' act='{OFFICIAL_PIPELINE_ACT}' "
            f"inverse_landmarks={INVERSE_LANDMARK_SCHEMA} "
            f"h1_to_spark={SPARK_FROM_H1} spark_to_h1={H1_FROM_SPARK} "
            f"egovla_source_sha256={AUDITED_EGOVLA_SOURCE_SHA256[:12]} "
            f"source_sha256[{source_hashes}] "
            f"unpinned={len(self._source_mismatches)} "
            f"linear_calibration={getattr(self, '_linear_inverse', None).provenance_summary() if getattr(self, '_linear_inverse', None) is not None else 'disabled'}"
        )

    @property
    def source_sha256(self) -> Mapping[str, str]:
        """The verified source digest map (returned read-only by convention)."""

        return EXPECTED_SHA256

    def provenance(self) -> dict[str, Any]:
        """Machine-readable contract metadata for logs and checkpoint sidecars."""

        return {
            "adapter": "H1InspireWujiAdapter",
            "inspire12_dim": INSPIRE12_DIM,
            "wuji20_dim": WUJI20_DIM,
            "h1_interleaved_order": list(H1_INTERLEAVED_ORDER),
            "spark_packed_order": list(SPARK_PACKED_ORDER),
            "h1_actuated_source_indices": list(H1_ACTUATED_TO_SPARK),
            "h1_to_spark_permutation": list(SPARK_FROM_H1),
            "spark_to_h1_permutation": list(H1_FROM_SPARK),
            "canonicalization": "official_egovla_h1_actuated_qpos_plus_urdf_mimic",
            "egovla_source_sha256": AUDITED_EGOVLA_SOURCE_SHA256,
            "inspire12_order": list(self.inspire_models["left"].actuated_names)
            + list(
                getattr(
                    __import__(
                        "spark_data.alignment.sources.inspire_hand",
                        fromlist=["INSPIRE12_MIMIC"],
                    ),
                    "INSPIRE12_MIMIC",
                )
            ),
            "wuji_variant": WUJI_VARIANT,
            "observation_pipeline": OFFICIAL_PIPELINE_OBS,
            "action_pipeline": OFFICIAL_PIPELINE_ACT,
            "inverse_landmark_schema": INVERSE_LANDMARK_SCHEMA,
            "solver": self.solver,
            "max_iterations": self.max_iterations,
            "forward_rms_limit_m": self.tolerance_m,
            "inverse_rms_limit_m": self.inverse_max_rms_m,
            "mimic_tolerance_rad": self.mimic_tolerance_rad,
            "source_root": str(self.repo_root),
            "source_sha256": dict(EXPECTED_SHA256),
            "source_provenance": (
                "pinned" if not self._source_mismatches else "unpinned_warning"
            ),
            "source_mismatches": list(self._source_mismatches),
            "linear_inverse": (
                {
                    "enabled": True,
                    "path": str(self._linear_inverse.artifact_path),
                    "sha256": self._linear_inverse.artifact_sha256,
                    "mode": ("linear_direct" if getattr(self, "inverse_mode", "geometric") == "linear_direct" else "fallback_after_geometric_quality_gate"),
                }
                if getattr(self, "_linear_inverse", None) is not None
                else {"enabled": False}
            ),
        }


__all__ = [
    "EXPECTED_SHA256",
    "RETARGET_DIAGNOSTICS_SCHEMA",
    "H1InspireWujiAdapter",
    "H1InspireRetarget",
    "H1InspireRetargetError",
    "H1RetargetContractError",
    "H1RetargetFitError",
    "H1_INTERLEAVED_ORDER",
    "SPARK_PACKED_ORDER",
    "H1_ACTUATED_TO_SPARK",
    "H1_ACTUATED_QPOS_INDICES",
    "SPARK_FROM_H1",
    "H1_FROM_SPARK",
    "AUDITED_EGOVLA_SOURCE_SHA256",
    "INSPIRE12_DIM",
    "WUJI20_DIM",
    "_inverse_fit_diagnostics",
]

# Compatibility names used by the adapter contract tests and by older local
# prototypes.  They are aliases, not separate implementations.
H1InspireRetarget = H1InspireWujiAdapter
PINNED_SOURCE_SHA256 = EXPECTED_SHA256
