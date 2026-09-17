"""Calibrated MANO45 runtime codec for the Spark0 Wuji Hand2.

This module deliberately does not duplicate the retargeting implementation.
For a MANO45 checkpoint it loads the exact ``add_mano``/``mano_wuji_map``
toolchain selected by ``mano_tools_root`` and validates its public kinematic
contract before serving an action.  The legacy Wuji20 adapter never imports
this module.

The two frame conventions that matter at the policy boundary are:

* ``mano/state/*_ee_poses`` stores MANO root rotation, but its xyz is anchored
  directly on the dataset Link7 xyz by ``add_mano.anchor_mano_wrist_to_ee``.
* XPolicyLab commands Link7.  Decoding therefore recovers Link7 orientation
  from the predicted MANO root and the observation-time MANO-root-in-Link7
  offset.  Using the newly solved finger q here would conjugate extra
  rotation onto the wrist.  It must not apply the old fixed Link7-to-wrist
  transform.

Action decoding is explicitly selected per deployment.  ``geometric`` is the
historical default.  ``linear`` uses a hash-locked side-specific paired-data
decoder directly; a safety-gate rejection falls back to the historical bounded
geometric inverse, while an accepted linear prediction is never IK-refined.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.VITRA.mano45_linear_inverse import (
    LinearInverseContractError,
    LinearInverseSafetyError,
    Mano45LinearInverse,
)

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # validated explicitly when the MANO45 codec is constructed
    _threadpool_limits = None


SIDES = ("left", "right")
HAND_DOF = 20
MANO_POSE_DIM = 45
MANO_BETA_DIM = 10
DEFAULT_HAND_VARIANT = "wujihand2"
DEFAULT_LEFT_CONVENTION = "mirror_model"
MAX_INVERSE_ITERATIONS = 12
DEFAULT_INVERSE_TIMEOUT_S = 2.0


class ManoRuntimeContractError(RuntimeError):
    """The configured external retargeting toolchain is absent/incompatible."""


class ManoRuntimeFitError(RuntimeError):
    """A finite Wuji/MANO fit could not be produced for a runtime frame."""


@dataclass(frozen=True)
class ManoFit:
    """One fitted MANO hand in the environment frame."""

    root_in_env: np.ndarray
    hand_pose_rotvec: np.ndarray
    betas: np.ndarray
    rms_m: float


@dataclass(frozen=True)
class WujiDecode:
    """One decoded Link7/Wuji target in native simulator order."""

    link7_in_env: np.ndarray
    q_stage_major: np.ndarray
    rms_m: float
    solve_elapsed_s: float = 0.0
    solver_reported_valid: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix, got {matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    return matrix


def _require_vector(value: Any, dimension: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite {dimension}-D, got {vector.shape}")
    return vector


def _discover_layout(configured_root: str | Path) -> tuple[Path, Path, dict[str, Path]]:
    """Resolve ``Spark-0`` repo root and its ``Spark_0`` Python package root."""

    requested = Path(configured_root).expanduser().resolve()
    candidates: list[tuple[Path, Path]] = []
    # The accepted explicit forms are Spark-0/, Spark-0/Spark_0/, or the
    # tools/data_alignment directory itself.  No implicit global path is used.
    candidates.append((requested, requested / "Spark_0"))
    candidates.append((requested.parent, requested))
    if requested.name == "data_alignment" and requested.parent.name == "tools":
        package_root = requested.parents[1]
        candidates.append((package_root.parent, package_root))

    for repo_root, package_root in candidates:
        files = {
            "add_mano.py": package_root / "tools/data_alignment/add_mano.py",
            "mano_wuji_map.py": package_root / "tools/data_alignment/mano_wuji_map.py",
            "mano.py": package_root / "tools/data_alignment/targets/mano.py",
            "wuji.py": package_root / "tools/data_alignment/targets/wuji.py",
            "mounts.py": repo_root / "data_trasfomer/to_spark_world_frame.py",
        }
        if all(path.is_file() for path in files.values()):
            return repo_root.resolve(), package_root.resolve(), files

    expected = requested / "Spark_0/tools/data_alignment/mano_wuji_map.py"
    raise ManoRuntimeContractError(
        "Invalid mano_tools_root. Expected the Spark-0 checkout (or its Spark_0 "
        f"package/tools directory) containing {expected}; configured={requested}"
    )


def _require_callable(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        raise ManoRuntimeContractError(
            f"External module {module.__name__} is missing callable {name}()"
        )
    return value


class Mano45RuntimeCodec:
    """Exact ``add_mano`` forward plus an explicit MANO-to-Wuji inverse.

    ``q`` is always the Wuji Hand2 Isaac articulation order used on disk:
    stage-major, with siblings ``index,middle,pinky,ring,thumb``.  The external
    ``WujiHandModel`` owns that order and the URDF joint limits.  Historical
    ``geometric`` mode uses MANO FK/Wuji IK; opt-in ``linear`` mode directly
    uses paired-data regression and invokes geometric mode only on rejection.
    """

    def __init__(
        self,
        mano_tools_root: str | Path,
        *,
        hand_variant: str = DEFAULT_HAND_VARIANT,
        left_convention: str = DEFAULT_LEFT_CONVENTION,
        max_iterations: int = 35,
        inverse_max_iterations: int = MAX_INVERSE_ITERATIONS,
        inverse_timeout_s: float = DEFAULT_INVERSE_TIMEOUT_S,
        tolerance_m: float = 0.03,
        inverse_mode: str = "geometric",
        linear_inverse_path: str | Path | None = None,
        linear_inverse_sha256: str | None = None,
    ) -> None:
        if not str(mano_tools_root).strip():
            raise ManoRuntimeContractError(
                "MANO45 inference requires an explicit mano_tools_root"
            )
        if hand_variant != DEFAULT_HAND_VARIANT:
            raise ManoRuntimeContractError(
                "MANO45 Spark0 checkpoints require hand_variant='wujihand2'; "
                f"got {hand_variant!r}"
            )
        if left_convention != DEFAULT_LEFT_CONVENTION:
            raise ManoRuntimeContractError(
                "The prepared Spark0 MANO data uses left_convention='mirror_model'; "
                f"got {left_convention!r}"
            )
        if max_iterations < 1 or not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError("max_iterations/tolerance_m must be positive")
        if not 1 <= int(inverse_max_iterations) <= MAX_INVERSE_ITERATIONS:
            raise ValueError(
                "inverse_max_iterations must be in "
                f"[1, {MAX_INVERSE_ITERATIONS}], got {inverse_max_iterations}"
            )
        if not np.isfinite(inverse_timeout_s) or inverse_timeout_s <= 0:
            raise ValueError("inverse_timeout_s must be finite and positive")
        if _threadpool_limits is None:
            raise ManoRuntimeContractError(
                "MANO45 runtime requires threadpoolctl so NumPy BLAS can be limited "
                "to one thread around the retargeting solvers. Install threadpoolctl; "
                "continuing without it can create hundreds of OpenBLAS threads."
            )

        repo_root, package_root, source_files = _discover_layout(mano_tools_root)
        # ``tools`` lives below package_root; the Link7 mount module is a sibling
        # package below repo_root.  Insert exact configured roots only.
        for path in (repo_root, package_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        try:
            map_module = importlib.import_module("tools.data_alignment.mano_wuji_map")
            mano_module = importlib.import_module("tools.data_alignment.targets.mano")
            wuji_module = importlib.import_module("tools.data_alignment.targets.wuji")
            mount_module = importlib.import_module("data_trasfomer.to_spark_world_frame")
        except Exception as exc:  # pragma: no cover - message is the useful contract
            raise ManoRuntimeContractError(
                f"Could not import MANO/Wuji tools from configured root {repo_root}: {exc}"
            ) from exc
        imported_sources = {
            "mano_wuji_map.py": map_module,
            "mano.py": mano_module,
            "wuji.py": wuji_module,
            "mounts.py": mount_module,
        }
        for label, module in imported_sources.items():
            imported_path = Path(getattr(module, "__file__", "")).resolve()
            if imported_path != source_files[label].resolve():
                raise ManoRuntimeContractError(
                    f"Imported {module.__name__} from {imported_path}, not configured "
                    f"mano_tools_root source {source_files[label]}"
                )

        self._ManoFitter = getattr(mano_module, "ManoFitter", None)
        self._ManoModel = getattr(mano_module, "ManoModel", None)
        self._WujiHandModel = getattr(wuji_module, "WujiHandModel", None)
        if not all(callable(value) for value in (self._ManoFitter, self._ManoModel, self._WujiHandModel)):
            raise ManoRuntimeContractError(
                "External toolchain must export ManoFitter, ManoModel, and WujiHandModel"
            )
        self._default_mano_asset = _require_callable(mano_module, "default_mano_asset")
        self._forward_hand_wrist_anchored = _require_callable(
            mano_module, "forward_hand_wrist_anchored"
        )
        self._world_mano_to_wuji = _require_callable(
            map_module, "world_mano_joints21_to_wuji_q"
        )
        self._mano_joints21_to_add_mano_chains = _require_callable(
            map_module, "mano_joints21_to_add_mano_chains"
        )
        self._canonicalize_hand_points = _require_callable(
            map_module, "canonicalize_hand_points"
        )
        signature = inspect.signature(self._world_mano_to_wuji)
        required_parameters = {"side", "variant", "initial", "model"}
        if not required_parameters.issubset(signature.parameters):
            raise ManoRuntimeContractError(
                "world_mano_joints21_to_wuji_q() has an incompatible API; "
                f"required parameters={sorted(required_parameters)}, signature={signature}"
            )

        self.repo_root = repo_root
        self.package_root = package_root
        self.source_hashes = {name: _sha256(path) for name, path in source_files.items()}
        self.hand_variant = hand_variant
        self.left_convention = left_convention
        self.max_iterations = int(max_iterations)
        self.inverse_max_iterations = int(inverse_max_iterations)
        self.inverse_timeout_s = float(inverse_timeout_s)
        self.tolerance_m = float(tolerance_m)
        self.last_inverse_solve_elapsed_s: dict[str, float] = {}
        self.last_inverse_backend: dict[str, str] = {}
        self._warned_inverse_valid_false: set[str] = set()
        self.linear_inverse_counts = {
            side: {"accepted": 0, "fallback": 0} for side in SIDES
        }
        self.inverse_mode = str(inverse_mode)
        if self.inverse_mode not in ("geometric", "linear"):
            raise ManoRuntimeContractError(
                "MANO45 inverse_mode must be 'geometric' or 'linear', got "
                f"{self.inverse_mode!r}"
            )
        self.linear_inverse = None
        if self.inverse_mode == "linear":
            if linear_inverse_path is None or linear_inverse_sha256 is None:
                raise ManoRuntimeContractError(
                    "MANO45 inverse_mode='linear' requires both linear_inverse_path "
                    "and linear_inverse_sha256"
                )
            try:
                self.linear_inverse = Mano45LinearInverse.load(
                    linear_inverse_path,
                    expected_sha256=linear_inverse_sha256,
                )
            except LinearInverseContractError as exc:
                raise ManoRuntimeContractError(str(exc)) from exc

        asset_path = Path(self._default_mano_asset()).resolve()
        if not asset_path.is_file():
            raise ManoRuntimeContractError(f"MANO asset does not exist: {asset_path}")
        self.source_hashes["mano_asset.npz"] = _sha256(asset_path)
        self.mano_model = self._ManoModel(asset_path)
        self.mano_fitter = self._ManoFitter(self.mano_model)
        fit_signature = inspect.signature(self.mano_fitter.fit)
        if not {"previous", "max_iterations", "tolerance_m", "mirror"}.issubset(
            fit_signature.parameters
        ):
            raise ManoRuntimeContractError(
                f"ManoFitter.fit() has an incompatible API: {fit_signature}"
            )
        if not callable(getattr(self.mano_fitter, "_palm_frame", None)) or np.asarray(
            getattr(self.mano_fitter, "rest_palm", None)
        ).shape != (3, 3):
            raise ManoRuntimeContractError(
                "ManoFitter must expose the add_mano palm-frame/rest-palm contract"
            )

        hand_mount_r = getattr(mount_module, "HAND_MOUNT_R", None)
        hand_mount_t = getattr(mount_module, "HAND_MOUNT_T", None)
        if not isinstance(hand_mount_r, Mapping) or not isinstance(hand_mount_t, Mapping):
            raise ManoRuntimeContractError(
                "to_spark_world_frame.py must export HAND_MOUNT_R/HAND_MOUNT_T"
            )
        self.mounts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.hand_models: dict[str, Any] = {}
        expected_joint_order = tuple(
            (finger, stage)
            for stage in range(4)
            for finger in (1, 2, 4, 3, 0)
        )
        for side in SIDES:
            rotation = np.asarray(hand_mount_r[side], dtype=np.float64)
            translation = np.asarray(hand_mount_t[side], dtype=np.float64)
            if rotation.shape != (3, 3) or translation.shape != (3,):
                raise ManoRuntimeContractError(f"Invalid {side} Link7 hand mount shapes")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
                raise ManoRuntimeContractError(f"{side} Link7 hand mount is not a rotation")
            self.mounts[side] = (rotation, translation)

            hand = self._WujiHandModel(side, variant=hand_variant)
            variant = getattr(hand, "variant", None)
            if (
                getattr(variant, "name", None) != DEFAULT_HAND_VARIANT
                or tuple(getattr(variant, "joint_order", ())) != expected_joint_order
            ):
                raise ManoRuntimeContractError(
                    f"{side} Wuji model is not the expected Isaac stage-major Hand2 contract"
                )
            if np.asarray(hand.lower).shape != (HAND_DOF,) or np.asarray(hand.upper).shape != (HAND_DOF,):
                raise ManoRuntimeContractError(f"{side} Wuji model has invalid joint limits")
            self.hand_models[side] = hand
            urdf_path = Path(hand.urdf_path).resolve()
            if not urdf_path.is_file():
                raise ManoRuntimeContractError(f"{side} Wuji Hand2 URDF is missing: {urdf_path}")
            self.source_hashes[f"wuji_{side}.urdf"] = _sha256(urdf_path)
            if self.linear_inverse is not None:
                try:
                    self.linear_inverse.validate_urdf_contract(
                        side=side,
                        lower=hand.lower,
                        upper=hand.upper,
                        urdf_sha256=self.source_hashes[f"wuji_{side}.urdf"],
                    )
                except LinearInverseContractError as exc:
                    raise ManoRuntimeContractError(str(exc)) from exc

    @staticmethod
    def _require_side(side: str) -> str:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        return side

    def _checked_q(self, side: str, value: Any, *, input_state: bool) -> np.ndarray:
        q = _require_vector(value, HAND_DOF, f"{side} Wuji q")
        model = self.hand_models[side]
        lower = np.asarray(model.lower, dtype=np.float64)
        upper = np.asarray(model.upper, dtype=np.float64)
        # Simulator observations may differ from a URDF endpoint by tiny float
        # noise.  Anything larger is a representation/order error, not something
        # to hide with clipping.
        slack = 1e-4 if input_state else 1e-8
        if np.any(q < lower - slack) or np.any(q > upper + slack):
            label = "runtime observation" if input_state else "IK result"
            raise ManoRuntimeFitError(
                f"{side} {label} violates Wuji Hand2 URDF limits in stage-major order"
            )
        return np.clip(q, lower, upper)

    def _mano_root_rotation_in_link7(self, side: str, q_stage_major: Any) -> np.ndarray:
        """Closed-form MANO-root rotation for q with identity Link7.

        This is exactly the orientation part of ``ManoFitter.fit``: construct
        the mounted Wuji palm frame and register it to ``rest_palm``.  The left
        branch applies the same mirror-fit/mirror-parameter convention as the
        source.  No 45-D hand-pose DLS is needed to recover Link7 orientation.
        """

        side = self._require_side(side)
        q = self._checked_q(side, q_stage_major, input_state=False)
        hand = self.hand_models[side]
        points = np.asarray(hand.forward_points(q, canonical=False), dtype=np.float64)
        mount_rotation, mount_translation = self.mounts[side]
        points = points @ mount_rotation.T + mount_translation
        mirror_x = np.diag([-1.0, 1.0, 1.0])
        if side == "left":
            points = points.copy()
            points[..., 0] *= -1.0
        palm = self.mano_fitter._palm_frame(
            points[1, 1], points[4, 1], points[2, 1], points[0, 0]
        )
        if palm is None:
            raise ManoRuntimeFitError(f"{side} mounted Wuji palm frame is degenerate")
        rotation = palm @ np.asarray(self.mano_fitter.rest_palm, dtype=np.float64).T
        if side == "left":
            # ManoFitter.fit mirrors returned axis-angle as (x,-y,-z), which
            # is the proper-rotation conjugation F @ R @ F.
            rotation = mirror_x @ rotation @ mirror_x
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ManoRuntimeFitError(f"{side} MANO root registration is not a rotation")
        return Rotation.from_matrix(rotation).as_matrix()

    def _link7_from_integrated_root(
        self,
        side: str,
        root: np.ndarray,
        q_for_offset: Any,
    ) -> np.ndarray:
        """Map an integrated MANO root back to Link7 with a frozen palm offset.

        ``add_mano`` anchors xyz on Link7.  The orientation offset is
        q-dependent; callers must pass the observation-time q so finger IK
        updates cannot accumulate extra wrist rotation onto the EE target.
        """

        root_rotation_in_link7 = self._mano_root_rotation_in_link7(side, q_for_offset)
        link7 = np.eye(4, dtype=np.float64)
        link7[:3, :3] = root[:3, :3] @ root_rotation_in_link7.T
        link7[:3, 3] = root[:3, 3]
        return link7

    def wuji_to_mano(
        self,
        *,
        side: str,
        q_stage_major: Any,
        link7_in_env: Any,
        previous_hand_pose: Optional[Any] = None,
    ) -> ManoFit:
        """Run the same Wuji-FK + world mount + MANO fit as ``add_mano``."""

        side = self._require_side(side)
        q = self._checked_q(side, q_stage_major, input_state=True)
        link7 = _require_matrix(link7_in_env, f"{side} Link7 pose")
        previous = None
        if previous_hand_pose is not None:
            previous = _require_vector(
                previous_hand_pose, MANO_POSE_DIM, f"{side} previous MANO pose"
            )
            if side == "left":
                # ManoFitter.fit(mirror=True) mirrors its returned pose back to
                # true-left semantics, but ``previous`` seeds the internal
                # right-model solver *before* that final mirror.  The mapping
                # (x,y,z)->(x,-y,-z) is an involution, so convert cached
                # true-left values back into solver space here.
                previous = previous.reshape(15, 3).copy()
                previous[:, 1:] *= -1.0
                previous = previous.reshape(MANO_POSE_DIM)

        hand = self.hand_models[side]
        points = np.asarray(hand.forward_points(q, canonical=False), dtype=np.float64)
        if points.shape != (5, 5, 3):
            raise ManoRuntimeContractError(
                f"WujiHandModel.forward_points() returned {points.shape}, expected (5,5,3)"
            )
        mount_rotation, mount_translation = self.mounts[side]
        points_link7 = points @ mount_rotation.T + mount_translation
        points_env = points_link7 @ link7[:3, :3].T + link7[:3, 3]
        # NumPy/OpenBLAS may otherwise create one worker per host CPU for each
        # tiny DLS system (64+ threads per hand on the evaluation host).  Keep
        # the scope around the external numerical solve only so Torch/PaliGemma
        # execution is unaffected.
        with _threadpool_limits(limits=1, user_api="blas"):
            result, rms, valid, _iterations = self.mano_fitter.fit(
                points_env,
                np.ones((5, 5), dtype=np.float64),
                previous=previous,
                max_iterations=self.max_iterations,
                tolerance_m=self.tolerance_m,
                mirror=(side == "left"),
            )
        global_orient = _require_vector(
            result.get("global_orient"), 3, f"{side} fitted MANO global orient"
        )
        hand_pose = _require_vector(
            result.get("hand_pose"), MANO_POSE_DIM, f"{side} fitted MANO hand pose"
        )
        betas = _require_vector(
            result.get("betas"), MANO_BETA_DIM, f"{side} fitted MANO betas"
        )
        if not bool(valid) or not np.isfinite(rms) or float(rms) > self.tolerance_m:
            raise ManoRuntimeFitError(
                f"{side} Wuji-to-MANO fit failed quality gate "
                f"(rms={float(rms):.6f} m, limit={self.tolerance_m:.6f} m)"
            )

        root = np.eye(4, dtype=np.float64)
        root[:3, :3] = Rotation.from_rotvec(global_orient).as_matrix()
        # add_mano anchors the stored MANO wrist xyz directly to Link7 xyz.
        root[:3, 3] = link7[:3, 3]
        return ManoFit(root, hand_pose, betas, float(rms))

    def mano_to_wuji(
        self,
        *,
        side: str,
        root_in_env: Any,
        hand_pose_rotvec: Any,
        betas: Any,
        initial_q_stage_major: Any,
    ) -> WujiDecode:
        """Decode MANO to Wuji q20, then recover the absolute Link7 target."""

        side = self._require_side(side)
        root = _require_matrix(root_in_env, f"{side} MANO root")
        hand_pose = _require_vector(
            hand_pose_rotvec, MANO_POSE_DIM, f"{side} MANO hand pose"
        )
        beta = _require_vector(betas, MANO_BETA_DIM, f"{side} MANO betas")
        initial = self._checked_q(side, initial_q_stage_major, input_state=True)

        global_orient = Rotation.from_matrix(root[:3, :3]).as_rotvec()
        joints21 = self._forward_hand_wrist_anchored(
            self.mano_model,
            is_right=(side == "right"),
            global_orient=global_orient.reshape(1, 3),
            hand_pose=hand_pose.reshape(1, MANO_POSE_DIM),
            betas=beta.reshape(1, MANO_BETA_DIM),
            wrist=root[:3, 3].reshape(1, 3),
            left_convention=self.left_convention,
        )
        if isinstance(joints21, Mapping):
            joints21 = joints21.get("joints21")
        joints21 = np.asarray(joints21, dtype=np.float64)
        if joints21.shape != (1, 21, 3) or not np.all(np.isfinite(joints21)):
            raise ManoRuntimeFitError(
                f"{side} MANO FK returned invalid joints shape {joints21.shape}"
            )
        if getattr(self, "inverse_mode", "geometric") == "linear":
            try:
                decoded = self._linear_mano_to_wuji(
                    side=side,
                    root=root,
                    hand_pose=hand_pose,
                    joints21=joints21[0],
                    initial=initial,
                )
            except ManoRuntimeFitError as exc:
                self.linear_inverse_counts[side]["fallback"] += 1
                self.last_inverse_backend[side] = "geometric_fallback"
                print(
                    "[VITRA] MANO45 linear safety fallback",
                    f"side={side}",
                    f"reason={exc}",
                    f"accepted={self.linear_inverse_counts[side]['accepted']}",
                    f"fallback={self.linear_inverse_counts[side]['fallback']}",
                    flush=True,
                )
            else:
                self.linear_inverse_counts[side]["accepted"] += 1
                self.last_inverse_backend[side] = "linear"
                return decoded
        solve_started = time.perf_counter()
        try:
            with _threadpool_limits(limits=1, user_api="blas"):
                q, valid, rms = self._world_mano_to_wuji(
                    joints21[0],
                    side=side,
                    variant=self.hand_variant,
                    initial=initial,
                    max_iterations=self.inverse_max_iterations,
                    tolerance_m=self.tolerance_m,
                    model=self.hand_models[side],
                )
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            solve_elapsed_s = time.perf_counter() - solve_started
            self.last_inverse_solve_elapsed_s[side] = solve_elapsed_s
            raise ManoRuntimeFitError(
                f"{side} MANO-to-Wuji IK numerical solve failed after "
                f"{solve_elapsed_s:.6f} s: {exc}"
            ) from exc
        solve_elapsed_s = time.perf_counter() - solve_started
        self.last_inverse_solve_elapsed_s[side] = solve_elapsed_s
        if getattr(self, "inverse_mode", "geometric") == "geometric":
            self.last_inverse_backend[side] = "geometric"
        print(
            "[VITRA] MANO45 inverse",
            f"side={side}",
            f"elapsed_s={solve_elapsed_s:.6f}",
            f"iterations_limit={self.inverse_max_iterations}",
            f"rms_m={float(rms):.6f}",
            flush=True,
        )
        # This audit limit is intentionally checked after the external solve:
        # Python cannot safely preempt a native BLAS call in-process.  BLAS=1
        # plus the hard iteration cap bounds the work; the elapsed gate makes a
        # host regression explicit instead of silently serving a stale chunk.
        if solve_elapsed_s > self.inverse_timeout_s:
            raise ManoRuntimeFitError(
                f"{side} MANO-to-Wuji IK exceeded the post-solve time limit "
                f"(elapsed={solve_elapsed_s:.6f} s, limit={self.inverse_timeout_s:.6f} s, "
                f"iterations_limit={self.inverse_max_iterations})"
            )
        q_array = np.asarray(q, dtype=np.float64)
        if not np.all(np.isfinite(q_array)):
            raise ManoRuntimeFitError(
                f"{side} MANO-to-Wuji IK returned non-finite q"
            )
        q = self._checked_q(side, q_array, input_state=False)
        if not np.isfinite(rms) or float(rms) > self.tolerance_m:
            raise ManoRuntimeFitError(
                f"{side} MANO-to-Wuji IK failed quality gate "
                f"(rms={float(rms):.6f} m, limit={self.tolerance_m:.6f} m)"
            )
        solver_reported_valid = bool(valid)
        if not solver_reported_valid:
            warned_sides = getattr(self, "_warned_inverse_valid_false", set())
            if side not in warned_sides:
                print(
                    "[VITRA] WARNING MANO45 inverse solver reported valid=False, "
                    "but finite in-limit q and RMS passed the runtime quality gate",
                    f"side={side} rms_m={float(rms):.6f}",
                    flush=True,
                )
                warned_sides.add(side)
                self._warned_inverse_valid_false = warned_sides

        # Recover Link7 from the integrated MANO root.  Freeze the palm-frame
        # offset to the observation-time q: a newly solved finger q would
        # otherwise conjugate extra rotation onto Link7 every step and look
        # like open-loop wrist spin under EgoVLA EE-IK.
        link7 = self._link7_from_integrated_root(side, root, initial)
        return WujiDecode(
            link7,
            q.astype(np.float32),
            float(rms),
            solve_elapsed_s,
            solver_reported_valid,
        )

    def _linear_mano_to_wuji(
        self,
        *,
        side: str,
        root: np.ndarray,
        hand_pose: np.ndarray,
        joints21: np.ndarray,
        initial: np.ndarray,
    ) -> WujiDecode:
        """Decode with the paired linear inverse, without geometric IK refinement."""

        linear_inverse = self.linear_inverse
        if linear_inverse is None:  # pragma: no cover - caller guards this branch
            raise ManoRuntimeContractError("linear MANO inverse is not configured")
        hand = self.hand_models[side]
        started = time.perf_counter()
        try:
            prediction = linear_inverse.predict(
                side=side,
                hand_pose_rotvec=hand_pose,
                previous_q_stage_major=initial,
                lower=hand.lower,
                upper=hand.upper,
            )
        except LinearInverseSafetyError as exc:
            raise ManoRuntimeFitError(str(exc)) from exc

        q = self._checked_q(side, prediction.q_stage_major, input_state=False)
        points, weights = self._mano_joints21_to_add_mano_chains(
            joints21, model=hand
        )
        local, geometry_ok = self._canonicalize_hand_points(
            points,
            joints21[0],
            weights,
            knuckle_index=1,
        )
        if not geometry_ok:
            raise ManoRuntimeFitError(
                f"{side} MANO linear inverse could not canonicalize target geometry"
            )
        geometry_valid, rms, _min_distance = hand.evaluate(q, local, weights)
        geometry_limit = linear_inverse.geometry_limit_m(side)
        if not geometry_valid or not np.isfinite(rms) or float(rms) > geometry_limit:
            raise ManoRuntimeFitError(
                f"{side} MANO linear inverse failed geometry cycle gate "
                f"(rms={float(rms):.6f} m, limit={geometry_limit:.6f} m)"
            )

        link7 = self._link7_from_integrated_root(side, root, initial)
        elapsed_s = time.perf_counter() - started
        self.last_inverse_solve_elapsed_s[side] = elapsed_s
        accepted_next = self.linear_inverse_counts[side]["accepted"] + 1
        if accepted_next == 1 or accepted_next % 1000 == 0:
            print(
                "[VITRA] MANO45 linear inverse summary",
                f"side={side}",
                f"accepted={accepted_next}",
                f"fallback={self.linear_inverse_counts[side]['fallback']}",
                f"elapsed_s={elapsed_s:.6f}",
                f"rms_m={float(rms):.6f}",
                f"ood_rms={prediction.ood_rms:.4f}",
                f"cycle_rms={prediction.cycle_rms:.4f}",
                f"step_ratio={prediction.step_delta_ratio:.4f}",
                f"urdf_clipped={prediction.clipped_to_urdf}",
                flush=True,
            )
        return WujiDecode(
            link7,
            q.astype(np.float32),
            float(rms),
            elapsed_s,
            True,
        )

    def provenance_summary(self) -> str:
        hashes = ",".join(
            f"{name}={digest[:12]}" for name, digest in sorted(self.source_hashes.items())
        )
        inverse = (
            self.linear_inverse.provenance_summary()
            if self.linear_inverse is not None
            else (
                f"geometric_diagnostic inverse_max_iterations={self.inverse_max_iterations} "
                f"inverse_timeout_s={self.inverse_timeout_s:.3f}"
            )
        )
        counts = ",".join(
            f"{side}:accepted={self.linear_inverse_counts[side]['accepted']},"
            f"fallback={self.linear_inverse_counts[side]['fallback']}"
            for side in SIDES
        )
        return (
            f"tools_root={self.repo_root} source_sha256[{hashes}] "
            f"inverse_mode={self.inverse_mode} inverse=[{inverse}] "
            f"counts=[{counts}] blas_threads=1"
        )
