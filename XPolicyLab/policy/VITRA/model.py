"""XPolicyLab inference adapter for Microsoft VITRA.

The XPolicyLab boundary is absolute dual-arm EE control.  VITRA keeps its
native representation internally: per hand, the state is an absolute
camera-frame wrist pose plus absolute hand joints, while each predicted action
contains a wrist step delta plus the next absolute hand-joint target.  The
shared Spark0 dataset module owns the coordinate and Wuji/MANO mappings so that
training and inference cannot silently diverge.
"""

from __future__ import annotations

import inspect
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info


# Required by the XPolicyLab integration contract: model.py lives at
# XPolicyLab/policy/VITRA/model.py, therefore parents[2] is the checkout root.
_XPOLICYLAB_ROOT = Path(__file__).resolve().parents[2]
_POLICY_DIR = Path(__file__).resolve().parent
_VITRA_ROOT = _POLICY_DIR / "VITRA"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_DEFAULT_MAPPING_PATH = _POLICY_DIR / "mapping_wuji20_mano45.json"
_BUNDLED_MANO_LINEAR_INVERSE_PATH = (
    _POLICY_DIR / "artifacts/mano45_to_wuji20_linear_v1.json"
)
_BUNDLED_MANO_LINEAR_INVERSE_SHA256 = (
    "f196701addfcddc7bfac70a6caf722b237a398af104aa6a15993a60ccea93940"
)

# The released EgoVLA H1 simulator does not expose camera calibration through
# its native observation dictionary. Keep the policy-side fallback explicit,
# immutable, and hash locked; a unit matrix (or a guessed FOV) would silently
# change the MANO/Wuji camera-frame geometry. The artifact is only loaded for
# the exact ``ego_h1_inspire`` environment and is never used for other robots.
_DEFAULT_H1_CAMERA_CALIBRATION_PATH = (
    _POLICY_DIR / "artifacts/ego_h1_inspire_main_camera_v1.json"
)
_H1_CAMERA_CALIBRATION_SCHEMA = 1
_H1_CAMERA_SCALE_RULE = (
    "K_out = diag(width/1280, height/720, 1) @ K_source"
)
# EgoVLA VITRA training stores native 384x384 RGB. Isaac live ``fixed_rgb`` is
# 1280x720. Stretch the live frame to the training size before the VLM so
# PaliGemma does not see a different aspect-ratio path than the HDF5 loader.
_H1_POLICY_IMAGE_HW = (384, 384)
_BUNDLED_H1_CAMERA_CALIBRATION_SHA256 = (
    "50cb0135ffae63531e75d15adb791426d02789638ead7958567f1a0766170578"
)
# ``EVAL_ENV_TYPE`` is resolved by the benchmark launcher as ``sim``,
# ``debug`` or ``real_world`` (with an unset value defaulting to ``sim``).
# Keep the policy-side interpretation local as well: a policy server can be
# started independently of the environment client and must still fail closed
# before a simulator camera profile is ever applied to a real robot.
_H1_CAMERA_SIM_ENV_TYPES = frozenset(("", "sim", "debug"))
_H1_CAMERA_REAL_ENV_TYPES = frozenset(("real", "real_world"))
_H1_CAMERA_FALLBACK_SCOPES = frozenset(("simulator", "runtime"))

_STEP_RE = re.compile(r"(?:^|[-_.])step[=_-]?(\d+)(?:$|[-_.])")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _as_optional_path(value: Any, *, base: Path = _POLICY_DIR) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _step_number(path: Path) -> int:
    """Return the largest numeric ``step=...`` component in a path."""
    values: list[int] = []
    for part in path.parts:
        values.extend(int(value) for value in _STEP_RE.findall(part))
    return max(values, default=-1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_h1_camera_fallback_policy(
    model_cfg: Mapping[str, Any],
) -> tuple[str, str, bool]:
    """Resolve the environment mode and whether simulator fallback is legal.

    The benchmark client defaults an unset ``EVAL_ENV_TYPE`` to simulation.
    We mirror that rule here because policy servers may be launched in a
    separate process.  ``runtime`` scope is an explicit opt-in to require
    calibration metadata even in simulation/debug (useful for hardware-like
    test harnesses).  A real/real_world mode always disables the static
    simulator artifact, regardless of scope.
    """

    raw_mode = os.environ.get("EVAL_ENV_TYPE", "")
    mode = str(raw_mode).strip().lower().replace("-", "_")
    if mode not in _H1_CAMERA_SIM_ENV_TYPES | _H1_CAMERA_REAL_ENV_TYPES:
        raise ValueError(
            "Unknown EVAL_ENV_TYPE for VITRA H1 camera fallback: "
            f"{raw_mode!r}; expected sim, debug, real, or real_world"
        )
    resolved_mode = "sim" if mode == "" else ("real_world" if mode in _H1_CAMERA_REAL_ENV_TYPES else mode)

    raw_scope = model_cfg.get("h1_camera_calibration_scope", "simulator")
    scope = str(raw_scope).strip().lower().replace("-", "_")
    if scope not in _H1_CAMERA_FALLBACK_SCOPES:
        raise ValueError(
            "h1_camera_calibration_scope must be 'simulator' or 'runtime', "
            f"got {raw_scope!r}"
        )
    allow_fallback = scope == "simulator" and mode in _H1_CAMERA_SIM_ENV_TYPES
    return resolved_mode, scope, allow_fallback


def _latest_camera_extrinsics(value: Any, name: str) -> np.ndarray:
    """Select one runtime camera pose/matrix without changing its convention."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (7,) or array.shape == (4, 4):
        result = array
    elif array.ndim >= 2 and array.shape[-1] == 7:
        result = array.reshape(-1, 7)[-1]
    elif array.ndim >= 3 and array.shape[-2:] == (4, 4):
        result = array.reshape(-1, 4, 4)[-1]
    else:
        raise ValueError(
            f"{name} must be a pose7 or 4x4 matrix (optionally batched), got {array.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(result, dtype=np.float64)


def _latest_camera_intrinsics(value: Any, name: str) -> np.ndarray:
    """Select one runtime K/[fx,fy,cx,cy] value without flattening a matrix."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, 3) or array.shape == (4,):
        result = array
    elif array.ndim >= 3 and array.shape[-2:] == (3, 3):
        result = array.reshape(-1, 3, 3)[-1]
    elif array.ndim >= 2 and array.shape[-1] == 4:
        result = array.reshape(-1, 4)[-1]
    else:
        raise ValueError(
            f"{name} must be a 3x3 matrix or [fx,fy,cx,cy] (optionally batched), got {array.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    if result.shape == (4,):
        fx, fy, cx, cy = result.tolist()
        result = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return np.asarray(result, dtype=np.float64)


def _validate_camera_extrinsics(value: Any, name: str) -> np.ndarray:
    result = _latest_camera_extrinsics(value, name)
    if result.shape == (7,):
        quaternion = result[3:]
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm <= 1e-8 or abs(norm - 1.0) > 2e-3:
            raise ValueError(f"{name} quaternion must be finite and unit length")
    else:
        if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=2e-5):
            raise ValueError(f"{name} has an invalid homogeneous last row")
        rotation = result[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError(f"{name} rotation is not orthonormal")
        if float(np.linalg.det(rotation)) <= 0.0:
            raise ValueError(f"{name} rotation must have positive determinant")
    return result


def _validate_camera_intrinsics(
    value: Any, name: str, *, image_height: int, image_width: int
) -> np.ndarray:
    result = _latest_camera_intrinsics(value, name)
    if result.shape != (3, 3):
        raise ValueError(f"{name} must resolve to a 3x3 matrix, got {result.shape}")
    if result[0, 0] <= 0.0 or result[1, 1] <= 0.0:
        raise ValueError(f"{name} fx/fy must be positive")
    if not np.allclose(result[2], [0.0, 0.0, 1.0], atol=2e-5):
        raise ValueError(f"{name} has an invalid projective bottom row")
    if abs(float(result[0, 1])) > 1e-4:
        raise ValueError(f"{name} skew must be zero for the audited pinhole camera")
    cx, cy = float(result[0, 2]), float(result[1, 2])
    if not (0.0 <= cx <= image_width and 0.0 <= cy <= image_height):
        raise ValueError(
            f"{name} principal point ({cx}, {cy}) is outside image {image_width}x{image_height}"
        )
    return result


def _load_h1_camera_calibration(
    path: Any, asserted_sha256: Any
) -> dict[str, Any]:
    """Load and validate the hash-locked official EgoVLA H1 camera profile."""
    resolved = _as_optional_path(path)
    if resolved is None:
        raise ValueError(
            "ego_h1_inspire requires h1_camera_calibration_path; refusing an implicit camera frame"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"H1 camera calibration artifact is missing: {resolved}")
    expected = str(asserted_sha256 or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError(
            "h1_camera_calibration_sha256 must be a 64-character SHA-256 digest"
        )
    actual = _sha256_file(resolved)
    if actual != expected:
        raise ValueError(
            "H1 camera calibration SHA-256 mismatch: "
            f"expected={expected} actual={actual} path={resolved}"
        )
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"H1 camera calibration is not valid JSON: {resolved}") from exc
    if not isinstance(profile, dict):
        raise ValueError("H1 camera calibration root must be an object")
    if int(profile.get("schema_version", -1)) != _H1_CAMERA_CALIBRATION_SCHEMA:
        raise ValueError("unsupported H1 camera calibration schema_version")
    if profile.get("profile") != "egovla_ego_h1_inspire_main_camera_v1":
        raise ValueError("unexpected H1 camera calibration profile")
    if profile.get("env_cfg_type") != "ego_h1_inspire":
        raise ValueError("H1 camera calibration is not for ego_h1_inspire")
    if profile.get("camera") != "cam_head" or profile.get("color_order") != "RGB":
        raise ValueError("H1 camera calibration must describe the RGB cam_head view")

    source = profile.get("source")
    if not isinstance(source, dict):
        raise ValueError("H1 camera calibration is missing source metadata")
    resolution = source.get("resolution")
    if not isinstance(resolution, dict):
        raise ValueError("H1 camera calibration is missing source resolution")
    try:
        source_width = int(resolution["width"])
        source_height = int(resolution["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("H1 camera calibration source resolution is invalid") from exc
    if source_width <= 0 or source_height <= 0:
        raise ValueError("H1 camera calibration source resolution must be positive")
    if source.get("resize_mode") != "stretch":
        raise ValueError("H1 camera calibration requires the explicit stretch resize rule")
    if source.get("allowed_runtime_sizes") is None:
        raise ValueError("H1 camera calibration must list allowed runtime image sizes")
    allowed_sizes: set[tuple[int, int]] = set()
    for item in source["allowed_runtime_sizes"]:
        if not isinstance(item, dict):
            raise ValueError("H1 camera allowed_runtime_sizes entries must be objects")
        try:
            width, height = int(item["width"]), int(item["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("H1 camera allowed runtime size is invalid") from exc
        if width <= 0 or height <= 0:
            raise ValueError("H1 camera allowed runtime sizes must be positive")
        allowed_sizes.add((height, width))
    if (source_height, source_width) not in allowed_sizes:
        raise ValueError("H1 camera source resolution must be an allowed runtime size")
    if _H1_POLICY_IMAGE_HW not in allowed_sizes:
        raise ValueError(
            "H1 camera calibration must allow the 384x384 EgoVLA training image size"
        )

    intrinsics = profile.get("intrinsics")
    if not isinstance(intrinsics, dict) or intrinsics.get("scale_rule") != _H1_CAMERA_SCALE_RULE:
        raise ValueError("H1 camera calibration has an unexpected intrinsic scale rule")
    source_k = _validate_camera_intrinsics(
        intrinsics.get("matrix"),
        "H1 calibration intrinsics",
        image_height=source_height,
        image_width=source_width,
    )
    extrinsics = profile.get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise ValueError("H1 camera calibration is missing extrinsics")
    if (
        extrinsics.get("frame") != "camera_to_env"
        or extrinsics.get("axes") != "x_right_y_up_z_back"
        or extrinsics.get("order") != "x,y,z,qw,qx,qy,qz"
    ):
        raise ValueError("H1 camera calibration extrinsics convention is not audited")
    source_pose = _validate_camera_extrinsics(
        extrinsics.get("pose7"), "H1 calibration extrinsics"
    )
    if source_pose.shape != (7,):
        raise ValueError("H1 calibration extrinsics must use pose7")
    return {
        "path": resolved,
        "sha256": actual,
        "profile": str(profile["profile"]),
        "source_width": source_width,
        "source_height": source_height,
        "policy_hw": _H1_POLICY_IMAGE_HW,
        "allowed_sizes": frozenset(allowed_sizes),
        "source_intrinsics": source_k,
        "source_extrinsics": source_pose,
        "scale_rule": _H1_CAMERA_SCALE_RULE,
    }


def _resolve_weights(model_cfg: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve the standard run root, then select its newest numeric checkpoint."""
    root = resolve_checkpoint_root(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
    )

    if root.is_file():
        if root.suffix != ".pt":
            raise ValueError(f"VITRA checkpoint must be a .pt file, got: {root}")
        weights = root
        metadata_root = _find_metadata_root(weights, root.parent)
        return weights, metadata_root

    candidates: list[Path] = []
    direct = root / "weights.pt"
    if direct.is_file():
        candidates.append(direct)
    candidates.extend(path for path in root.rglob("weights.pt") if path.is_file())

    # A released checkpoint may retain its descriptive filename instead of the
    # FSDP ``weights.pt`` name.  Only use this fallback when it is unambiguous.
    if not candidates:
        released = [path for path in root.glob("*.pt") if path.name != "optimizer.pt"]
        if len(released) == 1:
            candidates = released

    if not candidates:
        raise FileNotFoundError(f"No VITRA weights found below checkpoint root: {root}")

    # Numeric global step is authoritative.  mtime only breaks ties (for
    # example a resumed run with two timestamp directories at the same step).
    weights = max(candidates, key=lambda path: (_step_number(path), path.stat().st_mtime_ns, str(path)))
    # Keep the logical checkpoint path when ``weights.pt`` is a symlink.  The
    # run's config/statistics live beside that link, while the released tensor
    # file may live in the shared pretrain directory.
    # A Web/evaluator plan may pass the epoch checkpoint directory itself
    # (``.../checkpoints/epoch=33-step=80000.ckpt``), rather than the run
    # directory containing ``config.json``.  Keep the exact selected weights,
    # but promote the metadata search root to the nearest ancestor that owns a
    # saved config.  This prevents falling back to the unrelated one-byte
    # policy-level statistics file and never copies or rewrites checkpoint data.
    return weights.absolute(), _find_metadata_root(weights, root.resolve())


def _find_metadata_root(weights: Path, fallback: Path) -> Path:
    """Find the nearest saved-run metadata directory for selected weights.

    Checkpoint artifacts are commonly nested as ``run/checkpoints/epoch.ckpt``
    while ``config.json`` lives at ``run``.  Search only ancestors up to this
    policy directory (or the explicit fallback for an external artifact), so a
    similarly named config from another checkout cannot be mixed in.
    """

    fallback = fallback.resolve()
    try:
        policy_root = _POLICY_DIR.resolve()
        weights_path = weights.resolve()
        if policy_root == weights_path or policy_root in weights_path.parents:
            for directory in _ancestors_until(weights.parent, policy_root):
                if any(
                    (directory / name).is_file()
                    for name in ("config.json", "configs_config.json")
                ):
                    return directory.resolve()
    except OSError:
        # Preserve the old explicit-root behavior when an external filesystem
        # disappears during discovery; the subsequent config error is clearer.
        pass
    return fallback


def _first_existing(paths: list[Optional[Path]]) -> Optional[Path]:
    for path in paths:
        if path is not None and path.is_file():
            return path.resolve()
    return None


_SHARED_MOUNT_ALIASES = (
    ("/mnt/xspark-data/", "/personal/"),
    ("/personal/", "/mnt/xspark-data/"),
)


def _resolve_existing_shared_path(
    value: str,
    aliases: tuple[tuple[str, str], ...] = _SHARED_MOUNT_ALIASES,
) -> str:
    """Resolve equivalent shared-storage mount names without changing files.

    Training hosts expose the shared tree as both ``/mnt/xspark-data`` and
    ``/personal``, while some evaluation nodes expose only ``/personal``.  A
    saved run config is immutable provenance, so deployment translates a path
    only when the saved spelling is absent and the equivalent alias exists.
    """

    if not isinstance(value, str) or not value.startswith("/"):
        return value
    if Path(value).exists():
        return value
    for source_prefix, target_prefix in aliases:
        if value.startswith(source_prefix):
            candidate = target_prefix + value[len(source_prefix) :]
            if Path(candidate).exists():
                return candidate
    return value


def _rewrite_saved_config_shared_paths(
    value: Any,
    *,
    aliases: tuple[tuple[str, str], ...] = _SHARED_MOUNT_ALIASES,
    key_path: str = "",
) -> list[tuple[str, str, str]]:
    """Rewrite existing shared-mount aliases in a loaded config in place."""

    changes: list[tuple[str, str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{key_path}.{key}" if key_path else str(key)
            if isinstance(item, str):
                resolved = _resolve_existing_shared_path(item, aliases)
                if resolved != item:
                    value[key] = resolved
                    changes.append((child_path, item, resolved))
            else:
                changes.extend(
                    _rewrite_saved_config_shared_paths(
                        item, aliases=aliases, key_path=child_path
                    )
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{key_path}[{index}]"
            if isinstance(item, str):
                resolved = _resolve_existing_shared_path(item, aliases)
                if resolved != item:
                    value[index] = resolved
                    changes.append((child_path, item, resolved))
            else:
                changes.extend(
                    _rewrite_saved_config_shared_paths(
                        item, aliases=aliases, key_path=child_path
                    )
                )
    return changes


def _ancestors_until(path: Path, stop: Path) -> list[Path]:
    result: list[Path] = []
    current = path if path.is_dir() else path.parent
    stop = stop.resolve()
    while True:
        result.append(current)
        if current == stop or current.parent == current:
            break
        current = current.parent
    return result


def _resolve_config_and_statistics(
    model_cfg: dict[str, Any], weights: Path, checkpoint_root: Path
) -> tuple[Path, Path]:
    explicit_config = _as_optional_path(
        model_cfg.get("model_config_path") or model_cfg.get("config_path")
    )
    search_dirs = _ancestors_until(weights.parent, checkpoint_root)
    config_path = _first_existing(
        [explicit_config]
        + [directory / name for directory in search_dirs for name in ("config.json", "configs_config.json")]
    )
    if config_path is None:
        raise FileNotFoundError(
            "Could not find the VITRA model config. Set model_config_path in deploy.yml "
            f"or place config.json beside the run/checkpoint (weights={weights})."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        saved_config = json.load(handle)

    explicit_stats = _as_optional_path(
        model_cfg.get("statistics_path")
        or model_cfg.get("stats_path")
        or model_cfg.get("norm_stats_path")
    )
    configured_stats = _as_optional_path(saved_config.get("statistics_path"), base=config_path.parent)
    stats_path = _first_existing(
        [explicit_stats, configured_stats]
        + [
            directory / name
            for directory in search_dirs
            for name in ("teledata_statistics.json", "dataset_statistics.json")
        ]
    )
    if stats_path is None:
        raise FileNotFoundError(
            "Could not find VITRA native-space normalization statistics. Set "
            "statistics_path in deploy.yml or place teledata_statistics.json beside the run."
        )
    return config_path, stats_path


def _validate_statistics_identity(
    model_cfg: dict[str, Any],
    saved_config: dict[str, Any],
    config_path: Path,
    statistics_path: Path,
    representation: str,
) -> str:
    """Bind deployment statistics to the immutable training-run identity."""

    actual_sha256 = _sha256_file(statistics_path)
    saved_sha256 = saved_config.get("statistics_sha256")
    asserted_sha256 = model_cfg.get("statistics_sha256")
    for label, expected in (
        ("saved config statistics_sha256", saved_sha256),
        ("deployment statistics_sha256", asserted_sha256),
    ):
        if expected in (None, ""):
            continue
        expected = str(expected)
        if not _SHA256_RE.fullmatch(expected):
            raise ValueError(f"{label} is not a SHA-256 digest: {expected!r}")
        if actual_sha256.lower() != expected.lower():
            raise ValueError(
                f"Statistics identity mismatch for {statistics_path}: "
                f"actual_sha256={actual_sha256}, {label}={expected}"
            )

    explicit_value = (
        model_cfg.get("statistics_path")
        or model_cfg.get("stats_path")
        or model_cfg.get("norm_stats_path")
    )
    if explicit_value not in (None, "") and saved_sha256 in (None, ""):
        saved_path = _as_optional_path(
            saved_config.get("statistics_path"), base=config_path.parent
        )
        same_file = False
        if saved_path is not None and saved_path.is_file():
            try:
                same_file = statistics_path.samefile(saved_path)
            except OSError:
                same_file = False
        if not same_file:
            raise ValueError(
                "statistics_path is an assertion, not an override. The saved config has "
                "no statistics_sha256 and the explicit file is not the saved statistics file."
            )
    if representation == "mano45" and saved_sha256 in (None, ""):
        raise ValueError(
            "MANO45 checkpoint config must contain statistics_sha256; refusing an "
            "unbound normalization file"
        )
    return actual_sha256


def _call_supported(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Pass optional adapter metadata only when the shared function accepts it."""
    signature = inspect.signature(function)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **kwargs)


def _extract_head_camera(obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    vision = obs.get("vision", obs)
    for camera_name in (
        "cam_head",
        "head_camera",
        "camera_head",
        "cam_high",
        "top_camera",
        "head",
    ):
        if camera_name not in vision:
            continue
        camera = vision[camera_name]
        camera_meta = camera if isinstance(camera, dict) else {}
        image: Any = camera
        if isinstance(camera, dict):
            for image_key in ("color", "rgb", "image", "colors"):
                if image_key in camera:
                    image = camera[image_key]
                    break
            else:
                raise KeyError(f"Camera {camera_name!r} has no decoded RGB image field")

        # The policy server has already decoded encoded image bits.  Rejecting
        # them here catches a broken boundary without silently introducing a
        # second decoder or an RGB/BGR conversion.
        if isinstance(image, (bytes, bytearray, memoryview)):
            raise TypeError("VITRA model.py received encoded image bytes; server-side RGB decoding is required")
        image_np = np.asarray(image)
        if image_np.ndim != 3:
            raise ValueError(f"Head RGB image must be HWC, got shape {image_np.shape}")
        if image_np.shape[-1] not in (1, 3) and image_np.shape[0] in (1, 3):
            image_np = np.transpose(image_np, (1, 2, 0))
        if image_np.shape[-1] != 3:
            raise ValueError(f"Head RGB image must have three channels, got shape {image_np.shape}")
        return np.ascontiguousarray(image_np.astype(np.uint8, copy=False)), camera_meta
    raise KeyError(
        "Missing head RGB camera; tried cam_head/head_camera/camera_head/cam_high/top_camera/head"
    )


def _camera_metadata_keys(camera: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    extrinsics = [
        key
        for key in ("extrinsics", "extrinsic", "extrinsics_matrix", "extrinsic_matrix")
        if key in camera
    ]
    intrinsics = [
        key
        for key in ("intrinsics", "intrinsic", "intrinsics_matrix", "intrinsic_matrix")
        if key in camera
    ]
    return extrinsics, intrinsics


def _validate_runtime_camera_metadata(
    camera: Mapping[str, Any], image: np.ndarray
) -> bool:
    """Validate supplied calibration; return False only when both fields absent."""
    extrinsic_keys, intrinsic_keys = _camera_metadata_keys(camera)
    if not extrinsic_keys and not intrinsic_keys:
        return False
    if len(extrinsic_keys) != 1 or len(intrinsic_keys) != 1:
        raise ValueError(
            "Runtime cam_head must provide exactly one extrinsics and one intrinsics field"
        )
    height, width = (int(image.shape[0]), int(image.shape[1]))
    shape_value = camera.get("shape")
    if shape_value is not None:
        shape = np.asarray(shape_value).reshape(-1)
        if shape.size != 2 or tuple(int(v) for v in shape) != (height, width):
            raise ValueError(
                f"Runtime cam_head shape metadata disagrees with RGB image: {shape_value!r} vs {(height, width)}"
            )
    frame = camera.get("frame", camera.get("extrinsics_frame"))
    if frame not in (None, "camera_to_env"):
        raise ValueError(
            f"Runtime cam_head extrinsics must use frame='camera_to_env', got {frame!r}"
        )
    axes = camera.get("camera_axes", camera.get("extrinsics_axes"))
    if axes not in (None, "x_right_y_up_z_back"):
        raise ValueError(
            "Runtime cam_head extrinsics axes do not match the audited Spark0 convention"
        )
    _validate_camera_extrinsics(camera[extrinsic_keys[0]], "Runtime cam_head extrinsics")
    _validate_camera_intrinsics(
        camera[intrinsic_keys[0]],
        "Runtime cam_head intrinsics",
        image_height=height,
        image_width=width,
    )
    return True


def _head_camera_key(vision: Mapping[str, Any]) -> str:
    for key in (
        "cam_head",
        "head_camera",
        "camera_head",
        "cam_high",
        "top_camera",
        "head",
    ):
        if key in vision:
            return key
    raise KeyError("VITRA observation is missing a head camera entry")


def _stretch_rgb_hw(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Stretch-resize RGB HWC to the EgoVLA training size. Channel order is unchanged."""
    current = np.ascontiguousarray(image)
    if current.shape[0] == height and current.shape[1] == width:
        return current
    import cv2

    resized = cv2.resize(
        current, (int(width), int(height)), interpolation=cv2.INTER_AREA
    )
    if resized.ndim != 3 or resized.shape[-1] != 3:
        raise ValueError(
            f"H1 policy resize must keep RGB HWC, got shape {getattr(resized, 'shape', None)}"
        )
    return np.ascontiguousarray(resized.astype(np.uint8, copy=False))


def _copy_observation_with_head_image(
    observation: Mapping[str, Any],
    image: np.ndarray,
    *,
    camera_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    vision = observation.get("vision")
    if not isinstance(vision, Mapping):
        raise KeyError("VITRA observation must contain a vision mapping")
    head_key = _head_camera_key(vision)
    head_value = vision[head_key]
    head = dict(head_value) if isinstance(head_value, Mapping) else {"color": image}
    head["color"] = image
    head["shape"] = np.asarray([int(image.shape[0]), int(image.shape[1])], dtype=np.int32)
    if camera_updates:
        head.update(camera_updates)
    copied = dict(observation)
    copied_vision = dict(vision)
    copied_vision[head_key] = head
    copied["vision"] = copied_vision
    return copied


def _scale_runtime_intrinsics(
    camera: Mapping[str, Any],
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> np.ndarray:
    _, intrinsic_keys = _camera_metadata_keys(camera)
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    current = _validate_camera_intrinsics(
        camera[intrinsic_keys[0]],
        "Runtime cam_head intrinsics",
        image_height=src_h,
        image_width=src_w,
    )
    scaled = np.diag(
        [dst_w / float(src_w), dst_h / float(src_h), 1.0]
    ) @ np.asarray(current, dtype=np.float64)
    return _validate_camera_intrinsics(
        scaled,
        "H1 policy-resize intrinsics",
        image_height=dst_h,
        image_width=dst_w,
    ).astype(np.float32)


def _prepare_h1_camera_observation(
    observation: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    *,
    allow_fallback: bool = True,
    eval_env_mode: str = "sim",
) -> dict[str, Any]:
    """Attach audited H1 camera metadata and stretch live RGB to training size.

    The common EgoVLA bridge intentionally remains policy-neutral and currently
    forwards only decoded images. This function is therefore a narrow VITRA
    boundary shim: supplied metadata is validated and preserved when the image
    is already the training size; 1280x720 Isaac frames are stretch-resized to
    the 384x384 HDF5 training size. A completely absent calibration pair is
    synthesized from the hash-locked profile after that resize.
    """
    if not isinstance(observation, Mapping):
        raise TypeError("VITRA observation must be a mapping")
    working: Mapping[str, Any] = observation
    image, camera = _extract_head_camera(dict(working))
    if calibration is not None:
        policy_hw = calibration.get("policy_hw")
        src_hw = (int(image.shape[0]), int(image.shape[1]))
        allowed_sizes = calibration.get("allowed_sizes") or frozenset()
        if (
            isinstance(policy_hw, tuple)
            and len(policy_hw) == 2
            and src_hw != policy_hw
            and src_hw in allowed_sizes
        ):
            has_runtime_calibration = _validate_runtime_camera_metadata(camera, image)
            resized = _stretch_rgb_hw(image, int(policy_hw[0]), int(policy_hw[1]))
            updates: dict[str, Any] = {"calibration_resize_mode": "stretch"}
            if has_runtime_calibration:
                updates["intrinsics"] = _scale_runtime_intrinsics(
                    camera, src_hw=src_hw, dst_hw=policy_hw
                )
            working = _copy_observation_with_head_image(
                working, resized, camera_updates=updates
            )
            image, camera = _extract_head_camera(dict(working))
    if _validate_runtime_camera_metadata(camera, image):
        return dict(working)

    if not allow_fallback:
        raise ValueError(
            "H1 camera calibration fallback is disabled for "
            f"EVAL_ENV_TYPE={eval_env_mode!r}; runtime cam_head must provide "
            "validated intrinsics and extrinsics (real calibration required)"
        )
    if calibration is None:
        raise ValueError(
            "H1 camera observation has no runtime calibration and no simulator "
            "fallback profile is configured"
        )

    height, width = int(image.shape[0]), int(image.shape[1])
    allowed_sizes = calibration["allowed_sizes"]
    if (height, width) not in allowed_sizes:
        allowed_text = ", ".join(
            f"{h}x{w}" for h, w in sorted(allowed_sizes)
        )
        raise ValueError(
            "H1 camera calibration fallback refuses unknown RGB size "
            f"{height}x{width}; allowed sizes: {allowed_text}"
        )
    source_height = int(calibration["source_height"])
    source_width = int(calibration["source_width"])
    scale_x = width / float(source_width)
    scale_y = height / float(source_height)
    source_k = np.asarray(calibration["source_intrinsics"], dtype=np.float64)
    scaled_k = np.diag([scale_x, scale_y, 1.0]) @ source_k
    scaled_k = _validate_camera_intrinsics(
        scaled_k,
        "H1 fallback intrinsics",
        image_height=height,
        image_width=width,
    ).astype(np.float32)
    source_pose = np.asarray(calibration["source_extrinsics"], dtype=np.float64)
    # Revalidate before copying so a mutated in-memory calibration cannot cross
    # the model boundary even when the artifact was verified at construction.
    source_pose = _validate_camera_extrinsics(source_pose, "H1 fallback extrinsics")

    return _copy_observation_with_head_image(
        working,
        image,
        camera_updates={
            "intrinsics": scaled_k,
            "extrinsics": source_pose.astype(np.float32, copy=True),
            "frame": "camera_to_env",
            "camera_axes": "x_right_y_up_z_back",
            "calibration_profile": calibration["profile"],
            "calibration_sha256": calibration["sha256"],
            "calibration_resize_mode": "stretch",
        },
    )


def _extract_instruction(obs: dict[str, Any], default_prompt: str) -> str:
    value = obs.get("instruction", obs.get("instructions", default_prompt))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default_prompt
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value or default_prompt).strip()
    if "Left hand:" in text and "Right hand:" in text:
        return text
    return f"Left hand: {text} Right hand: {text}"


def _extract_fov(image: np.ndarray, camera: dict[str, Any], fallback: np.ndarray) -> np.ndarray:
    value = None
    for key in ("intrinsic_matrix", "intrinsics", "K"):
        if key in camera:
            value = camera[key]
            break
    if value is None:
        result = np.asarray(fallback, dtype=np.float32).reshape(-1)
        if result.size != 2:
            raise ValueError(f"default_fov must contain [horizontal, vertical], got {result.shape}")
        return result.copy()

    intrinsics = np.asarray(value, dtype=np.float32)
    # Runtime camera buffers may retain a leading history/environment axis.
    # Select the latest matrix/vector before shape validation; flattening a
    # 3x3 K and reshaping it to FOV(2) was the previous failure mode.
    if intrinsics.ndim >= 3 and intrinsics.shape[-2:] == (3, 3):
        intrinsics = intrinsics.reshape(-1, 3, 3)[-1]
    elif intrinsics.ndim >= 2 and intrinsics.shape[-1] == 4:
        intrinsics = intrinsics.reshape(-1, 4)[-1]
    if intrinsics.shape == (4,):
        fx, fy, cx, cy = intrinsics.tolist()
        intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Head-camera intrinsics must be (3,3) or [fx,fy,cx,cy], got {intrinsics.shape}")
    height, width = image.shape[:2]
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("Head-camera fx/fy must be positive")
    return np.asarray(
        [
            2.0 * np.arctan(width / (2.0 * intrinsics[0, 0])),
            2.0 * np.arctan(height / (2.0 * intrinsics[1, 1])),
        ],
        dtype=np.float32,
    )


def _normalize_env_idx_list(value: Any, fallback: list[int]) -> list[int]:
    if value is None:
        return list(fallback)
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1).tolist()]
    return [int(item) for item in value]


def _latest_runtime_vector(value: Any, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (dimension,):
        result = array
    elif array.ndim >= 2 and array.shape[-1] == dimension:
        result = array.reshape(-1, dimension)[-1]
    else:
        raise ValueError(f"{name} must end in dimension {dimension}, got {array.shape}")
    result = np.asarray(result, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _resolve_data_representation(
    model_cfg: dict[str, Any], saved_config: dict[str, Any]
) -> str:
    """Resolve representation from the checkpoint config, rejecting overrides.

    Old released Wuji checkpoints predate ``data_representation`` and retain
    their historical Wuji20 behavior.  A MANO45 run must declare itself in its
    saved config; a deployment override may confirm but never change that
    declaration.
    """

    saved = saved_config.get("data_representation")
    if saved is None and isinstance(saved_config.get("train_dataset"), dict):
        saved = saved_config["train_dataset"].get("representation")
    override = model_cfg.get("data_representation")
    if saved is None:
        if override not in (None, "", "wuji20"):
            raise ValueError(
                "Cannot deploy a legacy checkpoint as MANO45: its saved config has no "
                "data_representation declaration"
            )
        return "wuji20"
    saved = str(saved)
    if saved not in ("wuji20", "mano45", "inspire12"):
        raise ValueError(f"Unsupported saved data_representation={saved!r}")
    if override not in (None, "") and str(override) != saved:
        raise ValueError(
            "Deployment data_representation disagrees with the checkpoint: "
            f"override={override!r}, saved={saved!r}"
        )
    return saved


def _resolve_mano_inverse_settings(
    model_cfg: Mapping[str, Any],
    saved_config: Mapping[str, Any],
    *,
    saved_config_path: Path,
) -> tuple[str, Optional[Path], Optional[str]]:
    """Resolve an explicit MANO inverse A/B variant without changing old runs.

    Deployment values win when non-empty; otherwise the immutable run config is
    used.  Runs that predate this field remain ``geometric``.  A linear variant
    is never inferred from the presence of a bundled artifact: it must name and
    hash-lock its artifact explicitly.
    """

    deployed_mode = model_cfg.get("mano_inverse_mode")
    saved_mode = saved_config.get("mano_inverse_mode")
    mode = str(
        deployed_mode
        if deployed_mode not in (None, "")
        else saved_mode
        if saved_mode not in (None, "")
        else "geometric"
    )
    if mode not in ("geometric", "linear"):
        raise ValueError(
            "mano_inverse_mode must be 'geometric' or 'linear', got "
            f"{mode!r}"
        )

    deployed_path = model_cfg.get("mano_linear_inverse_path")
    saved_path = saved_config.get("mano_linear_inverse_path")
    if deployed_path not in (None, ""):
        artifact_path = _as_optional_path(deployed_path)
    elif saved_path not in (None, ""):
        artifact_path = _as_optional_path(saved_path, base=saved_config_path.parent)
    else:
        artifact_path = None

    deployed_sha = model_cfg.get("mano_linear_inverse_sha256")
    saved_sha = saved_config.get("mano_linear_inverse_sha256")
    artifact_sha = (
        str(deployed_sha)
        if deployed_sha not in (None, "")
        else str(saved_sha)
        if saved_sha not in (None, "")
        else None
    )
    if mode == "linear":
        if artifact_path is None or artifact_sha is None:
            raise ValueError(
                "mano_inverse_mode='linear' requires explicit "
                "mano_linear_inverse_path and mano_linear_inverse_sha256"
            )
        if not _SHA256_RE.fullmatch(artifact_sha):
            raise ValueError(
                "mano_linear_inverse_sha256 is not a 64-character SHA-256 digest"
            )
    return mode, artifact_path, artifact_sha


def _resolve_h1_inverse_settings(
    model_cfg: Mapping[str, Any],
    saved_config: Mapping[str, Any],
    *,
    saved_config_path: Path,
) -> tuple[str, Optional[Path], Optional[str]]:
    """Resolve the optional run-specific Wuji20 -> Inspire6 calibration.

    The setting is intentionally H1-only and opt-in.  Older DexBench/Wuji
    runs have no such fields and remain on the historical geometric adapter.
    Deployment values take precedence over the immutable run config; a linear
    fallback always requires an explicit path and SHA-256.
    """

    deployed_mode = model_cfg.get("h1_inverse_mode")
    saved_mode = saved_config.get("h1_inverse_mode")
    mode = str(
        deployed_mode
        if deployed_mode not in (None, "")
        else saved_mode
        if saved_mode not in (None, "")
        else "geometric"
    ).strip().lower()
    if mode in {"linear", "linear_fallback", "calibrated_linear_fallback"}:
        mode = "linear_fallback"
    elif mode in {"geometric", ""}:
        mode = "geometric"
    else:
        raise ValueError(
            "h1_inverse_mode must be 'geometric' or 'linear_fallback', got "
            f"{mode!r}"
        )

    deployed_path = model_cfg.get("h1_linear_inverse_path")
    saved_path = saved_config.get("h1_linear_inverse_path")
    if deployed_path not in (None, ""):
        path = _as_optional_path(deployed_path)
    elif saved_path not in (None, ""):
        path = _as_optional_path(saved_path, base=saved_config_path.parent)
    else:
        path = None
    deployed_sha = model_cfg.get("h1_linear_inverse_sha256")
    saved_sha = saved_config.get("h1_linear_inverse_sha256")
    sha = (
        str(deployed_sha)
        if deployed_sha not in (None, "")
        else str(saved_sha)
        if saved_sha not in (None, "")
        else None
    )
    if mode == "linear_fallback":
        if path is None or sha is None:
            raise ValueError(
                "h1_inverse_mode='linear_fallback' requires explicit "
                "h1_linear_inverse_path and h1_linear_inverse_sha256"
            )
        if not _SHA256_RE.fullmatch(sha):
            raise ValueError("h1_linear_inverse_sha256 is not a 64-character SHA-256 digest")
    return mode, path, sha


class Model(ModelTemplate):
    """VITRA policy-server model implementing the XPolicyLab ModelTemplate."""

    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = str(model_cfg["action_type"])
        self.env_cfg_type = str(model_cfg["env_cfg_type"])
        if self.action_type != "ee":
            raise ValueError(
                "VITRA's XPolicyLab boundary is action_type='ee'. Its internal wrist "
                "prediction is a step delta that this adapter reconstructs to absolute EE poses."
            )

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        arm_dims = list(self.robot_action_dim_info["arm_dim"])
        ee_dims = list(self.robot_action_dim_info["ee_dim"])
        if len(arm_dims) != 2 or len(ee_dims) != 2 or any(dim <= 0 for dim in (*arm_dims, *ee_dims)):
            raise ValueError(
                "VITRA Spark0 integration requires a valid dual-arm/dual-hand robot registration; "
                f"got arm_dim={arm_dims}, ee_dim={ee_dims} for env_cfg_type={self.env_cfg_type!r}."
            )
        self.arm_dims = arm_dims
        # ``ee_dims`` is the external XPolicyLab contract.  The immutable
        # VITRA checkpoint still consumes/produces internal Wuji20 values;
        # H1 Inspire12 is converted only at this policy boundary.
        self.ee_dims = ee_dims
        self._internal_ee_dims = [20, 20]
        if self.env_cfg_type == "ego_h1_inspire":
            if self.ee_dims != [12, 12]:
                raise ValueError(
                    "ego_h1_inspire must register external ee_dim=[12,12], "
                    f"got {self.ee_dims}"
                )
        elif self.ee_dims != self._internal_ee_dims:
            raise ValueError(
                "VITRA Spark0 requires the dual Wuji Hand2 contract for non-H1 "
                f"robots: ee_dim=[20,20], got {self.ee_dims}"
            )

        self._h1_camera_calibration: dict[str, Any] | None = None
        self._h1_camera_env_mode = "sim"
        self._h1_camera_calibration_scope = "runtime"
        self._h1_camera_fallback_allowed = False
        if self.env_cfg_type == "ego_h1_inspire":
            (
                self._h1_camera_env_mode,
                self._h1_camera_calibration_scope,
                self._h1_camera_fallback_allowed,
            ) = _resolve_h1_camera_fallback_policy(model_cfg)
            calibration_path = model_cfg.get("h1_camera_calibration_path")
            if calibration_path in (None, ""):
                calibration_path = os.environ.get("VITRA_H1_CAMERA_CALIBRATION_PATH")
            if calibration_path in (None, ""):
                calibration_path = _DEFAULT_H1_CAMERA_CALIBRATION_PATH
            calibration_sha256 = model_cfg.get("h1_camera_calibration_sha256")
            if calibration_sha256 in (None, ""):
                calibration_sha256 = os.environ.get("VITRA_H1_CAMERA_CALIBRATION_SHA256")
            if calibration_sha256 in (None, ""):
                # This fallback is only for the bundled path. A caller-supplied
                # path still has to provide its own digest and will fail closed
                # against this value if it omits one.
                calibration_sha256 = _BUNDLED_H1_CAMERA_CALIBRATION_SHA256
            self._h1_camera_calibration = _load_h1_camera_calibration(
                calibration_path, calibration_sha256
            )

        if not _VITRA_ROOT.is_dir():
            raise FileNotFoundError(f"Missing official VITRA checkout: {_VITRA_ROOT}")
        if str(_VITRA_ROOT) not in sys.path:
            sys.path.insert(0, str(_VITRA_ROOT))

        weights, checkpoint_root = _resolve_weights(self.model_cfg)
        config_path, statistics_path = _resolve_config_and_statistics(
            self.model_cfg, weights, checkpoint_root
        )
        with config_path.open("r", encoding="utf-8") as handle:
            configs = json.load(handle)
        shared_path_changes = _rewrite_saved_config_shared_paths(configs)
        if shared_path_changes:
            print(
                "[VITRA] resolved shared-storage path aliases",
                ", ".join(
                    f"{key}: {source} -> {target}"
                    for key, source, target in shared_path_changes
                ),
            )
        self.representation = _resolve_data_representation(self.model_cfg, configs)
        if self.representation == "mano45" and model_cfg.get("mapping_path") not in (None, ""):
            raise ValueError(
                "MANO45 must not receive mapping_path; the legacy sparse Wuji20 JSON "
                "mapping is incompatible with the fitted MANO codec"
            )
        statistics_sha256 = _validate_statistics_identity(
            self.model_cfg,
            configs,
            config_path,
            statistics_path,
            self.representation,
        )
        configs["model_load_path"] = str(weights)
        configs["statistics_path"] = str(statistics_path)

        if self.representation == "wuji20":
            self.mapping_path = (
                _as_optional_path(model_cfg.get("mapping_path")) or _DEFAULT_MAPPING_PATH
            )
        elif self.representation == "inspire12":
            self.mapping_path = _as_optional_path(model_cfg.get("mapping_path")) or (
                _POLICY_DIR / "mapping_inspire12_mano45.json"
            )
        else:
            self.mapping_path = None
        self.default_prompt = str(
            model_cfg.get("prompt") or "Perform the instructed bimanual manipulation task."
        )
        default_fov = model_cfg.get("default_fov", [1.0, 0.8])
        self.default_fov = np.asarray(default_fov, dtype=np.float32).reshape(2)
        self.num_ddim_steps = int(model_cfg.get("num_ddim_steps", 10))
        self.cfg_scale = float(model_cfg.get("cfg_scale", 5.0))
        self.sample_times = int(model_cfg.get("sample_times", 1))
        if self.cfg_scale <= 1.0:
            raise ValueError("Official VITRA inference requires cfg_scale > 1.0")
        if self.sample_times != 1:
            raise ValueError("VITRA XPolicyLab adapter currently requires sample_times=1")

        # Import only after the official checkout is on sys.path.  Keeping these
        # dependencies out of module import makes adapter discovery lightweight.
        import torch
        from vitra.datasets.spark0_dataset import (
            REPRESENTATION_MANO45,
            REPRESENTATION_WUJI20,
            _validate_statistics_contract,
            human_action_to_native,
            human_action_to_mano,
            mano_action_chunk_to_xpolicy,
            mano_state_to_human,
            native_action_chunk_to_xpolicy,
            native_state_to_human,
            pose7_to_matrix,
            runtime_observation_to_mano,
            runtime_observation_to_native,
        )
        from vitra.models import load_model
        from vitra.utils.data_utils import load_normalizer, read_dataset_statistics

        if not torch.cuda.is_available():
            raise RuntimeError("VITRA inference requires a CUDA GPU")
        # The evaluator masks CUDA_VISIBLE_DEVICES to the requested policy GPU;
        # inside that process the selected physical card is always ordinal 0.
        torch.cuda.set_device(0)
        self._torch = torch
        self._pose7_to_matrix = pose7_to_matrix
        statistics = read_dataset_statistics(str(statistics_path))
        if self.representation == "inspire12":
            from vitra.datasets.egovla_inspire_dataset import (
                _validate_statistics_contract as _validate_inspire_statistics_contract,
            )

            _validate_inspire_statistics_contract(
                str(statistics_path), self.representation, statistics
            )
        else:
            _validate_statistics_contract(
                str(statistics_path), self.representation, statistics
            )
        self._mano_codec = None
        if self.representation == REPRESENTATION_WUJI20:
            self._runtime_observation_to_native = runtime_observation_to_native
            self._native_state_to_human = native_state_to_human
            self._human_action_to_native = human_action_to_native
            self._native_action_chunk_to_xpolicy = native_action_chunk_to_xpolicy
        elif self.representation == REPRESENTATION_MANO45:
            mano_tools_root = _as_optional_path(model_cfg.get("mano_tools_root"))
            if mano_tools_root is None:
                raise ValueError(
                    "MANO45 checkpoint requires mano_tools_root pointing to the "
                    "Spark-0 retargeting checkout"
                )
            from XPolicyLab.policy.VITRA.mano45_runtime import Mano45RuntimeCodec

            inverse_mode, linear_inverse_path, asserted_inverse_sha256 = (
                _resolve_mano_inverse_settings(
                    model_cfg,
                    configs,
                    saved_config_path=config_path,
                )
            )

            self._mano_codec = Mano45RuntimeCodec(
                mano_tools_root,
                inverse_max_iterations=int(
                    model_cfg.get("mano_inverse_max_iterations", 12)
                ),
                inverse_timeout_s=float(
                    model_cfg.get("mano_inverse_timeout_s", 2.0)
                ),
                inverse_mode=inverse_mode,
                linear_inverse_path=linear_inverse_path,
                linear_inverse_sha256=asserted_inverse_sha256,
            )
            self._runtime_observation_to_native = runtime_observation_to_mano
            self._native_state_to_human = mano_state_to_human
            self._human_action_to_native = human_action_to_mano
            self._native_action_chunk_to_xpolicy = mano_action_chunk_to_xpolicy
        elif self.representation == "inspire12":
            from vitra.datasets.egovla_inspire_dataset import (
                human_action_to_inspire,
                inspire_action_chunk_to_xpolicy,
                inspire_state_to_human,
                runtime_observation_to_inspire,
            )

            self._internal_ee_dims = [12, 12]
            self._runtime_observation_to_native = runtime_observation_to_inspire
            self._native_state_to_human = inspire_state_to_human
            self._human_action_to_native = human_action_to_inspire
            self._native_action_chunk_to_xpolicy = inspire_action_chunk_to_xpolicy
        else:  # guarded by _resolve_data_representation
            raise AssertionError(f"Unhandled VITRA representation {self.representation!r}")

        # EgoVLA's H1 environment exposes six actuated Inspire joints plus six
        # mimic joints per hand.  Keep VITRA's internal Wuji20/MANO45 ABI
        # untouched and bridge the external 12-D fields with the verified
        # Spark-0 kinematic chain at the policy boundary.  inspire12 already
        # trains and decodes those 12 channels directly, so no Wuji IK.
        self._h1_adapter = None
        if self.env_cfg_type == "ego_h1_inspire" and self.representation != "inspire12":
            h1_root_value = model_cfg.get("h1_retarget_root")
            if h1_root_value in (None, ""):
                h1_root_value = os.environ.get("VITRA_H1_RETARGET_ROOT")
            h1_root = _as_optional_path(h1_root_value)
            if h1_root is None:
                raise ValueError(
                    "ego_h1_inspire requires an explicit h1_retarget_root (or "
                    "VITRA_H1_RETARGET_ROOT) containing the verified Spark-0 "
                    "Inspire12<->Wuji20 conversion chain"
                )
            from XPolicyLab.policy.VITRA.h1_inspire_retarget import (
                H1InspireWujiAdapter,
            )

            h1_inverse_mode, h1_linear_path, h1_linear_sha256 = (
                _resolve_h1_inverse_settings(
                    model_cfg,
                    configs,
                    saved_config_path=config_path,
                )
            )
            h1_linear_provenance: dict[str, Any] | None = None
            if h1_inverse_mode == "linear_fallback":
                manifest_sha = configs.get("data_manifest_sha256")
                if manifest_sha in (None, ""):
                    raise ValueError(
                        "H1 linear calibration requires the 0902 data_manifest_sha256"
                    )
                h1_linear_provenance = {
                    "statistics_sha256": statistics_sha256,
                    "manifest_sha256": str(manifest_sha),
                    "config_sha256": _sha256_file(config_path),
                    "representation": "mano45",
                    "normalization": True,
                    "runtime_action_type": "ee",
                    "model_chunk": int(configs.get("fwd_pred_next_n", self.model_cfg.get("execute_action_chunk", 16))),
                }

            self._h1_adapter = H1InspireWujiAdapter(
                h1_root,
                solver=str(model_cfg.get("h1_retarget_solver", "analytic")),
                max_iterations=int(model_cfg.get("h1_retarget_max_iterations", 35)),
                tolerance_m=float(model_cfg.get("h1_retarget_tolerance_m", 0.03)),
                inverse_max_rms_m=float(
                    model_cfg.get("h1_inverse_max_rms_m", 0.08)
                ),
                mimic_tolerance_rad=float(
                    model_cfg.get("h1_mimic_tolerance_rad", 2e-5)
                ),
                linear_inverse_path=h1_linear_path
                if h1_inverse_mode == "linear_fallback"
                else None,
                linear_inverse_sha256=h1_linear_sha256
                if h1_inverse_mode == "linear_fallback"
                else None,
                linear_inverse_provenance=h1_linear_provenance,
            )
            # The same checkpoint is mounted as /personal on the Web host and
            # /mnt/xspark-data on A800 nodes. Compare filesystem identities.
            paired_binding = next((
                binding for path, binding in (model_cfg.get("h1_paired12_by_checkpoint") or {}).items()
                if Path(path).resolve() == weights.resolve()
            ), {})
            paired_path = model_cfg.get("h1_paired12_path") or os.environ.get("VITRA_H1_PAIRED12_PATH") or paired_binding.get("path")
            if paired_path:
                from XPolicyLab.policy.VITRA.h1_paired12_inverse import Paired12Adapter, Paired12Decoder
                paired_sha = model_cfg.get("h1_paired12_sha256") or os.environ.get("VITRA_H1_PAIRED12_SHA256") or paired_binding.get("sha256")
                paired_decoder = Paired12Decoder(paired_path, paired_sha, {
                    "config_sha256": _sha256_file(config_path),
                    "statistics_sha256": statistics_sha256,
                    "manifest_sha256": str(configs.get("data_manifest_sha256")),
                    "checkpoint_weights_path": str(weights.resolve()),
                    "representation": self.representation,
                    "action_type": "ee",
                    "model_chunk": int(configs.get("fwd_pred_next_n", 16)),
                    "state_action_per_hand": [61, 51],
                })
                self._h1_adapter = Paired12Adapter(self._h1_adapter, paired_decoder)
                print(f"[VITRA] explicit paired12 conditional decoder path={paired_path} sha256={paired_sha}; no exact recovery of discarded joints is claimed", flush=True)
        self.normalizer = load_normalizer(configs)
        self.model = load_model(configs).cuda().eval()

        self.chunk_size = int(getattr(self.model, "chunk_size", configs.get("fwd_pred_next_n", 16)))
        self.execute_action_chunk = int(model_cfg.get("execute_action_chunk", self.chunk_size))
        if not 1 <= self.execute_action_chunk <= self.chunk_size:
            raise ValueError(
                f"execute_action_chunk must be in [1, {self.chunk_size}], got {self.execute_action_chunk}"
            )

        self._observations: dict[int, dict[str, Any]] = {}
        self._raw_observations: dict[int, dict[str, Any]] = {}
        self._mano_fit_previous: dict[int, dict[str, np.ndarray]] = {}
        self._mano_pending_keyframes: dict[
            int, list[tuple[int, dict[str, np.ndarray]]]
        ] = {}
        self._mano_next_tick: dict[int, int] = {}
        self._mano_warned_odd_query: set[int] = set()
        self._latest_env_idx_list: list[int] = [0]
        print(
            "[VITRA] initialized",
            f"weights={weights}",
            f"statistics={statistics_path}",
            f"statistics_sha256={statistics_sha256}",
            f"representation={self.representation}",
            f"chunk={self.chunk_size}",
            flush=True,
        )
        if self._mano_codec is not None:
            print(f"[VITRA] MANO45 codec {self._mano_codec.provenance_summary()}", flush=True)
            if self.execute_action_chunk % 2:
                raise ValueError(
                    "MANO45 execute_action_chunk must be even so every policy query "
                    "lands on an add_mano stride-2 fitted frame; got "
                    f"{self.execute_action_chunk}"
                )
        if self._h1_adapter is not None:
            print(
                "[VITRA] H1 external hand adapter",
                self._h1_adapter.provenance_summary(),
                flush=True,
            )
        if self._h1_camera_calibration is not None:
            print(
                "[VITRA] H1 camera calibration",
                f"profile={self._h1_camera_calibration['profile']}",
                f"sha256={self._h1_camera_calibration['sha256']}",
                f"path={self._h1_camera_calibration['path']}",
                f"env_mode={self._h1_camera_env_mode}",
                f"scope={self._h1_camera_calibration_scope}",
                f"fallback_allowed={self._h1_camera_fallback_allowed}",
                flush=True,
            )

    def _internalize_observation(
        self, obs: dict[str, Any], env_idx: int
    ) -> dict[str, Any]:
        """Convert external H1 Inspire12 state fields to internal Wuji20.

        Only hand joint-state fields are transformed.  EE pose, arm state,
        image, and instruction retain the common EgoVLA contract.  The input
        dictionary is never mutated, and a failed two-hand conversion rolls
        back the adapter's warm-start state.
        """

        adapter = getattr(self, "_h1_adapter", None)
        if adapter is None:
            return obs
        state = obs.get("state")
        if not isinstance(state, Mapping):
            raise KeyError("Runtime observation must contain a state mapping")
        snapshot = adapter.snapshot(env_idx)
        try:
            copied = dict(obs)
            copied_state = dict(state)
            for side in ("left", "right"):
                q12 = _latest_runtime_vector(
                    state[f"{side}_ee_joint_state"],
                    12,
                    f"state/{side}_ee_joint_state (ego_h1_inspire)",
                )
                q20 = adapter.h1_to_wuji(
                    q12, side=side, env_idx=env_idx
                )
                q20 = np.asarray(q20, dtype=np.float32).reshape(-1)
                if q20.size != 20:
                    raise ValueError(
                        f"H1 adapter returned {side} Wuji state with {q20.size} "
                        "values; expected exactly 20"
                    )
                copied_state[f"{side}_ee_joint_state"] = q20
            copied["state"] = copied_state
            return copied
        except Exception:
            adapter.restore(env_idx, snapshot)
            raise

    def _externalize_action_chunk(
        self, action_chunk: Any, env_idx: int
    ) -> list[dict[str, Any]]:
        """Convert internal Wuji20 hand commands to external Inspire12."""

        adapter = getattr(self, "_h1_adapter", None)
        if adapter is None:
            return action_chunk
        snapshot = adapter.snapshot(env_idx)
        try:
            result: list[dict[str, Any]] = []
            for step_index, action in enumerate(action_chunk):
                if not isinstance(action, Mapping):
                    raise TypeError(f"Action step {step_index} is not a mapping")
                copied = dict(action)
                for side in ("left", "right"):
                    q20 = _latest_runtime_vector(
                        action[f"{side}_ee_joint_state"],
                        20,
                        f"action/{side}_ee_joint_state (internal Wuji20)",
                    )
                    q12 = adapter.wuji_to_h1(
                        q20, side=side, env_idx=env_idx
                    )
                    q12 = np.asarray(q12, dtype=np.float32).reshape(-1)
                    if q12.size != 12:
                        raise ValueError(
                            f"H1 adapter returned {side} Inspire action with "
                            f"{q12.size} values; expected exactly 12"
                        )
                    copied[f"{side}_ee_joint_state"] = q12
                result.append(copied)
            return result
        except Exception:
            adapter.restore(env_idx, snapshot)
            raise

    def _encode_observation(self, obs: dict[str, Any], env_idx: int = 0) -> dict[str, Any]:
        image, camera = _extract_head_camera(obs)
        converted = _call_supported(
            self._runtime_observation_to_native,
            obs,
            mapping_path=self.mapping_path,
            codec=self._mano_codec,
            previous_fit=self._mano_fit_previous.get(env_idx),
        )
        if isinstance(converted, dict):
            if "native_state" not in converted:
                raise KeyError("runtime_observation_to_native() must return dict['native_state']")
            native_state = np.asarray(converted["native_state"], dtype=np.float32).reshape(-1)
            if self.representation == "mano45":
                fit_state = converted.get("fit_state")
                if not isinstance(fit_state, dict) or set(fit_state) != {"left", "right"}:
                    raise ValueError("MANO45 converter must return left/right fit_state")
                self._mano_fit_previous[env_idx] = {
                    # add_mano keeps the DLS warm-start in float64 until the
                    # final dataset write.  Preserve that solver state here;
                    # only the model-visible Euler features are float32.
                    side: np.asarray(fit_state[side], dtype=np.float64).reshape(45)
                    for side in ("left", "right")
                }
            context = converted.get("context", converted)
            image = np.asarray(converted.get("image", image), dtype=np.uint8)
            instruction = str(
                converted.get("instruction", _extract_instruction(obs, self.default_prompt))
            )
            fov = np.asarray(
                converted.get("fov", _extract_fov(image, camera, self.default_fov)),
                dtype=np.float32,
            ).reshape(-1)
            if fov.size != 2:
                raise ValueError(f"VITRA FOV must contain [horizontal, vertical], got {fov.shape}")
        else:
            native_state = np.asarray(converted, dtype=np.float32).reshape(-1)
            context = obs
            instruction = _extract_instruction(obs, self.default_prompt)
            fov = _extract_fov(image, camera, self.default_fov)

        expected_state_dim = int(np.asarray(self.normalizer.state_mean).size)
        if native_state.size != expected_state_dim:
            raise ValueError(
                f"Native VITRA state has {native_state.size} values, statistics expect {expected_state_dim}"
            )
        normalized_native = self.normalizer.normalize_state(native_state)
        converted_state = _call_supported(
            self._native_state_to_human,
            normalized_native,
            mapping=self.mapping_path,
            mapping_path=self.mapping_path,
            return_mask=True,
        )
        state_mask = None
        if isinstance(converted_state, tuple):
            human_state, state_mask = converted_state[:2]
        elif isinstance(converted_state, dict):
            human_state = converted_state.get("state", converted_state.get("human_state"))
            state_mask = converted_state.get("state_mask")
        else:
            human_state = converted_state
        human_state = np.asarray(human_state, dtype=np.float32).reshape(-1)
        if human_state.size != 212:
            raise ValueError(f"VITRA unified state must be 212-D, got {human_state.size}")
        if state_mask is None:
            state_mask = np.zeros(212, dtype=bool)
            state_mask[:102] = True
        else:
            state_mask = np.asarray(state_mask, dtype=bool).reshape(212)

        action_mask = np.zeros((self.chunk_size, 192), dtype=bool)
        action_mask[:, :102] = True
        return {
            # Server-decoded buffers can be read-only views. VITRA/Pillow may
            # retain the array during preprocessing, so give it an owned RGB
            # copy without changing channel order.
            "image": np.array(image, dtype=np.uint8, copy=True, order="C"),
            "instruction": instruction,
            "current_state": human_state,
            "current_state_mask": state_mask,
            "action_mask": action_mask,
            "fov": fov,
            "context": context,
            "env_idx": int(env_idx),
        }

    def _snapshot_mano_state(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        state = obs.get("state")
        if not isinstance(state, Mapping):
            raise KeyError("Runtime MANO observation must contain obs['state']")
        snapshot: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            snapshot[f"{side}_ee_pose"] = _latest_runtime_vector(
                state[f"{side}_ee_pose"], 7, f"state/{side}_ee_pose"
            ).copy()
            snapshot[f"{side}_ee_joint_state"] = _latest_runtime_vector(
                state[f"{side}_ee_joint_state"], 20, f"state/{side}_ee_joint_state"
            ).copy()
        return snapshot

    def _advance_mano_fit_history(
        self, env_idx: int, snapshot: Mapping[str, np.ndarray]
    ) -> None:
        if self._mano_codec is None:
            raise RuntimeError("MANO fit history used without a MANO45 codec")
        previous = self._mano_fit_previous.get(env_idx, {})
        current: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            fit = self._mano_codec.wuji_to_mano(
                side=side,
                q_stage_major=snapshot[f"{side}_ee_joint_state"],
                link7_in_env=self._pose7_to_matrix(snapshot[f"{side}_ee_pose"]),
                previous_hand_pose=previous.get(side),
            )
            current[side] = np.asarray(
                fit.hand_pose_rotvec, dtype=np.float64
            ).reshape(45)
        self._mano_fit_previous[env_idx] = current

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        self._latest_env_idx_list = [
            int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)
        ]
        calibration = getattr(self, "_h1_camera_calibration", None)
        prepared_obs_list = [
            _prepare_h1_camera_observation(
                obs,
                calibration,
                allow_fallback=getattr(self, "_h1_camera_fallback_allowed", False),
                eval_env_mode=getattr(self, "_h1_camera_env_mode", "sim"),
            )
            if getattr(self, "env_cfg_type", None) == "ego_h1_inspire"
            else obs
            for obs in obs_list
        ]
        internal_obs_list = [
            self._internalize_observation(obs, env_idx)
            for obs, env_idx in zip(prepared_obs_list, self._latest_env_idx_list)
        ]
        if self.representation == "mano45":
            # add_mano was generated with stride=2: frames 0,2,... are fitted
            # sequentially and odd frames are interpolated.  The eval loop
            # still sends frames 1..15 while executing a 16-step chunk.  Keep
            # compact even-frame state snapshots and replay those fits lazily
            # before the next query, reproducing the offline warm-start history
            # without retaining eight camera images per environment.
            for env_idx, obs in zip(self._latest_env_idx_list, internal_obs_list):
                tick = self._mano_next_tick.get(env_idx, 0)
                self._mano_next_tick[env_idx] = tick + 1
                self._raw_observations[env_idx] = obs
                if tick % 2 == 0:
                    self._mano_pending_keyframes.setdefault(env_idx, []).append(
                        (tick, self._snapshot_mano_state(obs))
                    )
                self._observations.pop(env_idx, None)
        else:
            self._observations = {
                env_idx: self._encode_observation(obs, env_idx)
                for env_idx, obs in zip(self._latest_env_idx_list, internal_obs_list)
            }

    def _predict_one(self, payload: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        torch = self._torch
        current_state = torch.from_numpy(payload["current_state"]).unsqueeze(0)
        current_state_mask = torch.from_numpy(payload["current_state_mask"]).unsqueeze(0)
        action_mask = torch.from_numpy(payload["action_mask"]).unsqueeze(0)
        fov = torch.from_numpy(payload["fov"]).unsqueeze(0)

        with torch.inference_mode():
            normalized_human_action = self.model.predict_action(
                image=payload["image"],
                instruction=payload["instruction"],
                current_state=current_state,
                current_state_mask=current_state_mask,
                action_mask_torch=action_mask,
                num_ddim_steps=self.num_ddim_steps,
                cfg_scale=self.cfg_scale,
                fov=fov,
                sample_times=self.sample_times,
            )
        normalized_human_action = np.asarray(normalized_human_action, dtype=np.float32)
        if normalized_human_action.ndim != 3 or normalized_human_action.shape[0] != 1:
            raise ValueError(
                "VITRA predict_action must return [1,T,192], got "
                f"{normalized_human_action.shape}"
            )
        normalized_native_action = _call_supported(
            self._human_action_to_native,
            normalized_human_action[0],
            mapping=self.mapping_path,
            mapping_path=self.mapping_path,
        )
        if isinstance(normalized_native_action, dict):
            normalized_native_action = normalized_native_action.get(
                "native_action", normalized_native_action.get("action")
            )
        normalized_native_action = np.asarray(normalized_native_action, dtype=np.float32)
        expected_action_dim = int(np.asarray(self.normalizer.action_mean).size)
        if normalized_native_action.ndim != 2 or normalized_native_action.shape[1] != expected_action_dim:
            raise ValueError(
                "Mapped native VITRA action must be [T,D] with "
                f"D={expected_action_dim}, got {normalized_native_action.shape}"
            )
        native_action = self.normalizer.unnormalize_action(normalized_native_action)
        action_chunk = _call_supported(
            self._native_action_chunk_to_xpolicy,
            native_action[: self.execute_action_chunk],
            payload["context"],
            mapping_path=self.mapping_path,
            codec=self._mano_codec,
        )
        action_chunk = self._externalize_action_chunk(
            action_chunk, int(payload.get("env_idx", 0))
        )
        return self._validate_action_chunk(action_chunk)

    def _validate_action_chunk(self, action_chunk: Any) -> list[dict[str, np.ndarray]]:
        if not isinstance(action_chunk, (list, tuple)) or not action_chunk:
            raise ValueError("native_action_chunk_to_xpolicy() must return a non-empty action list")
        expected = {
            "left_ee_pose": 7,
            "right_ee_pose": 7,
            "left_ee_joint_state": self.ee_dims[0],
            "right_ee_joint_state": self.ee_dims[1],
        }
        result: list[dict[str, np.ndarray]] = []
        for step_index, action in enumerate(action_chunk):
            if not isinstance(action, dict):
                raise TypeError(f"Action step {step_index} is not a dictionary")
            clean: dict[str, np.ndarray] = {}
            for key, dimension in expected.items():
                if key not in action:
                    raise KeyError(f"Action step {step_index} is missing {key!r}")
                value = np.asarray(action[key], dtype=np.float32).reshape(-1)
                if value.size != dimension or not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"Action {key} at step {step_index} must be finite {dimension}-D, got {value.shape}"
                    )
                clean[key] = value
            result.append(clean)
        return result

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if not self._latest_env_idx_list:
            raise RuntimeError("Call update_obs() before get_action()")
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list: Any = None) -> list[list[dict[str, np.ndarray]]]:
        indices = _normalize_env_idx_list(env_idx_list, self._latest_env_idx_list)
        result: list[list[dict[str, np.ndarray]]] = []
        # Official VITRA predict_action supports B=1 only, so batched XPolicyLab
        # evaluation intentionally runs one independent prediction per env.
        for env_idx in indices:
            if env_idx not in self._observations:
                if self.representation == "mano45" and env_idx in self._raw_observations:
                    latest_tick = self._mano_next_tick[env_idx] - 1
                    pending = self._mano_pending_keyframes.get(env_idx, [])
                    # The latest even keyframe is encoded below with its image
                    # and camera; replay only preceding keyframes here.
                    history = pending[:-1] if pending and pending[-1][0] == latest_tick else pending
                    previous_before = {
                        side: value.copy()
                        for side, value in self._mano_fit_previous.get(env_idx, {}).items()
                    }
                    had_previous = env_idx in self._mano_fit_previous
                    try:
                        for _tick, snapshot in history:
                            self._advance_mano_fit_history(env_idx, snapshot)
                        if latest_tick % 2:
                            raise RuntimeError(
                                "MANO45 policy query landed on an odd observation tick "
                                f"{latest_tick} for env_idx={env_idx}; this violates the "
                                "add_mano stride-2 runtime contract"
                            )
                        self._observations[env_idx] = self._encode_observation(
                            self._raw_observations[env_idx], env_idx
                        )
                    except Exception:
                        # Replaying a fit sequence is stateful.  A failed request
                        # must be retryable from the identical solver seed instead
                        # of advancing the history twice.
                        if had_previous:
                            self._mano_fit_previous[env_idx] = previous_before
                        else:
                            self._mano_fit_previous.pop(env_idx, None)
                        self._observations.pop(env_idx, None)
                        raise
                    self._mano_pending_keyframes[env_idx] = []
                else:
                    raise RuntimeError(
                        f"No buffered observation for env_idx={env_idx}; call update_obs_batch() first"
                    )
            result.append(self._predict_one(self._observations[env_idx]))
        return result

    def reset(self) -> None:
        self._observations.clear()
        self._raw_observations.clear()
        self._mano_fit_previous.clear()
        self._mano_pending_keyframes.clear()
        self._mano_next_tick.clear()
        self._mano_warned_odd_query.clear()
        self._latest_env_idx_list = [0]
        adapter = getattr(self, "_h1_adapter", None)
        if adapter is not None:
            adapter.reset()
        if hasattr(self.model, "reset"):
            self.model.reset()
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()
        print("[VITRA] reset", flush=True)
