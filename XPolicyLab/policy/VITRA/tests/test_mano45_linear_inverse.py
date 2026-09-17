from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.VITRA.mano45_linear_inverse import (
    LinearInverseResult,
    LinearInverseSafetyError,
    Mano45LinearInverse,
    WUJI_STAGE_MAJOR_ORDER,
)
from XPolicyLab.policy.VITRA.mano45_runtime import (
    Mano45RuntimeCodec,
    ManoRuntimeFitError,
)
import XPolicyLab.policy.VITRA.mano45_runtime as mano_runtime
from XPolicyLab.policy.VITRA.model import (
    _BUNDLED_MANO_LINEAR_INVERSE_PATH,
    _BUNDLED_MANO_LINEAR_INVERSE_SHA256,
    _resolve_mano_inverse_settings,
)


def _artifact() -> Mano45LinearInverse:
    return Mano45LinearInverse.load(
        _BUNDLED_MANO_LINEAR_INVERSE_PATH,
        expected_sha256=_BUNDLED_MANO_LINEAR_INVERSE_SHA256,
    )


def _center_pose_and_q(
    artifact: Mano45LinearInverse, side: str
) -> tuple[np.ndarray, np.ndarray]:
    model = artifact.models[side]
    rotvec = Rotation.from_euler(
        "xyz", model.reverse_center.reshape(15, 3)
    ).as_rotvec().reshape(45)
    q = model.reverse_weights[0].copy()
    return rotvec, q


def test_production_artifact_hash_contract_split_and_heldout_metrics() -> None:
    assert _BUNDLED_MANO_LINEAR_INVERSE_PATH.is_file()
    actual = hashlib.sha256(_BUNDLED_MANO_LINEAR_INVERSE_PATH.read_bytes()).hexdigest()
    assert actual == _BUNDLED_MANO_LINEAR_INVERSE_SHA256
    payload = json.loads(_BUNDLED_MANO_LINEAR_INVERSE_PATH.read_text(encoding="utf-8"))
    assert payload["training_provenance"]["episode_split"] == {
        "train": "0-79",
        "calibration": "80-89",
        "test": "90-99",
    }
    assert tuple(payload["wuji_stage_major_order"]) == WUJI_STAGE_MAJOR_ORDER
    expected_runtime_accepts = {"left": 14811, "right": 14776}
    artifact = _artifact()
    for side in ("left", "right"):
        metrics = payload["heldout_metrics"][side]
        assert metrics["q_abs_rad"]["mean"] < 0.01
        assert metrics["runtime_pre_geometry_gate_accepted_frames"] == (
            expected_runtime_accepts[side]
        )
        assert metrics["runtime_pre_geometry_gate_accept_fraction"] == pytest.approx(
            expected_runtime_accepts[side] / 14816
        )
        artifact.validate_urdf_contract(
            side=side,
            lower=artifact.urdf_lower,
            upper=artifact.urdf_upper,
            urdf_sha256=artifact.urdf_sha256[side],
        )

    with pytest.raises(RuntimeError, match="URDF hash differs"):
        artifact.validate_urdf_contract(
            side="right",
            lower=artifact.urdf_lower,
            upper=artifact.urdf_upper,
            urdf_sha256="0" * 64,
        )


def test_exact_runtime_validation_report_is_hash_linked_and_passes_gates() -> None:
    report_path = _BUNDLED_MANO_LINEAR_INVERSE_PATH.with_suffix(
        ".runtime_validation.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["artifact_sha256"] == _BUNDLED_MANO_LINEAR_INVERSE_SHA256
    assert report["episode_split"] == "90-99"
    assert report["frame_stride"] == 8
    for side in ("left", "right"):
        metrics = report["sides"][side]
        assert metrics["attempted"] >= 1800
        assert metrics["full_runtime_accept_fraction"] > 0.997
        assert metrics["geometry_rms_mm"]["max"] < (
            report["geometry_limit_m"] * 1000.0
        )
        assert metrics["accepted_q_abs_rad"]["mean"] < 0.01
        assert metrics["link7_rotation_error_deg"]["p95"] < 0.5
        assert metrics["link7_position_error_m"]["max"] < 1e-6


def test_artifact_hash_mismatch_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="hash mismatch"):
        Mano45LinearInverse.load(
            _BUNDLED_MANO_LINEAR_INVERSE_PATH,
            expected_sha256="0" * 64,
        )


def test_inverse_mode_defaults_geometric_and_supports_saved_or_deploy_ab(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run" / "config.json"
    config_path.parent.mkdir()
    mode, path, digest = _resolve_mano_inverse_settings(
        {}, {}, saved_config_path=config_path
    )
    assert (mode, path, digest) == ("geometric", None, None)

    saved = {
        "mano_inverse_mode": "linear",
        "mano_linear_inverse_path": "artifacts/inverse.json",
        "mano_linear_inverse_sha256": "1" * 64,
    }
    mode, path, digest = _resolve_mano_inverse_settings(
        {}, saved, saved_config_path=config_path
    )
    assert mode == "linear"
    assert path == (config_path.parent / "artifacts/inverse.json").resolve()
    assert digest == "1" * 64

    # An explicit deployment mode enables an independent historical A/B run;
    # the old saved config itself is not rewritten.
    mode, path, digest = _resolve_mano_inverse_settings(
        {"mano_inverse_mode": "geometric"},
        saved,
        saved_config_path=config_path,
    )
    assert mode == "geometric"
    assert path == (config_path.parent / "artifacts/inverse.json").resolve()
    assert digest == "1" * 64

    with pytest.raises(ValueError, match="requires explicit"):
        _resolve_mano_inverse_settings(
            {"mano_inverse_mode": "linear"},
            {},
            saved_config_path=config_path,
        )


def test_linear_inverse_accepts_center_pose_and_enforces_urdf_clip() -> None:
    artifact = _artifact()
    rotvec, q_raw = _center_pose_and_q(artifact, "left")
    lower = q_raw - 1.0
    upper = q_raw + 1.0
    result = artifact.predict(
        side="left",
        hand_pose_rotvec=rotvec,
        previous_q_stage_major=q_raw,
        lower=lower,
        upper=upper,
    )
    np.testing.assert_allclose(result.q_stage_major, q_raw, atol=1e-7)
    assert result.clipped_to_urdf is False

    clipped_upper = upper.copy()
    clipped_upper[0] = q_raw[0] - 0.01
    clipped_previous = q_raw.copy()
    clipped_previous[0] = clipped_upper[0]
    clipped = artifact.predict(
        side="left",
        hand_pose_rotvec=rotvec,
        previous_q_stage_major=clipped_previous,
        lower=lower,
        upper=clipped_upper,
    )
    assert clipped.clipped_to_urdf is True
    assert clipped.q_stage_major[0] == pytest.approx(clipped_upper[0])

    unsafe_upper = upper.copy()
    unsafe_upper[0] = q_raw[0] - 0.10
    with pytest.raises(LinearInverseSafetyError, match="materially exceeds"):
        artifact.predict(
            side="left",
            hand_pose_rotvec=rotvec,
            previous_q_stage_major=q_raw,
            lower=lower,
            upper=unsafe_upper,
        )


def test_linear_inverse_rejects_ood_and_velocity() -> None:
    artifact = _artifact()
    rotvec, q_raw = _center_pose_and_q(artifact, "right")
    bounds_lower = np.full(20, -10.0)
    bounds_upper = np.full(20, 10.0)
    with pytest.raises(LinearInverseSafetyError, match="per-step velocity"):
        artifact.predict(
            side="right",
            hand_pose_rotvec=rotvec,
            previous_q_stage_major=q_raw + 2.0,
            lower=bounds_lower,
            upper=bounds_upper,
        )

    model = artifact.models["right"]
    ood_euler = model.reverse_center.copy()
    # This component has a small fitted standard deviation, so a valid
    # principal Euler value of +1 rad is well beyond the calibrated domain.
    ood_euler[0] += 1.0
    ood_rotvec = Rotation.from_euler(
        "xyz", ood_euler.reshape(15, 3)
    ).as_rotvec().reshape(45)
    with pytest.raises(LinearInverseSafetyError, match="outside the calibrated"):
        artifact.predict(
            side="right",
            hand_pose_rotvec=ood_rotvec,
            previous_q_stage_major=q_raw,
            lower=bounds_lower,
            upper=bounds_upper,
        )


class _FakeLinearInverse:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return LinearInverseResult(
            q_stage_major=np.full(20, 0.2),
            mano_euler45=np.zeros(45),
            ood_rms=0.1,
            ood_abs=0.2,
            cycle_rms=0.1,
            step_delta_ratio=0.2,
            limit_excess_rad=0.0,
            clipped_to_urdf=False,
        )

    def geometry_limit_m(self, side):
        return 0.015


def test_runtime_linear_path_never_calls_geometric_ik(capsys) -> None:
    codec = Mano45RuntimeCodec.__new__(Mano45RuntimeCodec)
    hand = SimpleNamespace(
        lower=np.full(20, -1.0),
        upper=np.full(20, 1.0),
        evaluate=lambda q, target, weights: (True, 0.003, 0.01),
    )
    codec.hand_models = {"right": hand}
    codec.mano_model = object()
    codec.left_convention = "mirror_model"
    codec.tolerance_m = 0.03
    codec.last_inverse_solve_elapsed_s = {}
    codec.last_inverse_backend = {}
    codec.linear_inverse_counts = {
        "left": {"accepted": 0, "fallback": 0},
        "right": {"accepted": 0, "fallback": 0},
    }
    codec.inverse_mode = "linear"
    codec.linear_inverse = _FakeLinearInverse()
    codec._forward_hand_wrist_anchored = (
        lambda model, **kwargs: np.zeros((1, 21, 3), dtype=np.float64)
    )
    codec._mano_joints21_to_add_mano_chains = (
        lambda joints, model: (np.zeros((5, 5, 3)), np.ones((5, 5)))
    )
    codec._canonicalize_hand_points = (
        lambda points, wrist, weights, knuckle_index: (points, True)
    )
    codec._mano_root_rotation_in_link7 = MethodType(
        lambda self, side, q: np.eye(3), codec
    )
    codec._world_mano_to_wuji = lambda *args, **kwargs: pytest.fail(
        "production linear path must not call geometric IK"
    )

    decoded = codec.mano_to_wuji(
        side="right",
        root_in_env=np.eye(4),
        hand_pose_rotvec=np.zeros(45),
        betas=np.zeros(10),
        initial_q_stage_major=np.zeros(20),
    )
    np.testing.assert_allclose(decoded.q_stage_major, 0.2)
    assert decoded.rms_m == pytest.approx(0.003)
    assert decoded.solver_reported_valid is True
    assert codec.last_inverse_backend["right"] == "linear"
    assert codec.linear_inverse_counts["right"] == {"accepted": 1, "fallback": 0}
    output = capsys.readouterr().out
    assert "linear inverse summary side=right accepted=1 fallback=0" in output


def test_runtime_linear_safety_failure_falls_back_to_geometric_ik(
    monkeypatch, capsys
) -> None:
    class _Reject(_FakeLinearInverse):
        def predict(self, **kwargs):
            raise LinearInverseSafetyError("injected OOD")

    codec = Mano45RuntimeCodec.__new__(Mano45RuntimeCodec)
    codec.hand_models = {
        "right": SimpleNamespace(lower=np.full(20, -1.0), upper=np.full(20, 1.0))
    }
    codec.mano_model = object()
    codec.left_convention = "mirror_model"
    codec.hand_variant = "wujihand2"
    codec.inverse_max_iterations = 12
    codec.inverse_timeout_s = 2.0
    codec.tolerance_m = 0.03
    codec.last_inverse_solve_elapsed_s = {}
    codec.last_inverse_backend = {}
    codec._warned_inverse_valid_false = set()
    codec.linear_inverse_counts = {
        "left": {"accepted": 0, "fallback": 0},
        "right": {"accepted": 0, "fallback": 0},
    }
    codec.inverse_mode = "linear"
    codec.linear_inverse = _Reject()
    codec._forward_hand_wrist_anchored = (
        lambda model, **kwargs: np.zeros((1, 21, 3), dtype=np.float64)
    )
    codec._world_mano_to_wuji = (
        lambda points, **kwargs: (np.full(20, 0.3), True, 0.002)
    )
    codec._mano_root_rotation_in_link7 = MethodType(
        lambda self, side, q: np.eye(3), codec
    )
    monkeypatch.setattr(
        mano_runtime,
        "_threadpool_limits",
        lambda **kwargs: nullcontext(),
    )
    decoded = codec.mano_to_wuji(
        side="right",
        root_in_env=np.eye(4),
        hand_pose_rotvec=np.zeros(45),
        betas=np.zeros(10),
        initial_q_stage_major=np.zeros(20),
    )
    np.testing.assert_allclose(decoded.q_stage_major, 0.3)
    assert codec.last_inverse_backend["right"] == "geometric_fallback"
    assert codec.linear_inverse_counts["right"] == {"accepted": 0, "fallback": 1}
    output = capsys.readouterr().out
    assert "linear safety fallback" in output
    assert "reason=injected OOD" in output
