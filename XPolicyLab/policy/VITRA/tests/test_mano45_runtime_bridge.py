from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_POLICY_DIR = Path(__file__).resolve().parents[1]
for _path in (_POLICY_DIR / "VITRA", _POLICY_DIR.parents[2]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vitra.datasets.spark0_dataset import (
    MANO_ACTION_DUAL_DIM,
    MANO_STATE_DUAL_DIM,
    human_action_to_mano,
    mano_action_chunk_to_xpolicy,
    mano_euler45_to_rotvec45,
    mano_rotvec45_to_euler45,
    pose7_to_matrix,
    runtime_observation_to_mano,
)
from XPolicyLab.policy.VITRA.model import (
    Model,
    _BUNDLED_H1_CAMERA_CALIBRATION_SHA256,
    _DEFAULT_H1_CAMERA_CALIBRATION_PATH,
    _load_h1_camera_calibration,
    _prepare_h1_camera_observation,
    _resolve_h1_camera_fallback_policy,
    _resolve_data_representation,
    _resolve_existing_shared_path,
    _rewrite_saved_config_shared_paths,
    _validate_statistics_identity,
)
import XPolicyLab.policy.VITRA.mano45_runtime as mano_runtime
from XPolicyLab.policy.VITRA.mano45_runtime import (
    Mano45RuntimeCodec,
    ManoRuntimeFitError,
)
from XPolicyLab.policy.VITRA.deploy import _attach_env_indices


def _pose7(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.concatenate([translation, quaternion_xyzw[[3, 0, 1, 2]]]).astype(np.float32)


class _FakeCodec:
    def __init__(self) -> None:
        self.decode_initials: dict[str, list[np.ndarray]] = {"left": [], "right": []}

    def wuji_to_mano(
        self, *, side, q_stage_major, link7_in_env, previous_hand_pose=None
    ):
        q = np.asarray(q_stage_major, dtype=np.float64)
        # Preserve Link7 xyz and use a proper root rotation; local values merely
        # make the two hands observably different in the state vector.
        root = np.asarray(link7_in_env, dtype=np.float64).copy()
        root[:3, :3] = Rotation.from_euler(
            "xyz", [0.1, -0.2, 0.3 if side == "right" else -0.3]
        ).as_matrix()
        hand_pose = np.zeros(45, dtype=np.float64)
        hand_pose[:20] = q
        return SimpleNamespace(
            root_in_env=root,
            hand_pose_rotvec=hand_pose,
            betas=np.zeros(10, dtype=np.float64),
            rms_m=0.001,
        )

    def mano_to_wuji(
        self,
        *,
        side,
        root_in_env,
        hand_pose_rotvec,
        betas,
        initial_q_stage_major,
    ):
        initial = np.asarray(initial_q_stage_major, dtype=np.float32).copy()
        self.decode_initials[side].append(initial)
        # Deliberately retain vector order. A hidden finger-major permutation
        # would make the warm-start assertions below fail.
        q = initial + np.arange(20, dtype=np.float32) / 1000.0 + 0.01
        return SimpleNamespace(
            link7_in_env=np.asarray(root_in_env, dtype=np.float64).copy(),
            q_stage_major=q,
            rms_m=0.002,
        )


class _HistoryCodec:
    def __init__(self) -> None:
        self.calls: dict[str, list[int]] = {"left": [], "right": []}

    def wuji_to_mano(
        self, *, side, q_stage_major, link7_in_env, previous_hand_pose=None
    ):
        tick = int(round(float(np.asarray(q_stage_major)[0])))
        self.calls[side].append(tick)
        pose = np.full(45, tick / 100.0, dtype=np.float32)
        return SimpleNamespace(hand_pose_rotvec=pose)


class _FailingDecodeCodec(_FakeCodec):
    tolerance_m = 0.03

    def __init__(self, fail_side: str, fail_side_call: int) -> None:
        super().__init__()
        self.fail_side = fail_side
        self.fail_side_call = fail_side_call
        self.side_calls = {"left": 0, "right": 0}
        self.calls: list[tuple[str, int]] = []

    def mano_to_wuji(self, *, side, **kwargs):
        side_call = self.side_calls[side]
        self.side_calls[side] += 1
        self.calls.append((side, side_call))
        if side == self.fail_side and side_call == self.fail_side_call:
            raise ManoRuntimeFitError(f"injected {side} failure at call {side_call}")
        return super().mano_to_wuji(side=side, **kwargs)


def _observation() -> dict:
    left_q = np.arange(20, dtype=np.float32) / 100.0
    right_q = np.arange(20, dtype=np.float32)[::-1] / 100.0
    return {
        "instruction": "pick up the objects",
        "vision": {
            "cam_head": {
                "color": np.zeros((32, 48, 3), dtype=np.uint8),
                "extrinsics": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
                "intrinsics": np.array([40, 41, 24, 16], dtype=np.float32),
            }
        },
        "state": {
            "left_ee_pose": _pose7(
                np.array([-0.2, 0.3, 0.4]), Rotation.from_euler("z", 0.2).as_matrix()
            ),
            "right_ee_pose": _pose7(
                np.array([0.2, 0.3, 0.4]), Rotation.from_euler("z", -0.2).as_matrix()
            ),
            "left_ee_joint_state": left_q,
            "right_ee_joint_state": right_q,
        },
    }


def _observation_without_camera_metadata(height: int = 720, width: int = 1280) -> dict:
    observation = _observation()
    camera = observation["vision"]["cam_head"]
    camera.pop("extrinsics")
    camera.pop("intrinsics")
    camera["color"] = np.zeros((height, width, 3), dtype=np.uint8)
    camera["shape"] = np.asarray([height, width], dtype=np.int32)
    return observation


def test_h1_camera_profile_is_hash_locked_and_scales_explicitly() -> None:
    profile = _load_h1_camera_calibration(
        _DEFAULT_H1_CAMERA_CALIBRATION_PATH,
        _BUNDLED_H1_CAMERA_CALIBRATION_SHA256,
    )
    observation = _observation_without_camera_metadata()
    prepared = _prepare_h1_camera_observation(observation, profile)
    camera = prepared["vision"]["cam_head"]
    assert camera["color"].shape == (384, 384, 3)
    np.testing.assert_array_equal(camera["shape"], [384, 384])
    np.testing.assert_allclose(
        camera["intrinsics"],
        np.array(
            [[146.59986, 0.0, 192.0], [0.0, 260.6219733, 192.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        rtol=0,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        camera["extrinsics"],
        profile["source_extrinsics"],
        rtol=0,
        atol=1e-7,
    )
    assert camera["frame"] == "camera_to_env"
    assert camera["calibration_sha256"] == _BUNDLED_H1_CAMERA_CALIBRATION_SHA256
    assert "extrinsics" not in observation["vision"]["cam_head"]
    assert "intrinsics" not in observation["vision"]["cam_head"]

    square = _observation_without_camera_metadata(384, 384)
    square_prepared = _prepare_h1_camera_observation(square, profile)
    np.testing.assert_allclose(
        square_prepared["vision"]["cam_head"]["intrinsics"],
        np.array(
            [[146.59986, 0.0, 192.0], [0.0, 260.6219733, 192.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        rtol=0,
        atol=2e-4,
    )


def test_h1_live_720p_is_stretched_to_training_384() -> None:
    import cv2

    profile = _load_h1_camera_calibration(
        _DEFAULT_H1_CAMERA_CALIBRATION_PATH,
        _BUNDLED_H1_CAMERA_CALIBRATION_SHA256,
    )
    observation = _observation_without_camera_metadata(720, 1280)
    live = np.zeros((720, 1280, 3), dtype=np.uint8)
    live[:, :, 0] = 200
    live[:, :, 2] = 30
    observation["vision"]["cam_head"]["color"] = live
    prepared = _prepare_h1_camera_observation(observation, profile)
    camera = prepared["vision"]["cam_head"]
    expected = cv2.resize(live, (384, 384), interpolation=cv2.INTER_AREA)
    assert camera["color"].shape == (384, 384, 3)
    np.testing.assert_array_equal(camera["color"], expected)
    assert observation["vision"]["cam_head"]["color"].shape == (720, 1280, 3)
    np.testing.assert_allclose(
        camera["intrinsics"],
        np.array(
            [[146.59986, 0.0, 192.0], [0.0, 260.6219733, 192.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        rtol=0,
        atol=2e-4,
    )

    with_meta = _observation_without_camera_metadata(720, 1280)
    with_meta["vision"]["cam_head"]["color"] = live.copy()
    with_meta["vision"]["cam_head"]["extrinsics"] = profile["source_extrinsics"].astype(
        np.float32
    )
    with_meta["vision"]["cam_head"]["intrinsics"] = np.array(
        [[488.66618945, 0.0, 640.0], [0.0, 488.66618945, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    prepared_meta = _prepare_h1_camera_observation(with_meta, profile)
    np.testing.assert_array_equal(prepared_meta["vision"]["cam_head"]["color"], expected)
    np.testing.assert_allclose(
        prepared_meta["vision"]["cam_head"]["intrinsics"],
        np.array(
            [[146.59986, 0.0, 192.0], [0.0, 260.6219733, 192.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        rtol=0,
        atol=2e-4,
    )


def test_h1_camera_profile_preserves_valid_metadata_and_rejects_partial_or_unknown() -> None:
    profile = _load_h1_camera_calibration(
        _DEFAULT_H1_CAMERA_CALIBRATION_PATH,
        _BUNDLED_H1_CAMERA_CALIBRATION_SHA256,
    )
    original = _observation()
    prepared = _prepare_h1_camera_observation(original, profile)
    assert prepared is not original
    assert prepared["vision"]["cam_head"] is original["vision"]["cam_head"]
    np.testing.assert_array_equal(
        prepared["vision"]["cam_head"]["intrinsics"],
        original["vision"]["cam_head"]["intrinsics"],
    )
    partial = _observation_without_camera_metadata()
    partial["vision"]["cam_head"]["intrinsics"] = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="exactly one extrinsics and one intrinsics"):
        _prepare_h1_camera_observation(partial, profile)
    unknown = _observation_without_camera_metadata(480, 640)
    with pytest.raises(ValueError, match="unknown RGB size"):
        _prepare_h1_camera_observation(unknown, profile)


def test_h1_camera_profile_rejects_hash_mismatch() -> None:
    wrong = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_h1_camera_calibration(_DEFAULT_H1_CAMERA_CALIBRATION_PATH, wrong)


def test_h1_camera_fallback_is_disabled_for_real_environment(
    monkeypatch,
) -> None:
    profile = _load_h1_camera_calibration(
        _DEFAULT_H1_CAMERA_CALIBRATION_PATH,
        _BUNDLED_H1_CAMERA_CALIBRATION_SHA256,
    )
    observation = _observation_without_camera_metadata()
    monkeypatch.setenv("EVAL_ENV_TYPE", "real")
    mode, scope, allowed = _resolve_h1_camera_fallback_policy(
        {"h1_camera_calibration_scope": "simulator"}
    )
    assert (mode, scope, allowed) == ("real_world", "simulator", False)
    with pytest.raises(ValueError, match="real calibration required"):
        _prepare_h1_camera_observation(
            observation,
            profile,
            allow_fallback=allowed,
            eval_env_mode=mode,
        )

    # A real runtime observation that carries both fields is still accepted;
    # the guard only blocks accidental simulator fallback.
    valid = _observation()
    prepared = _prepare_h1_camera_observation(
        valid,
        profile,
        allow_fallback=allowed,
        eval_env_mode=mode,
    )
    assert prepared["vision"]["cam_head"] is valid["vision"]["cam_head"]


@pytest.mark.parametrize("raw_mode", ["", "sim", "debug"])
def test_h1_camera_simulator_modes_allow_profile_fallback(
    monkeypatch, raw_mode: str
) -> None:
    monkeypatch.setenv("EVAL_ENV_TYPE", raw_mode)
    mode, scope, allowed = _resolve_h1_camera_fallback_policy(
        {"h1_camera_calibration_scope": "simulator"}
    )
    assert mode in {"sim", "debug"}
    assert scope == "simulator"
    assert allowed is True


def test_h1_camera_runtime_scope_disables_fallback_even_in_sim(monkeypatch) -> None:
    monkeypatch.setenv("EVAL_ENV_TYPE", "sim")
    mode, scope, allowed = _resolve_h1_camera_fallback_policy(
        {"h1_camera_calibration_scope": "runtime"}
    )
    assert (mode, scope, allowed) == ("sim", "runtime", False)


def test_h1_camera_unknown_environment_mode_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("EVAL_ENV_TYPE", "hardware_lab")
    with pytest.raises(ValueError, match="Unknown EVAL_ENV_TYPE"):
        _resolve_h1_camera_fallback_policy(
            {"h1_camera_calibration_scope": "simulator"}
        )


def test_mano_euler_rotvec_roundtrip_is_per_joint() -> None:
    rng = np.random.default_rng(2)
    rotvec = rng.normal(scale=0.3, size=(4, 45))
    recovered = mano_euler45_to_rotvec45(mano_rotvec45_to_euler45(rotvec))
    expected_matrix = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    actual_matrix = Rotation.from_rotvec(recovered.reshape(-1, 3)).as_matrix()
    np.testing.assert_allclose(actual_matrix, expected_matrix, atol=1e-6)


def test_human_action_to_mano_is_contiguous_not_sparse() -> None:
    model_action = np.arange(3 * 192, dtype=np.float32).reshape(3, 192)
    extracted = human_action_to_mano(model_action)
    assert extracted.shape == (3, MANO_ACTION_DUAL_DIM)
    np.testing.assert_array_equal(extracted, model_action[:, :102])


def test_saved_representation_cannot_be_overridden() -> None:
    saved = {"data_representation": "mano45"}
    assert _resolve_data_representation({}, saved) == "mano45"
    with pytest.raises(ValueError, match="disagrees"):
        _resolve_data_representation({"data_representation": "wuji20"}, saved)


def test_saved_config_resolves_existing_shared_mount_alias(tmp_path: Path) -> None:
    source_root = tmp_path / "mnt" / "xspark-data"
    target_root = tmp_path / "personal"
    target_model = target_root / "models" / "paligemma"
    target_model.mkdir(parents=True)
    saved_model = source_root / "models" / "paligemma"
    aliases = ((f"{source_root}/", f"{target_root}/"),)

    assert _resolve_existing_shared_path(str(saved_model), aliases) == str(target_model)
    config = {
        "vlm": {"pretrained_model_name_or_path": str(saved_model)},
        "unrelated": "namespace/model",
    }
    changes = _rewrite_saved_config_shared_paths(config, aliases=aliases)
    assert config["vlm"]["pretrained_model_name_or_path"] == str(target_model)
    assert config["unrelated"] == "namespace/model"
    assert changes == [
        (
            "vlm.pretrained_model_name_or_path",
            str(saved_model),
            str(target_model),
        )
    ]


def test_batch_observations_keep_authoritative_ids_after_dropout() -> None:
    observations = [{"state": {}}, {"state": {}}]
    bound = _attach_env_indices(observations, [1, 3])
    assert [item["env_idx"] for item in bound] == [1, 3]
    assert all("env_idx" not in item for item in observations)
    with pytest.raises(ValueError, match="disagrees"):
        _attach_env_indices([{"env_idx": 2}], [3])


def test_statistics_hash_mismatch_fails_before_load(tmp_path: Path) -> None:
    statistics = tmp_path / "teledata_statistics.json"
    statistics.write_text('{"representation":"mano45"}\n', encoding="utf-8")
    actual = hashlib.sha256(statistics.read_bytes()).hexdigest()
    wrong = ("0" if actual[0] != "0" else "1") + actual[1:]
    with pytest.raises(ValueError, match="Statistics identity mismatch"):
        _validate_statistics_identity(
            {},
            {"statistics_path": str(statistics), "statistics_sha256": wrong},
            tmp_path / "config.json",
            statistics,
            "mano45",
        )


def test_runtime_bridge_builds_state122_and_warm_starts_dual_chunk() -> None:
    codec = _FakeCodec()
    converted = runtime_observation_to_mano(_observation(), codec=codec)
    assert converted["native_state"].shape == (MANO_STATE_DUAL_DIM,)
    assert converted["fov"].shape == (2,)
    assert converted["context"]["representation"] == "mano45"

    action = np.zeros((3, MANO_ACTION_DUAL_DIM), dtype=np.float32)
    # Exercise sequential root integration while keeping the local rotations
    # finite/proper.
    action[:, 0] = 0.01
    action[:, 51 + 1] = -0.02
    commands = mano_action_chunk_to_xpolicy(
        action, converted["context"], codec=codec
    )
    assert len(commands) == 3
    for command in commands:
        assert command["left_ee_pose"].shape == (7,)
        assert command["right_ee_pose"].shape == (7,)
        assert command["left_ee_joint_state"].shape == (20,)
        assert command["right_ee_joint_state"].shape == (20,)

    obs = _observation()
    for side in ("left", "right"):
        current = obs["state"][f"{side}_ee_joint_state"]
        np.testing.assert_allclose(codec.decode_initials[side][0], current)
        np.testing.assert_allclose(
            codec.decode_initials[side][1], commands[0][f"{side}_ee_joint_state"]
        )
        np.testing.assert_allclose(
            codec.decode_initials[side][2], commands[1][f"{side}_ee_joint_state"]
        )

    # Root xyz is the Link7 xyz anchor; no 128.5-mm fixed wrist translation is
    # subtracted on decode.
    np.testing.assert_allclose(
        commands[0]["left_ee_pose"][:3],
        obs["state"]["left_ee_pose"][:3] + np.array([0.01, 0.0, 0.0]),
        atol=1e-6,
    )


def _assert_command_holds_observation(command: dict, observation: dict) -> None:
    for side in ("left", "right"):
        np.testing.assert_allclose(
            pose7_to_matrix(command[f"{side}_ee_pose"]),
            pose7_to_matrix(observation["state"][f"{side}_ee_pose"]),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            command[f"{side}_ee_joint_state"],
            observation["state"][f"{side}_ee_joint_state"],
        )


def test_mano_chunk_first_row_failure_holds_observed_both_hands(capsys) -> None:
    observation = _observation()
    codec = _FailingDecodeCodec(fail_side="left", fail_side_call=0)
    converted = runtime_observation_to_mano(observation, codec=codec)
    action = np.zeros((4, MANO_ACTION_DUAL_DIM), dtype=np.float32)
    commands = mano_action_chunk_to_xpolicy(action, converted["context"], codec=codec)

    assert len(commands) == 4
    assert codec.calls == [("left", 0)]
    for command in commands:
        _assert_command_holds_observation(command, observation)
    assert capsys.readouterr().out.count("MANO45 degraded chunk") == 1


def test_mano_chunk_mid_row_one_hand_failure_atomically_holds_both(capsys) -> None:
    codec = _FailingDecodeCodec(fail_side="right", fail_side_call=1)
    converted = runtime_observation_to_mano(_observation(), codec=codec)
    action = np.zeros((4, MANO_ACTION_DUAL_DIM), dtype=np.float32)
    action[:, 0] = 0.01
    action[:, 51 + 1] = -0.02
    commands = mano_action_chunk_to_xpolicy(action, converted["context"], codec=codec)

    assert len(commands) == 4
    # Row 0 committed. Row 1 tried left and then failed right; rows 2/3 never
    # call either solver and both hands hold the complete row-0 command.
    assert codec.calls == [("left", 0), ("right", 0), ("left", 1), ("right", 1)]
    for held in commands[1:]:
        for key in commands[0]:
            np.testing.assert_array_equal(held[key], commands[0][key])
    assert capsys.readouterr().out.count("MANO45 degraded chunk") == 1


def test_model_replays_stride2_pending_history_and_reset() -> None:
    model = Model.__new__(Model)
    model.representation = "mano45"
    model._mano_codec = _HistoryCodec()
    model._pose7_to_matrix = pose7_to_matrix
    model._observations = {}
    model._raw_observations = {}
    model._mano_fit_previous = {}
    model._mano_pending_keyframes = {}
    model._mano_next_tick = {}
    model._mano_warned_odd_query = set()
    model._latest_env_idx_list = [0]
    model.model = SimpleNamespace()

    def _encode_latest(self, obs, env_idx=0):
        # The real encoder performs the final frame's fit after all preceding
        # even snapshots have advanced the cached previous pose.
        self._advance_mano_fit_history(env_idx, self._snapshot_mano_state(obs))
        return {"tick": int(obs["state"]["left_ee_joint_state"][0])}

    model._encode_observation = MethodType(_encode_latest, model)
    model._predict_one = lambda payload: [payload["tick"]]

    def _tick_observation(tick: int) -> dict:
        pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        q = np.full(20, tick, dtype=np.float32)
        return {
            "env_idx": 0,
            "state": {
                "left_ee_pose": pose,
                "right_ee_pose": pose,
                "left_ee_joint_state": q,
                "right_ee_joint_state": q,
            },
        }

    model.update_obs(_tick_observation(0))
    assert model.get_action_batch([0]) == [[0]]
    for tick in range(1, 17):
        model.update_obs(_tick_observation(tick))
    assert model.get_action_batch([0]) == [[16]]
    assert model._mano_codec.calls["left"] == list(range(0, 17, 2))
    assert model._mano_codec.calls["right"] == list(range(0, 17, 2))

    model.reset()
    assert model._observations == {}
    assert model._raw_observations == {}
    assert model._mano_fit_previous == {}
    assert model._mano_pending_keyframes == {}
    assert model._mano_next_tick == {}


def test_left_previous_pose_is_mirrored_back_to_solver_space(monkeypatch) -> None:
    blas_limits: list[tuple[int, str]] = []

    @contextmanager
    def _single_blas_thread(*, limits, user_api):
        blas_limits.append((limits, user_api))
        yield

    monkeypatch.setattr(mano_runtime, "_threadpool_limits", _single_blas_thread)
    codec = Mano45RuntimeCodec.__new__(Mano45RuntimeCodec)
    hand = SimpleNamespace(
        lower=np.full(20, -2.0),
        upper=np.full(20, 2.0),
        forward_points=lambda q, canonical=False: np.zeros((5, 5, 3), dtype=np.float64),
    )
    captured: dict[str, np.ndarray] = {}

    class _Fitter:
        def fit(self, points, confidence, **kwargs):
            captured["previous"] = kwargs["previous"].copy()
            return (
                {
                    "global_orient": np.zeros(3),
                    "hand_pose": np.zeros(45),
                    "betas": np.zeros(10),
                },
                0.0,
                True,
                1,
            )

    codec.hand_models = {"left": hand}
    codec.mounts = {"left": (np.eye(3), np.zeros(3))}
    codec.mano_fitter = _Fitter()
    codec.max_iterations = 35
    codec.tolerance_m = 0.03
    previous = np.tile(np.array([0.1, 0.2, -0.3]), 15)
    codec.wuji_to_mano(
        side="left",
        q_stage_major=np.zeros(20),
        link7_in_env=np.eye(4),
        previous_hand_pose=previous,
    )
    expected = previous.reshape(15, 3).copy()
    expected[:, 1:] *= -1.0
    np.testing.assert_allclose(captured["previous"], expected.reshape(45))
    assert blas_limits == [(1, "blas")]


def _stub_inverse_codec() -> Mano45RuntimeCodec:
    codec = Mano45RuntimeCodec.__new__(Mano45RuntimeCodec)
    codec.hand_models = {
        "right": SimpleNamespace(
            lower=np.full(20, -2.0),
            upper=np.full(20, 2.0),
        )
    }
    codec.mano_model = object()
    codec.left_convention = "mirror_model"
    codec.hand_variant = "wujihand2"
    codec.inverse_max_iterations = 12
    codec.inverse_timeout_s = 2.0
    codec.tolerance_m = 0.03
    codec.last_inverse_solve_elapsed_s = {}
    codec.last_inverse_backend = {}
    codec._forward_hand_wrist_anchored = (
        lambda model, **kwargs: np.zeros((1, 21, 3), dtype=np.float64)
    )
    codec._mano_root_rotation_in_link7 = MethodType(
        lambda self, side, q: np.eye(3), codec
    )
    return codec


def test_inverse_is_blas_single_threaded_and_iteration_capped(monkeypatch) -> None:
    blas_limits: list[tuple[int, str]] = []
    solver_kwargs: dict[str, object] = {}

    @contextmanager
    def _single_blas_thread(*, limits, user_api):
        blas_limits.append((limits, user_api))
        yield

    def _solve(points, **kwargs):
        solver_kwargs.update(kwargs)
        return np.zeros(20), True, 0.001

    monkeypatch.setattr(mano_runtime, "_threadpool_limits", _single_blas_thread)
    codec = _stub_inverse_codec()
    codec._world_mano_to_wuji = _solve
    decoded = codec.mano_to_wuji(
        side="right",
        root_in_env=np.eye(4),
        hand_pose_rotvec=np.zeros(45),
        betas=np.zeros(10),
        initial_q_stage_major=np.zeros(20),
    )
    assert blas_limits == [(1, "blas")]
    assert solver_kwargs["max_iterations"] == 12
    assert decoded.solve_elapsed_s >= 0.0
    assert codec.last_inverse_solve_elapsed_s["right"] == decoded.solve_elapsed_s


def test_link7_orientation_freezes_palm_offset_to_initial_q(monkeypatch) -> None:
    @contextmanager
    def _single_blas_thread(*, limits, user_api):
        yield

    offset_q: list[np.ndarray] = []

    def _offset(self, side, q):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        offset_q.append(q.copy())
        return Rotation.from_rotvec([0.0, 0.0, float(q[0])]).as_matrix()

    monkeypatch.setattr(mano_runtime, "_threadpool_limits", _single_blas_thread)
    codec = _stub_inverse_codec()
    codec._mano_root_rotation_in_link7 = MethodType(_offset, codec)
    codec._world_mano_to_wuji = (
        lambda points, **kwargs: (np.full(20, 0.4), True, 0.001)
    )
    root = np.eye(4, dtype=np.float64)
    root[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 0.05]).as_matrix()
    root[:3, 3] = (0.3, -0.1, 0.5)
    initial = np.zeros(20, dtype=np.float64)
    decoded = codec.mano_to_wuji(
        side="right",
        root_in_env=root,
        hand_pose_rotvec=np.zeros(45),
        betas=np.zeros(10),
        initial_q_stage_major=initial,
    )
    assert offset_q and np.allclose(offset_q[-1], initial)
    np.testing.assert_allclose(decoded.link7_in_env[:3, :3], root[:3, :3], atol=1e-9)
    np.testing.assert_allclose(decoded.link7_in_env[:3, 3], root[:3, 3], atol=1e-9)


def test_inverse_elapsed_limit_fails_after_audited_solve(monkeypatch) -> None:
    @contextmanager
    def _single_blas_thread(*, limits, user_api):
        yield

    elapsed_clock = iter((10.0, 12.5))
    monkeypatch.setattr(mano_runtime, "_threadpool_limits", _single_blas_thread)
    monkeypatch.setattr(mano_runtime.time, "perf_counter", lambda: next(elapsed_clock))
    codec = _stub_inverse_codec()
    codec.inverse_timeout_s = 2.0
    codec._world_mano_to_wuji = (
        lambda points, **kwargs: (np.zeros(20), True, 0.001)
    )
    with pytest.raises(
        mano_runtime.ManoRuntimeFitError, match="exceeded the post-solve time limit"
    ):
        codec.mano_to_wuji(
            side="right",
            root_in_env=np.eye(4),
            hand_pose_rotvec=np.zeros(45),
            betas=np.zeros(10),
            initial_q_stage_major=np.zeros(20),
        )
    assert codec.last_inverse_solve_elapsed_s["right"] == 2.5


def test_inverse_accepts_valid_false_only_when_numerically_safe(
    monkeypatch, capsys
) -> None:
    @contextmanager
    def _single_blas_thread(*, limits, user_api):
        yield

    monkeypatch.setattr(mano_runtime, "_threadpool_limits", _single_blas_thread)
    codec = _stub_inverse_codec()
    codec._world_mano_to_wuji = (
        lambda points, **kwargs: (np.zeros(20), False, 0.001)
    )
    decoded = codec.mano_to_wuji(
        side="right",
        root_in_env=np.eye(4),
        hand_pose_rotvec=np.zeros(45),
        betas=np.zeros(10),
        initial_q_stage_major=np.zeros(20),
    )
    assert decoded.solver_reported_valid is False
    assert capsys.readouterr().out.count("reported valid=False") == 1

    codec._world_mano_to_wuji = (
        lambda points, **kwargs: (np.full(20, np.nan), False, 0.001)
    )
    with pytest.raises(ManoRuntimeFitError, match="non-finite q"):
        codec.mano_to_wuji(
            side="right",
            root_in_env=np.eye(4),
            hand_pose_rotvec=np.zeros(45),
            betas=np.zeros(10),
            initial_q_stage_major=np.zeros(20),
        )

    def _singular_solve(points, **kwargs):
        raise np.linalg.LinAlgError("injected singular system")

    codec._world_mano_to_wuji = _singular_solve
    with pytest.raises(ManoRuntimeFitError, match="numerical solve failed"):
        codec.mano_to_wuji(
            side="right",
            root_in_env=np.eye(4),
            hand_pose_rotvec=np.zeros(45),
            betas=np.zeros(10),
            initial_q_stage_major=np.zeros(20),
        )


def test_inverse_iteration_hard_cap_and_threadpoolctl_gate(monkeypatch) -> None:
    with pytest.raises(ValueError, match="inverse_max_iterations"):
        Mano45RuntimeCodec("/definitely/missing", inverse_max_iterations=13)
    monkeypatch.setattr(mano_runtime, "_threadpool_limits", None)
    with pytest.raises(mano_runtime.ManoRuntimeContractError, match="threadpoolctl"):
        Mano45RuntimeCodec("/definitely/missing")


@pytest.mark.skipif(
    not os.environ.get("VITRA_MANO_TOOLS_ROOT"),
    reason="set VITRA_MANO_TOOLS_ROOT for the real add_mano/Wuji assets",
)
def test_real_codec_root_identity_and_synthetic_roundtrip() -> None:
    """Gate intended for the policy environment with Zijian's pinned tools."""

    from XPolicyLab.policy.VITRA.mano45_runtime import Mano45RuntimeCodec

    codec = Mano45RuntimeCodec(Path(os.environ["VITRA_MANO_TOOLS_ROOT"]))
    rng = np.random.default_rng(11)
    expected_stage_major = tuple(
        (finger, stage)
        for stage in range(4)
        for finger in (1, 2, 4, 3, 0)
    )
    for side in ("left", "right"):
        hand = codec.hand_models[side]
        assert tuple(hand.variant.joint_order) == expected_stage_major
        q = hand.lower + (hand.upper - hand.lower) * rng.uniform(0.25, 0.75, 20)
        link7 = np.eye(4)
        link7[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
        link7[:3, 3] = [-0.2 if side == "left" else 0.2, 0.3, 0.4]
        fit = codec.wuji_to_mano(side=side, q_stage_major=q, link7_in_env=link7)
        # Closed-form frame identity is exact before the morphology-changing IK.
        expected_root_rotation = (
            link7[:3, :3] @ codec._mano_root_rotation_in_link7(side, q)
        )
        root_identity_error = Rotation.from_matrix(
            fit.root_in_env[:3, :3] @ expected_root_rotation.T
        ).magnitude()
        assert root_identity_error < 1e-6
        decoded = codec.mano_to_wuji(
            side=side,
            root_in_env=fit.root_in_env,
            hand_pose_rotvec=fit.hand_pose_rotvec,
            betas=fit.betas,
            initial_q_stage_major=q,
        )
        assert decoded.rms_m <= codec.tolerance_m
        assert float(np.mean(np.abs(decoded.q_stage_major - q))) < 0.15
        np.testing.assert_allclose(decoded.link7_in_env[:3, 3], link7[:3, 3], atol=1e-8)
        rotation_error = Rotation.from_matrix(
            decoded.link7_in_env[:3, :3] @ link7[:3, :3].T
        ).magnitude()
        # MANO/Wuji morphology makes the IK approximate (and non-invertible),
        # but this still gates the gross ~140-degree legacy frame error.
        assert rotation_error < 0.15


@pytest.mark.skipif(
    not (
        os.environ.get("VITRA_MANO_TOOLS_ROOT")
        and os.environ.get("VITRA_MANO_HELDOUT_HDF5")
    ),
    reason="set VITRA_MANO_TOOLS_ROOT and VITRA_MANO_HELDOUT_HDF5",
)
def test_real_codec_heldout_mano_to_stage_major_wuji() -> None:
    """Held-out gate against an immutable add_mano episode."""

    import h5py

    from XPolicyLab.policy.VITRA.mano45_runtime import Mano45RuntimeCodec

    codec = Mano45RuntimeCodec(Path(os.environ["VITRA_MANO_TOOLS_ROOT"]))
    path = Path(os.environ["VITRA_MANO_HELDOUT_HDF5"])
    q_errors: list[float] = []
    root_position_errors: list[float] = []
    local_rotation_errors: list[float] = []
    with h5py.File(path, "r") as handle:
        length = int(handle["state/left_ee_poses"].shape[0])
        # add_mano used stride=2; replay its consecutive solver keyframes so
        # the cached previous pose has the identical history/convention.
        indices = np.arange(0, min(max(0, length - 1), 18), 2, dtype=int)
        previous_pose = {"left": None, "right": None}
        for index in indices:
            for side in ("left", "right"):
                q_true = np.asarray(
                    handle[f"state/{side}_ee_joint_states"][index], dtype=np.float64
                )
                link7 = pose7_to_matrix(handle[f"state/{side}_ee_poses"][index])
                fit = codec.wuji_to_mano(
                    side=side,
                    q_stage_major=q_true,
                    link7_in_env=link7,
                    previous_hand_pose=previous_pose[side],
                )
                previous_pose[side] = fit.hand_pose_rotvec
                stored_root = pose7_to_matrix(
                    handle[f"mano/state/{side}_ee_poses"][index]
                )
                stored_pose = np.asarray(
                    handle[f"mano/state/{side}_ee_joint_states"][index],
                    dtype=np.float64,
                )
                stored_beta = np.asarray(
                    handle[f"mano/state/{side}_hand_betas"][index], dtype=np.float64
                )
                # Both the runtime forward and stored add_mano contract anchor
                # MANO root xyz exactly on Link7 xyz.
                np.testing.assert_allclose(fit.root_in_env[:3, 3], stored_root[:3, 3], atol=1e-8)
                fitted_local = Rotation.from_rotvec(
                    fit.hand_pose_rotvec.reshape(15, 3)
                ).as_matrix()
                stored_local = Rotation.from_rotvec(stored_pose.reshape(15, 3)).as_matrix()
                local_rotation_errors.extend(
                    Rotation.from_matrix(fitted_local @ stored_local.transpose(0, 2, 1))
                    .magnitude()
                    .tolist()
                )
                decoded = codec.mano_to_wuji(
                    side=side,
                    root_in_env=stored_root,
                    hand_pose_rotvec=stored_pose,
                    betas=stored_beta,
                    initial_q_stage_major=q_true,
                )
                q_errors.append(float(np.mean(np.abs(decoded.q_stage_major - q_true))))
                root_position_errors.append(
                    float(np.linalg.norm(decoded.link7_in_env[:3, 3] - link7[:3, 3]))
                )
    # This threshold is intentionally much tighter than the old sparse-map
    # held-out MAE (~0.22 rad), while allowing MANO/Wuji morphology mismatch.
    assert float(np.median(q_errors)) < 0.12
    assert max(root_position_errors) < 1e-7
    assert float(np.mean(local_rotation_errors)) < 1e-5
