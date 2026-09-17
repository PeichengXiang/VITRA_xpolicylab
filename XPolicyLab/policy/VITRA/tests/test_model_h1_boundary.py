"""Policy-local tests for the VITRA H1 boundary.

These tests deliberately do not construct :class:`Model` through its normal
initializer (and therefore never load a checkpoint or a GPU model).  A small
stateful adapter double exercises the exact boundary methods that the runtime
uses around the real, hash-pinned Spark-0 retargeter.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

_WORKSPACE = Path(__file__).resolve().parents[4]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from XPolicyLab.policy.VITRA.model import Model  # noqa: E402
from XPolicyLab.policy.VITRA import model as vitra_model  # noqa: E402


class _BoundaryAdapterDouble:
    """Deterministic 12<->20 adapter with observable transactional state."""

    def __init__(self) -> None:
        self.counter = 0
        self.snapshot_calls: list[int] = []
        self.restore_calls: list[tuple[int, dict[str, int]]] = []
        self.h1_calls: list[tuple[str, int]] = []
        self.wuji_calls: list[tuple[str, int]] = []

    def snapshot(self, env_idx: int) -> dict[str, int]:
        self.snapshot_calls.append(int(env_idx))
        return {"env_idx": int(env_idx), "counter": self.counter}

    def restore(self, env_idx: int, snapshot: dict[str, int]) -> None:
        self.restore_calls.append((int(env_idx), dict(snapshot)))
        self.counter = int(snapshot["counter"])

    def h1_to_wuji(self, q12: np.ndarray, side: str, env_idx: int = 0) -> np.ndarray:
        q12 = np.asarray(q12, dtype=np.float64)
        if q12.shape != (12,):
            raise ValueError("double expects Inspire12")
        self.counter += 1
        self.h1_calls.append((side, int(env_idx)))
        # Keep the first twelve coordinates recognizable and append eight
        # deterministic Wuji-only coordinates.  No production code relies on
        # this artificial mapping; it only makes shape/order assertions clear.
        tail = np.arange(8, dtype=np.float64) + (100.0 if side == "left" else 200.0)
        return np.concatenate((q12, tail))

    def wuji_to_h1(self, q20: np.ndarray, side: str, env_idx: int = 0) -> np.ndarray:
        q20 = np.asarray(q20, dtype=np.float64)
        if q20.shape != (20,):
            raise ValueError("double expects Wuji20")
        self.counter += 1
        self.wuji_calls.append((side, int(env_idx)))
        return q20[:12].copy()


def _bare_model(adapter: _BoundaryAdapterDouble | None = None) -> Model:
    """Build only the attributes used by the H1 boundary methods."""

    model = Model.__new__(Model)
    model._h1_adapter = adapter or _BoundaryAdapterDouble()
    model.ee_dims = [12, 12]
    return model


def _observation(left: np.ndarray | None = None, right: np.ndarray | None = None) -> dict:
    left = np.arange(12, dtype=np.float32) / 10.0 if left is None else left
    right = np.arange(12, dtype=np.float32)[::-1] / 10.0 if right is None else right
    return {
        "instruction": "test boundary",
        "state": {
            "left_ee_joint_state": np.asarray(left, dtype=np.float32),
            "right_ee_joint_state": np.asarray(right, dtype=np.float32),
            "left_arm_joint_state": np.zeros(7, dtype=np.float32),
            "right_arm_joint_state": np.zeros(7, dtype=np.float32),
        },
        "extra": {"keep": True},
    }


def _action_chunk() -> list[dict[str, np.ndarray]]:
    steps: list[dict[str, np.ndarray]] = []
    for step in range(2):
        steps.append(
            {
                "left_ee_pose": np.array([step, 0, 0, 1, 0, 0, 0], dtype=np.float32),
                "right_ee_pose": np.array([step, 0, 0, 1, 0, 0, 0], dtype=np.float32),
                "left_ee_joint_state": np.arange(20, dtype=np.float32) + step,
                "right_ee_joint_state": np.arange(20, dtype=np.float32)[::-1] - step,
            }
        )
    return steps


def test_internalize_observation_converts_both_hands_without_mutating_input() -> None:
    adapter = _BoundaryAdapterDouble()
    model = _bare_model(adapter)
    observation = _observation()
    original = copy.deepcopy(observation)

    converted = model._internalize_observation(observation, env_idx=7)

    assert converted is not observation
    assert converted["state"] is not observation["state"]
    assert adapter.h1_calls == [("left", 7), ("right", 7)]
    np.testing.assert_array_equal(
        converted["state"]["left_ee_joint_state"][0:12],
        original["state"]["left_ee_joint_state"],
    )
    np.testing.assert_array_equal(
        converted["state"]["right_ee_joint_state"][0:12],
        original["state"]["right_ee_joint_state"],
    )
    assert converted["state"]["left_ee_joint_state"].shape == (20,)
    assert converted["state"]["right_ee_joint_state"].shape == (20,)
    assert converted["extra"] == observation["extra"]
    # The caller-owned observation, including its arrays, remains byte-for-byte
    # unchanged after the policy-side conversion.
    np.testing.assert_array_equal(
        observation["state"]["left_ee_joint_state"],
        original["state"]["left_ee_joint_state"],
    )
    np.testing.assert_array_equal(
        observation["state"]["right_ee_joint_state"],
        original["state"]["right_ee_joint_state"],
    )


def test_externalize_action_chunk_converts_both_hands_and_validates_external_abi() -> None:
    adapter = _BoundaryAdapterDouble()
    model = _bare_model(adapter)
    actions = _action_chunk()
    original = copy.deepcopy(actions)

    converted = model._externalize_action_chunk(actions, env_idx=3)
    validated = model._validate_action_chunk(converted)

    assert len(validated) == 2
    assert adapter.wuji_calls == [("left", 3), ("right", 3), ("left", 3), ("right", 3)]
    for step, clean in enumerate(validated):
        assert clean["left_ee_joint_state"].shape == (12,)
        assert clean["right_ee_joint_state"].shape == (12,)
        assert clean["left_ee_pose"].shape == (7,)
        assert clean["right_ee_pose"].shape == (7,)
        np.testing.assert_array_equal(
            clean["left_ee_joint_state"], original[step]["left_ee_joint_state"][:12]
        )
        np.testing.assert_array_equal(
            clean["right_ee_joint_state"], original[step]["right_ee_joint_state"][:12]
        )
    # Conversion copies each action dictionary and never rewrites the internal
    # Wuji20 arrays supplied by the model output.
    for before, after in zip(original, actions):
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])


@pytest.mark.parametrize(
    "bad_observation",
    [
        _observation(right=np.zeros(11, dtype=np.float32)),
        _observation(right=np.array([np.nan] + [0.0] * 11, dtype=np.float32)),
    ],
)
def test_internalize_rejects_bad_shape_or_nonfinite_and_rolls_back(
    bad_observation: dict,
) -> None:
    adapter = _BoundaryAdapterDouble()
    model = _bare_model(adapter)
    original = copy.deepcopy(bad_observation)

    with pytest.raises((KeyError, ValueError)):
        model._internalize_observation(bad_observation, env_idx=11)

    assert adapter.snapshot_calls == [11]
    assert len(adapter.restore_calls) == 1
    assert adapter.restore_calls[0][0] == 11
    # Left conversion increments the double before right-side validation fails;
    # transactional restore must remove that partial state.
    assert adapter.counter == adapter.restore_calls[0][1]["counter"] == 0
    np.testing.assert_array_equal(
        bad_observation["state"]["left_ee_joint_state"],
        original["state"]["left_ee_joint_state"],
    )


def test_externalize_rejects_bad_internal_shape_and_rolls_back() -> None:
    adapter = _BoundaryAdapterDouble()
    model = _bare_model(adapter)
    actions = _action_chunk()
    actions[1]["right_ee_joint_state"] = np.zeros(19, dtype=np.float32)
    original = copy.deepcopy(actions)

    with pytest.raises(ValueError):
        model._externalize_action_chunk(actions, env_idx=13)

    assert adapter.snapshot_calls == [13]
    assert len(adapter.restore_calls) == 1
    assert adapter.counter == 0
    for before, after in zip(original, actions):
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda action: action.pop("right_ee_pose"), "missing"),
        (
            lambda action: action.__setitem__(
                "left_ee_joint_state", np.zeros(11, dtype=np.float32)
            ),
            "finite 12-D",
        ),
        (
            lambda action: action.__setitem__(
                "right_ee_joint_state",
                np.array([np.nan] + [0.0] * 11, dtype=np.float32),
            ),
            "finite 12-D",
        ),
    ],
)
def test_validate_action_chunk_is_strict_and_fail_closed(mutator, match: str) -> None:
    model = _bare_model(_BoundaryAdapterDouble())
    action = model._externalize_action_chunk(_action_chunk()[:1], env_idx=0)[0]
    mutator(action)
    with pytest.raises((KeyError, ValueError), match=match):
        model._validate_action_chunk([action])


def test_epoch_checkpoint_directory_resolves_run_metadata(tmp_path, monkeypatch) -> None:
    """A Web-style ``epoch=*.ckpt`` path must stay paired with its run config."""

    policy_dir = tmp_path / "policy"
    checkpoints_dir = policy_dir / "checkpoints"
    run_dir = policy_dir / "checkpoints" / "run_TB64"
    epoch_dir = run_dir / "checkpoints" / "epoch=33-step=80000.ckpt"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "weights.pt").write_bytes(b"weights")
    (run_dir / "stats.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.json").write_text(
        '{"statistics_path": "stats.json"}', encoding="utf-8"
    )
    monkeypatch.setattr(vitra_model, "_POLICY_DIR", policy_dir)
    monkeypatch.setattr(vitra_model, "_CHECKPOINTS_DIR", checkpoints_dir)

    weights, metadata_root = vitra_model._resolve_weights(
        {"model_path": str(epoch_dir)}
    )
    assert weights == (epoch_dir / "weights.pt").absolute()
    assert metadata_root == run_dir.resolve()
    config_path, statistics_path = vitra_model._resolve_config_and_statistics(
        {}, weights, metadata_root
    )
    assert config_path == (run_dir / "config.json").resolve()
    assert statistics_path == (run_dir / "stats.json").resolve()
