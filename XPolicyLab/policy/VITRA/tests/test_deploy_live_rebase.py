import contextlib
import importlib.util
import io
import math
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "deploy.py"
if not MODULE_PATH.is_file():
    # Keep the standalone review bundle executable before installation under
    # policy/VITRA/tests/.
    MODULE_PATH = Path(__file__).with_name("deploy.py")
SPEC = importlib.util.spec_from_file_location("vitra_deploy_live_rebase", MODULE_PATH)
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


def quat_z(degrees):
    radians = math.radians(degrees) / 2.0
    return np.array([math.cos(radians), 0.0, 0.0, math.sin(radians)])


def pose(x, degrees=0.0, *, y=0.0):
    return np.concatenate([[x, y, 0.0], quat_z(degrees)]).astype(np.float32)


def observation(env_idx, x, degrees=0.0):
    return {
        "env_idx": env_idx,
        "state": {
            "left_ee_pose": pose(x, degrees),
            "right_ee_pose": pose(x, degrees, y=1.0),
        },
    }


def action(x, degrees=0.0, *, hand_marker=0.0):
    return {
        "left_ee_pose": pose(x, degrees),
        "right_ee_pose": pose(x, degrees, y=1.0),
        "left_ee_joint_state": np.full(12, hand_marker, dtype=np.float32),
        "right_ee_joint_state": np.full(12, hand_marker + 0.5, dtype=np.float32),
    }


def assert_quaternion_equivalent(test_case, actual, expected, atol=1e-6):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    actual /= np.linalg.norm(actual)
    expected /= np.linalg.norm(expected)
    test_case.assertAlmostEqual(abs(float(np.dot(actual, expected))), 1.0, delta=atol)


def rotation_matrix_wxyz(value):
    w, x, y, z = np.asarray(value, dtype=np.float64) / np.linalg.norm(value)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class FakeSingleClient:
    def __init__(self, actions):
        self.actions = actions
        self.updated = []
        self.reset_count = 0

    def call(self, *, func_name, obs=None):
        if func_name == "reset":
            self.reset_count += 1
            return None
        if func_name == "update_obs":
            self.updated.append(obs)
            return None
        if func_name == "get_action":
            return self.actions
        raise AssertionError(func_name)


class FakeSingleEnv:
    def __init__(self, live_xs, live_angles=None):
        self.live_xs = live_xs
        self.live_angles = live_angles or [0.0] * len(live_xs)
        self.steps = 0
        self.taken = []

    def is_episode_end(self):
        return self.steps >= len(self.live_xs)

    def get_obs(self):
        return observation(0, self.live_xs[self.steps], self.live_angles[self.steps])

    def take_action(self, value):
        self.taken.append(value)
        self.steps += 1


class FakeBatchClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.updated_env_ids = []
        self.get_action_env_ids = []

    def call(self, *, func_name, obs=None):
        if func_name == "reset":
            return None
        if func_name == "update_obs_batch":
            self.updated_env_ids.append([item["env_idx"] for item in obs])
            return None
        if func_name == "get_action_batch":
            self.get_action_env_ids.append(list(obs))
            return self.chunks
        raise AssertionError(func_name)


class FakeBatchEnv:
    _RUNNING = ([0, 1, 2], [0, 2], [2], [])
    _LIVE_X = (
        {0: 0.0, 1: 100.0, 2: 200.0},
        {0: 10.0, 2: 1000.0},
        {2: 2000.0},
    )

    def __init__(self):
        self.steps = 0
        self.taken = []

    def is_episode_end(self):
        return self.steps >= 3

    def get_running_env_idx_list(self):
        return list(self._RUNNING[self.steps])

    def get_obs_batch(self, env_idx_list):
        return [
            observation(env_idx, self._LIVE_X[self.steps][env_idx])
            for env_idx in env_idx_list
        ]

    def take_action_batch(self, action_list, env_idx_list):
        self.taken.append((list(env_idx_list), list(action_list)))
        self.steps += 1


class LiveRebaseTests(unittest.TestCase):
    def live_rebase_environment(self):
        return mock.patch.dict(
            os.environ,
            {
                "VITRA_LIVE_REBASE": "1",
                "EGOVLA_ENV_CFG_TYPE": "ego_h1_inspire",
                "EGOVLA_ACTION_TYPE": "ee",
            },
            clear=True,
        )

    def test_flag_is_explicit_and_scope_fails_closed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(deploy._live_rebase_enabled())

        cases = [
            ({"VITRA_LIVE_REBASE": "true"}, "exactly 0 or 1"),
            ({"VITRA_LIVE_REBASE": "1"}, "supported only"),
            (
                {
                    "VITRA_LIVE_REBASE": "1",
                    "EGOVLA_ENV_CFG_TYPE": "tianji_marvin_wuji",
                    "EGOVLA_ACTION_TYPE": "ee",
                },
                "supported only",
            ),
            (
                {
                    "VITRA_LIVE_REBASE": "1",
                    "EGOVLA_ENV_CFG_TYPE": "ego_h1_inspire",
                    "EGOVLA_ACTION_TYPE": "joint",
                },
                "supported only",
            ),
        ]
        for environment, message in cases:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(RuntimeError, message):
                        deploy._live_rebase_enabled()

        with self.live_rebase_environment(), contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertTrue(deploy._live_rebase_enabled())
            self.assertIn("live_rebase=1", log.getvalue())

    def test_residual_uses_previous_original_and_preserves_hands(self):
        previous = observation(0, 1.0, 30.0)
        original = action(4.0, 100.0, hand_marker=7.0)
        live = observation(0, 10.0, -20.0)
        original_pose_before = original["left_ee_pose"].copy()

        rebased = deploy._live_rebase_action(original, previous, live)

        np.testing.assert_allclose(rebased["left_ee_pose"][:3], [13.0, 0.0, 0.0])
        np.testing.assert_allclose(rebased["right_ee_pose"][:3], [13.0, 1.0, 0.0])
        assert_quaternion_equivalent(self, rebased["left_ee_pose"][3:], quat_z(50.0))
        self.assertIs(
            rebased["left_ee_joint_state"], original["left_ee_joint_state"]
        )
        self.assertIs(
            rebased["right_ee_joint_state"], original["right_ee_joint_state"]
        )
        np.testing.assert_array_equal(original["left_ee_pose"], original_pose_before)

    def test_random_wxyz_composition_matches_left_multiplied_rotation_matrices(self):
        random = np.random.default_rng(20260916)
        for _ in range(200):
            previous_quaternion = random.normal(size=4)
            command_quaternion = random.normal(size=4)
            live_quaternion = random.normal(size=4)
            previous_position = random.normal(size=3)
            command_position = random.normal(size=3)
            live_position = random.normal(size=3)

            original = action(0.0)
            original["left_ee_pose"] = np.concatenate(
                [command_position, command_quaternion]
            )
            previous = observation(0, 0.0)
            previous["state"]["left_ee_pose"] = np.concatenate(
                [previous_position, previous_quaternion]
            )
            live = observation(0, 0.0)
            live["state"]["left_ee_pose"] = np.concatenate(
                [live_position, live_quaternion]
            )

            result = deploy._live_rebase_action(original, previous, live)[
                "left_ee_pose"
            ]
            np.testing.assert_allclose(
                result[:3],
                live_position + command_position - previous_position,
                atol=2e-7,
            )
            expected_rotation = (
                rotation_matrix_wxyz(command_quaternion)
                @ rotation_matrix_wxyz(previous_quaternion).T
                @ rotation_matrix_wxyz(live_quaternion)
            )
            np.testing.assert_allclose(
                rotation_matrix_wxyz(result[3:]), expected_rotation, atol=2e-6
            )

    def test_single_loop_reads_latest_observation_each_step(self):
        commands = [
            action(1.0, 30.0, hand_marker=1.0),
            action(2.0, 60.0, hand_marker=2.0),
            action(4.0, 90.0, hand_marker=3.0),
        ]
        task_env = FakeSingleEnv([0.0, 10.0, 20.0], [0.0, 10.0, 20.0])
        client = FakeSingleClient(commands)
        with self.live_rebase_environment(), contextlib.redirect_stderr(io.StringIO()) as log:
            deploy.eval_one_episode(task_env, client)

        np.testing.assert_allclose(
            [item["left_ee_pose"][0] for item in task_env.taken],
            [1.0, 11.0, 22.0],
        )
        for item, expected_degrees in zip(task_env.taken, [30.0, 40.0, 50.0]):
            assert_quaternion_equivalent(
                self, item["left_ee_pose"][3:], quat_z(expected_degrees)
            )
        for sent, original in zip(task_env.taken, commands):
            np.testing.assert_array_equal(
                sent["left_ee_joint_state"], original["left_ee_joint_state"]
            )
        self.assertEqual(
            [item["state"]["left_ee_pose"][0] for item in client.updated],
            [0.0, 10.0, 20.0],
        )
        self.assertIn("chunk_range=[0,3)", log.getvalue())

    def test_batch_loop_keeps_original_history_aligned_after_envs_finish(self):
        chunks = [
            [action(1.0, hand_marker=1), action(3.0, hand_marker=2), action(6.0, hand_marker=3)],
            [action(101.0, hand_marker=11), action(103.0, hand_marker=12), action(106.0, hand_marker=13)],
            [action(201.0, hand_marker=21), action(203.0, hand_marker=22), action(206.0, hand_marker=23)],
        ]
        task_env = FakeBatchEnv()
        client = FakeBatchClient(chunks)
        with self.live_rebase_environment(), contextlib.redirect_stderr(io.StringIO()) as log:
            deploy.eval_one_episode_batch(task_env, client)

        self.assertEqual([ids for ids, _ in task_env.taken], [[0, 1, 2], [0, 2], [2]])
        np.testing.assert_allclose(
            [item["left_ee_pose"][0] for item in task_env.taken[0][1]],
            [1.0, 101.0, 201.0],
        )
        np.testing.assert_allclose(
            [item["left_ee_pose"][0] for item in task_env.taken[1][1]],
            [12.0, 1002.0],
        )
        np.testing.assert_allclose(
            [item["left_ee_pose"][0] for item in task_env.taken[2][1]],
            [2003.0],
        )
        self.assertEqual(client.updated_env_ids, [[0, 1, 2], [0, 2], [2]])
        self.assertEqual(client.get_action_env_ids, [[0, 1, 2]])
        np.testing.assert_array_equal(
            task_env.taken[1][1][1]["left_ee_joint_state"],
            chunks[2][1]["left_ee_joint_state"],
        )
        self.assertIn("returned_chunk_sizes=[3, 3, 3]", log.getvalue())

    def test_disabled_single_loop_passes_action_through_unchanged(self):
        sentinel_action = {"sentinel": object()}
        task_env = FakeSingleEnv([0.0])
        client = FakeSingleClient([sentinel_action])
        with mock.patch.dict(os.environ, {"VITRA_LIVE_REBASE": "0"}, clear=True):
            deploy.eval_one_episode(task_env, client)
        self.assertIs(task_env.taken[0], sentinel_action)

    def test_invalid_pose_or_hand_fails_before_execution(self):
        original = action(1.0)
        original["left_ee_pose"][3:] = 0.0
        with self.assertRaisesRegex(ValueError, "zero norm"):
            deploy._live_rebase_action(
                original, observation(0, 0.0), observation(0, 0.0)
            )

        original = action(1.0)
        original["left_ee_joint_state"] = np.zeros(11)
        with self.assertRaisesRegex(ValueError, "dim 12"):
            deploy._live_rebase_action(
                original, observation(0, 0.0), observation(0, 0.0)
            )


if __name__ == "__main__":
    unittest.main()
