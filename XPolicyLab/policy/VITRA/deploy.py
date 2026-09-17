"""Standard XPolicyLab evaluation loops for VITRA."""

import os
import sys
from collections.abc import Mapping, Sequence

import numpy as np


_LIVE_REBASE_ENV = "VITRA_LIVE_REBASE"
_LIVE_REBASE_ENV_CFG = "ego_h1_inspire"
_LIVE_REBASE_ACTION_TYPE = "ee"
_POSE_KEYS = ("left_ee_pose", "right_ee_pose")
_HAND_KEYS = ("left_ee_joint_state", "right_ee_joint_state")


def _live_rebase_enabled():
    """Return the explicit live-rebase opt-in and reject ambiguous settings."""

    raw = os.environ.get(_LIVE_REBASE_ENV, "0").strip()
    if raw in {"", "0"}:
        return False
    if raw != "1":
        raise RuntimeError(
            f"{_LIVE_REBASE_ENV} must be exactly 0 or 1, got {raw!r}"
        )

    env_cfg_type = os.environ.get("EGOVLA_ENV_CFG_TYPE", "").strip()
    action_type = os.environ.get("EGOVLA_ACTION_TYPE", "").strip()
    if env_cfg_type != _LIVE_REBASE_ENV_CFG or action_type != _LIVE_REBASE_ACTION_TYPE:
        raise RuntimeError(
            f"{_LIVE_REBASE_ENV}=1 is supported only for "
            f"EGOVLA_ENV_CFG_TYPE={_LIVE_REBASE_ENV_CFG!r} and "
            f"EGOVLA_ACTION_TYPE={_LIVE_REBASE_ACTION_TYPE!r}; got "
            f"env_cfg_type={env_cfg_type!r}, action_type={action_type!r}"
        )
    print(
        "[VITRA] live_rebase=1 "
        f"env_cfg_type={env_cfg_type} action_type={action_type}",
        file=sys.stderr,
        flush=True,
    )
    return True


def _log_live_rebase_chunk(chunk_size, *, batch_sizes=None):
    details = ""
    if batch_sizes is not None:
        details = f" returned_chunk_sizes={list(batch_sizes)}"
    print(
        f"[VITRA] live_rebase=1 chunk_range=[0,{int(chunk_size)}){details}",
        file=sys.stderr,
        flush=True,
    )


def _normalise_quaternion_wxyz(value, *, label):
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.size != 4 or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{label} must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} quaternion has zero norm")
    return quaternion / norm


def _quaternion_multiply_wxyz(left, right):
    """Hamilton product for two normalised-or-normalisable wxyz quaternions."""

    lw, lx, ly, lz = _normalise_quaternion_wxyz(left, label="left quaternion")
    rw, rx, ry, rz = _normalise_quaternion_wxyz(right, label="right quaternion")
    product = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return _normalise_quaternion_wxyz(product, label="quaternion product")


def _quaternion_inverse_wxyz(value):
    quaternion = _normalise_quaternion_wxyz(value, label="quaternion inverse input")
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)


def _pose7(source, key, *, label):
    if not isinstance(source, Mapping):
        raise TypeError(f"{label} must be a mapping")
    container = source.get("state", source)
    if not isinstance(container, Mapping) or key not in container:
        raise KeyError(f"{label} is missing {key!r}")
    pose = np.asarray(container[key], dtype=np.float64).reshape(-1)
    if pose.size != 7 or not np.all(np.isfinite(pose)):
        raise ValueError(f"{label}.{key} must be a finite xyz+wxyz pose with dim 7")
    pose = pose.copy()
    pose[3:] = _normalise_quaternion_wxyz(
        pose[3:], label=f"{label}.{key}"
    )
    return pose


def _validate_hand_targets(action):
    for key in _HAND_KEYS:
        if key not in action:
            raise KeyError(f"VITRA live rebase action is missing {key!r}")
        value = np.asarray(action[key])
        if value.size != 12 or not np.all(np.isfinite(value)):
            raise ValueError(
                f"VITRA live rebase {key} must be finite with dim 12, got {value.shape}"
            )


def _live_rebase_action(original_action, previous_original, live_observation):
    """Apply one original-command residual to the latest real EE poses.

    VITRA's Inspire decoder uses additive world-frame translation and
    left-multiplied rotation. ``previous_original`` therefore advances along
    the untouched model command sequence; it must never be the rebased output.
    Hand-joint targets remain byte-for-byte the original action values.
    """

    if not isinstance(original_action, Mapping):
        raise TypeError("VITRA live rebase action must be a mapping")
    _validate_hand_targets(original_action)
    rebased = dict(original_action)
    for key in _POSE_KEYS:
        command = _pose7(original_action, key, label="original action")
        previous = _pose7(previous_original, key, label="previous original")
        live = _pose7(live_observation, key, label="live observation")

        delta_position = command[:3] - previous[:3]
        delta_rotation = _quaternion_multiply_wxyz(
            command[3:], _quaternion_inverse_wxyz(previous[3:])
        )
        target_quaternion = _quaternion_multiply_wxyz(delta_rotation, live[3:])
        # q and -q encode the same rotation.  Follow the original command's
        # hemisphere to keep logs and downstream interpolation deterministic.
        if float(np.dot(target_quaternion, command[3:])) < 0.0:
            target_quaternion = -target_quaternion

        rebased[key] = np.concatenate(
            [live[:3] + delta_position, target_quaternion]
        ).astype(np.float32)
    return rebased


def _validate_live_rebase_batch(actions, env_idx_list):
    if not isinstance(actions, Sequence) or len(actions) != len(env_idx_list):
        raise RuntimeError(
            "VITRA live rebase requires one action chunk per running env: "
            f"got {len(actions) if isinstance(actions, Sequence) else type(actions).__name__} "
            f"for {len(env_idx_list)} envs"
        )
    if any(not isinstance(chunk, Sequence) or not chunk for chunk in actions):
        raise RuntimeError("VITRA live rebase received an empty batched action chunk")


def _attach_env_indices(obs_list, env_idx_list):
    """Bind batched observations to the environment ids owned by TASK_ENV.

    Some environments omit ``obs['env_idx']``.  Enumeration is not a stable
    substitute after one member of a batch finishes, and MANO45 keeps a
    stride-2 solver history per id.  Copy the outer dict so this metadata does
    not mutate the environment's observation object.
    """

    if len(obs_list) != len(env_idx_list):
        raise ValueError(
            "Observation batch length does not match running env ids: "
            f"{len(obs_list)} != {len(env_idx_list)}"
        )
    bound = []
    for observation, env_idx in zip(obs_list, env_idx_list):
        if not isinstance(observation, dict):
            raise TypeError("VITRA batched observations must be dictionaries")
        copied = dict(observation)
        declared = copied.get("env_idx")
        if declared is not None and int(declared) != int(env_idx):
            raise ValueError(
                f"Observation env_idx={declared} disagrees with TASK_ENV id={env_idx}"
            )
        copied["env_idx"] = int(env_idx)
        bound.append(copied)
    return bound


def eval_one_episode(TASK_ENV, model_client):
    live_rebase = _live_rebase_enabled()
    model_client.call(func_name="reset")
    logged_chunk_sizes = set()

    while not TASK_ENV.is_episode_end():
        query_obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=query_obs)
        actions = model_client.call(func_name="get_action")
        if live_rebase:
            if not isinstance(actions, Sequence) or not actions:
                raise RuntimeError("VITRA live rebase received an empty action chunk")
            if len(actions) not in logged_chunk_sizes:
                _log_live_rebase_chunk(len(actions))
                logged_chunk_sizes.add(len(actions))

        previous_original = query_obs
        live_obs = query_obs
        for action_idx, action in enumerate(actions):
            action_to_execute = (
                _live_rebase_action(action, previous_original, live_obs)
                if live_rebase
                else action
            )
            TASK_ENV.take_action(action_to_execute)
            previous_original = action
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            live_obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=live_obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    live_rebase = _live_rebase_enabled()
    model_client.call(func_name="reset")
    logged_chunk_signatures = set()

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        if not env_idx_list:
            break
        obs_list = _attach_env_indices(
            TASK_ENV.get_obs_batch(env_idx_list), env_idx_list
        )
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        # XPolicyLab's RPC payload field is named ``obs`` for arbitrary method
        # arguments. Model.get_action_batch receives this value as env_idx_list.
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        if not actions or not actions[0]:
            raise RuntimeError("VITRA returned an empty batched action chunk")
        if live_rebase:
            _validate_live_rebase_batch(actions, env_idx_list)

        chunk_size = min(len(env_actions) for env_actions in actions)
        if live_rebase:
            signature = tuple(len(env_actions) for env_actions in actions)
            if signature not in logged_chunk_signatures:
                _log_live_rebase_chunk(chunk_size, batch_sizes=signature)
                logged_chunk_signatures.add(signature)
        previous_original_by_env = {
            int(env_idx): obs
            for env_idx, obs in zip(env_idx_list, obs_list)
        }
        live_obs_by_env = dict(previous_original_by_env)
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            if live_rebase:
                action_to_execute = [
                    _live_rebase_action(
                        action,
                        previous_original_by_env[int(env_idx)],
                        live_obs_by_env[int(env_idx)],
                    )
                    for action, env_idx in zip(current_action_list, env_idx_list)
                ]
            else:
                action_to_execute = current_action_list
            TASK_ENV.take_action_batch(action_to_execute, env_idx_list)
            for env_idx, action in zip(env_idx_list, current_action_list):
                previous_original_by_env[int(env_idx)] = action

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [
                index for index, env_idx in enumerate(env_idx_list) if env_idx in running
            ]
            actions = [actions[index] for index in active_batch_idx]
            env_idx_list = [env_idx_list[index] for index in active_batch_idx]
            if not env_idx_list:
                break
            obs_list = _attach_env_indices(
                TASK_ENV.get_obs_batch(env_idx_list), env_idx_list
            )
            live_obs_by_env = {
                int(env_idx): obs
                for env_idx, obs in zip(env_idx_list, obs_list)
            }
            model_client.call(
                func_name="update_obs_batch",
                obs=obs_list,
            )
