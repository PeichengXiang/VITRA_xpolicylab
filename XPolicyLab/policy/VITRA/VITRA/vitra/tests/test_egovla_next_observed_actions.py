from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


POLICY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POLICY_ROOT))

# spark0_dataset imports this decoder while defining shared pose helpers.  The
# EgoVLA test stores decoded uint8 images, so the external bitstream decoder is
# never called; a narrow stub keeps this source-snapshot test self-contained.
process_data = types.ModuleType("XPolicyLab.utils.process_data")
process_data.decode_image_bit = lambda value: value
xpolicylab = types.ModuleType("XPolicyLab")
xpolicylab.__path__ = []
xpolicylab_utils = types.ModuleType("XPolicyLab.utils")
xpolicylab_utils.__path__ = []
sys.modules.setdefault("XPolicyLab", xpolicylab)
sys.modules.setdefault("XPolicyLab.utils", xpolicylab_utils)
sys.modules.setdefault("XPolicyLab.utils.process_data", process_data)

from vitra.datasets import egovla_inspire_dataset as dataset  # noqa: E402


def _pose7(position: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat()
    return np.concatenate([position, quat_xyzw[[3, 0, 1, 2]]]).astype(np.float32)


class EgoVLANextObservedActionTest(unittest.TestCase):
    FRAME_COUNT = 17

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.mapping_path = self.root / "mapping.json"
        self.episode_path = self.root / "Open-Drawer" / "episode_0.hdf5"
        self.episode_path.parent.mkdir(parents=True)
        self._write_mapping()
        self.observed_pose, self.qpos = self._write_episode()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_mapping(self) -> None:
        mapping = {
            "codec_id": dataset.INSPIRE2MANO_CODEC_ID,
            "joint_mapping": [
                {
                    "source_index": source,
                    "mano_pose_index": destination,
                    "sign": sign,
                }
                for source, (destination, sign) in enumerate(
                    zip(dataset._INSPIRE2MANO_DST, dataset._INSPIRE2MANO_SIGNS)
                )
            ],
            "calibration": {
                "left_ee_to_vitra_wrist": np.eye(4).tolist(),
                "right_ee_to_vitra_wrist": np.eye(4).tolist(),
            },
        }
        self.mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    def _write_episode(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        frame = np.arange(self.FRAME_COUNT, dtype=np.float64)
        observed_pose = {
            "left": np.stack(
                [
                    _pose7(
                        np.array([0.40 + 0.004 * i, 0.15 + 0.001 * i, 0.90 - 0.002 * i]),
                        np.array([0.003 * i, -0.002 * i, 0.004 * i]),
                    )
                    for i in frame
                ]
            ),
            "right": np.stack(
                [
                    _pose7(
                        np.array([0.42 - 0.003 * i, -0.16 + 0.002 * i, 0.88 + 0.001 * i]),
                        np.array([-0.002 * i, 0.003 * i, -0.005 * i]),
                    )
                    for i in frame
                ]
            ),
        }

        qpos = np.zeros((self.FRAME_COUNT, dataset.QPOS_DIM), dtype=np.float32)
        for side_index, side in enumerate(dataset.SIDES):
            indices = list(dataset.HAND_QPOS_INDICES[side])
            values = (
                0.1 * side_index
                + 0.01 * frame[:, None]
                + 0.001 * np.arange(dataset.INSPIRE_HAND_DIM)[None, :]
            )
            qpos[:, indices] = values.astype(np.float32)

        # Deliberately poison controller targets.  A next-observed-state label
        # must be independent of both arrays.
        controller_action = np.full_like(qpos, 123.0)
        controller_pose = {
            side: np.stack(
                [
                    _pose7(
                        np.array([4.0 + i, -3.0, 2.0]),
                        np.array([0.4, -0.3, 0.2]),
                    )
                    for i in frame
                ]
            )
            for side in dataset.SIDES
        }

        with h5py.File(self.episode_path, "w") as handle:
            handle.create_dataset("action", data=controller_action)
            observations = handle.create_group("observations")
            observations.create_dataset("qpos", data=qpos)
            images = observations.create_group("images")
            images.create_dataset(
                "main",
                shape=(self.FRAME_COUNT, *dataset.EGOVLA_IMAGE_HW, 3),
                dtype=np.uint8,
                compression="gzip",
            )
            for side in dataset.SIDES:
                observations.create_dataset(f"{side}_ee_pose", data=observed_pose[side])
                observations.create_dataset(
                    f"{side}_target_ee_pose", data=controller_pose[side]
                )
        return observed_pose, qpos

    def _dataset(self) -> dataset.EgoVLAInspireDatasetCore:
        return dataset.EgoVLAInspireDatasetCore(
            str(self.root),
            action_past_window_size=0,
            action_future_window_size=15,
            load_images=False,
            mapping_path=str(self.mapping_path),
        )

    def test_sixteen_step_labels_decode_to_next_observed_trajectory(self) -> None:
        core = self._dataset()
        sample = core[0]
        np.testing.assert_array_equal(sample["action_mask"], np.ones((16, 2), dtype=bool))

        context = {
            "camera_vitra_to_env": dataset.OFFICIAL_CAMERA_VITRA,
            "current_wrist_in_vitra_camera": {
                side: core._wrist_in_camera(self.observed_pose[side][0])
                for side in dataset.SIDES
            },
            "ee_to_vitra_wrist": {side: np.eye(4) for side in dataset.SIDES},
        }
        commands = dataset.inspire_action_chunk_to_xpolicy(sample["action_list"], context)
        self.assertEqual(len(commands), 16)

        for step, command in enumerate(commands, start=1):
            for side in dataset.SIDES:
                actual = dataset.pose7_to_matrix(command[f"{side}_ee_pose"])
                expected = dataset.pose7_to_matrix(self.observed_pose[side][step])
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.0)
                np.testing.assert_allclose(
                    command[f"{side}_ee_joint_state"],
                    dataset.unpack_inspire12(self.qpos[step], side),
                    atol=0.0,
                    rtol=0.0,
                )

    def test_terminal_and_out_of_range_rows_are_masked(self) -> None:
        core = self._dataset()

        penultimate = core[self.FRAME_COUNT - 2]
        np.testing.assert_array_equal(
            penultimate["action_mask"][0], np.array([True, True])
        )
        self.assertFalse(penultimate["action_mask"][1:].any())

        terminal = core[self.FRAME_COUNT - 1]
        self.assertFalse(terminal["action_mask"].any())
        np.testing.assert_array_equal(
            terminal["action_list"], np.zeros_like(terminal["action_list"])
        )


if __name__ == "__main__":
    unittest.main()
