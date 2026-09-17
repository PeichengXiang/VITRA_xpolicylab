"""Contract tests for the policy-side H1/Inspire retarget adapter."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_POLICY_DIR = Path(__file__).resolve().parents[1]
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from h1_inspire_retarget import (  # noqa: E402
    H1InspireRetarget,
    H1InspireRetargetError,
    H1RetargetFitError,
    H1_FROM_SPARK,
    INSPIRE6_DIM,
    INSPIRE12_DIM,
    PINNED_SOURCE_SHA256,
    SPARK_FROM_H1,
    WUJI20_DIM,
)


SPARK_ROOT = Path(
    os.environ.get("SPARK0_ROOT", "/personal/zijian/Spark_0/Spark-0")
).expanduser()


class _UnreachableWujiModel:
    """Small deterministic Wuji fixture for the inverse quality gate test."""

    def __init__(self) -> None:
        self.lower = np.full(WUJI20_DIM, -2.5, dtype=np.float64)
        self.upper = np.full(WUJI20_DIM, 2.5, dtype=np.float64)
        self.neutral = np.zeros(WUJI20_DIM, dtype=np.float64)
        self.points = np.zeros((5, 5, 3), dtype=np.float64)
        for finger in range(5):
            for point in range(5):
                self.points[finger, point] = (0.02 * finger, 0.01 * point, 0.0)
        # This artificial downstream fixture uses the common wrist landmark.
        self.points[:, 0] = 0.0

    def forward_points(self, _q20: np.ndarray, canonical: bool = True) -> np.ndarray:
        del canonical
        return self.points.copy()


class _UnreachableInspireModel:
    """A six-DOF model whose target is deterministically 80.271 mm away."""

    actuated_names = tuple(f"inspire_q{index}" for index in range(INSPIRE6_DIM))

    def __init__(self, source: _UnreachableWujiModel) -> None:
        self.neutral = np.zeros(INSPIRE6_DIM, dtype=np.float64)
        self.lower = np.zeros(INSPIRE6_DIM, dtype=np.float64)
        self.upper = np.ones(INSPIRE6_DIM, dtype=np.float64)
        self._source = source

    def forward_points(self, _q6: np.ndarray, canonical: bool = True) -> np.ndarray:
        del canonical
        return self._source.points.copy()

    def scale_target(self, source_points: np.ndarray) -> np.ndarray:
        # Every point has the same 80.271 mm x-offset.  This keeps the
        # residual exactly reproducible while preserving distinct finger
        # spacing, so collision diagnostics remain meaningful.
        target = np.asarray(source_points, dtype=np.float64).copy()
        target[..., 0] += 0.080271
        return target

    def solve(self, *_args: object, **_kwargs: object) -> tuple[np.ndarray, bool, float, int]:
        # This is an intentionally unreachable Wuji pose.  The test verifies
        # that the adapter reports and rejects it; it does not relax the
        # production 80 mm gate.
        return self.neutral.copy(), False, 0.080271, 35

    def pack_inspire12(self, q6: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(q6, dtype=np.float64), np.zeros(6)])


class DeterministicUnreachableFitTests(unittest.TestCase):
    """Regression test for structured diagnostics on an over-limit fit."""

    @staticmethod
    def _adapter() -> H1InspireRetarget:
        adapter = object.__new__(H1InspireRetarget)
        adapter.solver = "analytic"
        adapter.max_iterations = 35
        adapter.tolerance_m = 0.03
        adapter.inverse_max_rms_m = 0.08
        adapter.mimic_tolerance_rad = 2e-5
        adapter._previous_wuji = {}
        adapter._previous_inspire = {}
        adapter.last_diagnostics = {}
        for side in ("left", "right"):
            wuji = _UnreachableWujiModel()
            adapter.wuji_models = getattr(adapter, "wuji_models", {})
            adapter.wuji_models[side] = wuji
            adapter.inspire_models = getattr(adapter, "inspire_models", {})
            adapter.inspire_models[side] = _UnreachableInspireModel(wuji)
        return adapter

    def test_both_sides_reject_overlimit_fit_with_replayable_diagnostics(self) -> None:
        import json

        adapter = self._adapter()
        reports: list[str] = []
        for env_idx, side in enumerate(("left", "right")):
            with self.assertRaises(H1RetargetFitError):
                adapter.wuji_to_h1(np.zeros(WUJI20_DIM), side=side, env_idx=env_idx)
            self.assertNotIn(env_idx, adapter._previous_inspire)
            report = adapter.last_diagnostics[f"{env_idx}:{side}:wuji_to_h1"]
            self.assertEqual(report["schema"], "h1_retarget_diagnostics.v1")
            self.assertEqual(report["direction"], "wuji20_to_inspire12")
            self.assertEqual(report["side"], side)
            self.assertNotIn("quality_gate_override", report)
            self.assertEqual(report["residual"]["available_points"], 25)
            self.assertAlmostEqual(report["residual"]["rms_m"], 0.080271, places=9)
            self.assertAlmostEqual(report["residual"]["max_m"], 0.080271, places=9)
            self.assertAlmostEqual(report["quality_gate"]["rms_returned_m"], 0.080271, places=9)
            self.assertFalse(report["quality_gate"]["passed"])
            self.assertIn("solver_rms_limit_exceeded", report["constraints"]["failed"])
            self.assertIn("inverse_quality_gate_exceeded", report["constraints"]["failed"])
            self.assertEqual(
                set(report["residual"]["per_finger"]),
                {"thumb", "index", "middle", "ring", "pinky"},
            )
            self.assertTrue(all(
                value["count"] == 5
                for value in report["residual"]["per_finger"].values()
            ))
            reports.append(json.dumps(report, sort_keys=True, separators=(",", ":")))

        with self.assertRaises(H1RetargetFitError):
            adapter.wuji_to_h1(np.zeros(WUJI20_DIM), side="left", env_idx=0)
        self.assertEqual(
            reports[0],
            json.dumps(
                adapter.last_diagnostics["0:left:wuji_to_h1"],
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def test_nonfinite_inverse_rms_still_fail_closed(self) -> None:
        adapter = self._adapter()
        inspire = adapter.inspire_models["left"]

        def _nan_solve(*_args: object, **_kwargs: object):
            return inspire.neutral.copy(), False, float("nan"), 35

        inspire.solve = _nan_solve  # type: ignore[method-assign]
        with self.assertRaises(H1RetargetFitError):
            adapter.wuji_to_h1(np.zeros(WUJI20_DIM), side="left", env_idx=0)

    def test_linear_fallback_uses_euclidean_point_rms(self) -> None:
        adapter = self._adapter()
        adapter._linear_inverse = SimpleNamespace(
            predict=lambda side, q: SimpleNamespace(q6=np.zeros(INSPIRE6_DIM))
        )
        # 80.271 mm Euclidean point error exceeds 80 mm. Coordinate RMS
        # would incorrectly divide by sqrt(3) and accept this calibration.
        with self.assertRaisesRegex(H1RetargetFitError, "linear calibration failed quality gate"):
            adapter._linear_wuji_to_h1(
                np.zeros(WUJI20_DIM), side="left", env_idx=0,
                source_points=adapter.wuji_models["left"].forward_points(np.zeros(WUJI20_DIM)),
                geometric_diagnostics={},
            )


@unittest.skipUnless(
    (SPARK_ROOT / "Spark_data" / "src").is_dir(),
    "Spark-0 checkout is not mounted",
)
class H1InspireRetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Forward keeps the published 30 mm gate; the inverse has an explicit
        # 80 mm morphology gate because Wuji2 has more DoF than Inspire12.
        cls.adapter = H1InspireRetarget(
            SPARK_ROOT,
            max_iterations=35,
            tolerance_m=0.03,
            inverse_max_rms_m=0.08,
        )

    def _inspire_sample(self) -> np.ndarray:
        samples = []
        for side in ("left", "right"):
            model = self.adapter.inspire_models[side]
            q6 = np.clip(
                model.neutral
                + np.asarray([0.08, 0.25, 0.20, 0.15, 0.12, 0.06]),
                model.lower,
                model.upper,
            )
            # The Spark kernel returns its packed order.  The public adapter
            # accepts EgoVLA's H1 interleaved order, so apply the audited
            # inverse permutation for the test fixture.
            spark_q12 = np.asarray(model.pack_inspire12(q6), dtype=np.float64)
            samples.append(spark_q12[list(H1_FROM_SPARK)])
        return np.stack(samples)

    def test_pinned_provenance_and_dimensions(self) -> None:
        self.assertEqual(self.adapter.source_sha256, PINNED_SOURCE_SHA256)
        self.assertEqual(
            self.adapter.provenance()["inspire12_order"][0],
            "thumb_proximal_yaw_joint",
        )
        self.assertEqual(
            tuple(self.adapter.provenance()["h1_to_spark_permutation"]),
            SPARK_FROM_H1,
        )

    def test_h1_order_and_official_mimic_canonicalization(self) -> None:
        """Raw H1 mimic slots are regenerated by the audited official reader."""

        spark_model = self.adapter.inspire_models["left"]
        q6 = np.clip(
            spark_model.neutral
            + np.asarray([0.06, 0.21, 0.18, 0.14, 0.11, 0.05]),
            spark_model.lower,
            spark_model.upper,
        )
        spark_q12 = np.asarray(spark_model.pack_inspire12(q6), dtype=np.float64)
        h1_q12 = spark_q12[list(H1_FROM_SPARK)]
        # Simulated H1 exposes independent intermediate drives.  Deliberately
        # perturb those six redundant slots; the official qpos reader ignores
        # them and recomputes the URDF mimics from the six actuated values.
        h1_q12[[1, 3, 5, 7, 10, 11]] += np.asarray(
            [0.17, -0.08, 0.11, -0.05, 0.09, -0.07]
        )
        self.adapter.h1_to_wuji(h1_q12, "left", 23)
        canonical = self.adapter.last_diagnostics["23:left:h1_to_wuji"]
        self.assertEqual(canonical["input_order"], "ego_h1_inspire_interleaved")
        self.assertEqual(canonical["mimic_source"], "official_inspire_urdf")
        # The result must be finite and the public output remains Wuji20.
        self.assertEqual(self.adapter.h1_to_wuji(h1_q12, "left", 24).shape, (20,))

    def test_forward_retry_recovers_without_relaxing_the_fit_gate(self) -> None:
        adapter = self.adapter
        adapter.reset(700)
        q12 = self._inspire_sample()[0]
        model = adapter.wuji_models["left"]
        original_solve = model.solve
        calls = []

        def limited_first_solve(*args, **kwargs):
            calls.append(kwargs.copy())
            if len(calls) == 1:
                # Same failure class as the observed 30.374 mm runtime fit.
                return model.neutral.copy(), False, 0.030374, 35
            return original_solve(*args, **kwargs)

        with patch.object(model, "solve", side_effect=limited_first_solve):
            with self.assertWarnsRegex(RuntimeWarning, "fit recovered"):
                result = adapter.h1_to_wuji(q12, "left", 700)
        self.assertEqual(result.shape, (WUJI20_DIM,))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["tolerance_m"], 0.03)
        self.assertEqual(calls[1]["tolerance_m"], 0.03)
        self.assertEqual(calls[1]["max_iterations"], 105)
        self.assertLessEqual(adapter.last_diagnostics["700:left:h1_to_wuji"]["rms_m"], 0.03)

    def test_forward_unreachable_pose_still_fails_with_replayable_diagnostics(self) -> None:
        adapter = self.adapter
        adapter.reset(701)
        q12 = self._inspire_sample()[0]
        model = adapter.wuji_models["left"]
        with patch.object(model, "solve", return_value=(model.neutral.copy(), False, 0.030374, 35)) as solve:
            with self.assertRaises(H1RetargetFitError) as failure:
                adapter.h1_to_wuji(q12, "left", 701)
        self.assertEqual(solve.call_count, 11)
        self.assertNotIn(701, adapter._previous_wuji)
        diagnostic = failure.exception.diagnostics
        np.testing.assert_allclose(diagnostic["input_h1_q12"], q12)
        self.assertEqual(diagnostic["limit_m"], 0.03)
        self.assertEqual(len(diagnostic["attempts"]), 11)

    def test_sort_cans_observation_escapes_local_minimum_reproducibly(self) -> None:
        # Exact right-hand observation/prior from 8w job 49eff227, rollout 2.
        # Warm/continuation/neutral all fail at >30 mm in the original code.
        q12 = np.array([
            -.3006523549556732, .5800400972366333, 2.5551068782806396,
            -.9258460998535156, -2.156353235244751, -3.0614211559295654,
            -.36138707399368286, -5.21496057510376, .5311524868011475,
            1.372093677520752, -.7323694825172424, .014571934938430786,
        ])
        previous = np.array([
            .6100858860620113, .6954597943065921, .21366200978011918,
            .5850692919117185, 1.291, .07551686771338215,
            .14697076417576274, .2399546435144842, .22476211221996412,
            .698, 1.3324758377628365, 1.318624992929014,
            .5074919060699482, 1.2002221771849428, -1.047,
            -.1645124355387999, -.052359999890542436,
            -.09582237168304908, -.07399700261875669, -1.047,
        ])
        adapter = self.adapter
        before_random = np.random.get_state()
        results = []
        for env_idx in (710, 711):
            adapter.reset(env_idx)
            adapter._previous_wuji[env_idx] = {"right": previous.copy()}
            with self.assertWarnsRegex(RuntimeWarning, "multistart_"):
                results.append(adapter.h1_to_wuji(q12, "right", env_idx))
            diagnostic = adapter.last_diagnostics[f"{env_idx}:right:h1_to_wuji"]
            self.assertTrue(all(not trial["valid"] for trial in diagnostic["attempts"][:3]))
            self.assertLessEqual(diagnostic["rms_m"], .03)
            self.assertEqual(adapter.tolerance_m, .03)
            self.assertLessEqual(len(diagnostic["attempts"]), 11)
            model = adapter.wuji_models["right"]
            self.assertTrue(np.all(results[-1] >= model.lower - 1e-6))
            self.assertTrue(np.all(results[-1] <= model.upper + 1e-6))
        np.testing.assert_array_equal(results[0], results[1])
        for old, new in zip(before_random, np.random.get_state()):
            np.testing.assert_equal(old, new)

    def test_exhausted_multistart_preserves_last_accepted_state(self) -> None:
        adapter = self.adapter
        model = adapter.wuji_models["left"]
        prior = model.neutral.copy()
        adapter._previous_wuji[712] = {"left": prior.copy()}
        with patch.object(model, "solve", return_value=(prior.copy(), False, .04, 35)) as solve:
            with self.assertRaises(H1RetargetFitError):
                adapter.h1_to_wuji(self._inspire_sample()[0], "left", 712)
        self.assertEqual(solve.call_count, 11)
        np.testing.assert_array_equal(adapter._previous_wuji[712]["left"], prior)

    def test_nonfinite_forward_fit_does_not_trigger_multistart(self) -> None:
        adapter = self.adapter
        adapter.reset(713)
        model = adapter.wuji_models["left"]
        with patch.object(model, "solve", return_value=(model.neutral.copy(), False, float("nan"), 35)) as solve:
            with self.assertRaises(H1RetargetFitError):
                adapter.h1_to_wuji(self._inspire_sample()[0], "left", 713)
        self.assertEqual(solve.call_count, 1)
        self.assertNotIn(713, adapter._previous_wuji)

    def test_real_failed_action_uses_matching_wrist_landmarks(self) -> None:
        # Captured from c4cb97ff, seen room 1/table 1, rollout 2: the old
        # chain correspondence produced 127.7949 mm RMS and aborted evaluation.
        q20 = np.array([
            1.0944470167160034, .7905531525611877, 1.3087079524993896,
            1.462129831314087, -1.0027894973754883, -.33123186230659485,
            .30283603072166443, .6980000138282776, .6980000138282776,
            .6980000138282776, 2.0940001010894775, 1.7601027488708496,
            2.0940001010894775, 2.0940001010894775, -.5272721648216248,
            .4457082748413086, .31127697229385376, .7864334583282471,
            .8208107352256775, -.29869720339775085,
        ])
        self.adapter.reset(702)
        for side in ("left", "right"):
            inspire = self.adapter.inspire_models[side]
            points = self.adapter.wuji_models[side].forward_points(q20)
            self.assertTrue(np.all(np.linalg.norm(points[:, 0], axis=-1) > .01))
            _, _, old_rms, _ = inspire.solve(points, np.ones((5, 5)), max_iterations=105)
            self.assertGreater(old_rms, .12)
            result = self.adapter.wuji_to_h1(q20, side, 702)
            self.assertEqual(result.shape, (INSPIRE12_DIM,))
            self.assertTrue(np.isfinite(result).all())
            report = self.adapter.last_diagnostics[f"702:{side}:wuji_to_h1"]
            self.assertLess(report["residual"]["rms_m"], .06)
            self.assertEqual(report["landmark_schema"], "wuji-palm-origin-mcp-chain-v1")
            self.assertEqual(self.adapter.inverse_max_rms_m, .08)

    def test_raw_h1_source_range_is_not_checked_against_spark_target_limits(self) -> None:
        """The pinned EgoVLA reader forwards source H1 angles to FK unchanged."""

        model = self.adapter.inspire_models["left"]
        # These are valid-looking H1 source angles but deliberately fall
        # outside the narrower Spark target URDF bounds (negative yaw and
        # >1.47-rad finger flexion).  The official egovla.py chain has no
        # source-side Spark limit gate; it must still produce a finite Wuji20.
        spark_q6 = np.asarray([-0.05, 0.40, 1.55, 1.55, 1.55, 0.40])
        h1_q12 = np.asarray(model.pack_inspire12(spark_q6))[list(H1_FROM_SPARK)]
        converted = self.adapter.h1_to_wuji(h1_q12, "left", 31)
        self.assertEqual(converted.shape, (WUJI20_DIM,))
        self.assertTrue(np.isfinite(converted).all())

    def test_bidirectional_batch_and_single_shapes(self) -> None:
        q12 = np.repeat(self._inspire_sample()[None, ...], 3, axis=0)
        q20 = self.adapter.inspire12_to_wuji20(q12)
        self.assertEqual(q20.shape, (3, 2, WUJI20_DIM))
        self.assertEqual(self.adapter.inspire12_to_wuji20(q12[0]).shape, (2, WUJI20_DIM))
        q12_back = self.adapter.wuji20_to_inspire12(q20)
        self.assertEqual(q12_back.shape, (3, 2, INSPIRE12_DIM))
        self.assertTrue(np.isfinite(q12_back).all())

    def test_nonfinite_dimension_limit_and_mimic_fail_closed(self) -> None:
        q12 = self._inspire_sample()
        bad = q12.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(H1InspireRetargetError):
            self.adapter.inspire12_to_wuji20(bad)
        with self.assertRaises(H1InspireRetargetError):
            self.adapter.inspire12_to_wuji20(np.zeros((2, 11)))
        bad = q12.copy()
        # H1 source angles are not checked against Spark's target URDF limits:
        # the audited EgoVLA reader passes them to FK unchanged.  Exercise the
        # target-side limit gate instead with an invalid Wuji action.
        bad20 = self.adapter.inspire12_to_wuji20(q12)
        bad20[0, 0] = self.adapter.wuji_models["left"].upper[0] + 0.1
        with self.assertRaises(H1InspireRetargetError):
            self.adapter.wuji20_to_inspire12(bad20)
        # The six H1 intermediate slots are simulator-side redundant drives;
        # changing them must not alter the official canonicalization result.
        perturbed = q12.copy()
        perturbed[0, [1, 3, 5, 7, 10, 11]] += 0.01
        self.adapter.reset()
        reference = self.adapter.inspire12_to_wuji20(q12)
        self.adapter.reset()
        actual = self.adapter.inspire12_to_wuji20(perturbed)
        np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=0)

    def test_stateful_single_hand_abi_and_reset(self) -> None:
        q12_pair = self._inspire_sample()
        left_a = self.adapter.h1_to_wuji(q12_pair[0], "left", 17)
        right_a = self.adapter.h1_to_wuji(q12_pair[1], side="right", env_idx=17)
        self.assertEqual(left_a.shape, (WUJI20_DIM,))
        self.assertEqual(right_a.shape, (WUJI20_DIM,))
        self.assertTrue(np.isfinite(left_a).all())
        self.assertTrue(np.isfinite(right_a).all())
        left_batch = self.adapter.h1_to_wuji(
            np.stack([q12_pair[0], q12_pair[0]]), side="left", env_idx=18
        )
        self.assertEqual(left_batch.shape, (2, WUJI20_DIM))
        q12_back = self.adapter.wuji_to_h1(left_a, "left", 17)
        self.assertEqual(q12_back.shape, (INSPIRE12_DIM,))
        self.assertTrue(np.isfinite(q12_back).all())
        snapshot = self.adapter.snapshot(17)
        wuji_snapshot = snapshot.get("wuji", snapshot.get("previous_wuji"))
        self.assertIsInstance(wuji_snapshot, dict)
        self.adapter.h1_to_wuji(q12_pair[0], "left", 17)
        self.adapter.restore(17, snapshot)
        self.assertTrue(
            np.array_equal(
                self.adapter._previous_wuji[17]["left"],
                wuji_snapshot["left"],
            )
        )
        self.adapter.reset(17)
        self.assertNotIn(17, self.adapter._previous_wuji)
        self.assertNotIn(17, self.adapter._previous_inspire)
        self.adapter.reset()
        self.assertEqual(self.adapter._previous_wuji, {})
        self.assertEqual(self.adapter._previous_inspire, {})


if __name__ == "__main__":
    unittest.main()
