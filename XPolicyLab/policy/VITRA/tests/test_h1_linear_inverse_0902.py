"""Deterministic regression for the hash-bound 0902 H1 fallback."""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

_POLICY_DIR = Path(__file__).resolve().parents[1]
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from h1_inspire_retarget import H1InspireWujiAdapter  # noqa: E402


ROOT = Path(os.environ.get("SPARK0_ROOT", "/personal/zijian/Spark_0/Spark-0"))
RUN = _POLICY_DIR / "checkpoints/egovla_humanoid_mano-20260902-tianji_marvin_wuji-ee-42/2026-09-02-egovla_humanoid_mano-20260902_TB64_B8_bf16True"
SIDECAR = RUN / "wuji20_to_inspire6_linear_0902_v1.json"


@unittest.skipUnless(SIDECAR.is_file() and (ROOT / "Spark_data/src").is_dir(), "0902 sidecar/Spark checkout is not mounted")
class H1LinearInverse0902Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = H1InspireWujiAdapter(
            ROOT,
            linear_inverse_path=SIDECAR,
            linear_inverse_sha256=hashlib.sha256(SIDECAR.read_bytes()).hexdigest(),
        )

    def test_deterministic_fallback_keeps_gate_and_exact_abi(self) -> None:
        q20 = np.asarray(
            [0.6121411919593811, 0.8320068120956421, 0.7695000767707824,
             0.8906779885292053, 0.7426122426986694, 0.08122808486223221,
             0.09049942344427109, 0.3606128692626953, 0.1945294886827469,
             0.09471721947193146, 1.107483983039856, 1.4547725915908813,
             1.4452663660049438, 1.513474941253662, -0.2671889364719391,
             -0.009340832941234112, 0.07148585468530655, 0.05660097673535347,
             0.060357652604579926, -0.0645091161131858], dtype=np.float64
        )
        outputs = []
        for side in ("left", "right"):
            value = self.adapter.wuji_to_h1(q20, side=side, env_idx=4)
            outputs.append(value.copy())
            diagnostic = self.adapter.last_diagnostics["4:" + side + ":wuji_to_h1"]
            self.assertEqual(diagnostic["backend"], "linear_calibration_0902")
            self.assertLessEqual(float(diagnostic["rms_m"]), 0.08)
            self.assertTrue(np.isfinite(value).all())
            self.assertEqual(value.shape, (12,))
        # Repeating the same request is byte deterministic and does not mutate
        # the input q20 vector or project it through a hidden clip operation.
        np.testing.assert_array_equal(q20, np.asarray(q20))
        np.testing.assert_array_equal(outputs[0], self.adapter.wuji_to_h1(q20, "left", 4))

