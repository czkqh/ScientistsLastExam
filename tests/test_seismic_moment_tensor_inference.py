import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "benchmarks/EarthScience/SeismicMomentTensorInference/verification/evaluator.py"


def load_eval():
    spec = importlib.util.spec_from_file_location("seismic_mti_eval", EVAL)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class SeismicMomentTensorInvariantTests(unittest.TestCase):
    def test_p_radiation_matches_public_equation(self):
        e = load_eval()
        tensor = [0.2, -0.1, 0.05, 0.03, -0.02, 0.04]
        station = np.asarray([[120.0, -80.0]])
        source = np.asarray([15.0, -12.0])
        amplitude, arrival = e._radiation(tensor, 30.0, station, "P", source)
        displacement = np.r_[station[0] - source, -30.0]
        distance = np.linalg.norm(displacement)
        direction = displacement / distance
        expected = 9000.0 * direction @ e._matrix(tensor) @ direction / distance**2
        self.assertAlmostEqual(amplitude[0], expected, places=14)
        self.assertAlmostEqual(arrival[0], distance / 6.0, places=14)

    def test_magnitude_scale_and_source_location_are_active(self):
        e = load_eval()
        tensor = np.asarray([0.2, -0.1, -0.1, 0.03, -0.02, 0.04])
        stations = [[120.0, -80.0], [-90.0, 170.0]]
        low, t0 = e._radiation(tensor, 30.0, stations, "P", [0.0, 0.0])
        high, t1 = e._radiation(10.0 * tensor, 30.0, stations, "P", [0.0, 0.0])
        _, shifted = e._radiation(tensor, 30.0, stations, "P", [40.0, -20.0])
        np.testing.assert_allclose(high, 10.0 * low)
        self.assertFalse(np.allclose(t0, shifted))

    def test_baseline_is_valid_and_zero(self):
        e = load_eval()
        path = ROOT / "benchmarks/EarthScience/SeismicMomentTensorInference/solution.py"
        spec = importlib.util.spec_from_file_location("seismic_baseline", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        out = e.evaluate(mod.infer_source)
        self.assertEqual(out["valid"], 1.0)
        self.assertEqual(out["combined_score"], 0.0)

    def test_ten_malformed_candidates_fail_closed(self):
        e = load_eval()
        bad = [
            lambda *a: None,
            lambda *a: {},
            lambda *a: "wrong",
            lambda *a: {"abstain": False, "source_xy_km": [0, 0], "moment_tensor": [0], "depth_km": 20, "magnitude": 3, "confidence": 1},
            lambda *a: {"abstain": False, "source_xy_km": [0, 0], "moment_tensor": [0]*5+[float("nan")], "depth_km": 20, "magnitude": 3, "confidence": 1},
            lambda *a: {"abstain": False, "moment_tensor": [0]*6, "depth_km": 20, "magnitude": 3, "confidence": 1},
            lambda *a: {"abstain": False, "source_xy_km": [80, 0], "moment_tensor": [0]*6, "depth_km": 20, "magnitude": 3, "confidence": 1},
            lambda *a: {"abstain": True, "source_xy_km": [0, 0], "moment_tensor": [1]*6, "depth_km": 20, "magnitude": 0, "confidence": 0},
            lambda *a: {"abstain": True, "source_xy_km": [0, 0], "moment_tensor": [0]*6, "depth_km": 20, "magnitude": 0, "confidence": 2},
            lambda *a: (_ for _ in ()).throw(RuntimeError("candidate error")),
        ]
        for candidate in bad:
            metrics = e.evaluate(candidate)
            self.assertEqual(metrics["valid"], 0.0)
            self.assertEqual(metrics["combined_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
