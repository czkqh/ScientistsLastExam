import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "benchmarks/EarthScience/SeismicMomentTensorInference/verification/evaluator.py"


def load_eval():
    spec = importlib.util.spec_from_file_location("seismic_mti_eval", EVAL)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class SeismicMomentTensorInvariantTests(unittest.TestCase):
    def test_radiation_is_symmetric_tensor_invariant(self):
        e = load_eval()
        tensor = [0.2, -0.1, 0.05, 0.03, -0.02, 0.04]
        a, t = e._radiation(tensor, 30.0, [[-100, 40], [120, -80]], "P")
        swapped = [0.2, -0.1, 0.05, 0.03, -0.02, 0.04]
        b, u = e._radiation(swapped, 30.0, [[-100, 40], [120, -80]], "P")
        self.assertEqual(a.tolist(), b.tolist()); self.assertEqual(t.tolist(), u.tolist())

    def test_baseline_is_valid_and_zero(self):
        e = load_eval()
        path = ROOT / "benchmarks/EarthScience/SeismicMomentTensorInference/solution.py"
        spec = importlib.util.spec_from_file_location("seismic_baseline", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        out = e.evaluate(mod.infer_source)
        self.assertEqual(out["valid"], 1.0)
        self.assertEqual(out["combined_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
