"""Executable 192-strategy low-dimensional shortcut probe."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np

TENSORS = (
    (0.7071, -0.7071, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.7071, 0.0, 0.0),
    (0.0, -0.7071, 0.7071, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.7071, 0.0),
    (-0.7071, 0.0, 0.7071, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.7071),
)
DEPTHS = tuple(np.linspace(12.0, 48.0, 8))
THRESHOLDS = (0.006, 0.012, 0.024, 0.048)
STATIONS = ((-240.0, -100.0), (-170.0, 190.0), (20.0, 250.0),
            (210.0, 150.0), (250.0, -70.0), (100.0, -240.0),
            (-80.0, -250.0), (-230.0, -170.0))


def _evaluator():
    path = Path(__file__).with_name("evaluator.py")
    spec = importlib.util.spec_from_file_location("seismic_shortcut_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(tensor, depth, threshold):
    def infer_source(station_bounds, wave_types, observe, budget_units):
        del station_bounds, wave_types, budget_units
        p = observe(STATIONS, "P")
        observe(STATIONS, "S")
        if float(np.sqrt(np.mean(np.asarray(p["amplitude"]) ** 2))) < threshold:
            return {"moment_tensor": np.zeros(6), "depth_km": depth,
                    "magnitude": 0.0, "confidence": 0.0, "abstain": True}
        return {"moment_tensor": np.asarray(tensor), "depth_km": depth,
                "magnitude": 3.2, "confidence": 0.5, "abstain": False}
    return infer_source


def run_probe():
    evaluator = _evaluator()
    rows = []
    for tensor in TENSORS:
        for depth in DEPTHS:
            for threshold in THRESHOLDS:
                metrics = evaluator.evaluate(candidate(tensor, depth, threshold))
                rows.append((float(metrics["combined_score"]), tensor, depth, threshold))
    return max(rows, key=lambda row: row[0]), len(rows)


if __name__ == "__main__":
    best, count = run_probe()
    print({"strategies": count, "best_combined_score": best[0],
           "depth_km": best[2], "amplitude_threshold": best[3]})
