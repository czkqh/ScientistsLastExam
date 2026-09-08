"""Weak but valid baseline: one broad survey, then calibrated refusal."""
import numpy as np


def infer_source(station_bounds, wave_types, observe, budget_units):
    del wave_types, budget_units
    x0, x1 = station_bounds[0]
    y0, y1 = station_bounds[1]
    stations = [[float(x0), float(y0)], [float(x1), float(y0)],
                [float(x1), float(y1)], [float(x0), float(y1)]]
    observe(stations, "P")
    return {"moment_tensor": np.zeros(6), "depth_km": 30.0,
            "magnitude": 0.0, "confidence": 0.0, "abstain": True}
