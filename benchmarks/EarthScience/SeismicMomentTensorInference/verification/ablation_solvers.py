"""Simple ablations used to document the capability ladder."""
import numpy as np


def p_only_refusal(station_bounds, wave_types, observe, budget_units):
    del station_bounds, wave_types, budget_units
    observe([[-250, -50], [-150, 220], [80, 250], [250, 0]], "P")
    return {"source_xy_km": np.zeros(2), "moment_tensor": np.zeros(6), "depth_km": 30.0, "magnitude": 0.0, "confidence": 0.0, "abstain": True}


def single_azimuth_refusal(station_bounds, wave_types, observe, budget_units):
    del station_bounds, wave_types, budget_units
    observe([[-250, 0], [-100, 0], [100, 0], [250, 0]], "P")
    return {"source_xy_km": np.zeros(2), "moment_tensor": np.zeros(6), "depth_km": 30.0, "magnitude": 0.0, "confidence": 0.0, "abstain": True}
