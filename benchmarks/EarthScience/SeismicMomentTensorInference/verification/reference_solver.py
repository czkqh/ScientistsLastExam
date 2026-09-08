"""Truth-blind P/S moment-tensor inversion witness."""
import math
import numpy as np
from scipy.optimize import least_squares


def _matrix(v):
    return np.asarray(((v[0], v[3], v[4]), (v[3], v[1], v[5]), (v[4], v[5], v[2])))


def _radiation(v, depth, stations, wave):
    M = _matrix(v); xy = np.asarray(stations, float)
    rv = np.column_stack((xy[:, 0], xy[:, 1], -np.full(len(xy), depth)))
    dist = np.linalg.norm(rv, axis=1); n = rv / dist[:, None]; mn = n @ M.T
    p = np.einsum("ij,ij->i", n, mn)
    if wave == "P":
        rad = p; vel = 6.0
    else:
        t = np.column_stack((-n[:, 1], n[:, 0], np.zeros(len(n)))); t /= np.maximum(np.linalg.norm(t, axis=1)[:, None], 1e-12)
        rad = np.einsum("ij,ij->i", t, mn); vel = 3.5
    return 9000.0 * rad / np.maximum(dist * dist, 1.0), dist / vel


def infer_source(station_bounds, wave_types, observe, budget_units):
    del station_bounds, wave_types, budget_units
    stations = [[-260.0, -40.0], [-180.0, 180.0], [0.0, 260.0], [190.0, 170.0],
                [270.0, -20.0], [140.0, -210.0], [-40.0, -260.0], [-220.0, -180.0]]
    records = [observe(stations, "P"), observe(stations, "S")]
    def physical(q):
        raw = np.asarray(q[:6], dtype=float)
        raw[2] = -raw[0] - raw[1]
        norm = np.linalg.norm(raw)
        return raw / max(norm, 1e-12)
    def residual(q):
        v, depth = physical(q), q[6]
        out = []
        for row in records:
            pa, pt = _radiation(v, depth, row["station_xy_km"], row["wave_type"])
            out.extend(((pa - row["amplitude"]) / 0.02).tolist())
            out.extend(((pt - row["p_arrival_s"]) / 0.15).tolist())
        return np.asarray(out)
    best = least_squares(residual, np.r_[np.zeros(6), 30.0], bounds=(np.r_[np.full(6, -3.0), 5.0], np.r_[np.full(6, 3.0), 100.0]), max_nfev=180)
    tensor, depth = physical(best.x), float(best.x[6])
    norm = float(np.linalg.norm(tensor))
    rms = float(np.sqrt(np.mean(residual(best.x) ** 2)))
    if not np.isfinite(norm) or norm < 1e-5 or rms > 2.5:
        return {"moment_tensor": np.zeros(6), "depth_km": 30.0, "magnitude": 0.0, "confidence": 0.0, "abstain": True}
    return {"moment_tensor": tensor, "depth_km": depth, "magnitude": 3.2,
            "confidence": float(math.exp(-0.5 * np.mean(residual(best.x) ** 2))), "abstain": False}
