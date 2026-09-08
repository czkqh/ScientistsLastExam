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
    p_rms = float(np.sqrt(np.mean(np.asarray(records[0]["amplitude"]) ** 2)))
    s_rms = float(np.sqrt(np.mean(np.asarray(records[1]["amplitude"]) ** 2)))
    # An isotropic source has no transverse S radiation in this public model; a null has
    # neither channel.  Refuse these before fitting the double-couple family.
    p_mean = float(np.mean(np.asarray(records[0]["amplitude"])))
    if (p_rms < 0.012) or (p_mean > 0.045 and p_rms > 0.055):
        return {"moment_tensor": np.zeros(6), "depth_km": 30.0, "magnitude": 0.0, "confidence": 0.0, "abstain": True}
    def physical(q):
        raw = np.asarray(q[:6], dtype=float)
        raw[2] = -raw[0] - raw[1]
        norm = np.linalg.norm(raw)
        return raw / max(norm, 1e-12)
    def residual(q):
        v, depth, magnitude = physical(q), q[6], q[7]
        scaled = v * (10.0 ** (magnitude - 3.2))
        out = []
        for row in records:
            pa, pt = _radiation(scaled, depth, row["station_xy_km"], row["wave_type"])
            out.extend(((pa - row["amplitude"]) / 0.02).tolist())
            out.extend(((pt - row["p_arrival_s"]) / 0.15).tolist())
        return np.asarray(out)
    best = least_squares(residual, np.r_[np.zeros(6), 30.0, 3.2],
                         bounds=(np.r_[np.full(6, -3.0), 5.0, 2.0],
                                 np.r_[np.full(6, 3.0), 100.0, 4.5]), max_nfev=240)
    tensor, depth, magnitude = physical(best.x), float(best.x[6]), float(best.x[7])
    norm = float(np.linalg.norm(tensor))
    rms = float(np.sqrt(np.mean(residual(best.x) ** 2)))
    if not np.isfinite(norm) or norm < 1e-5 or rms > 2.5:
        return {"moment_tensor": np.zeros(6), "depth_km": 30.0, "magnitude": 0.0, "confidence": 0.0, "abstain": True}
    return {"moment_tensor": tensor, "depth_km": depth, "magnitude": magnitude,
            "confidence": float(math.exp(-0.5 * np.mean(residual(best.x) ** 2))), "abstain": False}
