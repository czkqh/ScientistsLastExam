"""Deterministic reduced-order active seismic moment-tensor oracle."""
from __future__ import annotations

import hashlib
import math

import numpy as np

STATION_BOUNDS = np.asarray(((-300.0, 300.0), (-300.0, 300.0)))
WAVE_TYPES = ("P", "S")
BUDGET = 8
MIN_STATIONS, MAX_STATIONS = 4, 12
VP, VS = 6.0, 3.5

DEV = ((2101, "double_couple"), (2102, "double_couple"), (2103, "double_couple"),
       (2104, "double_couple"), (2105, "null"), (2106, "isotropic"))
HELD = ((3101, "double_couple"), (3102, "double_couple"), (3103, "double_couple"),
        (3104, "null"), (3105, "isotropic"))


def _tensor(seed, kind):
    rng = np.random.default_rng(seed)
    if kind == "null":
        return np.zeros(6), 28.0, 0.0
    m = rng.normal(size=6)
    if kind == "double_couple":
        a, b = rng.normal(size=(2, 3))
        a /= np.linalg.norm(a); b -= a * np.dot(a, b); b /= np.linalg.norm(b)
        M = np.outer(a, b) + np.outer(b, a)
    else:
        a, b = rng.normal(size=(2, 3))
        a /= np.linalg.norm(a); b -= a * np.dot(a, b); b /= np.linalg.norm(b)
        M = np.outer(a, b) + np.outer(b, a) + 0.75 * np.eye(3)
    M /= max(np.linalg.norm(M), 1e-12)
    return np.asarray((M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2])), float(rng.uniform(12.0, 48.0)), 3.2


def _matrix(v):
    v = np.asarray(v, dtype=float)
    return np.asarray(((v[0], v[3], v[4]), (v[3], v[1], v[5]), (v[4], v[5], v[2])))


def _radiation(tensor, depth, stations, wave):
    M = _matrix(tensor)
    xy = np.asarray(stations, dtype=float)
    rvec = np.column_stack((xy[:, 0], xy[:, 1], -np.full(len(xy), depth)))
    distance = np.linalg.norm(rvec, axis=1)
    n = rvec / distance[:, None]
    Mn = n @ M.T
    p = np.einsum("ij,ij->i", n, Mn)
    if wave == "P":
        radiation = p
        velocity = VP
    else:
        tang = np.column_stack((-n[:, 1], n[:, 0], np.zeros(len(n))))
        norm = np.linalg.norm(tang, axis=1)
        tang[norm < 1e-12] = (1.0, 0.0, 0.0)
        tang /= np.maximum(norm[:, None], 1e-12)
        radiation = np.einsum("ij,ij->i", tang, Mn)
        velocity = VS
    amplitude = 9000.0 * radiation / np.maximum(distance * distance, 1.0)
    arrival = distance / velocity
    return amplitude, arrival


def _seed(world_seed, call, stations, wave):
    payload = np.asarray(stations, dtype="<f8").tobytes() + wave.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") + world_seed * 1009 + call * 7919


class _World:
    def __init__(self, seed, kind):
        self.seed, self.kind = int(seed), str(kind)
        self.tensor, self.depth, self.magnitude = _tensor(self.seed, self.kind)
        self.calls, self.used, self.failed = 0, 0, None
        self.records = []

    def observe(self, stations, wave_type):
        try:
            xy = np.asarray(stations, dtype=float)
        except Exception as exc:
            self.failed = "stations must be numeric"; raise ValueError(self.failed) from exc
        if xy.ndim != 2 or xy.shape[1] != 2 or not MIN_STATIONS <= len(xy) <= MAX_STATIONS:
            self.failed = "stations must have shape (4-12,2)"; raise ValueError(self.failed)
        if np.any(~np.isfinite(xy)) or np.any(xy < STATION_BOUNDS[:, 0]) or np.any(xy > STATION_BOUNDS[:, 1]):
            self.failed = "station outside public bounds"; raise ValueError(self.failed)
        if len({tuple(row) for row in xy.tolist()}) != len(xy):
            self.failed = "stations must be distinct"; raise ValueError(self.failed)
        if wave_type not in WAVE_TYPES:
            self.failed = "wave_type must be P or S"; raise ValueError(self.failed)
        cost = int(math.ceil(len(xy) / 4.0))
        if self.used + cost > BUDGET:
            self.failed = "budget exceeded"; raise ValueError(self.failed)
        self.used += cost; self.calls += 1
        amp, arr = _radiation(self.tensor, self.depth, xy, wave_type)
        rng = np.random.default_rng(_seed(self.seed, self.calls, xy, wave_type))
        amp_noise, time_noise = 0.006, 0.045
        amp_obs = amp + rng.normal(0.0, amp_noise, len(xy))
        arr_obs = arr + rng.normal(0.0, time_noise, len(xy))
        row = {"station_xy_km": xy.copy(), "wave_type": wave_type,
               "p_arrival_s": arr_obs, "amplitude": amp_obs,
               "noise_std": {"amplitude": amp_noise, "p_arrival_s": time_noise},
               "budget_cost": cost, "budget_used": self.used}
        self.records.append(row)
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in row.items()}


def _validate(out):
    if not isinstance(out, dict): raise ValueError("submission must be a dictionary")
    abstain = out.get("abstain")
    if not isinstance(abstain, (bool, np.bool_)): raise ValueError("abstain must be boolean")
    tensor = np.asarray(out.get("moment_tensor"), dtype=float).ravel()
    if tensor.shape != (6,) or not np.all(np.isfinite(tensor)): raise ValueError("moment_tensor must contain six finite values")
    depth = float(out.get("depth_km")); magnitude = float(out.get("magnitude")); confidence = float(out.get("confidence"))
    if not all(math.isfinite(x) for x in (depth, magnitude, confidence)) or not 0.0 <= confidence <= 1.0:
        raise ValueError("depth, magnitude and confidence must be finite; confidence in [0,1]")
    if abstain:
        if np.max(np.abs(tensor)) > 1e-8: raise ValueError("abstention requires zero tensor")
    elif not (0.1 <= depth <= 100.0 and 0.0 <= magnitude <= 10.0):
        raise ValueError("claimed depth or magnitude outside bounds")
    return tensor, depth, magnitude, confidence, bool(abstain)


def _quality(world, tensor, depth, magnitude, records):
    if world.kind != "double_couple": return 1.0 if np.max(np.abs(tensor)) < 1e-8 else 0.0
    truth = world.tensor
    scale = max(np.linalg.norm(truth), 1e-9)
    tensor_q = math.exp(-0.5 * float(np.mean(((tensor - truth) / 0.10) ** 2)))
    depth_q = math.exp(-0.5 * ((depth - world.depth) / 3.0) ** 2)
    mag_q = math.exp(-0.5 * ((magnitude - world.magnitude) / 0.35) ** 2)
    if not records: return 0.0
    residuals = []
    for row in records:
        pred_a, pred_t = _radiation(tensor, depth, row["station_xy_km"], row["wave_type"])
        residuals.extend(((pred_a - row["amplitude"]) / 0.02).tolist())
        residuals.extend(((pred_t - row["p_arrival_s"]) / 0.15).tolist())
    fit_q = math.exp(-0.5 * float(np.mean(np.asarray(residuals) ** 2)))
    return float((tensor_q * depth_q * mag_q * fit_q) ** 0.25)


def _run(fn, spec, split, index):
    world = _World(*spec)
    try:
        out = fn(STATION_BOUNDS.tolist(), list(WAVE_TYPES), world.observe, BUDGET)
        tensor, depth, magnitude, confidence, abstain = _validate(out)
    except Exception as exc:
        return {"split": split, "world_index": index, "valid": False, "failure_kind": str(type(exc).__name__),
                "kind": world.kind, "abstain": False, "claimed": False, "mechanism_quality": 0.0,
                "correct_refusal": False, "false_discovery": False, "attempted_discovery": False}
    supported = world.kind == "double_couple"
    claimed = not abstain
    correct_refusal = (not supported) and abstain
    false_discovery = (not supported) and claimed
    quality = _quality(world, tensor, depth, magnitude, world.records) if claimed else (1.0 if correct_refusal else 0.0)
    return {"split": split, "world_index": index, "valid": True, "failure_kind": "", "kind": world.kind,
            "abstain": abstain, "claimed": claimed, "mechanism_quality": quality,
            "correct_refusal": correct_refusal, "false_discovery": false_discovery,
            "attempted_discovery": claimed, "confidence": confidence, "budget_used": world.used,
            "experiment_calls": world.calls}


def _metrics(rows, specs):
    unsupported = sum(kind != "double_couple" for _, kind in specs)
    raw = float(np.mean([r["mechanism_quality"] for r in rows]))
    baseline = unsupported / len(rows)
    claims = sum(r["claimed"] for r in rows)
    return {"mechanism": float(np.clip((raw - baseline) / max(1e-12, 1.0 - baseline), 0, 1)),
            "fdr": sum(r["false_discovery"] for r in rows) / max(claims, 1),
            "refusal": sum(r["correct_refusal"] for r in rows) / max(unsupported, 1),
            "coverage": sum(r["claimed"] and r["kind"] == "double_couple" for r in rows) / max(1, len(rows)-unsupported)}


def evaluate(infer_source):
    dev = [_run(infer_source, s, "development", i) for i, s in enumerate(DEV)]
    held = [_run(infer_source, s, "heldout", i) for i, s in enumerate(HELD)]
    dm, hm = _metrics(dev, DEV), _metrics(held, HELD)
    valid = all(r["valid"] for r in dev + held)
    result = {"combined_score": dm["mechanism"] if valid else 0.0, "raw_score": dm["mechanism"] if valid else 0.0,
              "valid": 1.0 if valid else 0.0, "feasibility_rate": float(np.mean([r["valid"] for r in dev])),
              "development_mechanism_score": dm["mechanism"], "heldout_mechanism_score": hm["mechanism"],
              "development_false_discovery_rate": dm["fdr"], "heldout_false_discovery_rate": hm["fdr"],
              "development_correct_refusal_rate": dm["refusal"], "heldout_correct_refusal_rate": hm["refusal"],
              "development_attempt_rate": float(np.mean([r["attempted_discovery"] for r in dev])),
              "heldout_attempt_rate": float(np.mean([r["attempted_discovery"] for r in held])),
              "development_discovery_coverage": dm["coverage"], "heldout_discovery_coverage": hm["coverage"],
              "robustness_score": hm["mechanism"], "per_world": dev + held}
    if not valid: result["error_message"] = "candidate invalid"
    return result
