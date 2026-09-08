# SeismicMomentTensorInference - active source-mechanism inversion

## Scientific setting

Earthquake waveforms encode a source's moment tensor through directional P- and S-wave
radiation. Recovering that tensor and hypocentral depth from a small, noisy network is a
standard inverse problem: azimuthal coverage controls identifiability, while a nearly
collinear network cannot distinguish source mechanisms. This is a deterministic reduced-order
experiment, not a claim about operational earthquake warnings.

## Your task

```python
def infer_source(station_bounds, wave_types, observe, budget_units):
    """Return a source model or an explicit refusal.

    Return {"moment_tensor": [Mxx, Myy, Mzz, Mxy, Mxz, Myz],
      "source_xy_km": [x_km, y_km], "depth_km": float, "magnitude": float,
            "confidence": float in [0, 1], "abstain": bool}

    observe(stations, wave_type) returns a dictionary with keys
    station_xy_km, wave_type, p_arrival_s, amplitude, noise_std,
    budget_cost, budget_used.
    """
```

The public `station_bounds` is `[[-300, 300], [-300, 300]]` km and `wave_types` is
`["P", "S"]`. A request contains 4-12 distinct stations in the bounds and one wave type;
its cost is `ceil(number_of_stations / 4)`. Total cost may not exceed `budget_units` (8).
Observations contain one signed radiation amplitude and an arrival time per station.
`station_xy_km` has shape `(n,2)`; `p_arrival_s`, `amplitude`, and `noise_std` are length-`n`
arrays. `noise_std` is the pointwise amplitude standard deviation; arrival-time noise has public
standard deviation `0.045 s`. `wave_type` is one string and both budget fields are scalar
integers. Moment components use an arbitrary but fixed normalized scale. The six output keys
above are the complete seven-key submission contract.

The horizontal source location lies in `[-60,60]` km on each axis. For station position `s`,
source position `q`, depth `d`, displacement `(dx,dy)=s-q`, distance
`r = sqrt(dx^2+dy^2+d^2)`, ray direction `n=(dx,dy,-d)/r`, symmetric tensor `M`, and
horizontal transverse direction `t=(-n_y,n_x,0)/sqrt(n_x^2+n_y^2)`, the public forward model is

```text
A_P = 9000 * 10^(magnitude-3.2) * (n^T M n) / r^2
A_S = 9000 * 10^(magnitude-3.2) * (t^T M n) / r^2
T_P = r / 6.0,                 T_S = r / 3.5.
```

Supported `M` is a trace-free double-couple tensor normalized to unit Frobenius norm. Hidden
locations span the public square, depths span 12-48 km and magnitudes span 2.8-3.6. Independent
Gaussian noise has the disclosed standard deviations. The tensor carries mechanism orientation
while `magnitude` carries scale.

## Scoring

The development `combined_score` is a continuous normalized mean of moment-tensor recovery,
horizontal location, depth, magnitude and waveform prediction on held-out azimuths. Two unsupported
worlds are included: a null signal and an isotropic-plus-double-couple source outside the
declared family. Correct abstention is
rewarded; false discovery, invalid output, or budget overspend scores zero. Mechanism,
false-discovery, refusal and attempted-discovery rates are reported separately; all-abstain is
exactly zero. Held-out stations, arrival residuals and shifted noise are evaluator-only.

## Rules

- Only edit `solution.py`; keep the `infer_source` signature and all output key names.
- Use deterministic Python, NumPy and SciPy only. Do not read `verification/` or
  `frontier_eval/`, use the network, or create processes.
- A non-abstaining result must have two finite bounded `source_xy_km` values, six finite tensor
  values, positive finite depth and magnitude, and
  confidence in `[0,1]`. If the measurements do not support the declared double-couple family,
  abstain with a zero tensor.
- `sle.contract_lint` is importable and free to call for shape checks.

## Relationship to nearby tasks

This is distinct from `EarthScience/GravityInversion`: gravity uses static surface fields and
rectangular density bodies, whereas this task uses directional seismic radiation, P/S timing,
and a source-mechanism tensor. It is distinct from `Physics/PTAHellingsDowns`, which classifies
long-baseline pulsar correlations rather than inverting a local earthquake source.

References: Aki & Richards, *Quantitative Seismology* (2002), DOI `10.1111/j.1365-246X.1987.tb05275.x`;
Vavryčuk, *Earth and Planetary Science Letters* (2015), DOI `10.1016/j.epsl.2015.02.022`.
