# SeismicMomentTensorInference calibration record

## Scientific context
Moment tensors separate shear faulting from isotropic or compensated sources. P/S radiation
patterns and differential arrivals provide complementary constraints on mechanism and depth.

## Oracle and normalization
The frozen oracle uses a deterministic point-source radiation model with inverse-distance
amplitude, P/S velocity-dependent arrivals, Gaussian noise seeded from the world and query,
and a source-family check. The clipped score is normalized from the
all-abstain baseline to an exact supported-family reference; refusal and false-discovery rates
remain separate. This is a reduced-order benchmark, not field deployment.

## Baseline and reference
The baseline spends one P-wave survey and abstains on every world, so combined score is 0. The
reference performs two azimuthally diverse surveys, solves bounded nonlinear least squares,
and abstains when the signal is null or the isotropic P-wave offset is inconsistent with the
supported family. On the frozen executable revision it scores `0.864846` development and
`0.824390` held-out, with zero false discovery and complete unsupported-world refusal on both
splits. The baseline scores `0.000000` on both splits.

## Ablations and shortcuts
P-only fitting loses depth/magnitude separation; a single azimuth loses off-diagonal tensor
components; ignoring arrival time loses depth. A fixed four-corner survey is inferior to
adaptive radial coverage. Public procedural worlds require a future server-held split.

## Model calibration
Thinking-disabled `deepseek-v4-flash` (seed 43) and `deepseek-v4-pro` (seed 42) each produced one
valid first proposal at budget one. Both scored `0.000000`, below the reference's `0.864846`.
Earlier seed-41 draws and a Flash seed-42 draw are protocol diagnostics only: an underspecified
`noise_std` container caused two runtime failures, then Flash generated syntactically incomplete
code under an 8k output limit. The public container shape was fixed before the admitted draws;
the Flash admitted draw used a local 16k output limit. See the compact experiment record.

## Construction and review
This fills the EarthScience x WavePropagation discovery/parameter-inversion gap and is distinct
from static gravity inversion and pulsar-correlation classification. Domain review, independent
waveform recomputation, server-held worlds and external seismic validation remain pending;
status is candidate.
