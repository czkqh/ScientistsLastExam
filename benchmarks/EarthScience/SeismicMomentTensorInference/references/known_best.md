# SeismicMomentTensorInference calibration record

## Scientific context
Moment tensors separate shear faulting from isotropic or compensated sources. P/S radiation
patterns and differential arrivals provide complementary constraints on mechanism and depth.

## Oracle and normalization
The frozen oracle uses a deterministic point-source radiation model with inverse-distance
amplitude, P/S velocity-dependent arrivals, Gaussian noise seeded from the world and query,
and a rank/condition check for identifiability. The clipped score is normalized from the
all-abstain baseline to an exact supported-family reference; refusal and false-discovery rates
remain separate. This is a reduced-order benchmark, not field deployment.

## Baseline and reference
The baseline spends one P-wave survey and abstains on every world, so combined score is 0. The
reference performs two azimuthally diverse surveys, solves bounded nonlinear least squares, and
abstains when the design matrix is rank deficient or the isotropic term is inconsistent with the
supported family. Development/held-out values are recomputed after final freeze.

## Ablations and shortcuts
P-only fitting loses depth/magnitude separation; a single azimuth loses off-diagonal tensor
components; ignoring arrival time loses depth. A fixed four-corner survey is inferior to
adaptive radial coverage. Public procedural worlds require a future server-held split.

## Model calibration
No paid frontier-model draw is included in this initial fork. DeepSeek Flash/Pro calibration
must record exact model ID, seed, proposal budget, thinking-disabled condition, valid proposal
count, split scores, and failure category without storing prompts or generated programs.

## Construction and review
This fills the EarthScience x WavePropagation discovery/parameter-inversion gap and is distinct
from static gravity inversion and pulsar-correlation classification. Domain review, independent
waveform recomputation, server-held worlds and external seismic validation remain pending;
status is candidate.
