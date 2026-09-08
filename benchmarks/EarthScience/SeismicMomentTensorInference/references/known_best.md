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
supported family. On the frozen executable revision it scores `0.904004` development and
`0.811379` held-out, with zero false discovery and complete unsupported-world refusal on both
splits. The baseline scores `0.000000` on both splits.

## Ablations and shortcuts
P-only fitting loses depth/magnitude separation; a single azimuth loses off-diagonal tensor
components; ignoring arrival time loses location and depth. Both shipped refusal ablations score
`0.000000`. The executable grid spans six constant focal mechanisms, eight depths and four
amplitude thresholds (192 strategies); its best development score is `0.000000`. Public
procedural worlds require a future server-held split.

## Model calibration
Thinking-disabled `deepseek-v4-flash` (seed 51) and `deepseek-v4-pro` (seed 52) each produced three
valid proposals at budget three. Their first proposals score approximately zero. Flash improves
to `0.800948` development and `0.422824` held-out; Pro improves to `0.353485` development and
`0.000000` held-out. Both terminal scores remain below the reference's `0.904004`. A pre-freeze
Pro draw reached `0.982356` when horizontal source location was fixed; this triggered the
scientifically material joint-location redesign and is excluded from final performance evidence.
See the compact experiment record.

## Robustness and limitations
The held-out split changes source locations, mechanisms, depths, magnitudes and noise draws. The
Flash terminal candidate loses unsupported-source refusal under that shift, while the reference
retains it, so the held-out axis is active. The homogeneous point-source model omits finite-fault
rupture, attenuation, anisotropy, 3-D velocity structure and instrument response. Those limits,
plus public procedural worlds, require server-held cases and independent waveform replication.

## Construction and review
This fills the EarthScience x WavePropagation discovery/parameter-inversion gap and is distinct
from static gravity inversion and pulsar-correlation classification. Domain review, independent
waveform recomputation, server-held worlds and external seismic validation remain pending;
status is candidate.
