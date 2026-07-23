# Experiment Log

All entries use the current repository outputs. No Stage 2 run has been started.

## Fixed-bank terminology

`G_sty` is the adapted Graphdeco source. `G_ord_ref` is an independent ordinary
reference and is not Gaussian-index paired with `G_sty`. `G_ord_canon` is fitted
as `T(G_sty, Delta*)` after correspondence. No real `G_ord_ref` has yet been
processed through the real correspondence pipeline.

## Source and Foreground Delta

### Graphdeco source adaptation

- Source: `output/elephant_source_graphdeco`, iteration `30000`.
- Gaussian count: `44764`.
- PLY schema: compatible with the local GaussianModel loader.

### Existing free xyz-only delta

- File: `output/elephant_source_graphdeco/mined_delta_xyz_only.pt`.
- Foreground energy: `61.6%`.
- Background energy: `38.4%`.
- `d_scaling` max norm: `0`.
- Result: FAIL because background motion is too large.

### Foreground-only xyz-only delta

- File: `output/elephant_source_graphdeco/mined_delta_foreground_xyz_only.pt`.
- Foreground energy: `100%`.
- Background energy: `0%`.
- `d_scaling` max norm: `0`.
- `d_xyz` mean norm: `0.01716731`.
- `d_xyz` p95 norm: `0.05971656`.
- Result: foreground gating and zero `d_scaling` PASS as implementation
  constraints, but the split-view reliability gate FAILS. This delta is not a
  successful teacher and is unusable for Stage 2.

## Part-Fit Candidates

All candidates fit the foreground-only pseudo-delta, not ChatGPT images.

| Candidate | Energy | Cosine | Explained variance | Result |
|---|---:|---:|---:|---|
| K16 hard translation | 0.138 | 0.371 | 0.136 | FAIL |
| K16 soft translation | 0.143 | 0.378 | 0.142 | FAIL |
| K16 hard affine | 0.283 | 0.532 | 0.281 | FAIL |
| K16 soft affine | 0.309 | 0.556 | 0.308 | FAIL |
| K24 soft affine | 0.343 | 0.585 | 0.341 | FAIL |
| K32 soft affine | 0.389 | 0.624 | 0.388 | FAIL |
| K16 soft affine refined | 1.000 | 0.502 | 0.003 | FAIL: wrong direction |
| K32 soft affine refined | 1.000 | 0.574 | 0.147 | FAIL: wrong direction |

The refined candidates preserve energy by adding an energy penalty, but their
cosine and explained variance remain below acceptance. This is the main failure
mode: high energy with insufficient directional agreement.

## Visual Runs

- Existing part-aware image-trained delta: x1 nearly identical to source; x2 subtle; x5/x10 coherent; `d_scaling=0`.
- Existing part-aware amplified images: `output/elephant_source_graphdeco/debug_partaware_delta_amplified/`.
- Existing part-aware target triptychs: `output/elephant_source_graphdeco/debug_triptych_partaware_xyz_only/`.
- Fitted part candidates have not yet been rendered in the current CUDA-disabled environment.

## Code Validation

- `python -m py_compile scripts/fit_part_delta_to_mined_delta.py scripts/interpolate_part_delta.py` passed.
- No Stage 2 training was run.

## Current Distillation Run

### CPU fitting and K sweep

- Ran hard and soft translation at K16.
- Ran hard and soft affine at K16.
- Ran soft affine at K24 and K32.
- Ran energy-refined soft affine at K16 and K32.
- All metrics are recorded in the corresponding `partfit_metrics.json` files.
- No candidate passed the full acceptance gate.

### Alpha rendering attempt

- Command targeted `output/elephant_source_graphdeco/partfit_affine_soft_k32_refined/partfit_affine_xyz_only.pt`.
- Rendering did not start because the active Python environment reports `torch.cuda.is_available() = False`.
- No visual PASS/FAIL was inferred from this failed render attempt.

## Stage 1.5 Reliability Experiment

### Split preparation

The deterministic definitions are A = key8 indices 0,2,4,6 and B = 1,3,5,7.
The source dataset matches 3/4 views in A and 4/4 in B. `03_standard.png`
(azimuth 090) has no exact source camera because that source frame was not
rendered. No fallback or random remapping was used.

### Split mining attempt

The required A mining command was started with the Graphdeco source, foreground
mask, xyz-only deformation, and the same weak-target weights as the full run.
It stopped in `safe_state` because the active environment reports
`torch.cuda.is_available() = False`. No subset delta, consistency metric,
consensus, or denoising result is claimed from this run.

The new CPU-side tools are syntax-checked and ready:

- `scripts/compare_mined_delta_consistency.py`
- `scripts/build_consensus_delta.py`
- `scripts/denoise_consensus_delta.py`
- `scripts/cluster_motion_aware_parts.py`

The reliability decision is **UNAVAILABLE**, not strong/mixed/weak: split
deltas do not yet exist and the target-camera set is incomplete.

### CPU tool smoke test (not a research result)

For interface validation only, the existing full-view delta was supplied twice
as A and B. It produced cosine `1.0`, as expected for identical inputs, and
was not used as evidence of split reproducibility. Consensus, denoising, and
motion-part outputs from this smoke test live under
`output/elephant_source_graphdeco/stage1_5_cpu_smoke/` and must not be used for
Stage 2.

### Updated CUDA run and decision

The exact eight elevation-0 key cameras were rebuilt into
`assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset/`.
Both deterministic subsets now match 4/4 views. The mined files are
`output/elephant_source_graphdeco/mined_delta_fg_subset_A.pt` and
`output/elephant_source_graphdeco/mined_delta_fg_subset_B.pt`.

| Metric | Result |
|---|---:|
| weighted global cosine | 0.1176 |
| confidence-weighted cosine | 0.1176 |
| median per-Gaussian cosine | 0.0749 |
| p25 / p75 per-Gaussian cosine | -0.4870 / 0.6035 |
| magnitude Pearson | 0.6313 |
| magnitude Spearman | 0.3623 |
| direction agreement, cosine > 0.5 | 30.3% |
| directional conflict fraction | 46.5% |
| high-confidence cosine | 0.0516 |

Both deltas are nonzero xyz-only signals; A mean norm is `0.01020`, B mean
norm is `0.00611`, and both have exact zero `d_scaling`. Nevertheless, the
direction is not reproducible across independent target subsets. Conflict
fractions are approximately 44-50% across the existing K16 parts.

Decision: **FAIL / weak-unreliable signal**. Consensus, denoising,
motion-aware clustering, and structured refitting were skipped. The next
research change must improve target geometric correspondence, not increase
part-model complexity or force energy normalization. Stage 2 remains paused.

## Synthetic Known-Delta Benchmark

### Data and deformation teachers

Created three controlled xyz-only teachers directly on the canonical 44,764
Gaussian Graphdeco bank:

- body roundness: 10,183 active foreground Gaussians;
- ear expansion: 1,950 active foreground Gaussians;
- segmented trunk bending: 2,956 active foreground Gaussians.

All teachers have exact zero `d_scaling` and exact zero background delta. Each
was rendered from the same eight exact key cameras. Target packages are under
`output/elephant_source_graphdeco/synthetic_known_delta/targets_*`.

### Body image-only recovery

The existing weak LPIPS/RGB Stage 1 objective was run against the perfectly
paired body-roundness renders. It failed to recover the known teacher:

| Metric | Image-only result |
|---|---:|
| global cosine | 0.4740 |
| energy ratio | 0.2739 |
| explained variance | 0.2166 |
| active-region cosine | 0.5133 |
| background energy | 0 |
| d_scaling | exact zero |

This is a useful negative control: pixel/perceptual image loss is not enough
for this source and small geometric change, even when target views are truly
paired.

### Correspondence-guided recovery

Added `--correspondence_path`, `--lambda_corr_3d`, and `--lambda_corr_2d` to
`train_delta_mining.py`. The synthetic correspondence contains exact target
xyz positions; projected 2D motion is computed per active camera from the
same camera matrices.

Full body recovery output:

```text
output/elephant_source_graphdeco/synthetic_known_delta/mined_recovered_body_roundness_corr.pt
output/elephant_source_graphdeco/synthetic_known_delta/body_roundness_corr_recovery_metrics.json
```

| Metric | Correspondence-guided result |
|---|---:|
| global cosine | 0.9614 |
| energy ratio | 0.8835 |
| explained variance | 0.9233 |
| active-region cosine | 0.9772 |
| background energy | 0 |
| d_scaling | exact zero |

Independent synthetic A/B split recovery also passed the directional gate:

- confidence-weighted cosine: `0.9881`;
- high-confidence cosine: `0.9808`;
- magnitude Pearson: `0.9532`;
- magnitude Spearman: `0.8364`;
- full-foreground direction conflict: `28.0%`.

The A/B outputs and diagnostic PLY are under
`output/elephant_source_graphdeco/synthetic_known_delta/consistency_corr_ab/`.

### Visual outputs

- ground truth x1/x2/x5/x10:
  `output/elephant_source_graphdeco/synthetic_known_delta/debug_ground_truth_amplified/`
- correspondence recovery x1/x2/x5/x10:
  `output/elephant_source_graphdeco/synthetic_known_delta/debug_recovered_corr_amplified/`

x1/x2 remain sharp and coherent; x5 remains interpretable; x10 shows the
expected large displacement/halo. No scale-blur is present.

## Correspondence Interfaces

Added modular interfaces:

- `scripts/align_ordinary_target.py`: uniform similarity alignment only;
- `scripts/define_semantic_anchors.py`: editable anchor schema;
- `scripts/evaluate_correspondence_quality.py`: support/confidence/residual gate.

No unified ordinary target 3DGS is currently available, so global alignment,
dense matching, and real ordinary-target quality metrics are not claimed.

## Fixed-bank Correspondence Refactor

Added the `correspondence/` package with a validated source-indexed bundle,
similarity alignment, anchor helpers, visibility/depth-lifting interfaces,
multi-view robust fusion, quality metrics, and separate 3D/2D Huber losses.
Added `CanonicalOrdinaryModel`, which preserves the source Gaussian count and
ordering while exporting `G_ord_canon` and `Delta*`. The old paired-point
alignment entry point is retained only as a compatibility wrapper; independent
references use `align_ordinary_reference.py`.

### CPU Regression

- Similarity recovery on a known rigid plus uniform-scale point set: PASS,
  recovered scale `2.0`, near-zero RMSE.
- Unified bundle validation with `[V,N,2]` target observations and multi-view
  fusion: PASS.
- Synthetic body-roundness canonical export: PASS; `N=44764`, ordering and
  PLY reload preserved, background delta zero, `d_scaling` zero.
- Canonical KNN edge-length ratio: median `0.9999998`, p05 `0.8355`, p95
  `1.2023`.

The earlier GPU synthetic correspondence recovery remains the latest recovery
metric: cosine `0.9614`, energy ratio `0.8835`, explained variance `0.9233`.
The refactor has not claimed a new GPU recovery run. Stage 2 and part
compression remain paused.

## Observed-2D Fixed-Bank Acceptance Experiment

Run date: 2026-07-22. The body-roundness synthetic teacher was projected with
the exact source cameras. Hidden `target_xyz` was used only to generate those
projections and to evaluate the recovered delta after training. Every optimizer
bundle used `observation_mode=observed_2d`, `target_xyz=None`, and contained
`target_xy [V,N,2]`, visibility, confidence, support counts, and camera names.

The deterministic seed was `20260722`. All runs used xyz-only deformation,
`max_d_xyz=0.08`, exact-zero scaling, `lambda_corr_2d=1`, no RGB/LPIPS/image
loss, `lambda_delta=0.0005`, `lambda_smooth=0.005`, 800 iterations, and fixed
Gaussian count/order. Conditions were clean 1/2/4/8 views, 0.5 px and 2 px
noise, 5% and 10% outliers, 70% visibility, and alternating A/B four-view
splits.

| condition | global cosine | active cosine | energy | explained variance | held-out projection |
|---|---:|---:|---:|---:|---:|
| clean 1 | 0.7512 | 0.7512 | 1.4349 | 0.3602 | 2.7492 px / 7 |
| clean 2 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 0.0267 px / 6 |
| clean 4 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.0094 px / 4 |
| clean 8 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 0.0013 px / 216 |
| noise 0.5 px | 0.9988 | 0.9995 | 1.0021 | 0.9975 | n/a |
| noise 2 px | 0.9770 | 0.9907 | 1.0221 | 0.9531 | n/a |
| outliers 5% | 0.9975 | 0.9979 | 0.9803 | 0.9949 | n/a |
| outliers 10% | 0.9909 | 0.9923 | 0.9540 | 0.9816 | n/a |
| visibility 70% | 0.9981 | 0.9981 | 0.9714 | 0.9959 | n/a |

All runs had zero background energy and exact-zero `d_scaling`. A/B full
foreground weighted cosine was `0.99998`, median per-Gaussian cosine `0.6962`,
and conflict `29.7%`. On the known active body-roundness region, weighted and
median cosine were `1.0` with zero conflict. The full-foreground statistic is
reported because it includes inactive near-zero entries rather than hiding
them.

The clean-8 visual panel is
`output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/novel_render_panels/0001_elevm020_az005_source_known_recovered.png`.
Source, known, and recovered renders are clear and coherent. Shared floating
background noise appears in all columns and is a source-model artifact.

**Decision: synthetic observed-2D gate PASS.** This validates recovery from
consistent 2D observations, not real generated style targets. Real target
generation, repeats, and Stage 2 remain unrun.

## Image-first Research Correction

The main route is now explicitly:

```text
stylized GLB
-> canonical stylized 3DGS and source views
-> generated target/content multi-view images
-> source-indexed observed-2D bundle
-> fixed-bank free delta
-> view/repeat/identity/intensity/structure reliability
-> stable style delta
-> future source-conditioned multiscale learning
```

An independently generated ordinary/content 3D model is optional and remains a
baseline/reference only. It is not required and does not provide inherent
Gaussian correspondence.

Added `style_data.StyleTaskRecord` for the experimental index
`(object, style_operation, intensity, repeat, view)`. Added the image-first
`ObservationBundle` schema with explicit `observed_2d`, `oracle_3d`, and
`hybrid` modes. `observed_2d` validates without `target_xyz` and refuses to
silently project oracle geometry. Legacy `CorrespondenceBundle` imports and old
synthetic files remain loadable.

Stage 1 now records observation mode and separates image/2D/3D responsibilities
through the `stage1/` modules. Added `StableStyleDelta` and future Stage 2
contracts only; no Stage 2 model or training was run.

## Image-Derived Observation Experiment

Run date: 2026-07-22. Added a pluggable matcher interface with an OpenCV
Farneback baseline, foreground/mask filtering, local projected-footprint median
flow, forward/backward cycle confidence, target foreground checks, and a coarse
projected depth-bin visibility estimate. Bundle metadata records
`matcher=opencv_farneback`, `visibility_method=projected_foreground_coarse_zbuffer`,
and `target_xyz_in_optimizer_input=false`. The matcher reads only images, masks,
source xyz/cameras, and source visibility; every optimizer bundle has
`target_xyz=None`.

| teacher/condition | median EPE | PCK@5 | PCK@10 | visible coverage |
|---|---:|---:|---:|---:|
| body clean 8 | 3.691 px | 0.539 | 0.719 | 0.356 |
| body brightness/contrast | 3.964 px | 0.529 | 0.711 | 0.358 |
| body blur/noise | 2.795 px | 0.567 | 0.738 | 0.366 |
| body eroded mask | 3.582 px | 0.543 | 0.724 | 0.352 |
| ear clean 8 | 0.060 px | 0.822 | 0.839 | 0.344 |
| trunk clean 8 | 0.109 px | 0.801 | 0.889 | 0.333 |

Body fails the provisional observation gate (`median EPE <=3`, `PCK@5 >=.80`,
coverage `>=.40`). Ear and trunk pass PCK@5 but fail coverage and have long
error tails. Body split observations were extracted: A median EPE `1.694 px`,
PCK@5 `.588`, coverage `.242`; B median EPE `4.824 px`, PCK@5 `.504`, coverage
`.289`.

The machine had no usable NVIDIA driver, so GPU Stage 1 recovery and delta A/B
were intentionally not run and are marked `not_run_cuda_unavailable`.

The historical manually generated target set was processed without GT leakage
using an explicit ordered key8 filename mapping. It produced overlays and a
diagnostic observed_2d bundle with support coverage `.4675`; it has no stable
style claim because there are no repeat generations or recovery validation.
Artifacts: `output/elephant_source_graphdeco/historical_image_observation_diagnostic/`.

## Geometry-Aware Audit and CPU Recovery

The v2 audit corrected the denominator and now reports all-Gaussian,
foreground, active, inactive, candidate, accepted-recall, confidence-rank,
zero-motion, and displacement-stratified metrics. The fixed radius-2 wording
was removed: the implemented fallback is a confidence-weighted local image
neighborhood; renderer radii are reserved for the CUDA path. CPU visibility now
uses explicit camera-space depth with a soft coarse-bin tolerance and records
that approximation.

Corrected Farneback active-region results:

| teacher | foreground coverage | active coverage | active median EPE | active PCK@5 | accepted recall |
|---|---:|---:|---:|---:|---:|
| body roundness | 0.721 | 0.772 | 9.617 px | 0.193 | 0.381 |
| ear expansion | 0.721 | 0.678 | 20.531 px | 0.091 | 0.371 |
| trunk bending | 0.721 | 0.918 | 5.353 px | 0.478 | 0.474 |

The zero-motion body active PCK@5 was `.162`, so the Farneback body result
only slightly improves over identity. This correction shows that prior global
PCK values were inflated by inactive/near-zero points.

DIS (`opencv_dis`, OpenCV `4.13.0`, medium preset) improved body active median
EPE to `3.548 px`, active PCK@5 to `.606`, and active accepted recall to `.573`,
but still failed the `.80` gate. No learned backend was available: transformers,
timm, kornia were absent, and torch/Hugging Face caches contained no RAFT,
DINO, or LoFTR weights. No weights were downloaded.

The CPU IRLS downstream diagnostic consumed only the saved body observed_2d
bundle. Active recovery results were:

| mode | active cosine | energy ratio | explained variance | background energy |
|---|---:|---:|---:|---:|
| point-only | 0.057 | 13.09 | -12.92 | 0 |
| silhouette-only | 0.041 | 4.24 | -4.17 | 0 |
| point + silhouette | 0.030 | 20.01 | -20.11 | 0 |

Thus neither the extracted point observations nor the current silhouette
constraint contain a recoverable 3D teacher under this baseline. No GPU Stage 1
run was possible because the NVIDIA driver remained unavailable. The image
observation route is not rejected in principle, but the current matcher and
visibility/observation quality are insufficient.

## Shared Style / Template Nuisance Factorization (2026-07-23)

The research interpretation was corrected: a generated ordinary target is a
conditional template, not an exact pointwise pair. The controlled benchmark
uses the existing fixed 44,764-Gaussian Graphdeco elephant bank and the hidden
body-roundness teacher. Five candidates were constructed as shared style plus
zero-centered moderate geometry nuisance; appearance nuisance was recorded in
metadata only. No mesh, independent target 3D, scaling delta, or background
motion was used.

Command:

```bash
python scripts/run_template_factorization_benchmark.py \
  --output_dir output/elephant_source_graphdeco/template_factorization_benchmark
```

The result is at
`output/elephant_source_graphdeco/template_factorization_benchmark/factorization_summary.json`.
The strongest nuisance-regression estimate achieved active cosine `1.0000`,
energy ratio `1.0000`, explained variance `1.0000`, and nuisance leakage
`0.000002`. The robust shared estimate achieved cosine `0.9990`, energy ratio
`0.9011`, explained variance `0.9955`, and nuisance leakage `0.0668`. A single
template was materially worse in energy (`0.4027`) and nuisance leakage
(`0.3924`). One-outlier robust recovery remained cosine `0.9992`, energy
`1.1078`, and explained variance `0.9955`; missing-region robustness remained
cosine `0.9897`. Background energy was 0 and `d_scaling` was exactly zero.
The controlled factorization gate passes, but this is not evidence of a real
style delta.

The CPU oracle observed-2D validation was run after the factorization code.
Point recovery reached active cosine `0.9999996`, energy ratio `1.0023`, and
explained variance `0.9999980`. The graph-coupled recovery reached active
cosine `0.9999998`, energy ratio `1.0017`, and explained variance `0.9999989`.
Both had zero background energy and exact-zero `d_scaling`. This validates the
CPU diagnostic on oracle target projections; image-derived observations remain
unvalidated for 3D recovery and CUDA is unavailable.

The real pilot manifest was changed from three nominal repeats to three
conditional template variants A/B/C. The folders remain empty and no target
images were generated. Stage 2 and part compression remain paused.

## Stochastic Template Factorization v2 (2026-07-23)

The v1 benchmark is now classified as a controlled implementation sanity
check. Its five nuisance coefficients were exactly symmetric and zero-centered,
the oracle coefficients were supplied to regression, R=5 with intercept plus
four features was exactly determined, and its compact/radial bases were
dependent. Its near-perfect result must not be presented as nontrivial
factorization validation.

The v2 benchmark was run with the fixed Graphdeco bank, body-roundness shared
teacher, independently sampled nuisance coefficients, six independent smooth
spatial nuisance bases, recovery-like noise, R=`3/5/8`, strengths
`0.25/0.5/1.0`, and seeds `11/29/47`. Primary no-label methods did not see
nuisance coefficients. Exact-label regression is marked upper bound and weak
labels are noisy/discretized/partly missing.

Artifact:
`output/elephant_source_graphdeco/template_factorization_benchmark_v2/benchmark_v2_summary.json`.

Selected no-label robust results (mean +/- std across seeds):

| nuisance | R | active cosine | explained variance | nuisance leakage |
|---:|---:|---:|---:|---:|
| 0.25 | 3 | 0.906 +/- 0.067 | 0.716 +/- 0.152 | 0.279 +/- 0.149 |
| 0.25 | 5 | 0.935 +/- 0.036 | 0.854 +/- 0.092 | 0.144 +/- 0.091 |
| 0.25 | 8 | 0.972 +/- 0.009 | 0.941 +/- 0.016 | 0.058 +/- 0.016 |
| 0.50 | 5 | 0.788 +/- 0.081 | 0.415 +/- 0.371 | 0.575 +/- 0.365 |
| 0.50 | 8 | 0.896 +/- 0.037 | 0.767 +/- 0.063 | 0.229 +/- 0.062 |

At moderate nuisance strength `0.25`, R=5/8 passes the provisional exact
candidate gate; R=3 is mixed. At strength `0.5`, R=5 fails and R=8 is
borderline. A deliberate nonzero shared nuisance bias produced a mean active
cosine `0.950` but is not identifiable as style versus systematic target bias
from one source without an additional prior, labels, or multiple source
objects. This failure is documented, not corrected by hidden information.

The saved oracle observed-2D candidate bundles contain `target_xyz=None`.
The full five-candidate recovery loop was memory-unstable on this CPU, so a
separate 4,096-active-Gaussian diagnostic was run with the graph-coupled CPU
solver. Recovered factorization achieved mean cosine `0.901`, energy `0.670`,
and explained variance `0.802`, versus a single recovered candidate cosine
`0.656`, energy `1.817`, and explained variance `-0.065`. This is a partial
recovered-candidate validation, not a full-bank acceptance result.

The real pilot now has `sample_A` through `sample_E`, all using the
same standardized prompt. Pre-generation observed nuisance fields are null;
the samples are independent conditional generations, not deliberately heavy,
slim, or large-ear variants. No real images were generated.

## Calibrated v3 and Full-Bank Recovery (2026-07-23)

Command:

```bash
python scripts/run_template_factorization_benchmark_v3.py
```

Artifact:
`output/elephant_source_graphdeco/template_factorization_benchmark_v3/benchmark_v3_summary.json`.

V3 reports per-candidate realized nuisance norm/style norm, per-template
nuisance energy/style energy, noise ratio, finite-sample nuisance mean, Gram
matrix, singular values/effective rank, total candidate residual, and support
condition. The target ratio is now a total realized norm ratio; at target
`0.5`, the mean nuisance energy/style energy is `0.25`, not `0.5`.

The mixed-support exact benchmark used five seeds and R=`3/5/8`. Robust active
cosine means at total norm ratio `0.5` were `0.965`, `0.976`, and `0.979`
respectively. The actual branches saved in the artifact include one/two
outliers, unequal confidence, missing local region, partially invalid delta,
poor recovered candidate, and the biased-nuisance limitation. All use zero
background and zero `d_scaling`.

The reusable geometry cache stores source projections, finite-difference
Jacobians, KNN neighbors, and graph degree once. The vectorized full-bank run
used 44,764 Gaussians and 8 cameras, K=`8`; cache build was `0.14 s`, candidate
recovery was approximately `0.05 s` each, and peak RSS was approximately
`424 MB`. Five observed_2d bundles were saved with `target_xyz=None`.

At mixed total nuisance norm ratio `0.5`, full-bank recovered factorization
reported:

| method | active cosine | centered explained variance | style leakage | background energy |
|---|---:|---:|---:|---:|
| single recovered candidate | 0.943 | 0.887 | 0.128 | 0 |
| robust shared | 0.977 | 0.953 | 0.059 | 0 |
| structured no-label | 0.994 | 0.987 | 0.017 | 0 |

The full-bank recovered gate passes for this controlled R=5 case. No real
target images or Stage 2 training were run. The deliberate systematic-bias
case remains non-identifiable from one source: a shared target bias cannot be
separated from style without a prior, labels, or multiple source objects.
