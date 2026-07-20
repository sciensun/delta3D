# Experiment Log

All entries use the current repository outputs. No Stage 2 run has been started.

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
- Result: foreground-only pseudo-delta PASS for structured fitting.

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
