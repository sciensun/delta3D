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
