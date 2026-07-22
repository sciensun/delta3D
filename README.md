# Delta3D: Image-Supervised Geometric Style Transformation on Fixed 3D Gaussian Banks

Delta3D studies source-conditioned geometric style operations on a fixed 3DGS Gaussian bank. The main route is image-first and remains 3DGS-only:

```text
stylized GLB
-> canonical stylized 3DGS and source views
-> generated target/content views preserving cameras, parts, identity, pose, and topology
-> confident source-to-target 2D observations
-> fixed-bank free delta
-> view/repeat/structure reliability analysis
-> stable style delta
-> future source-conditioned multiscale style learning
```

This repository does not claim real style transfer yet. Stage 2 training is paused until a reproducible stable style delta is available.

Status labels used below are: **Implemented** (code exists), **Validated**
(code or controlled benchmark passed), **Planned** (interface/design only), and
**Historical** (recorded result that is not an active method).

## Objective and Scope

The primary task is: given a new object and a learned geometric style operation, apply that operation to the object's stylized 3DGS representation. The current demo direction is stylized sculpture to a less stylized/ordinary appearance. The intended operation changes geometry while preserving object identity, pose, topology, and major parts.

The main representation is a fixed Gaussian bank. It does not use mesh deformation, Tripo, Gaussian densification, pruning, splitting, merging, or reordering. Gaussian scaling deformation is disabled by default and background Gaussians must remain exactly fixed.

## Core Insight

A single optimization result is not automatically a style delta:

```text
free_delta = stable_style_component
           + generation_randomness
           + matching_error
           + optimization_residual
```

A stable style delta is the source-conditioned component reproducible across reliable views, repeated target generations, and structurally related regions, associated with a style family and intensity.

## Inputs and Outputs

Inputs include a stylized GLB, a trained canonical stylized 3DGS model `G_sty`, generated target/content multi-view images, and a `style_data.StyleTaskRecord` describing object, style, intensity, repeat, cameras, prompts, and quality control.

Stage 1 outputs a free delta payload containing `d_xyz`, zero `d_scaling`, source xyz, foreground mask, observation mode, camera support, confidence, residual statistics, and loss metadata. Stage 1.5 produces a `StableStyleDelta` only after reliability gates pass. Future Stage 2 will output coarse, part, and detail delta components conditioned on style family and intensity.

## Algorithm Stages

1. **Source construction — Implemented/Validated.** Render the GLB with perspective cameras and train or load a source Graphdeco 3DGS.
2. **Target view generation — Planned.** Generate target images from the stylized object while preserving camera, parts, identity, pose, and topology.
3. **Image observations — Implemented as interfaces.** Store foreground-filtered 2D coordinates, visibility, confidence, camera names, and optional 3D oracle data in `ObservationBundle`.
4. **Free delta mining — Implemented.** Optimize per-Gaussian xyz deformation through the existing renderer. Image-only, observed-2D, oracle-3D, and hybrid modes are explicit.
5. **Reliability analysis — Implemented as tools/interfaces.** Compare view splits, repeated generations, identity controls, intensity paths, and local structure.
6. **Stable delta — Planned.** Fuse only reliable components into `StableStyleDelta`.
7. **Source-conditioned multiscale learning — Planned.** Learn coarse/part/detail style operations from stable deltas. No Stage 2 training is included here.

## Main Image-Observation Route

The primary route does not require an independently reconstructed ordinary 3D model. Target images are observations of the generated content object. Their Gaussian IDs are not assumed to correspond to source Gaussian IDs. The required intermediate artifact is a source-indexed observation bundle created from shared-view 2D matches and confidence/visibility checks.

`observed_2d` bundles are valid without `target_xyz`. The schema rejects missing `target_xy` in this mode and never silently projects an oracle target. `oracle_3d` and `hybrid` are available for synthetic regression and controlled experiments.

## Optional Content-3D Baseline

An independently generated content 3D model may be used as an optional baseline or auxiliary reference. It is named `G_ord_ref` and has no inherent Gaussian correspondence with `G_sty`. If used, it must be aligned and matched before it can supervise a fixed-bank canonical model. It is not a required main stage.

## Data and Task Schema

The experimental index is:

```text
(object, style_operation, intensity, repeat, view)
```

`StyleTaskRecord` contains object, source model, style family, source/target attributes, intensity, repeat ID, camera names, image roots, prompt, affected/preserved parts, and quality-control metadata. `ObservationBundle` contains source-indexed Gaussian geometry plus optional per-view `target_xy`, visibility, confidence, support count, reprojection residual, optional target xyz, and explicit validity masks.

## Repository Architecture

```text
train.py                         upstream source 3DGS training; untouched
train_delta_mining.py            backward-compatible Stage 1 CLI
stage1/                          configuration, objectives, regularizers, outputs
stage2/                          future contracts only; no model training
style_data/                      task records and manifests
correspondence/                  image/3D observation schema and losses
scene/free_delta_model.py        fixed-bank free xyz delta parameters
scene/canonical_ordinary_model.py optional fixed-bank canonical wrapper
scripts/                         rendering, diagnostics, synthetic benchmarks
docs/UPSTREAM_README.md           preserved upstream documentation
```

## Current Validation

### Validated synthetic fixed-bank recovery

The body-roundness known-delta benchmark recovered a controlled xyz-only delta:

```text
global cosine:        0.9614
energy ratio:         0.8835
explained variance:   0.9233
background energy:    0
d_scaling:            exactly zero
```

This validates the fixed-bank recovery formulation when correspondence is reliable. It does not validate real generated target images.

### Historical weak-target failure

Independent target-view subsets produced:

```text
weighted cosine:             0.1176
median per-Gaussian cosine:  0.0749
directional conflict:        46.5%
```

The old foreground-only free delta is not a stable style delta and is unusable for Stage 2. Foreground gating and zero `d_scaling` succeeded as constraints, but the style reliability gate failed.

### Validated observed-2D acceptance experiment

The controlled body-roundness benchmark also has an image-observation-only
run. The builder creates exact camera projections from the hidden synthetic
teacher, but the optimizer bundle contains only `target_xy`, visibility, and
confidence; `target_xyz` is not available to optimization and is used only by
the post-training evaluator.

```bash
python scripts/run_observed_2d_benchmark.py \
  --model_path output/elephant_source_graphdeco \
  --source_path assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset \
  --gt_delta_path output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_body_roundness.pt \
  --output_dir output/elephant_source_graphdeco/synthetic_observed_2d_benchmark
```

**Validated:** clean 8-view recovery reached global cosine `0.9999997`, active
region cosine `1.0`, explained variance `0.9999995`, energy ratio `0.9997`,
zero background energy, and exact-zero `d_scaling`. The A/B split had active
region cosine `1.0`; full-foreground weighted cosine was `0.99998` with 29.7%
conflict from inactive/near-zero entries. Novel-view projection RMSE was
`0.0013 px` over 216 cameras. These are controlled synthetic results, not a
claim of real style transfer.

The clean view trend was one/two/four/eight views: cosine `0.7512/0.99997/
1.0/1.0`, with one-view explained variance `0.3602` and 2.75 px held-out
projection error. Eight-view robustness remained strong at global cosine
`0.9988` for 0.5 px noise, `0.9770` for 2 px noise, `0.9975/0.9909` for 5%/10%
outliers, and `0.9981` with 70% visibility. Stage 2 remains **Planned** and
paused until real target repeats pass reliability gates.

## Actual Interfaces

Load a legacy synthetic oracle correspondence bundle:

```bash
python scripts/evaluate_correspondence_quality.py \
  --correspondence_path output/elephant_source_graphdeco/synthetic_known_delta/synthetic_correspondence_body_roundness.pt \
  --output_path /tmp/body_correspondence_quality.json
```

Stage 1 remains compatible with the existing entry point. Controlled observed-2D runs may use:

```bash
python train_delta_mining.py \
  -s path/to/source_dataset \
  --model_path output/source_model \
  --target_image_root path/to/target_views \
  --correspondence_path path/to/observation_bundle.pt \
  --observation_mode observed_2d \
  --lambda_corr_2d 1.0 \
  --max_d_scaling 0.0 \
  --disable_d_scaling
```

Use `oracle_3d` only for controlled known-delta tests. Do not use it to claim real image-only correspondence.

## Acceptance Gates

Do not promote a free delta to a stable style delta unless the source render remains sharp, background deformation energy is zero, `d_scaling` is exactly zero, independent view consistency is strong or improving, repeated target generations agree directionally, structurally related regions show compatible motion, identity-control deltas remain near zero, intensity paths are interpretable, and low-confidence/unmatched regions are not forced into a teacher.

Do not start Stage 2 before the stable-style gate passes.

## Roadmap

1. Prepare repeated target generations with strict camera and topology controls.
2. Build real `observed_2d` bundles with foreground, mutual-match, semantic, visibility, and confidence filtering.
3. Run view/repeat/identity/intensity reliability analysis.
4. Save a `StableStyleDelta` only if the gate passes.
5. Design and train source-conditioned multiscale style learning.

## Upstream Attribution

The original renderer and source dynamic reconstruction code are from [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians), implementing Yang et al. See `docs/UPSTREAM_README.md` for preserved upstream usage, datasets, results, and attribution.

```bibtex
@article{yang2023deformable3dgs,
  title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
  author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
  journal={arXiv preprint arXiv:2309.13101},
  year={2023}
}
```
