# Project Method

This repository implements an image-supervised, 3DGS-only style deformation
pipeline. The main target route is generated multi-view images, not an
independently reconstructed ordinary 3D model.

## Objective

The canonical source is a stylized Graphdeco 3DGS Gaussian bank:

```text
G_sty -> Delta* -> B(F, z)
```

Stage 1 mines a free deformation from source-to-target image observations.
Weak generated images are observations, not paired Gaussian ground truth.
Stage 1.5 tests view, repeat, identity, intensity, and structural reliability.
Stage 2 is intentionally paused until a stable, reproducible style Delta* exists.

## Current Source

The accepted source is the Graphdeco model adapted into:

```text
output/elephant_source_graphdeco/
```

It has 44,764 Gaussians and is loaded through the existing delta3D renderer.
The source builder is not retrained in the current experiment.

## Fixed-bank terminology

- `G_sty`: the canonical stylized source 3DGS loaded by delta3D.
- `G_ord_ref`: an optional independently generated ordinary/content 3D reference.
  It may have different Gaussian count, indexing, and distribution, so its
  Gaussian IDs must never be treated as paired with `G_sty`. It is not required
  by the main image-observation route.
- `G_ord_canon`: the ordinary model fitted on the unchanged `G_sty` bank:
  `G_ord_canon = T(G_sty, Delta*)`. It preserves Gaussian count, IDs, source
  foreground/background identity, source KNN graph, and source part identity.

The main supervision object is `(G_sty, image observations, Delta*)`.
If the optional reference route is used, `G_ord_ref` is only an auxiliary
reference used to construct observations and confidence; `G_ord_canon` is still
fitted on the fixed `G_sty` bank.

## Stage 1

The historical foreground-only pseudo-delta is:

```text
output/elephant_source_graphdeco/mined_delta_foreground_xyz_only.pt
```

Foreground gating and exact-zero `d_scaling` are successful implementation
constraints: it has `d_scaling = 0` and forces all background Gaussian
deformation to zero. However, independent target-view splits produced weighted
cosine `0.1176`, median per-Gaussian cosine `0.0749`, and 46.5% directional
conflict. It is **unusable for Stage 2** and is not a successful teacher.

The structured fitting experiment approximates the free delta with:

```text
d_hat_i = sum_k w_ik t_k
```

or:

```text
d_hat_i = sum_k w_ik [t_k + A_k (x_i - c_k)]
```

where `w_ik` is hard or spatial soft part assignment, `t_k` is a part
translation, `A_k` is a local affine matrix, and `c_k` is the part center.

No Gaussian scaling, rotation, mesh, or Tripo representation is used.

## Acceptance Gate

A fitted Delta* candidate must satisfy:

- foreground delta energy approximately 100%;
- `d_scaling` exactly zero;
- global cosine similarity with the foreground-only delta at least `0.75`;
- energy preservation ratio at least `0.60`;
- explained variance at least `0.50`;
- x1 visibly changes the source while remaining sharp;
- x5 remains coherent and interpretable;
- background Gaussians remain fixed.

Stage 2 remains disabled until this gate passes.

## Stage 1.5 Reliability

The next reliability experiment is split-view consistency: independently mine
xyz-only deltas from deterministic target subsets, compare their direction and
magnitude on foreground Gaussians, then optionally form a confidence-gated
consensus and graph-regularized teacher. The current target package is not fully
matched: there is no exact elevation-0/azimuth-90 source frame for
`03_standard.png`, so arbitrary sorted-order mapping is prohibited.

The exact-camera issue was subsequently resolved by rebuilding an eight-view
train-only subset. Independent subset mining then produced weighted cosine
`0.1176`, median per-Gaussian cosine `0.0749`, and 46.5% directional conflict.
This is a weak/unreliable teacher signal. Consensus and Stage 2 remain paused;
the next research change must improve target geometric correspondence.

## Main Image-Observation Pipeline

An independently generated ordinary object is not a geometric pair merely
because it shares an object category or a requested camera view. The intended
pipeline is:

```text
stylized GLB
-> canonical stylized 3DGS and source views
-> generated target/content multi-view images
-> foreground/mutual/semantic/visibility-filtered 2D observations
-> fixed-bank free Delta
-> view/repeat/structure reliability
-> stable style Delta*
-> future source-conditioned multiscale learning
```

The image-first schema accepts observed 2D target positions without target xyz.
Oracle 3D and hybrid modes exist only for controlled synthetic tests. Stage 1
adds projected 2D-motion loss when observed coordinates are present, keeps
optional 3D loss explicit, and records whether oracle 3D was available.
Optional alignment of `G_ord_ref` removes coordinate differences only; it must
not non-rigidly erase the intended style change.

The correspondence hierarchy is explicit for optional reference experiments:
semantic anchors, shared-view 2D matches, visibility checks, and confidence
fusion. Unmatched or low-confidence Gaussians remain unpaired and are handled by
confidence weighting and graph regularization; they are not forced to have a
target position.

This method currently assumes topology-preserving continuous geometry changes.
New or removed parts and major topology changes are out of scope. Once the
canonical identity is established, densification, pruning, splitting, merging,
and reordering are disabled.

## Synthetic Known-Delta Benchmark

The canonical Graphdeco Gaussian bank now has controlled xyz-only benchmark
teachers for body roundness, ear expansion, and segmented trunk bending under:

```text
output/elephant_source_graphdeco/synthetic_known_delta/
```

The body-roundness benchmark shows that image-only LPIPS/RGB optimization is
insufficient even with perfectly rendered targets, while direct correspondence
loss recovers the known geometry. No real target-image observation bundle has
yet passed the reliability gate, so stable style consensus and Stage 2 remain
disabled.

## Reliability Definition

For each `(object, style_operation, intensity, repeat, view)` task:

```text
free_delta = stable_style_delta + residual
```

The stable component must be reproducible across views and repeated target
generations, structurally coherent, and associated with a style operation and
intensity. Topology-preserving continuous changes are in scope. Major topology
changes, newly added parts, and removed parts are out of scope.

## Image-Derived Observation Status

The first image-derived baseline is implemented in
`correspondence/matching_backends.py` and
`correspondence/image_observations.py`. It uses same-view OpenCV Farneback
flow, forward/backward cycle confidence, foreground filtering, projected
footprint aggregation, and a coarse projected depth-bin visibility estimate.
This is an implemented diagnostic, not a validated learned matcher. Body
roundness failed the synthetic observation gate. The latest environment had no
working NVIDIA driver, so fixed-bank Stage 1 recovery was not run for these
image-derived bundles.

Oracle target projections remain a synthetic upper bound. Image-derived target
observations are matcher output. Real generated target observations require
repeated quality-controlled views and have not produced a stable style delta.
