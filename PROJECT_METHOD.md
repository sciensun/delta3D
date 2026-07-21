# Project Method

This repository implements a 3DGS-only two-stage deformation pipeline.

## Objective

The canonical source is a stylized Graphdeco 3DGS Gaussian bank:

```text
G_sty -> Delta* -> B(F, z)
```

Stage 1 mines a pseudo deformation from correspondence-filtered ordinary
reference observations. Weak manually generated target images are observations,
not paired Gaussian ground truth. Stage 2 is intentionally paused until Stage 1
produces a structured, reproducible Delta*.

## Current Source

The accepted source is the Graphdeco model adapted into:

```text
output/elephant_source_graphdeco/
```

It has 44,764 Gaussians and is loaded through the existing delta3D renderer.
The source builder is not retrained in the current experiment.

## Fixed-bank terminology

- `G_sty`: the canonical stylized source 3DGS loaded by delta3D.
- `G_ord_ref`: an independently generated ordinary reference model. It may have
  different Gaussian count, indexing, and distribution, so its Gaussian IDs
  must never be treated as paired with `G_sty`.
- `G_ord_canon`: the ordinary model fitted on the unchanged `G_sty` bank:
  `G_ord_canon = T(G_sty, Delta*)`. It preserves Gaussian count, IDs, source
  foreground/background identity, source KNN graph, and source part identity.

The actual paired training object is `(G_sty, G_ord_canon, Delta*)`.
`G_ord_ref` is only the teacher used to construct correspondence and confidence.

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

## Corrected Paired-Supervision Pipeline

An independently generated ordinary object is not a geometric pair merely
because it shares an object category or a requested camera view. The intended
pipeline is:

```text
stylized 3DGS
-> unified ordinary 3D representation
-> global similarity alignment
-> semantic anchors
-> dense multi-view correspondence
-> confidence-filtered paired data
-> Stage 1 Delta*
-> split-view reliability gate
-> structured compression
-> Stage 2
```

Global alignment removes coordinate-system differences only; it must not
non-rigidly warp away the intended stylized-to-ordinary deformation. The new
Stage 1 correspondence interface accepts lifted target xyz positions and
confidence, adds projected 2D-motion and 3D Huber losses, and keeps foreground
gating, exact-zero background motion, and exact-zero Gaussian scaling.

The correspondence hierarchy is explicit: similarity alignment, semantic
anchors, shared-view 2D matches, depth lifting, robust multi-view fusion, and
fixed-bank canonical fitting. Unmatched or low-confidence Gaussians remain
unpaired and are handled by confidence weighting and graph regularization; they
are not forced to have a target position.

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
loss recovers the known geometry. No ordinary generated target has yet passed
the real correspondence quality gate, so Stage 2 remains disabled.
