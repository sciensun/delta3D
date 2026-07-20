# Project Method

This repository implements a 3DGS-only two-stage deformation pipeline.

## Objective

The canonical source is a stylized Graphdeco 3DGS Gaussian bank:

```text
G_sty -> Delta* -> B(F, z)
```

Stage 1 mines a pseudo deformation from weak manually generated target images.
Stage 2 is intentionally paused until Stage 1 produces a structured, reusable
Delta*.

## Current Source

The accepted source is the Graphdeco model adapted into:

```text
output/elephant_source_graphdeco/
```

It has 44,764 Gaussians and is loaded through the existing delta3D renderer.
The source builder is not retrained in the current experiment.

## Stage 1

The successful foreground-only pseudo-delta is:

```text
output/elephant_source_graphdeco/mined_delta_foreground_xyz_only.pt
```

It has `d_scaling = 0` and forces all background Gaussian deformation to zero.

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
