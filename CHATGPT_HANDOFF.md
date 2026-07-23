# ChatGPT Handoff: Sparse Observation Recovery

## Status

**FAIL / PARTIAL.** The complete-observation sanity check passes, but the
20--40% sparse fixed-bank recovery gate fails. Stage 2 remains paused.

## Objective

Test whether sparse observed-2D target observations can recover dense xyz-only
deformation while keeping background and d_scaling exactly zero.

## Changes

- corrected the cached IRLS solver to accumulate nonlinear Newton increments;
- retained unobserved foreground as graph-constrained variables;
- added robust singular-normal fallback and convergence history;
- added confidence-aware structured factorization updates;
- added five-sample real-pilot QC preflight;
- added post-recovery five-candidate sparse factorization comparison;
- added track-aware support diagnostics, symmetric mutual-KNN graph, and
  control-node translation prior;
- added strict K-view support and camera conditioning diagnostics;
- added full-bank linearized PCG comparison and control representation ceilings.
- updated sparse benchmark and research documentation.

## Validation

`python scripts/run_sparse_observation_benchmark.py` ran on body roundness, ear
expansion, and trunk bending at 10/20/40/60/100 percent with five deterministic
seeds. At 20 percent active-cosine means were body `0.006`, ear `0.048`, trunk
`0.015`; at 40 percent `0.077`, `0.228`, `0.014`; at 100 percent all were
`1.000`. Direct full-observation body sanity: cosine `1.0`, energy ratio `1.0`,
near-zero reprojection residual. CPU syntax and 16 fixture-free tests passed;
three fixture-based tests were skipped because pytest is unavailable.

After sparse recovery, body active cosine for structured/mean-plus-graph was
`0.046/0.009` at 20%, `0.413/0.112` at 40%, and `0.995/0.994` at 100%.

Track-aware diagnostics show >=2-view support of about `0.199/0.397` at
track fractions `0.2/0.4`. Representative anchor cosine is `1.0`; graph
completion active cosine body/ear/trunk is `0.898/0.880/0.881` at `0.2` and
`0.972/0.963/0.959` at `0.4`. A 64-control graph reaches
`0.938/0.942/0.908` on the same teachers at `0.2`.

The current five-seed exact-K two-view fraction `0.20` means are body/ear/trunk
`0.899/0.889/0.871` for both random and maximum-center-baseline selection.
At 1 px noise body/ear are `0.898/0.888`; at 5% overconfident outliers they
fall to `-0.081/-0.011`. Full-bank IRLS-100 active cosine is `0.971` versus
linearized PCG `0.443`; PCG is faster but not yet an equivalent nonlinear
solver. The graph has 137,393 edges and 157 components.

## Limitation and decision

Status remains **PARTIAL**: independent per-view dropout is a FAIL stress test;
clean all-valid and exact-K track-aware recovery is promising; robust
noise/outlier and silhouette/hybrid gates are incomplete. No real target images
were generated and no style-transfer claim is made.

Track-aware implementation baseline: `a192585`.
Final HEAD is reported separately to avoid recursive metadata commits.
