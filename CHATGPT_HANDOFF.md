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

## Limitation and decision

Sparse points alone do not presently provide sufficient dense recovery. The
next experiment should add silhouette/boundary observations and stronger
multiscale graph priors, then rerun the sparse gate. No real target images were
generated and no style-transfer claim is made.

Implementation commit SHA: `4866c07`.
Final HEAD is reported separately to avoid recursive metadata commits.
