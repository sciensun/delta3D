# ChatGPT Handoff: Correspondence Reliability and Silhouette Audit

## Latest Result

Exhaustive pair hypotheses were added. On body exact-3 with 5% overconfident
outliers, `drop_track` rejected all injected outliers but falsely rejected
91.1% of clean views and reached active cosine 0.460; `keep_best_two` retained
24 tracks containing an outlier and reached 0.331. Neither passes. Recovered
silhouette metrics also degraded: body/ear/trunk IoU/F1/Chamfer changed from
pre `0.872/0.366/4.68`, `0.905/0.552/3.66`, `0.930/0.633/2.58` to post
`0.842/0.251/6.78`, `0.855/0.320/6.41`, `0.878/0.357/6.80`. Status remains
PARTIAL; independent correspondence redundancy is required next.

## Current Iteration

Status: **PARTIAL**. Clean track-aware recovery is promising, while outlier
robustness and the silhouette/hybrid gate remain incomplete. Exact-2 tracks do
not support within-track consensus.

The implementation now records true injected-outlier rejection separately from
clean false rejection, adds leave-one-view-out consensus for >=3-view tracks,
and replaces fixed circles with a foreground-filtered depth-sorted
variable-radius alpha-splat CPU fallback plus dynamic SDF sampling. One body
exact-3 5%-outlier diagnostic rejected 55.2% of injected outlier views and
falsely rejected 2.1% clean views, but active cosine was 0.004. Correct
silhouette-only active cosine is about -0.005/-0.020/-0.012 for body/ear/trunk;
target-mask IoU is 0.872/0.905/0.930. These do not pass the hybrid gate.

## Gated Workflow Result

The machine decision is `REAL_ASSET_BLOCKED`. The repaired exact-3 gate passes
for the available three-teacher records, but target_A and target_B contain no
images. Silhouette remains disabled because its independent synthetic
post-recovery gate failed. No real mining or Stage 2 training was started.

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
- added robust kernels and delayed cross-view track rejection;
- added synthetic silhouette generation and null/shuffled-control diagnostics.
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

At 1 px noise body/ear remain `0.898/0.888`. At 5% overconfident outliers,
Huber is `-0.081/-0.011`; delayed rejection improves body/ear/trunk to
`0.769/0.772/0.803`, still below the robust gate. Correct silhouette versus
source-silhouette null active cosine is body `0.000/-0.004`, ear
`-0.019/0.008`, trunk `0.004/-0.013`; no genuine silhouette gain is shown.

## Limitation and decision

Status remains **PARTIAL**: independent per-view dropout is a FAIL stress test;
clean all-valid and exact-K track-aware recovery is promising; robust
noise/outlier and silhouette/hybrid gates are incomplete. No real target images
were generated and no style-transfer claim is made.

Track-aware implementation baseline: `a192585`.
Final HEAD is reported separately to avoid recursive metadata commits.
