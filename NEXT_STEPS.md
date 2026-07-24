# Next Steps

## Active Decision

Validate sparse observation recovery before any real target generation. The
first implementation of IRLS was corrected to accumulate Newton increments;
the rerun reaches the oracle result at 100% observations, but 10--40% coverage
is still below the acceptance gate. Stage 2 and real generation remain paused.

The five samples must use the same standardized prompt and condition. They are
independent conditional generations, not deliberately heavy/slim/large-ear
variants. Record observed nuisance only after generation and quality control.

## Acceptance Evidence

- 44,764-Gaussian full-bank recovered R=5 active cosine: `0.994` structured;
- centered explained variance: `0.987`;
- style leakage: `0.017`;
- background energy: `0`;
- `d_scaling`: exactly zero;
- cache recovery: about `0.05 s` per candidate, peak RSS `424 MB`.

Sparse benchmark artifact:
`output/elephant_source_graphdeco/sparse_observation_benchmark/sparse_benchmark_summary.json`.
The corrected graph/IRLS solver has active cosine means at 20% coverage of
approximately `0.006` body, `0.048` ear, and `0.015` trunk; at 40% they are
approximately `0.077`, `0.228`, and `0.014`; at 100% they are `1.000` for
all three teachers. These low-coverage values fail the dense recovery gate.
The old low-support-clearing baseline cannot recover unobserved regions by
definition. A separate direct full-observation sanity check gives body active
cosine `1.0`, energy ratio `1.0`, and near-zero reprojection residual.

The next technical decision is whether to add silhouette/boundary observations
and stronger multiscale graph priors, or to require higher point coverage.

Post-recovery factorization confirms the same bottleneck: body active cosine
for structured/mean-plus-graph is `0.046/0.009` at 20%, `0.413/0.112` at 40%,
and `0.995/0.994` at 100%. Factorization is not the limiting component at low
coverage.

Track-aware support separates the old view-dropout result: track fractions
`0.2/0.4` retain approximately `0.199/0.397` foreground tracks with at least
two views. Representative anchor cosine is `1.0`; graph-completed active
cosine body/ear/trunk is `0.898/0.880/0.881` at `0.2` and
`0.972/0.963/0.959` at `0.4`. A 64-control translation graph reaches
`0.938/0.942/0.908` at `0.2`. These are promising diagnostics, not the
five-seed acceptance gate. The next active experiment is five-seed track-aware
validation plus silhouette/boundary observations.

The deduplicated five-seed clean track-dropout artifact contains 15 unique
records at fraction `0.20`. Active cosine mean/std/min/max are body
`0.899/0.002/0.897/0.904`, ear `0.889/0.007/0.879/0.898`, and trunk
`0.871/0.012/0.849/0.882`. Anchor cosine is `1.0` in these clean oracle
runs. This is promising but not the robust gate: baseline-conditioned K-view,
noise, outlier, and silhouette branches are still pending.

Exact-K two-view five-seed fraction `0.20` is also stable: body/ear/trunk
active cosine means are `0.899/0.889/0.871` for both random and
maximum-center-baseline selection. This synthetic equality does not replace
ray-angle and condition-number diagnostics.

This iteration completed the first robust correspondence comparison. At 1 px
noise, body/ear active cosine remains `0.898/0.888`. At 5% overconfident
outliers, Huber collapses to `-0.081/-0.011`; iterative rejection improves
body/ear/trunk to approximately `0.769/0.772/0.803` but does not satisfy the
0.85 robust gate. Correct target silhouette observations currently produce
body/ear/trunk cosine `0.000/-0.019/0.004`, versus source-silhouette null
controls `-0.004/0.008/-0.013`; silhouette implementation is therefore not
yet useful and needs better synthetic mask/gradient calibration.

## Real Pilot Layout

Expand the empty layout to `sample_A` through `sample_E`, each with the same
eight cameras and the same prompt. Keep pre-generation observed appearance,
geometry, and nuisance fields null. Only after five target sets pass semantic,
silhouette, view, and observation checks should an instance-stable delta be
considered. Cross-object style learning and Stage 2 remain blocked.
