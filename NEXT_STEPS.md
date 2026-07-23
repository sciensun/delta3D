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

## Real Pilot Layout

Expand the empty layout to `sample_A` through `sample_E`, each with the same
eight cameras and the same prompt. Keep pre-generation observed appearance,
geometry, and nuisance fields null. Only after five target sets pass semantic,
silhouette, view, and observation checks should an instance-stable delta be
considered. Cross-object style learning and Stage 2 remain blocked.
