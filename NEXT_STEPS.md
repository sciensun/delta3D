# Next Steps

## Active Decision

Select **R=5 standardized target samples** for the first real pilot. The v3
controlled exact results improve from R=3 to R=5, and the full-bank recovered
oracle-2D R=5 gate passes with structured no-label factorization. Do not
generate them in this iteration; Stage 2 remains paused.

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

## Real Pilot Layout

Expand the empty layout to `sample_A` through `sample_E`, each with the same
eight cameras and the same prompt. Keep pre-generation observed appearance,
geometry, and nuisance fields null. Only after five target sets pass semantic,
silhouette, view, and observation checks should an instance-stable delta be
considered. Cross-object style learning and Stage 2 remain blocked.
