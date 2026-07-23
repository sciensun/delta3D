# Next Steps

## Active Decision

Do not start Stage 2, real style training, or part compression. First validate
the controlled shared-style/template-nuisance factorization and the CPU
graph-coupled oracle recovery already added in this iteration.

The current conceptual model is:

```text
candidate_delta[o,s,r] = instance_style_delta[o,s]
                        + template_specific_delta[o,r]
                        + residual
```

The immediate active experiment is to inspect
`output/elephant_source_graphdeco/template_factorization_benchmark/factorization_summary.json`,
then validate the same factorization on oracle observed-2D recovered candidates.
Only after that should the three empty real target-template folders be filled
with actual images.

## Current Gates

- background delta must be exactly zero;
- `d_scaling` must be exactly zero;
- moderate-nuisance shared active cosine should be at least `0.90`;
- explained variance should be at least `0.80`;
- nuisance leakage should be at most `0.20`;
- the CPU solver must pass its oracle observed-2D regression before image-derived
  recovery is interpreted.

## Prepared Real Pilot

The empty conditional-template manifest is at:
`assets/prepared/big_carved_wooden_elephant_sculpture/real_pilot_blocky_to_rounded/target_template_manifest.json`.
It defines `template_A`, `template_B`, and `template_C` at intensity `0.5`.
They are target-template variants, not repeated identical generations. No
target images have been generated or fabricated.

Stage 2 remains paused until shared-style stability is demonstrated across
template variants and real observations pass quality control.
