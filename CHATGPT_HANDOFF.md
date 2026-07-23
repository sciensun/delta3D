# ChatGPT Handoff: Stochastic Target-Template Factorization

## Status

**PARTIAL.** The v1 sanity-test interpretation was corrected and v2 stochastic
factorization passed at low/moderate nuisance for R=5/8, but recovered-candidate
full-bank validation remains incomplete. Stage 2 was not started.

## Changes

- Corrected distinct global/foreground/active metric regions and labels.
- Normalized robust trimming by valid confidence.
- Added v2 random independent nuisance benchmark for R=3/5/8 and three seeds.
- Added biased-nuisance identifiability failure case.
- Added recovered oracle observed-2D candidate factorization diagnostic.
- Corrected real pilot to three identical standardized prompts with null
  pre-generation nuisance fields.

## Results

No-label robust active cosine / explained variance:
`R5, nuisance .25 = .935/.854`; `R8, nuisance .25 = .972/.941`;
`R5, nuisance .5 = .788/.415`; `R8, nuisance .5 = .896/.767`.
Sampled recovered-oracle diagnostic: cosine `.901`, energy `.670`, explained
variance `.802`, versus single candidate cosine `.656`.

## Limitation

The full-bank recovered-candidate CPU loop was memory-unstable; the reported
recovered result evaluates 4,096 active Gaussians. No real target images,
CUDA run, cross-object style representation, or Stage 2 model exists.

## Decision

Keep Stage 2 paused. Complete full-bank/documented sampled recovered-candidate
validation before the three-sample real pilot.

Implementation commit SHA: to be recorded after commit.
Final metadata commit SHA: to be recorded after metadata commit.
