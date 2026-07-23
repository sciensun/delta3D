# Next Steps

## Active Decision

Do not start Stage 2 or generate real targets yet. The stochastic benchmark
shows that no-label shared-style extraction is viable at moderate nuisance
with enough samples, but it is not reliable at stronger nuisance with only
three to five samples. The recovered oracle-2D diagnostic is only partial
because the full-bank CPU loop was memory-unstable.

Next, improve the recovered-candidate evaluation to a full-bank or documented
larger sampled run, then decide whether the initial real pilot needs three or
five standardized target samples. Do not use exact nuisance labels as the
primary method and do not interpret the biased-nuisance case as identifiable.

## Gates

- exact candidates: no-label active cosine >= 0.90, explained variance >= 0.80,
  nuisance leakage <= 0.20;
- recovered oracle candidates: active cosine >= 0.85, explained variance >=
  0.65, and improvement over a single recovered candidate;
- background delta exactly zero and `d_scaling` exactly zero;
- no Stage 2 or style-transfer claim from one source object.

## Real Pilot Layout

`assets/prepared/big_carved_wooden_elephant_sculpture/real_pilot_blocky_to_rounded/`
contains `sample_A`, `sample_B`, and `sample_C`. All use the same standardized
prompt and eight cameras. The folders are empty. Any observed color, texture,
body, ear, trunk, or limb nuisance will be recorded post-generation rather
than prescribed beforehand. Expand to five samples only if three-sample
stability is insufficient.
