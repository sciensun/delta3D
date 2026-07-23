# ChatGPT Handoff: Calibrated Full-Bank Target-Template Factorization

## Status

**PARTIAL / controlled full-bank gate PASS.** No real target images or Stage 2
training were run.

## Changes

- calibrated total realized nuisance norm rather than per-basis ratio;
- added graph-smoothed/decorrelated nuisance modes and support conditions;
- added actual outlier, confidence, missing-region, invalid, poor-candidate,
  label, and bias branches;
- added structured no-label factorization;
- added reusable/vectorized full-bank CPU geometry cache;
- corrected docs and selected five standardized pilot samples.

## Results

Full bank: 44,764 Gaussians, 8 views, R=5, total nuisance norm/style norm
`0.5` (energy ratio `0.25`). Structured no-label recovery: active cosine
`0.994`, centered explained variance `0.987`, style leakage `0.017`, background
energy `0`, and exact-zero `d_scaling`. Single candidate: cosine `0.943`,
explained variance `0.887`. Cache `0.14 s`, recovery about `0.05 s/candidate`,
peak RSS about `424 MB`.

## Limitation

This remains a one-object controlled oracle-observation result. Systematic
target-template bias is not identifiable from one source. Real standardized
targets, CUDA validation, cross-object representation, and Stage 2 remain
unimplemented.

## Decision

Use five identical-condition standardized target samples for the first real
pilot. Generate and evaluate them in the next iteration, not this one.

Implementation commit SHA: to be recorded after commit.
This handoff belongs to the final commit reported by cdx.
