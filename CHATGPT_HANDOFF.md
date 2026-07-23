# ChatGPT Handoff: Shared Style / Template Nuisance Factorization

## Status

**PARTIAL, controlled benchmark PASS.** No real style delta or Stage 2 model
was produced.

## Objective

Treat generated ordinary targets as conditional target-template samples rather
than exact pointwise object pairs. Estimate a shared instance-style component
from multiple candidate deltas while retaining template-specific nuisance.

## Changes

- Added `TargetTemplateRecord` with backward-compatible `StyleTaskRecord`
  loading.
- Added `stage1/template_factorization.py` with mean, median, geometric median,
  robust shared, and nuisance-regression estimators.
- Added the controlled five-template benchmark script and artifacts.
- Added graph-coupled CPU oracle observed-2D recovery.
- Changed the prepared elephant pilot from nominal repeats to empty conditional
  templates A/B/C with explicit allowed variations and invariants.
- Rewrote `NEXT_STEPS.md` so only the factorization/oracle decision is active.

## Validation

Controlled body-roundness factorization: nuisance regression active cosine
`1.0000`, explained variance `1.0000`, leakage `<0.00001`; robust shared cosine
`0.9990`, energy `0.9011`; one-template energy `0.4027`. The CPU oracle
observed-2D graph solver reached active cosine `0.9999998`, energy `1.0017`,
explained variance `0.9999989`. Background energy was zero and `d_scaling` was
exactly zero. CPU tests and syntax checks passed. CUDA was unavailable.

## Limitation and Decision

The controlled decomposition is validated, but it does not establish that real
generated target images contain a stable style component. Do not start Stage 2,
part compression, or real target generation until oracle-candidate
factorization and quality control are reviewed.

## Next Experiment

Apply the factorization to multiple oracle observed-2D recovered candidates,
then fill the three prepared target-template folders with real multi-view
conditional samples and evaluate them as a distribution.

## Artifacts

`output/elephant_source_graphdeco/template_factorization_benchmark/`
`assets/prepared/big_carved_wooden_elephant_sculpture/real_pilot_blocky_to_rounded/target_template_manifest.json`

Implementation commit SHA: `ba20e5ed518dc09bd83fd949e36fd6155bfe98a9`.
This handoff metadata is finalized by the follow-up metadata commit reported
by cdx; the implementation commit above contains the research changes.
