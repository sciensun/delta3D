# ChatGPT Handoff: Image-First Delta3D Refactor

## Status

**PARTIAL**. The architecture and schemas are refactored and CPU-validated.
Real generated target-image observations have not yet passed the reliability
gate, so there is no real stable style delta and Stage 2 remains paused.

## Objective

The main method is stylized GLB -> canonical stylized 3DGS/source views ->
generated target/content multi-view images -> confident observed-2D
source-to-target observations -> fixed-bank free delta -> view/repeat/structure
reliability -> stable style delta -> future source-conditioned multiscale
learning.

An independently generated content 3D is optional only. It is not a required
stage and does not provide inherent Gaussian correspondence.

## Important Changes

- Rewrote `README.md` as the Delta3D project entry point.
- Preserved the upstream README at `docs/UPSTREAM_README.md`.
- Added `style_data.StyleTaskRecord` plus JSON/JSONL manifest helpers.
- Refactored `correspondence/schema.py` around image-first `ObservationBundle`.
- Added explicit observation modes: `observed_2d`, `oracle_3d`, `hybrid`.
- `observed_2d` does not require `target_xyz` and rejects missing `target_xy`.
- Retained `CorrespondenceBundle` as a compatibility alias/loader.
- Split Stage 1 utilities into `stage1/config.py`, `objectives.py`,
  `regularizers.py`, `outputs.py`, and `miner.py` while preserving the
  `train_delta_mining.py` entry point.
- Added Stage 1.5 `StableStyleDelta` and confidence-composition interfaces.
- Added Stage 2 contracts only for coarse/part/detail components, style family,
  intensity, part masks, and local-frame metadata.
- Updated `PROJECT_METHOD.md`, `NEXT_STEPS.md`, and `EXPERIMENT_LOG.md`.

## Validation

CPU validation passed for StyleTaskRecord validation and JSON serialization,
observed-2D bundles without target xyz, oracle-3D and hybrid bundle validation,
legacy correspondence loading, empty valid masks, camera-name indexing,
informative failure when observed-2D lacks target xy, Stage 1 delta
serialization, StableStyleDelta serialization, exact zero background delta,
zero `d_scaling`, and synthetic canonical export/PLY reload where available.

The existing synthetic GPU benchmark remains the latest validated recovery:
cosine `0.9614`, energy ratio `0.8835`, explained variance `0.9233`, background
energy `0`, and `d_scaling=0`.

The historical weak-target split result remains failed: weighted cosine
`0.1176`, median per-Gaussian cosine `0.0749`, and directional conflict
`46.5%`.

## Limitations

- No real generated target-image observation bundle has been built and accepted.
- No real repeat-generation consensus has been measured.
- No stable style delta has been produced.
- Stage 2 has not been trained.
- `pytest` is unavailable in the current environment; direct Python CPU tests
  were used.
- `train.py` was not modified.

## Research Interpretation

The old free delta is not automatically a style delta. It may contain style,
generation randomness, matching error, and optimization residual. A stable
style delta requires agreement across views, repeats, and structurally related
regions, plus association with a style operation and intensity.

## Decision

Do not increase part count, force energy normalization, or start Stage 2. The
current implementation is ready for real observed-2D bundle construction, but
the research result is not yet a real style-transfer success.

## Next Experiment

Generate repeated target/content images from the stylized object with strict
camera, identity, pose, and topology preservation. Build source-indexed
`observed_2d` bundles with foreground filtering, mutual matching, semantic
constraints, visibility, and confidence. Run split/repeat/identity/intensity
reliability analysis before promoting any delta.

## Artifacts

`README.md`, `docs/UPSTREAM_README.md`, `style_data/`, `correspondence/`,
`stage1/`, `stage2/`, `scene/canonical_ordinary_model.py`, and
`output/elephant_source_graphdeco/synthetic_known_delta/`.

## Git

Commit SHA: `f6a4240d8b5cfea0a5979b25223f24f44a9bbca2`.
Push status: pushed to `delta3D/main`.
