# ChatGPT Handoff: Image-First Delta3D Refactor

## Status

**PARTIAL**. The oracle observed-2D optimizer benchmark passes, but the first
image-derived matcher baseline fails the body observation gate. No real stable
style delta exists and Stage 2 remains paused.

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

The clean-8 observed-2D-only run reached global cosine `0.9999997`, active
cosine `1.0`, energy ratio `0.9997`, explained variance `0.9999995`, zero
background energy, exact-zero scaling, and `0.0013 px` novel-view projection
RMSE over 216 cameras. The 1/2/4/8-view cosine trend was
`0.7512/0.99997/1.0/1.0`. Eight-view robustness remained strong at global
cosine `0.9988` for 0.5 px noise, `0.9770` for 2 px noise, `0.9975/0.9909`
for 5%/10% outliers, and `0.9981` with 70% visibility. A/B active-region
cosine was `1.0`; full-foreground weighted cosine was `0.99998` with 29.7%
conflict from inactive near-zero entries.

The image-derived Farneback run produced body clean-8 median EPE `3.691 px`,
PCK@5 `.539`, coverage `.356`; ear PCK@5 `.822`, coverage `.344`; trunk PCK@5
`.801`, coverage `.333`. Stage 1 recovery was not run because no NVIDIA
driver was available. The historical target diagnostic produced support
coverage `.4675` but is diagnostic-only and has no GT or repeat validation.

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

Improve image-derived matching first: test a local learned dense-feature/flow
backend or stronger visibility/mask model until the body synthetic observation
gate passes, then run GPU Stage 1 and A/B recovery. Only after that generate
three real repeated target sets from the prepared pilot manifest.

This iteration corrected support denominators and added DIS plus signed-distance
silhouette observations. Corrected Farneback active PCK@5 was body `.193`, ear
`.091`, trunk `.478`; DIS body `.606`. CPU active delta cosine was `.057`
point-only, `.041` silhouette-only, `.030` hybrid. No CUDA driver was present,
so differentiable Stage 1 was not run.

## Artifacts

`README.md`, `docs/UPSTREAM_README.md`, `style_data/`, `correspondence/`,
`stage1/`, `stage2/`, `scene/canonical_ordinary_model.py`, and
`output/elephant_source_graphdeco/synthetic_known_delta/`.

## Git

Commit SHA: see `git rev-parse HEAD` for this final commit.
Push status: pushed to `delta3D/main`.
