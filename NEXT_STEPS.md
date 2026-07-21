# Next Steps

## Current Decision

Do not proceed to Stage 2. No fitted translation or part-local affine delta
currently satisfies the cosine, energy, and explained-variance acceptance gate.

## Immediate Next Step: Build Correspondence

Do not render the old K32 candidate, increase part count, or revisit the old
missing-azimuth issue as an active task. Those instructions are historical
because the split-view reliability gate already failed. The active task is to
obtain one real `G_ord_ref`, align it with sparse semantic anchors, construct
shared-view correspondence, and fit `G_ord_canon` on the fixed `G_sty` bank.

Use `scripts/align_ordinary_reference.py` for an independent reference model,
`scripts/fit_similarity_from_corresponded_points.py` for paired point sets,
the `correspondence/` package for source-indexed confidence bundles, and
`scripts/build_canonical_ordinary.py` plus `scripts/validate_canonical_pair.py`
for fixed-bank export checks. No real ordinary reference has yet been processed,
so no real target quality or canonical ordinary reliability is claimed.

## Updated Stage 1.5 Result

The missing camera was resolved by rebuilding an eight-view train-only subset
from the exact source frames. A/B mining completed, but the independent delta
signal failed:

- weighted cosine: `0.1176`
- median per-Gaussian cosine: `0.0749`
- magnitude Pearson: `0.6313`
- direction agreement: `30.3%`
- conflict fraction: `46.5%`

Do not build a consensus teacher or proceed to Stage 2. Regenerate targets with
stronger camera and geometry preservation, or add a correspondence/feature
based geometric objective before rerunning Stage 1. More part clusters and
forced energy normalization are not justified by this evidence.

## Corrected Research Decision

The synthetic benchmark separates two problems:

1. image-only Stage 1 is inadequate even for a known paired deformation;
2. correspondence-guided Stage 1 recovers the known xyz-only deformation and
   passes the provisional benchmark gate.

Therefore do not tune part count or start Stage 2. The next target experiment
must first produce one unified ordinary 3DGS, then run global similarity
alignment, editable semantic anchors, dense multi-view matching, and a
correspondence quality gate. Only accepted lifted target xyz/confidence data
should be passed to the new Stage 1 correspondence losses.

New Stage 1 options:

```bash
--correspondence_path path/to/lifted_correspondence.pt \
--lambda_corr_3d 10.0 \
--lambda_corr_2d 1.0 \
--max_d_scaling 0.0 \
--disable_d_scaling
```

```bash
python scripts/interpolate_part_delta.py \
  -s "$DATA" \
  --model_path "$MODEL" \
  --load_iteration 30000 \
  --part_delta_path "$DELTA" \
  --out_dir "$MODEL/debug_partfit_affine_soft_k32_refined_interpolation" \
  --alphas 0.0 0.25 0.5 0.75 1.0 \
  --max_views 8 \
  --white_background
```

```bash
python scripts/debug_render_mined_delta_amplified.py \
  -s "$DATA" \
  --model_path "$MODEL" \
  --load_iteration 30000 \
  --mined_delta_path "$DELTA" \
  --out_dir "$MODEL/debug_partfit_affine_soft_k32_refined_amplified" \
  --amplify 1 2 5 10 \
  --white_background
```

## Research Decision

If the rendered fitted delta is coherent but metrics remain below threshold,
review the part representation. The current K-means parts may not correspond to
semantic elephant parts; inspect `part_clusters_color.ply` and consider
semantic or spatial part labels for ears, trunk, body, and base. Do not increase
Stage 2 complexity or enable scaling until a compact Delta* passes the gate.

## Historical: Stage 1.5 Reliability Before Further Modeling

1. Regenerate the missing exact source view at elevation 0, azimuth 90, or
   explicitly remove target `03_standard.png` and document a 7-view split.
2. Activate the CUDA environment used for the original delta mining run.
3. Mine A and B independently into `mined_delta_fg_subset_A.pt` and
   `mined_delta_fg_subset_B.pt`; do not overwrite the full-view delta.
4. Run `compare_mined_delta_consistency.py`. Cosine below 0.50 or broad
   directional conflict indicates weak target alignment is the bottleneck.
5. Only for mixed/strong consistency, run consensus, graph denoising,
   motion-aware clustering, and the existing part fit.
6. Keep Stage 2 paused until the acceptance gate is met without forced energy
   normalization.

Example after CUDA and camera matching are fixed:

```bash
python train_delta_mining.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --model_path output/elephant_source_graphdeco \
  --load_iteration 30000 \
  --target_image_root output/elephant_source_graphdeco/stage1_5_target_splits/subset_A \
  --iterations 3000 \
  --max_d_xyz 0.08 \
  --max_d_scaling 0.0 \
  --disable_d_scaling \
  --lambda_lpips 1.0 \
  --lambda_rgb_weak 0.05 \
  --lambda_mask 0.05 \
  --lambda_delta 0.0005 \
  --lambda_smooth 0.005 \
  --foreground_mask_path output/elephant_source_graphdeco/foreground_mask.pt \
  --save_delta_path output/elephant_source_graphdeco/mined_delta_fg_subset_A.pt \
  --white_background \
  --composite_target_white \
  --foreground_loss
```

Repeat the same command with `subset_B` and
`mined_delta_fg_subset_B.pt`. Do not run these until the missing azimuth-090
camera is resolved or explicitly excluded.

```bash
python scripts/compare_mined_delta_consistency.py \
  --delta A=output/elephant_source_graphdeco/mined_delta_fg_subset_A.pt \
  B=output/elephant_source_graphdeco/mined_delta_fg_subset_B.pt \
  --foreground_mask_path output/elephant_source_graphdeco/foreground_mask.pt \
  --part_labels_path output/elephant_source_graphdeco/part_labels.pt \
  --output_dir output/elephant_source_graphdeco/stage1_5_consistency
```

The older missing-azimuth instructions above are historical; the exact
eight-view split has now been rebuilt and evaluated. The current blocker is
target correspondence quality, not camera matching.
