# Next Steps

## Current Decision

Do not proceed to Stage 2. No fitted translation or part-local affine delta
currently satisfies the cosine, energy, and explained-variance acceptance gate.

## Immediate Validation

When a CUDA-visible Python environment is available, render the best current
candidate, K32 soft affine refined:

```bash
MODEL=output/elephant_source_graphdeco
DATA=assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent
DELTA=$MODEL/partfit_affine_soft_k32_refined/partfit_affine_xyz_only.pt
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
