# Two-Stage 3DGS Delta Pipeline

This is a 3DGS-only deformation demo. It does not introduce mesh reconstruction and does not use Tripo.

The first demo direction is:

```text
source 3DGS: stylized elephant sculpture
weak targets: less-stylized / more standard ChatGPT images
direction: stylized_to_standard
```

ChatGPT-generated targets are weak references, not pixel-aligned ground truth. The training scripts therefore avoid strong RGB L1/SSIM as the main supervision.

## Stage 0: Source 3DGS

The source assets live in:

```text
assets/3D/*.glb
```

A GLB is not a 3DGS checkpoint and cannot be loaded directly by `train_delta_mining.py`. First render the GLB, export an approximate Blender/NeRF-style image dataset, and train a source stylized 3DGS with the original repository pipeline.

For the elephant:

```bash
bash scripts/prepare_source_3dgs_from_glb.sh \
  --object_id big_carved_wooden_elephant_sculpture \
  --render_mode full36 \
  --resolution 1024
```

This prepares:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset/
```

Then train the source 3DGS:

```bash
python train.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset \
  --model_path output/elephant_source \
  --iterations 7000 \
  --warm_up 0 \
  --eval \
  --is_blender
```

After source training, the source model should be available under `--model_path`, for example:

```text
output/elephant_source/point_cloud/iteration_<N>/point_cloud.ply
```

If you already have rendered key8/full36 images and only need to export the approximate Blender-style dataset, use:

```bash
python scripts/export_key8_to_blender_dataset.py \
  --views_meta assets/prepared/big_carved_wooden_elephant_sculpture/renders_original/full36/views_meta.json \
  --out_dir assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset
```

Warning: the rendered views are orthographic, while this repo's Blender loader expects perspective `camera_angle_x`. The helper writes an approximate perspective dataset only.

Manual ChatGPT targets should be placed in:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual/
```

Supported target names include:

```text
01_standard.png
02_standard.png
...
08_standard.png
```

and names matching the source key8 view names.

## Stage 1: Free Delta Mining

Stage 1 learns per-Gaussian free deformation parameters for one fixed source 3DGS. This mines a pseudo-delta from weak target images; it is not a general model.

Example:

```bash
python train_delta_mining.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset \
  --model_path output/elephant_source \
  --target_image_root assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual \
  --object_id big_carved_wooden_elephant_sculpture \
  --direction stylized_to_standard \
  --iterations 3000 \
  --max_d_xyz 0.03 \
  --max_d_scaling 0.08 \
  --lambda_lpips 1.0 \
  --lambda_rgb_weak 0.02 \
  --lambda_mask 0.1 \
  --lambda_delta 0.01 \
  --lambda_smooth 0.05 \
  --save_delta_path output/elephant_source/mined_delta_latest.pt \
  --freeze_gaussians
```

Outputs:

```text
output/elephant_source/mined_delta_latest.pt
output/elephant_source/mined_delta_iter_<iteration>.pt
```

The delta file stores `d_xyz`, `d_scaling`, optional `d_rotation`, source Gaussian positions, source scaling when available, and metadata.

## Fixing Blurry Source 3DGS

If the GLB render is sharp but the source 3DGS render is blurry, do not do Stage 1 or Stage 2 yet. Stage 1 cannot mine a meaningful geometric delta if the canonical source 3DGS has already lost the geometry.

The previous orthographic-to-perspective approximate export can cause camera inconsistency and blurry source reconstruction. Prefer a true perspective Blender render where the rendered images and `transforms_train.json` use the exact same camera matrices.

Use more views than key8/full36 when needed. Good starting points are 108 views (`36 x 3 elevations`) or 216 views (`72 x 3 elevations`). Train the source 3DGS longer, for example 30000 iterations, then re-run GLB-vs-source comparison. Only when the source is sharp should you rerun xyz-only delta mining. Do not distill the old blur-by-scale delta.

Render a true perspective dataset:

```bash
python scripts/render_glb_perspective_dataset.py \
  --input_glb assets/3D/big_carved_wooden_elephant_sculpture.glb \
  --object_id big_carved_wooden_elephant_sculpture \
  --out_dir assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset \
  --resolution 1024 \
  --num_azimuth 72 \
  --elevations "-20,0,20" \
  --fov_degrees 35 \
  --white_background
```

Train source 3DGS from the perspective dataset:

```bash
python train.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset \
  --model_path output/elephant_source_perspective \
  --iterations 30000 \
  --warm_up 0 \
  --eval \
  --is_blender \
  --white_background \
  --resolution 2 \
  --densify_until_iter 7000
```

For a single-entry low-VRAM run, use:

```bash
bash scripts/run_perspective_source_pipeline.sh --train-source --low-vram
```

If this still runs out of memory, retry with `--train_resolution 4`, reduce render resolution to `--resolution 768`, or reduce views with `--num_azimuth 36 --elevations "-15,0,15"`.

Run the quality gate:

```bash
python scripts/source_quality_gate.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset \
  --model_path output/elephant_source_perspective \
  --original_render_root assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset/images \
  --out_dir output/elephant_source_perspective/debug_quality_gate \
  --max_views 12
```

After source quality passes, rerun xyz-only Stage 1:

```bash
python train_delta_mining.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset \
  --model_path output/elephant_source_perspective \
  --target_image_root assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual \
  --iterations 3000 \
  --max_d_xyz 0.08 \
  --max_d_scaling 0.0 \
  --disable_d_scaling \
  --lambda_lpips 1.0 \
  --lambda_rgb_weak 0.05 \
  --lambda_mask 0.05 \
  --lambda_delta 0.0005 \
  --lambda_smooth 0.005 \
  --save_delta_path output/elephant_source_perspective/mined_delta_xyz_only.pt
```

## Single-View And Few-View Overfit Diagnostics

Before further delta work, isolate whether source blur comes from the
dataset/camera/background pipeline or from the full multi-view source training
setup.

Create tiny overfit subsets from the perspective dataset:

```bash
python scripts/make_overfit_subset.py \
  --dataset_root assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --out_dir assets/prepared/big_carved_wooden_elephant_sculpture/overfit_1view \
  --num_views 1 \
  --copy_images
```

```bash
python scripts/make_overfit_subset.py \
  --dataset_root assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --out_dir assets/prepared/big_carved_wooden_elephant_sculpture/overfit_3view \
  --num_views 3 \
  --copy_images
```

```bash
python scripts/make_overfit_subset.py \
  --dataset_root assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --out_dir assets/prepared/big_carved_wooden_elephant_sculpture/overfit_8view \
  --num_views 8 \
  --copy_images
```

Train the 1-view overfit:

```bash
python train.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/overfit_1view \
  --model_path output/elephant_overfit_1view_noprune \
  --iterations 5000 \
  --warm_up 0 \
  --eval \
  --is_blender \
  --white_background \
  --resolution 1 \
  --densify_until_iter 0
```

Run the quality gate against the same subset:

```bash
python scripts/source_quality_gate.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/overfit_1view \
  --model_path output/elephant_overfit_1view_noprune \
  --original_render_root assets/prepared/big_carved_wooden_elephant_sculpture/overfit_1view/images \
  --out_dir output/elephant_overfit_1view_noprune/debug_quality_gate \
  --max_views 1 \
  --white_background
```

For the 1-view test, `--densify_until_iter 0` disables densification and
pruning. This avoids a degenerate failure where single-view training prunes all
Gaussians and the rasterizer backward pass receives empty tensors. Repeat the
same pattern for `overfit_3view` and `overfit_8view`; if those runs also prune
to zero, use the same no-prune setting first, then re-enable densification once
the loader/camera path is verified.

Interpretation:

- If 1-view overfit is still blurry, there is likely a loader/render/background/training issue.
- If 1-view is sharp but the full dataset is blurry, there is likely camera/transform inconsistency across views.
- If 1-view and 3-view are sharp but 216-view is blurry, inspect camera distribution, camera transforms, train/test split, and mask/background consistency.
- If all overfit tests are sharp but full reconstruction is not, try a different source builder or training recipe before any delta mining.

## Acceptance Criteria

Do not proceed to Stage 1 unless the source passes:

- source render is visually sharp;
- elephant silhouette is aligned with the GLB render;
- trunk, ears, legs, carved body surface, and base are recognizable;
- foreground mask IoU is preferably greater than `0.75`;
- p95 scale / bbox diagonal is preferably less than `0.03`;
- no large halo or fog around the silhouette.

Do not proceed to Stage 2 unless:

- source passes the source quality gate;
- xyz-only delta x1 remains sharp;
- amplified xyz-only delta shows coherent geometric motion;
- there is no blur-by-scale behavior.

## Using Graphdeco As Source 3DGS Builder

Graphdeco official 3DGS can be used only to build a sharper canonical
`G_sty`. The later delta pipeline remains unchanged:

```text
G_sty -> G_sty + Delta* -> B(F, delta_z)
```

First adapt the Graphdeco output into this repository's model layout. This does
not modify the PLY contents; it only checks fields and copies or symlinks the
PLY.

```bash
python scripts/adapt_graphdeco_source_to_delta3d.py \
  --graphdeco_model_path /home/shichang/gs_outputs/elephant_graphdeco \
  --graphdeco_iteration 30000 \
  --delta3d_model_path output/elephant_source_graphdeco \
  --source_dataset_path assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --copy
```

Then verify that the adapted Graphdeco PLY is still sharp when loaded through
delta3D's `GaussianModel` and rendered by delta3D's renderer:

```bash
python scripts/verify_adapted_source_render.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --model_path output/elephant_source_graphdeco \
  --load_iteration 30000 \
  --original_render_root assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent/images \
  --out_dir output/elephant_source_graphdeco/debug_verify_loaded_render \
  --max_views 12 \
  --white_background
```

If Graphdeco native render is sharp but the delta3D-loaded render is blurry,
the adapter/loading path is the problem. If the delta3D-loaded render is sharp,
proceed to xyz-only delta mining. Background differences are acceptable when
the foreground is sharp and aligned.

Optional target background standardization:

```bash
python scripts/composite_dataset_to_white.py \
  --input_dir assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual \
  --out_dir assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual_white \
  --recursive
```

Run xyz-only Stage 1 from the adapted Graphdeco source:

```bash
python train_delta_mining.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --model_path output/elephant_source_graphdeco \
  --load_iteration 30000 \
  --target_image_root assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual_white \
  --iterations 3000 \
  --max_d_xyz 0.08 \
  --max_d_scaling 0.0 \
  --disable_d_scaling \
  --lambda_lpips 1.0 \
  --lambda_rgb_weak 0.05 \
  --lambda_mask 0.05 \
  --lambda_delta 0.0005 \
  --lambda_smooth 0.005 \
  --save_delta_path output/elephant_source_graphdeco/mined_delta_xyz_only.pt \
  --white_background \
  --composite_target_white \
  --foreground_loss
```

After Stage 1, inspect amplified xyz-only delta:

```bash
python scripts/debug_render_mined_delta_amplified.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --model_path output/elephant_source_graphdeco \
  --load_iteration 30000 \
  --mined_delta_path output/elephant_source_graphdeco/mined_delta_xyz_only.pt \
  --out_dir output/elephant_source_graphdeco/debug_mined_delta_xyz_only_amplified \
  --amplify 1 2 5 10 \
  --white_background
```

Proceed to Stage 2 only if the adapted source render is sharp through the
delta3D renderer, xyz-only delta x1 remains sharp, amplified xyz-only delta
shows coherent geometric motion, there is no blur-by-scale, and the target
trend is geometric rather than only background or appearance.

## Structured Stage 1 Delta

Once the Graphdeco source is accepted as canonical `G_sty`, Stage 1 should mine
a structured, reusable `Delta*`:

```text
G_sty -> Delta* -> B(F,z)
```

Build a foreground Gaussian mask:

```bash
DATA=assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent
MODEL=output/elephant_source_graphdeco
```

```bash
python scripts/build_foreground_gaussian_mask.py \
  -s "$DATA" \
  --model_path "$MODEL" \
  --load_iteration 30000 \
  --out_dir "$MODEL" \
  --threshold 0.5 \
  --mask_dilate 7 \
  --max_views 216
```

Cluster foreground Gaussians into part-like units:

```bash
python scripts/cluster_foreground_gaussians.py \
  --model_path "$MODEL" \
  --load_iteration 30000 \
  --foreground_mask_path "$MODEL/foreground_mask.pt" \
  --out_dir "$MODEL" \
  --num_parts 16
```

Run foreground-only xyz delta mining:

```bash
python train_delta_mining.py \
  -s "$DATA" \
  --model_path "$MODEL" \
  --load_iteration 30000 \
  --target_image_root assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual_white \
  --iterations 3000 \
  --max_d_xyz 0.08 \
  --max_d_scaling 0.0 \
  --disable_d_scaling \
  --foreground_mask_path "$MODEL/foreground_mask.pt" \
  --lambda_lpips 1.0 \
  --lambda_rgb_weak 0.05 \
  --lambda_mask 0.05 \
  --lambda_delta 0.0005 \
  --lambda_smooth 0.005 \
  --save_delta_path "$MODEL/mined_delta_foreground_xyz_only.pt" \
  --white_background \
  --composite_target_white \
  --foreground_loss
```

Run part-aware Stage 1 experimentation:

```bash
python train_delta_mining.py \
  -s "$DATA" \
  --model_path "$MODEL" \
  --load_iteration 30000 \
  --target_image_root assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual_white \
  --iterations 3000 \
  --max_d_xyz 0.08 \
  --max_d_scaling 0.0 \
  --disable_d_scaling \
  --foreground_mask_path "$MODEL/foreground_mask.pt" \
  --part_labels_path "$MODEL/part_labels.pt" \
  --num_parts 16 \
  --part_assignment_temperature 0.15 \
  --lambda_lpips 1.0 \
  --lambda_rgb_weak 0.05 \
  --lambda_mask 0.05 \
  --lambda_delta 0.0005 \
  --lambda_smooth 0.005 \
  --save_delta_path "$MODEL/mined_delta_partaware_xyz_only.pt" \
  --white_background \
  --composite_target_white \
  --foreground_loss
```

Evaluate delta quality:

```bash
python scripts/evaluate_delta_quality.py \
  --mined_delta_path "$MODEL/mined_delta_partaware_xyz_only.pt" \
  --foreground_mask_path "$MODEL/foreground_mask.pt" \
  --part_labels_path "$MODEL/part_labels.pt" \
  --out_json "$MODEL/delta_quality_partaware_xyz_only.json"
```

Render amplified delta:

```bash
python scripts/debug_render_mined_delta_amplified.py \
  -s "$DATA" \
  --model_path "$MODEL" \
  --load_iteration 30000 \
  --mined_delta_path "$MODEL/mined_delta_partaware_xyz_only.pt" \
  --out_dir "$MODEL/debug_partaware_delta_amplified" \
  --amplify 1 2 5 10 \
  --white_background
```

Stage 1 passes only if at least 90% of delta energy lies in foreground,
`d_scaling` is zero, x1/x2/x5 amplified renders remain coherent, movement is
concentrated on interpretable parts, and the source stays sharp.

## Stage 1 Debugging Order

1. First compare original GLB render vs source 3DGS render. If source is blurry, fix source 3DGS before delta mining.

```bash
python scripts/compare_glb_render_vs_source_3dgs.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset \
  --model_path output/elephant_source \
  --original_render_root assets/prepared/big_carved_wooden_elephant_sculpture/renders_original/full36 \
  --out_dir output/elephant_source/debug_glb_vs_source \
  --max_views 8
```

Also inspect the trained source 3DGS:

```bash
python scripts/inspect_source_3dgs_quality.py \
  --model_path output/elephant_source \
  --load_iteration -1
```

2. Run forced delta debugging. Radial expansion/shrink and scale expansion/shrink must visibly change the render.

```bash
python scripts/debug_render_forced_delta.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset \
  --model_path output/elephant_source \
  --out_dir output/elephant_source/debug_forced_delta \
  --force_dx 0.2 \
  --radial_strength 0.15 \
  --force_scale 0.1 \
  --max_views 8
```

Each side-by-side image is:

```text
source | translation | radial expansion | radial shrink | scale expansion | scale shrink
```

The script also saves each panel as a separate PNG.

3. Inspect mined delta statistics. If `d_scaling` dominates, the method is learning blur-by-scale.

```bash
python scripts/inspect_mined_delta.py \
  --mined_delta_path output/elephant_source/mined_delta_latest.pt
```

4. Render amplified mined delta. If amplification produces a blurred blob, do not distill it.

```bash
python scripts/debug_render_mined_delta_amplified.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset \
  --model_path output/elephant_source \
  --mined_delta_path output/elephant_source/mined_delta_latest.pt \
  --out_dir output/elephant_source/debug_mined_delta_amplified \
  --amplify 1 2 5 10
```

5. Re-run Stage 1 with xyz-only:

```bash
python train_delta_mining.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset \
  --model_path output/elephant_source \
  --target_image_root assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual \
  --iterations 1000 \
  --max_d_xyz 0.08 \
  --max_d_scaling 0.0 \
  --disable_d_scaling \
  --lambda_lpips 1.0 \
  --lambda_rgb_weak 0.05 \
  --lambda_mask 0.05 \
  --lambda_delta 0.0005 \
  --lambda_smooth 0.005 \
  --save_delta_path output/elephant_source/mined_delta_xyz_only.pt
```

6. Only proceed to Stage 2 if:

- source render is clear;
- delta x1 is still clear;
- amplified delta shows coherent geometric motion, not blur;
- target trend is geometric, not just appearance.

If amplified delta looks random, the weak target images are not geometrically aligned enough. If only colors/textures differ, geometry deformation alone is the wrong supervision; add an optional appearance branch later or generate more geometry-specific targets.

## Stage 2: Distill B(F, delta_z) -> Pseudo-Delta

Stage 2 trains a label-conditioned local geometry deformation model:

```text
B(F_i, delta_z) -> delta_i
```

`F_i` contains Fourier xyz features, Gaussian scale/opacity when available, local covariance eigenvalue features, local density, and curvature proxies. `delta_z` is loaded from:

```text
configs/style_labels_elephant.json
```

For the current demo, `delta_z = target_style - source_style` means `stylized_to_standard`. Use `--invert_delta_z` later for the opposite direction.

Example:

```bash
python train_style_distill.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_full36_dataset \
  --model_path output/elephant_source \
  --mined_delta_path output/elephant_source/mined_delta_latest.pt \
  --label_config configs/style_labels_elephant.json \
  --iterations 3000 \
  --distill_lr 1e-3 \
  --save_style_model_path output/elephant_source/label_style_deform_latest.pt \
  --render_interpolation
```

If `--render_interpolation` is set, alpha renders are saved to:

```text
output/elephant_source/style_interpolation/alpha_000/
output/elephant_source/style_interpolation/alpha_025/
output/elephant_source/style_interpolation/alpha_050/
output/elephant_source/style_interpolation/alpha_075/
output/elephant_source/style_interpolation/alpha_100/
```

## Notes

- This is intentionally a robust, minimal first demo.
- Stage 1 may overfit one object; generalization is not required yet.
- Rotation deformation is disabled by default.
- Keep source colors and opacity conceptually fixed for this demo.
- Do not interpret weak ChatGPT images as pixel-aligned supervision.
