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
