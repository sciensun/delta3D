# Style-Conditioned Deformation Demo

This demo adds a minimal Scheme C training path in `train_style.py`. The original Deformable 3DGS scene is kept as the canonical 3DGS input, while a separate folder of stylized target views provides style supervision.

## Data Format

The original and stylized folders must share the same camera/view names:

```text
data/style_demo/chair001/
  original/
    transforms_train.json
    transforms_test.json
    train/r_000.png
    train/r_001.png
  stylized/
    round/
      alpha_1.0/
        train/r_000.png
        train/r_001.png
```

`train_style.py` looks up each target by `viewpoint_cam.image_name` in:

```text
<style_target_path>/train/
<style_target_path>/
<style_target_path>/images/
```

Common image extensions `.png`, `.jpg`, and `.jpeg` are supported.

## Weak Target Note

Generated standard/stylized images must be treated as weak references. They should be filtered by silhouette consistency and semantic consistency before training. Pixel-aligned L1/SSIM is only safe when target views are rendered from a true 3D paired model using the same cameras.

By default, `--weak_target true` avoids RGB L1/SSIM as the main supervision. It uses LPIPS when available, source-render mask consistency, deformation regularization, and identity regularization. A low-weight RGB term can be enabled with `--use_rgb_weak`.

## Sample Command

```bash
python train_style.py \
  -s data/style_demo/chair001/original \
  --style_target_path data/style_demo/chair001/stylized/round/alpha_1.0 \
  --style_name round \
  --style_names round \
  --style_alpha 1.0 \
  --model_path output/chair001_round_style \
  --iterations 12000 \
  --warm_up 1000 \
  --lambda_dssim 0.2 \
  --lambda_style 1.0 \
  --lambda_delta 0.001 \
  --lambda_id 0.1 \
  --max_d_xyz 0.03 \
  --max_d_scaling 0.08
```

Use `--freeze_gaussians` to train only the style deformation head and keep the canonical Gaussians fixed. Rotation deformation is disabled by default; enable it only for experiments with `--enable_style_rotation`.
