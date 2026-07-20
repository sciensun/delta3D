# Source 3DGS Builders

The two-stage delta pipeline is 3DGS-only, but the source stylized 3DGS does
not have to be produced by one hard-coded trainer.

## Required Source Interface

Stage 1 delta mining and later distillation require a Gaussian bank and the
matching cameras:

```text
xyz
scaling
rotation
opacity
SH/color
camera set
```

The current implementation loads these through this repository's
`Scene`/`GaussianModel` mechanism, so the most direct format is the local
GaussianModel-compatible PLY under:

```text
output/<model_path>/point_cloud/iteration_<N>/point_cloud.ply
```

## Builder A: Current Deformable-3D-Gaussians `train.py`

This is the default path. Render the GLB into a perspective Blender/NeRF-style
dataset, then train with `train.py`.

Use the source quality gate before any delta mining:

```bash
python scripts/source_quality_gate.py \
  -s assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent \
  --model_path output/elephant_source_perspective \
  --original_render_root assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent/images \
  --out_dir output/elephant_source_perspective/debug_quality_gate \
  --max_views 12 \
  --white_background
```

## Builder B: Vanilla Graphdeco 3DGS

Vanilla Graphdeco 3DGS can be used as an external source builder if it trains a
sharper canonical Gaussian bank from the same rendered perspective dataset.
The output must be converted or arranged into this repository's local
GaussianModel-compatible PLY layout before Stage 1 can consume it.

## Builder C: gsplat / Nerfstudio Splatfacto

gsplat or Nerfstudio Splatfacto can also be tested as external source builders.
The prepared perspective dataset is a Blender/NeRF-style image dataset with
`transforms_train.json`, `transforms_test.json`, and `images/`.

This repository does not install or run these systems automatically. Use:

```bash
bash scripts/prepare_gsplat_source_experiment.sh
```

to print the relevant paths and hand-off instructions.

## Conversion Boundary

The placeholder adapter is:

```bash
python scripts/convert_external_gaussian_to_local.py \
  --input path/to/external/output \
  --output output/converted_source/point_cloud/iteration_30000/point_cloud.ply \
  --source_format unknown
```

It currently exits with a TODO message. Do not fake conversion: external
builders use different field names and parameterizations for scaling,
rotation, opacity, and SH/color.

## Rule

Do not run Stage 1 delta mining until the chosen source builder passes visual
and numeric quality checks. Stage 1 cannot recover details that are already
lost in the canonical source 3DGS.
