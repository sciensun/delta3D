#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shichang/Deformable-3D-Gaussians}"
OBJECT_ID="${OBJECT_ID:-big_carved_wooden_elephant_sculpture}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/assets/prepared/${OBJECT_ID}/blender_perspective_dataset_transparent}"
EXPECTED_OUT="${EXPECTED_OUT:-${REPO_ROOT}/output/${OBJECT_ID}_external_source}"

usage() {
  cat <<EOF
Usage: bash scripts/prepare_gsplat_source_experiment.sh [options]

This script does not install or run gsplat/Nerfstudio. It only prints the
paths and hand-off instructions for trying an external 3DGS source builder.

Options:
  --dataset_dir PATH   Blender/NeRF-style perspective dataset.
  --expected_out PATH  Suggested external builder output location.
  --object_id NAME     Object id. Default: ${OBJECT_ID}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
    --expected_out) EXPECTED_OUT="$2"; shift 2 ;;
    --object_id) OBJECT_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "${DATASET_DIR}/transforms_train.json" ]]; then
  echo "Missing dataset: ${DATASET_DIR}/transforms_train.json" >&2
  echo "Create it first with scripts/render_glb_perspective_dataset.py or scripts/run_perspective_source_pipeline.sh." >&2
  exit 1
fi

cat <<EOF
External source-builder experiment plan
======================================

Dataset:
  ${DATASET_DIR}

Suggested output location:
  ${EXPECTED_OUT}

This repository's Stage 1/Stage 2 code should not depend on which source
builder produced the Gaussian bank. It only needs a compatible Gaussian set:
  - xyz
  - scaling
  - rotation
  - opacity
  - SH/color
  - the same camera set

Possible external builders:
  A. vanilla Graphdeco 3DGS
  B. gsplat
  C. Nerfstudio Splatfacto

For Nerfstudio/Splatfacto, use this dataset as a Blender/NeRF-style image
dataset if your local Nerfstudio version supports the transform format.
Then export the trained splat as a Gaussian PLY or another documented format.

After external training, convert or adapt the result back to this repository's
GaussianModel-compatible PLY format. A placeholder adapter exists at:
  scripts/convert_external_gaussian_to_local.py

No training has been started by this script.
EOF
