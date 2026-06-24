#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/shichang/Deformable-3D-Gaussians}
OBJECT_ID=${OBJECT_ID:-big_carved_wooden_elephant_sculpture}
INPUT_GLB=${INPUT_GLB:-"${REPO_ROOT}/assets/3D/${OBJECT_ID}.glb"}
DATASET_DIR=${DATASET_DIR:-"${REPO_ROOT}/assets/prepared/${OBJECT_ID}/blender_perspective_dataset"}
SOURCE_MODEL=${SOURCE_MODEL:-"${REPO_ROOT}/output/elephant_source_perspective"}
TARGET_ROOT=${TARGET_ROOT:-"${REPO_ROOT}/assets/prepared/${OBJECT_ID}/generated_standard/key8_manual"}
RESOLUTION=${RESOLUTION:-1024}
NUM_AZIMUTH=${NUM_AZIMUTH:-72}
ELEVATIONS=${ELEVATIONS:-"-20,0,20"}
FOV_DEGREES=${FOV_DEGREES:-35}
SOURCE_ITERS=${SOURCE_ITERS:-30000}
DELTA_ITERS=${DELTA_ITERS:-3000}
BLENDER_BIN=${BLENDER_BIN:-blender}
TRAIN_RESOLUTION=${TRAIN_RESOLUTION:-1}
DENSIFY_UNTIL=${DENSIFY_UNTIL:-15000}
PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE:-max_split_size_mb:128}

RUN_RENDER=1
RUN_SOURCE_TRAIN=0
RUN_QUALITY=0
RUN_DELTA=0
RUN_DELTA_DEBUG=0

usage() {
  cat <<EOF
Usage:
  bash scripts/run_perspective_source_pipeline.sh [options]

Default:
  Render the perspective dataset only, then print next-step commands.

Options:
  --all                 Run render, source training, quality gate, xyz-only Stage 1, and delta debug.
  --render-only         Render dataset only.
  --train-source        Also run source 3DGS training.
  --quality             Run source quality gate.
  --delta               Run xyz-only Stage 1 delta mining.
  --delta-debug         Render amplified xyz-only delta.
  --skip-render         Do not render dataset.
  --low-vram            Use safer VRAM settings: train --resolution 2, densify_until_iter 7000, allocator split 128MB.
  --object_id NAME      Default: ${OBJECT_ID}
  --input_glb PATH      Default: ${INPUT_GLB}
  --dataset_dir PATH    Default: ${DATASET_DIR}
  --source_model PATH   Default: ${SOURCE_MODEL}
  --target_root PATH    Default: ${TARGET_ROOT}
  --num_azimuth N       Default: ${NUM_AZIMUTH}
  --elevations CSV      Default: ${ELEVATIONS}
  --resolution N        Default: ${RESOLUTION}
  --train_resolution N  Passed to train.py --resolution. Use 2 or 4 to lower VRAM. Default: ${TRAIN_RESOLUTION}
  --densify_until N     Passed to train.py --densify_until_iter. Default: ${DENSIFY_UNTIL}
  --source_iters N      Default: ${SOURCE_ITERS}
  --delta_iters N       Default: ${DELTA_ITERS}
  --blender PATH        Default: ${BLENDER_BIN}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) RUN_SOURCE_TRAIN=1; RUN_QUALITY=1; RUN_DELTA=1; RUN_DELTA_DEBUG=1; shift ;;
    --render-only) RUN_RENDER=1; RUN_SOURCE_TRAIN=0; RUN_QUALITY=0; RUN_DELTA=0; RUN_DELTA_DEBUG=0; shift ;;
    --train-source) RUN_SOURCE_TRAIN=1; shift ;;
    --quality) RUN_QUALITY=1; shift ;;
    --delta) RUN_DELTA=1; shift ;;
    --delta-debug) RUN_DELTA_DEBUG=1; shift ;;
    --skip-render) RUN_RENDER=0; shift ;;
    --low-vram) TRAIN_RESOLUTION=2; DENSIFY_UNTIL=7000; PYTORCH_CUDA_ALLOC_CONF_VALUE=max_split_size_mb:128; shift ;;
    --object_id) OBJECT_ID="$2"; INPUT_GLB="${REPO_ROOT}/assets/3D/${OBJECT_ID}.glb"; DATASET_DIR="${REPO_ROOT}/assets/prepared/${OBJECT_ID}/blender_perspective_dataset"; TARGET_ROOT="${REPO_ROOT}/assets/prepared/${OBJECT_ID}/generated_standard/key8_manual"; shift 2 ;;
    --input_glb) INPUT_GLB="$2"; shift 2 ;;
    --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
    --source_model) SOURCE_MODEL="$2"; shift 2 ;;
    --target_root) TARGET_ROOT="$2"; shift 2 ;;
    --num_azimuth) NUM_AZIMUTH="$2"; shift 2 ;;
    --elevations) ELEVATIONS="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --train_resolution) TRAIN_RESOLUTION="$2"; shift 2 ;;
    --densify_until) DENSIFY_UNTIL="$2"; shift 2 ;;
    --source_iters) SOURCE_ITERS="$2"; shift 2 ;;
    --delta_iters) DELTA_ITERS="$2"; shift 2 ;;
    --blender) BLENDER_BIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cd "${REPO_ROOT}"

if [[ ! -f "${INPUT_GLB}" ]]; then
  echo "Missing GLB: ${INPUT_GLB}" >&2
  exit 1
fi

if ! command -v "${BLENDER_BIN}" >/dev/null 2>&1; then
  if [[ -x "${HOME}/software/blender-4.3.2-linux-x64/blender" ]]; then
    BLENDER_BIN="${HOME}/software/blender-4.3.2-linux-x64/blender"
  else
    echo "Missing Blender. Set --blender /path/to/blender or add blender to PATH." >&2
    exit 1
  fi
fi

echo "Repo:          ${REPO_ROOT}"
echo "Object:        ${OBJECT_ID}"
echo "Input GLB:     ${INPUT_GLB}"
echo "Dataset:       ${DATASET_DIR}"
echo "Source model:  ${SOURCE_MODEL}"
echo "Target root:   ${TARGET_ROOT}"
echo "Blender:       ${BLENDER_BIN}"
echo "Train res:     ${TRAIN_RESOLUTION}"
echo "Densify until: ${DENSIFY_UNTIL}"
echo "CUDA alloc:    ${PYTORCH_CUDA_ALLOC_CONF_VALUE}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF_VALUE}}"

if [[ "${RUN_RENDER}" -eq 1 ]]; then
  if [[ -f "${DATASET_DIR}/transforms_train.json" ]]; then
    echo "[render] Dataset exists, skipping render: ${DATASET_DIR}"
  else
    echo "[render] Creating perspective dataset"
    "${BLENDER_BIN}" -b --python scripts/render_glb_perspective_dataset.py -- \
      --input_glb "${INPUT_GLB}" \
      --object_id "${OBJECT_ID}" \
      --out_dir "${DATASET_DIR}" \
      --resolution "${RESOLUTION}" \
      --num_azimuth "${NUM_AZIMUTH}" \
      --elevations "${ELEVATIONS}" \
      --fov_degrees "${FOV_DEGREES}" \
      --white_background \
      --transparent_background
  fi
fi

SOURCE_TRAIN_CMD=(
  python train.py
  -s "${DATASET_DIR}"
  --model_path "${SOURCE_MODEL}"
  --iterations "${SOURCE_ITERS}"
  --warm_up 0
  --eval
  --is_blender
  --white_background
  --resolution "${TRAIN_RESOLUTION}"
  --densify_until_iter "${DENSIFY_UNTIL}"
)

if [[ "${RUN_SOURCE_TRAIN}" -eq 1 ]]; then
  echo "[train-source] Running source 3DGS training"
  "${SOURCE_TRAIN_CMD[@]}"
else
  printf '\nNext source training command:\n'
  printf ' %q' "${SOURCE_TRAIN_CMD[@]}"
  printf '\n'
fi

QUALITY_CMD=(
  python scripts/source_quality_gate.py
  -s "${DATASET_DIR}"
  --model_path "${SOURCE_MODEL}"
  --original_render_root "${DATASET_DIR}/images"
  --out_dir "${SOURCE_MODEL}/debug_quality_gate"
  --max_views 12
)

if [[ "${RUN_QUALITY}" -eq 1 ]]; then
  echo "[quality] Running source quality gate"
  "${QUALITY_CMD[@]}"
else
  printf '\nNext quality gate command:\n'
  printf ' %q' "${QUALITY_CMD[@]}"
  printf '\n'
fi

DELTA_PATH="${SOURCE_MODEL}/mined_delta_xyz_only.pt"
DELTA_CMD=(
  python train_delta_mining.py
  -s "${DATASET_DIR}"
  --model_path "${SOURCE_MODEL}"
  --target_image_root "${TARGET_ROOT}"
  --iterations "${DELTA_ITERS}"
  --max_d_xyz 0.08
  --max_d_scaling 0.0
  --disable_d_scaling
  --lambda_lpips 1.0
  --lambda_rgb_weak 0.05
  --lambda_mask 0.05
  --lambda_delta 0.0005
  --lambda_smooth 0.005
  --save_delta_path "${DELTA_PATH}"
)

if [[ "${RUN_DELTA}" -eq 1 ]]; then
  echo "[delta] Running xyz-only Stage 1"
  "${DELTA_CMD[@]}"
else
  printf '\nNext xyz-only Stage 1 command:\n'
  printf ' %q' "${DELTA_CMD[@]}"
  printf '\n'
fi

DELTA_DEBUG_CMD=(
  python scripts/debug_render_mined_delta_amplified.py
  -s "${DATASET_DIR}"
  --model_path "${SOURCE_MODEL}"
  --mined_delta_path "${DELTA_PATH}"
  --out_dir "${SOURCE_MODEL}/debug_mined_delta_xyz_only_amplified"
  --amplify 1 2 5 10
  --max_views 8
)

if [[ "${RUN_DELTA_DEBUG}" -eq 1 ]]; then
  echo "[delta-debug] Rendering amplified xyz-only delta"
  "${DELTA_DEBUG_CMD[@]}"
else
  printf '\nNext delta debug command:\n'
  printf ' %q' "${DELTA_DEBUG_CMD[@]}"
  printf '\n'
fi
