#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/shichang/Deformable-3D-Gaussians}
OBJECT_ID=${OBJECT_ID:-big_carved_wooden_elephant_sculpture}
INPUT_GLB=${INPUT_GLB:-"${REPO_ROOT}/assets/3D/${OBJECT_ID}.glb"}
DATASET_DIR=${DATASET_DIR:-"${REPO_ROOT}/assets/prepared/${OBJECT_ID}/blender_perspective_dataset"}
MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/output/elephant_source_perspective"}
RESOLUTION=${RESOLUTION:-1024}
NUM_AZIMUTH=${NUM_AZIMUTH:-72}
ELEVATIONS=${ELEVATIONS:-"-20,0,20"}
FOV_DEGREES=${FOV_DEGREES:-35}
ITERATIONS=${ITERATIONS:-30000}
TRAIN_RESOLUTION=${TRAIN_RESOLUTION:-2}
DENSIFY_UNTIL=${DENSIFY_UNTIL:-7000}
RUN_TRAIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --object_id) OBJECT_ID="$2"; INPUT_GLB="${REPO_ROOT}/assets/3D/${OBJECT_ID}.glb"; DATASET_DIR="${REPO_ROOT}/assets/prepared/${OBJECT_ID}/blender_perspective_dataset"; shift 2 ;;
    --input_glb) INPUT_GLB="$2"; shift 2 ;;
    --out_dir|--dataset_dir) DATASET_DIR="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --num_azimuth) NUM_AZIMUTH="$2"; shift 2 ;;
    --elevations) ELEVATIONS="$2"; shift 2 ;;
    --fov_degrees) FOV_DEGREES="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --train_resolution) TRAIN_RESOLUTION="$2"; shift 2 ;;
    --densify_until) DENSIFY_UNTIL="$2"; shift 2 ;;
    --run-train) RUN_TRAIN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "${REPO_ROOT}"

if [[ ! -f "${INPUT_GLB}" ]]; then
  echo "Missing GLB: ${INPUT_GLB}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_DIR}/transforms_train.json" ]]; then
  echo "[1/2] Rendering perspective dataset"
  blender -b --python scripts/render_glb_perspective_dataset.py -- \
    --input_glb "${INPUT_GLB}" \
    --object_id "${OBJECT_ID}" \
    --out_dir "${DATASET_DIR}" \
    --resolution "${RESOLUTION}" \
    --num_azimuth "${NUM_AZIMUTH}" \
    --elevations "${ELEVATIONS}" \
    --fov_degrees "${FOV_DEGREES}" \
    --white_background \
    --transparent_background
else
  echo "[1/2] Perspective dataset already exists: ${DATASET_DIR}"
fi

cat <<EOF

[2/2] Train source 3DGS with:

python train.py \\
  -s ${DATASET_DIR} \\
  --model_path ${MODEL_PATH} \\
  --iterations ${ITERATIONS} \\
  --warm_up 0 \\
  --eval \\
  --is_blender \\
  --white_background \\
  --resolution ${TRAIN_RESOLUTION} \\
  --densify_until_iter ${DENSIFY_UNTIL}

EOF

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  echo "Starting source 3DGS training because --run-train was set."
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
  python train.py \
    -s "${DATASET_DIR}" \
    --model_path "${MODEL_PATH}" \
    --iterations "${ITERATIONS}" \
    --warm_up 0 \
    --eval \
    --is_blender \
    --white_background \
    --resolution "${TRAIN_RESOLUTION}" \
    --densify_until_iter "${DENSIFY_UNTIL}"
else
  echo "Not starting training. Re-run with --run-train to execute the command."
fi
