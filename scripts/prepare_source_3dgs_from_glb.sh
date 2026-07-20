#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/shichang/Deformable-3D-Gaussians}
OBJECT_ID=${OBJECT_ID:-big_carved_wooden_elephant_sculpture}
INPUT_GLB=${INPUT_GLB:-"${REPO_ROOT}/assets/3D/${OBJECT_ID}.glb"}
PREPARED_ROOT=${PREPARED_ROOT:-"${REPO_ROOT}/assets/prepared/${OBJECT_ID}"}
RESOLUTION=${RESOLUTION:-1024}
RENDER_MODE=${RENDER_MODE:-full36}
SOURCE_MODEL_PATH=${SOURCE_MODEL_PATH:-"${REPO_ROOT}/output/${OBJECT_ID}_source"}
RUN_TRAIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --object_id)
      OBJECT_ID="$2"
      INPUT_GLB="${REPO_ROOT}/assets/3D/${OBJECT_ID}.glb"
      PREPARED_ROOT="${REPO_ROOT}/assets/prepared/${OBJECT_ID}"
      SOURCE_MODEL_PATH="${REPO_ROOT}/output/${OBJECT_ID}_source"
      shift 2
      ;;
    --input_glb)
      INPUT_GLB="$2"
      OBJECT_ID=$(basename "${INPUT_GLB}")
      OBJECT_ID="${OBJECT_ID%.*}"
      PREPARED_ROOT="${REPO_ROOT}/assets/prepared/${OBJECT_ID}"
      SOURCE_MODEL_PATH="${REPO_ROOT}/output/${OBJECT_ID}_source"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --render_mode)
      RENDER_MODE="$2"
      shift 2
      ;;
    --model_path)
      SOURCE_MODEL_PATH="$2"
      shift 2
      ;;
    --run-train)
      RUN_TRAIN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

if [[ ! -f "${INPUT_GLB}" ]]; then
  echo "Missing input GLB: ${INPUT_GLB}" >&2
  exit 1
fi

if command -v blender >/dev/null 2>&1; then
  BLENDER_BIN=$(command -v blender)
elif [[ -x "${HOME}/software/blender-4.3.2-linux-x64/blender" ]]; then
  BLENDER_BIN="${HOME}/software/blender-4.3.2-linux-x64/blender"
else
  echo "Missing dependency: blender is not available on PATH." >&2
  exit 1
fi

RENDER_DIR="${PREPARED_ROOT}/renders_original/${RENDER_MODE}"
DATASET_DIR="${PREPARED_ROOT}/blender_${RENDER_MODE}_dataset"
LOG_DIR="${PREPARED_ROOT}/logs"
mkdir -p "${RENDER_DIR}" "${DATASET_DIR}" "${LOG_DIR}"

echo "[1/2] Rendering ${RENDER_MODE} views from GLB"
"${BLENDER_BIN}" -b --python scripts/render_glb_views.py -- \
  --input_glb "${INPUT_GLB}" \
  --object_id "${OBJECT_ID}" \
  --mode "${RENDER_MODE}" \
  --out_dir "${RENDER_DIR}" \
  --resolution "${RESOLUTION}" \
  > "${LOG_DIR}/stage0_render_${RENDER_MODE}.log" 2>&1

echo "[2/2] Exporting approximate Blender/NeRF dataset"
python scripts/export_key8_to_blender_dataset.py \
  --views_meta "${RENDER_DIR}/views_meta.json" \
  --out_dir "${DATASET_DIR}" \
  > "${LOG_DIR}/stage0_export_dataset.log" 2>&1

cat <<EOF

Stage 0 dataset prepared.

GLB source:
  ${INPUT_GLB}

Rendered views:
  ${RENDER_DIR}

Approximate Blender dataset:
  ${DATASET_DIR}

Train source 3DGS with:

python train.py \\
  -s ${DATASET_DIR} \\
  --model_path ${SOURCE_MODEL_PATH} \\
  --iterations 7000 \\
  --warm_up 0 \\
  --eval \\
  --is_blender

After training, Stage 1 can load:
  ${SOURCE_MODEL_PATH}/point_cloud/iteration_<N>/point_cloud.ply
EOF

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  echo
  echo "[optional] Starting source 3DGS training because --run-train was set"
  python train.py \
    -s "${DATASET_DIR}" \
    --model_path "${SOURCE_MODEL_PATH}" \
    --iterations 7000 \
    --warm_up 0 \
    --eval \
    --is_blender
fi
