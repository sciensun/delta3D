#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/shichang/Deformable-3D-Gaussians
OBJECT_ID=big_carved_wooden_elephant_sculpture
PREPARED_ROOT="${REPO_ROOT}/assets/prepared/${OBJECT_ID}"
TRIPO_GLB="${PREPARED_ROOT}/tripo_standard/source_standard.glb"
OUT_DIR="${PREPARED_ROOT}/tripo_standard/renders_standard/full36"
LOG_DIR="${PREPARED_ROOT}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${REPO_ROOT}"

if [[ ! -f "${TRIPO_GLB}" ]]; then
  echo "Tripo output GLB not found:"
  echo "  ${TRIPO_GLB}"
  echo "Place the downloaded Tripo GLB there, then rerun:"
  echo "  bash scripts/render_tripo_output.sh"
  exit 1
fi

if ! command -v blender >/dev/null 2>&1; then
  echo "Missing dependency: blender is not available on PATH." >&2
  exit 1
fi

blender -b --python scripts/render_glb_views.py -- \
  --input_glb "${TRIPO_GLB}" \
  --object_id "${OBJECT_ID}_tripo_standard" \
  --mode full36 \
  --out_dir "${OUT_DIR}" \
  --resolution 1024 \
  > "${LOG_DIR}/render_tripo_standard_full36.log" 2>&1

echo "Rendered Tripo GLB full36 views:"
echo "  ${OUT_DIR}"
echo "Log:"
echo "  ${LOG_DIR}/render_tripo_standard_full36.log"
