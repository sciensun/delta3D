#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/shichang/Deformable-3D-Gaussians
INPUT_GLB=/home/shichang/Deformable-3D-Gaussians/assets/3D/big_carved_wooden_elephant_sculpture.glb
OBJECT_ID=big_carved_wooden_elephant_sculpture
PREPARED_ROOT="${REPO_ROOT}/assets/prepared/${OBJECT_ID}"
LOG_DIR="${PREPARED_ROOT}/logs"

mkdir -p \
  "${PREPARED_ROOT}/renders_original/full36" \
  "${PREPARED_ROOT}/renders_original/key8" \
  "${PREPARED_ROOT}/renders_original/tripo" \
  "${PREPARED_ROOT}/prompts" \
  "${PREPARED_ROOT}/chatgpt_upload_pack" \
  "${PREPARED_ROOT}/generated_standard/key8_manual" \
  "${PREPARED_ROOT}/tripo_input" \
  "${LOG_DIR}"

cd "${REPO_ROOT}"

if [[ ! -f "${INPUT_GLB}" ]]; then
  echo "Missing input GLB: ${INPUT_GLB}" >&2
  exit 1
fi

if ! command -v blender >/dev/null 2>&1; then
  echo "Missing dependency: blender is not available on PATH." >&2
  echo "Install Blender or add it to PATH, then rerun this script." >&2
  exit 1
fi

echo "[1/4] Rendering original GLB full36"
blender -b --python scripts/render_glb_views.py -- \
  --input_glb "${INPUT_GLB}" \
  --object_id "${OBJECT_ID}" \
  --mode full36 \
  --out_dir "${PREPARED_ROOT}/renders_original/full36" \
  --resolution 1024 \
  > "${LOG_DIR}/manual_render_full36.log" 2>&1

echo "[2/4] Rendering original GLB key8"
blender -b --python scripts/render_glb_views.py -- \
  --input_glb "${INPUT_GLB}" \
  --object_id "${OBJECT_ID}" \
  --mode key8 \
  --out_dir "${PREPARED_ROOT}/renders_original/key8" \
  --resolution 1024 \
  > "${LOG_DIR}/manual_render_key8.log" 2>&1

echo "[3/4] Rendering original GLB tripo candidates"
blender -b --python scripts/render_glb_views.py -- \
  --input_glb "${INPUT_GLB}" \
  --object_id "${OBJECT_ID}" \
  --mode tripo \
  --out_dir "${PREPARED_ROOT}/renders_original/tripo" \
  --resolution 1024 \
  > "${LOG_DIR}/manual_render_tripo.log" 2>&1

echo "[4/4] Generating prompts and ChatGPT upload pack"
python scripts/make_prompts.py \
  --views_meta "${PREPARED_ROOT}/renders_original/key8/views_meta.json" \
  --output "${PREPARED_ROOT}/prompts/prompts_standard_key8.json" \
  --manual_md "${PREPARED_ROOT}/prompts/chatgpt_manual_prompts.md" \
  --upload_pack "${PREPARED_ROOT}/chatgpt_upload_pack" \
  > "${LOG_DIR}/manual_make_prompts.log" 2>&1

cat <<EOF

Manual ChatGPT preparation complete.

Open ChatGPT, upload each image from chatgpt_upload_pack with the matching prompt text, download the generated images, and save them into generated_standard/key8_manual/ using the same numeric prefix.

Upload pack:
  ${PREPARED_ROOT}/chatgpt_upload_pack

Prompt JSON:
  ${PREPARED_ROOT}/prompts/prompts_standard_key8.json

Prompt Markdown:
  ${PREPARED_ROOT}/prompts/chatgpt_manual_prompts.md

Save manual ChatGPT outputs as:
  ${PREPARED_ROOT}/generated_standard/key8_manual/01_standard.png
  ${PREPARED_ROOT}/generated_standard/key8_manual/02_standard.png
  ...
  ${PREPARED_ROOT}/generated_standard/key8_manual/08_standard.png

Then collect results and prepare Tripo input:
  python scripts/collect_manual_chatgpt_results.py

Tripo-ready image will be:
  ${PREPARED_ROOT}/tripo_input/standard_front_3quarter.png

Logs:
  ${LOG_DIR}
EOF
