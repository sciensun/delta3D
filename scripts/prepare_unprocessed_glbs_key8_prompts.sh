#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/shichang/Deformable-3D-Gaussians}
ASSET_DIR=${ASSET_DIR:-"${REPO_ROOT}/assets/3D"}
PREPARED_ROOT=${PREPARED_ROOT:-"${REPO_ROOT}/assets/prepared"}
RESOLUTION=${RESOLUTION:-1024}
FORCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --dry-run|--dry_run)
      DRY_RUN=1
      shift
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

if command -v blender >/dev/null 2>&1; then
  BLENDER_BIN=$(command -v blender)
elif [[ -x "${HOME}/software/blender-4.3.2-linux-x64/blender" ]]; then
  BLENDER_BIN="${HOME}/software/blender-4.3.2-linux-x64/blender"
else
  echo "Missing dependency: blender is not available on PATH." >&2
  echo "Expected either 'blender' or ${HOME}/software/blender-4.3.2-linux-x64/blender" >&2
  exit 1
fi

if [[ ! -d "${ASSET_DIR}" ]]; then
  echo "Missing GLB asset directory: ${ASSET_DIR}" >&2
  exit 1
fi

shopt -s nullglob
glbs=("${ASSET_DIR}"/*.glb "${ASSET_DIR}"/*.GLB)
if [[ ${#glbs[@]} -eq 0 ]]; then
  echo "No GLB files found in ${ASSET_DIR}"
  exit 0
fi

processed=0
skipped=0

for glb in "${glbs[@]}"; do
  filename=$(basename "${glb}")
  object_id="${filename%.*}"
  object_category=$(echo "${object_id}" | tr '_-' ' ')
  object_root="${PREPARED_ROOT}/${object_id}"
  key8_dir="${object_root}/renders_original/key8"
  prompt_dir="${object_root}/prompts"
  upload_pack="${object_root}/chatgpt_upload_pack"
  log_dir="${object_root}/logs"
  prompt_json="${prompt_dir}/prompts_standard_key8.json"
  prompt_md="${prompt_dir}/chatgpt_manual_prompts.md"
  key8_count=0
  if [[ -d "${key8_dir}" ]]; then
    key8_count=$(find "${key8_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)
  fi

  if [[ "${FORCE}" -eq 0 && -f "${prompt_json}" && "${key8_count}" -ge 8 ]]; then
    echo "SKIP already processed: ${object_id}"
    skipped=$((skipped + 1))
    continue
  fi

  echo "PROCESS ${object_id}"
  echo "  GLB: ${glb}"
  echo "  Output: ${object_root}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  DRY RUN: would render key8 and generate prompts."
    processed=$((processed + 1))
    continue
  fi

  mkdir -p "${key8_dir}" "${prompt_dir}" "${upload_pack}" "${log_dir}"

  "${BLENDER_BIN}" -b --python scripts/render_glb_views.py -- \
    --input_glb "${glb}" \
    --object_id "${object_id}" \
    --mode key8 \
    --out_dir "${key8_dir}" \
    --resolution "${RESOLUTION}" \
    > "${log_dir}/render_key8.log" 2>&1

  python scripts/make_prompts.py \
    --views_meta "${key8_dir}/views_meta.json" \
    --output "${prompt_json}" \
    --manual_md "${prompt_md}" \
    --upload_pack "${upload_pack}" \
    --object_category "${object_category}" \
    > "${log_dir}/make_prompts.log" 2>&1

  processed=$((processed + 1))
done

cat <<EOF

Done.
Processed: ${processed}
Skipped: ${skipped}

For each processed object, check:
  assets/prepared/<object_id>/renders_original/key8/
  assets/prepared/<object_id>/prompts/prompts_standard_key8.json
  assets/prepared/<object_id>/prompts/chatgpt_manual_prompts.md
  assets/prepared/<object_id>/chatgpt_upload_pack/
EOF
