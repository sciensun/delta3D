#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="${IMAGE_DIR:-previews}"
EXCEL="${EXCEL:-models.xlsx}"
OUT_DIR="${OUT_DIR:-outputs}"
MATCH_MODE="${MATCH_MODE:-order}"
OVERWRITE="${OVERWRITE:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ARGS=()
if [[ "${OVERWRITE}" == "true" || "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

mkdir -p "${OUT_DIR}"

if [[ ! -d "${IMAGE_DIR}" ]]; then
  echo "ERROR: preview image directory not found: ${IMAGE_DIR}" >&2
  echo "Put preview images in ${IMAGE_DIR}/ or set IMAGE_DIR=/path/to/previews." >&2
  exit 1
fi

if [[ ! -f "${EXCEL}" ]]; then
  echo "ERROR: Excel file not found: ${EXCEL}" >&2
  echo "Put models.xlsx in the project root or set EXCEL=/path/to/models.xlsx." >&2
  exit 1
fi

if [[ ! -f "${OUT_DIR}/deformation_vocab.yaml" ]]; then
  cp deformation_vocab.yaml "${OUT_DIR}/deformation_vocab.yaml"
fi

"${PYTHON_BIN}" build_manifest.py \
  --image_dir "${IMAGE_DIR}" \
  --excel "${EXCEL}" \
  --out "${OUT_DIR}/manifest.csv" \
  --match_mode "${MATCH_MODE}" \
  "${ARGS[@]}"

"${PYTHON_BIN}" auto_tag_images.py \
  --manifest "${OUT_DIR}/manifest.csv" \
  --vocab "${OUT_DIR}/deformation_vocab.yaml" \
  --out "${OUT_DIR}/labels_auto.xlsx" \
  --manual_out "${OUT_DIR}/labels_for_manual.xlsx" \
  --jsonl "${OUT_DIR}/labels.jsonl" \
  "${ARGS[@]}"

"${PYTHON_BIN}" make_contact_sheet.py \
  --manifest "${OUT_DIR}/manifest.csv" \
  --labels "${OUT_DIR}/labels_auto.xlsx" \
  --out "${OUT_DIR}/contact_sheet.png" \
  "${ARGS[@]}"

echo "Done."
echo "Main outputs:"
echo "  ${OUT_DIR}/manifest.csv"
echo "  ${OUT_DIR}/labels_auto.xlsx"
echo "  ${OUT_DIR}/labels_for_manual.xlsx"
echo "  ${OUT_DIR}/labels.jsonl"
echo "  ${OUT_DIR}/contact_sheet.png"
echo "  ${OUT_DIR}/deformation_vocab.yaml"
