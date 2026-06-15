#!/usr/bin/env bash
set -euo pipefail

echo "scripts/prepare_elephant_data.sh now uses the manual ChatGPT pipeline."
echo "No OpenAI API key is read and no image generation API is called."
exec bash scripts/prepare_elephant_manual_chatgpt.sh "$@"
