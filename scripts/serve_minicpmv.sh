#!/usr/bin/env bash
# Download MiniCPM-V-4.5-Q4_K_M (if needed) and start the llama.cpp server.
#
# Usage:
#   ./scripts/serve_minicpmv.sh              # default port 8081
#   ./scripts/serve_minicpmv.sh --port 9090

set -euo pipefail

MODEL_DIR="${HOME}/models/minicpmv-4.5"
MODEL_FILE="MiniCPM-V-4_5-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-model-f16.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MMPROJ_PATH="${MODEL_DIR}/${MMPROJ_FILE}"
PORT=8081
CTX_SIZE=4096

# Parse optional --port flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Download model and mmproj if not present
if [[ ! -f "${MODEL_PATH}" ]] || [[ ! -f "${MMPROJ_PATH}" ]]; then
  echo "Downloading model files..."
  if ! command -v hf &>/dev/null; then
    echo "Installing huggingface_hub..."
    uv tool install huggingface_hub
  fi
  hf download openbmb/MiniCPM-V-4_5-gguf \
    "${MODEL_FILE}" "${MMPROJ_FILE}" \
    --local-dir "${MODEL_DIR}"
  echo "Download complete."
fi

echo "Starting llama-server on port ${PORT}..."
exec llama-server \
  --model "${MODEL_PATH}" \
  --mmproj "${MMPROJ_PATH}" \
  --port "${PORT}" \
  --ctx-size "${CTX_SIZE}"
