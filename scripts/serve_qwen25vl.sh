#!/usr/bin/env bash
# Download Qwen2.5-VL-7B-Instruct-Q4_K_M (if needed) and start the llama.cpp server.
#
# Usage:
#   ./scripts/serve_qwen25vl.sh              # default port 8080
#   ./scripts/serve_qwen25vl.sh --port 9090

set -euo pipefail

MODEL_DIR="${HOME}/models/qwen25vl-7b"
MODEL_FILE="Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MMPROJ_PATH="${MODEL_DIR}/${MMPROJ_FILE}"
PORT=8080
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
  hf download bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF \
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
