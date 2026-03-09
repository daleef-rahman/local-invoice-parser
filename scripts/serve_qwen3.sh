#!/usr/bin/env bash
# Download Qwen3-4B-Q4_K_M (if needed) and start the llama.cpp server.
#
# Usage:
#   ./scripts/serve_qwen3.sh              # default port 8080
#   ./scripts/serve_qwen3.sh --port 9090

set -euo pipefail

MODEL_DIR="${HOME}/models/qwen3-4b"
MODEL_FILE="Qwen_Qwen3-4B-Q4_K_M.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
PORT=8080
CTX_SIZE=4096

# Parse optional --port flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Download model if not present
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found at ${MODEL_PATH}, downloading..."
  if ! command -v hf &>/dev/null; then
    echo "Installing huggingface_hub..."
    uv tool install huggingface_hub
  fi
  hf download bartowski/Qwen_Qwen3-4B-GGUF \
    "${MODEL_FILE}" \
    --local-dir "${MODEL_DIR}"
  echo "Download complete."
fi

echo "Starting llama-server on port ${PORT}..."
exec llama-server \
  --model "${MODEL_PATH}" \
  --port "${PORT}" \
  --ctx-size "${CTX_SIZE}"
