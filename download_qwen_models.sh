#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Qwen model downloader (GGUF + mmproj + GPTQ/FP8)
#
# How to use:
# 1) Save this file as: download_qwen_models.sh
# 2) Make executable:    chmod +x download_qwen_models.sh
# 3) (Optional) Create project + venv and install deps with uv:
#      uv init
#      uv venv .venv
#      source .venv/bin/activate
#      uv add hf-transfer huggingface-hub
# 4) Set token:
#      export HF_TOKEN="your_hf_token_here"
# 5) Run:
#      ./download_qwen_models.sh
#
# Notes:
# - GGUF repos download BOTH Q4_K_M and mmproj files (for multimodal llama.cpp).
# - Re-running resumes partial downloads automatically.
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ".env" ]] && grep -qE '^HF_TOKEN=' ".env"; then
    # Load HF_TOKEN from local project env file if present.
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing dependency: 'hf' command not found."
  echo "Install with uv:"
  echo "  uv init"
  echo "  uv venv .venv"
  echo "  source .venv/bin/activate"
  echo "  uv add hf-transfer huggingface-hub"
  echo "Then activate your venv and run this script again."
  exit 127
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN found (from env or .env). Using token for downloads."
else
  echo "HF_TOKEN not found in environment or .env. Continuing without login."
  echo "For private/gated repos, set HF_TOKEN in .env or run:"
  echo "export HF_TOKEN=\"your_hf_token_here\""
fi

download_or_exit() {
  local repo="$1"
  shift

  if hf download "$repo" --type model "$@"; then
    return 0
  fi

  local rc=$?
  if [[ $rc -eq 130 ]]; then
    echo "Interrupted by Ctrl+C."
    exit 130
  fi

  echo "Failed: $repo (exit $rc)"
  exit "$rc"
}

GGUF_MODELS=(
  "unsloth/Qwen3.5-0.8B-GGUF"
  "unsloth/Qwen3.5-2B-GGUF"
  "unsloth/Qwen3.5-9B-GGUF"
  "unsloth/Qwen3.5-27B-GGUF"
  "unsloth/Qwen3.5-35B-A3B-GGUF"
  "unsloth/Qwen3.5-122B-A10B-GGUF"
)

HF_MODELS=(
  "Qwen/Qwen3.5-27B-GPTQ-Int4"
  "Qwen/Qwen3.5-27B-FP8"
  "Qwen/Qwen3.5-0.8B"
  "Qwen/Qwen3.5-4B"
  #"Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"

)

for REPO in "${GGUF_MODELS[@]}"; do
  echo "Downloading Q4_K_M for $REPO..."
  download_or_exit "$REPO" --include "*Q4_K_M*.gguf"

  echo "Downloading mmproj for $REPO..."
  download_or_exit "$REPO" --include "*mmproj*.gguf"

  echo "Done: $REPO"
  echo "----------------------------------------"
done

for REPO in "${HF_MODELS[@]}"; do
  echo "Downloading full repo for $REPO..."
  download_or_exit "$REPO"

  echo "Done: $REPO"
  echo "----------------------------------------"
done
