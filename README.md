# Qwen Image - Local Inference Setup

## YouTube link with explanation
- Qwen 3.5 Vision – The ONLY LOCAL Setup YOU NEED (No Ollama/LM Studio)!  https://youtu.be/-sl0oe3-Awc
- Qwen 3.5 Vision AI Speed Tuning: 30 Seconds → 2 Seconds https://youtu.be/thM6Sz_0YhE

## System Requirements

### Hardware & OS

- OS: Linux (tested on Ubuntu 24.04)
- Kernel: `6.14.0-29-generic #29~24.04.1-Ubuntu`
- GPU: NVIDIA GPU with CUDA support
- Python: 3.12+
- Tested driver: NVIDIA Driver `590.48.01` (supports up to CUDA `13.1`)
- Tested GPU: RTX PRO 6000 Blackwell Workstation Edition

## Prerequisites

- NVIDIA Container Toolkit
- Docker with Docker Compose
- `uv` (fast Python package manager)
- Hugging Face account with an access token

## Quick Setup

```bash
uv init
uv venv .venv
source .venv/bin/activate
uv add hf-transfer huggingface-hub openai
```

Add your token in `.env if you want faster download using download_qwen_models.sh etc.

```bash
HF_TOKEN=your_huggingface_token_here
```

## Download Models

Use the script:

```bash
chmod +x download_qwen_models.sh
./download_qwen_models.sh
```

The script reads `HF_TOKEN` from `.env` automatically.

If you do not want some models, open `download_qwen_models.sh` and comment out entries in:

- `GGUF_MODELS=(...)`
- `HF_MODELS=(...)`

Then run the script again.

## Inference Compose Files

Docker Compose examples are in `Inference_Single/` (vLLM and LLaMA.cpp variants).

### vLLM Profiles Added

- `Inference_Single/docker-compose vLLM Qwen3.5-0.8B.yaml` (`--max-model-len 131072`)
- `Inference_Single/docker-compose vLLM Qwen3.5-27B-GPTQ-Int4.yaml` (`--max-model-len 32768`)
- `Inference_Single/docker-compose vLLM Qwen3.5-27B-FP8.yaml` (`--max-model-len 32768`)

Each profile includes Qwen3.5 multimodal/runtime flags:

- `--mm-encoder-tp-mode data`
- `--mm-processor-cache-type shm`
- `--reasoning-parser qwen3`
- `--enable-prefix-caching`
- `--enable-auto-tool-choice`
- `--tool-call-parser qwen3_coder`

### llama.cpp profile notes

- All Qwen3.5 llama.cpp profiles include both `--model` and `--mmproj` for multimodal image support.
- Aliases are set per profile (for example `qwen35-122b-a10b-q4_k_m` for `Qwen3.5-122B-A10B`).
- Use the profile alias in `llamacpp_image_call.py` via `--model`.

Start one profile at a time (all use `localhost:8000`):

```bash
docker compose -f "Inference_Single/docker-compose vLLM Qwen3.5-0.8B.yaml" up -d
# or
docker compose -f "Inference_Single/docker-compose vLLM Qwen3.5-27B-GPTQ-Int4.yaml" up -d
# or
docker compose -f "Inference_Single/docker-compose vLLM Qwen3.5-27B-FP8.yaml" up -d
```

### 4B Lab Profile Defaults (Current)

`docker/docker-compose vLLM Qwen3.5-4B-lab.yaml` currently defaults to:

- `image: vllm/vllm-openai:latest`
- `--attention-backend ${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}`
- `--max-num-seqs 10`
- `--max-num-batched-tokens 10000`
- `--gpu-memory-utilization 0.4`
- `--max-model-len 10000`
- `--mm-processor-kwargs {"do_sample_frames":true,"fps":1.0}`
- `--no-enable-prefix-caching`
- `--mm-processor-cache-gb 0`

### Test Attention Backend (4B Lab)

The 4B lab profile supports temporary backend override via `VLLM_ATTENTION_BACKEND`.

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER \
docker compose -f "docker/docker-compose vLLM Qwen3.5-4B-lab.yaml" up -d --force-recreate
```

Check which backend was selected:

```bash
docker compose -f "docker/docker-compose vLLM Qwen3.5-4B-lab.yaml" logs vllm | rg "Using backend|Using FLASH_ATTN|FLASHINFER|TRITON_ATTN|FLEX_ATTENTION"
```

If you hit a crash inside `vllm/v1/attention/backends/flex_attention.py` with a
`view size is not compatible` error on long video requests, switch away from
`FLEX_ATTENTION` first. In this repo, the 4B lab default is `TRITON_ATTN`.
For long segmented videos, keep chunk parallel requests at `1` until you confirm
higher values are stable on your GPU/runtime combo.

Quick sweep example:

```bash
for b in FLASH_ATTN FLASHINFER TRITON_ATTN FLEX_ATTENTION; do
  echo "=== backend: $b ==="
  VLLM_ATTENTION_BACKEND="$b" \
  docker compose -f "docker/docker-compose vLLM Qwen3.5-4B-lab.yaml" up -d --force-recreate
  sleep 8
  docker compose -f "docker/docker-compose vLLM Qwen3.5-4B-lab.yaml" logs --tail=120 vllm | rg "Using backend|Error|exception" || true
done
```

`-O3` tuning in the 4B lab profile is currently commented out. To benchmark it,
uncomment `- -O${VLLM_OPT_LEVEL:-3}` in the compose file, then run with
`VLLM_OPT_LEVEL=3`.

Chunked prefill flags in the 4B lab profile are also commented out. If you need
to force ON/OFF behavior there, uncomment one of:

- `--enable-chunked-prefill`
- `--no-enable-chunked-prefill`

Stop:

```bash
docker compose -f "Inference_Single/docker-compose vLLM Qwen3.5-0.8B.yaml" down
```

OpenCode model IDs (provider `vllm-local`):

- `vllm-local/Qwen/Qwen3.5-0.8B`
- `vllm-local/Qwen/Qwen3.5-27B-GPTQ-Int4`
- `vllm-local/Qwen/Qwen3.5-27B-FP8`

## Python API Calls

vLLM image:

```bash
python3 vllm_image_call.py --image 2.png --model "Qwen/Qwen3.5-27B-FP8"
```

vLLM video:

```bash
python3 vllm_video_call.py --video 1.mp4 --model "Qwen/Qwen3.5-27B-FP8"
```

llama.cpp image (`llama_cpp_qwen3_5_122b_a10b_q4_k_m` profile):

```bash
python3 llamacpp_image_call.py --image 2.png --model "qwen35-122b-a10b-q4_k_m"
```

## Visual Experimentation App

Run the local FastAPI + Gradio experimentation app:

```bash
uv run python -m visual_experimentation_app.main
```

Detailed app docs and API routes:

- `visual_experimentation_app/README.md`
- Includes a prompt preset: `Benchmarking (Visible Chunk Summary)` for
  chunk-level benchmarking outputs (exactly 4 sentences, 6 bullet points, and
  8 keywords).

## References

- https://docs.vllm.ai/en/stable/features/multimodal_inputs/
- https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md
- https://blog.overshoot.ai/blog/qwen3.5-on-overshoot
- https://build.nvidia.com/nvidia/video-search-and-summarization
