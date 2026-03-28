# MM Lab (Local Multimodal Experiments)

This folder provides a local-only experimentation setup for Qwen3.5 multimodal testing with:

- Gradio UI
- Local FastAPI endpoints
- OpenAI-compatible calls to vLLM
- Timing capture
- JSON + JSONL run artifacts

## YouTube link with explanation
- Qwen 3.5 Vision AI Speed Tuning: 30 Seconds → 2 Seconds https://youtu.be/thM6Sz_0YhE
- Qwen 3.5 Vision – The ONLY LOCAL Setup YOU NEED (No Ollama/LM Studio)!  https://youtu.be/-sl0oe3-Awc


## Requirements

- Linux with NVIDIA driver + CUDA-capable GPU
- Docker Engine with Compose plugin (`docker compose`)
- NVIDIA Container Toolkit configured for Docker runtime
- Model weights available locally (see root `README.md` and `download_qwen_models.sh`)

References:

- Root setup and model download guide: `../README.md`
- MM lab vLLM compose profile: `../docker/docker-compose vLLM Qwen3.5-4B-lab.yaml`
- Additional inference compose profiles: `../Inference_Single/`
- https://docs.vllm.ai/en/stable/features/multimodal_inputs/
- https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md
- https://blog.overshoot.ai/blog/qwen3.5-on-overshoot
- https://build.nvidia.com/nvidia/video-search-and-summarization

## Start

### 1) Start vLLM (4B lab profile)

```bash
docker compose -f "docker/docker-compose vLLM Qwen3.5-4B-lab.yaml" up -d --build
```

### 2) Start MM lab server

```bash
uv run python -m visual_experimentation_app.main
```

Defaults:

- UI: `http://127.0.0.1:7870/`
- API: `http://127.0.0.1:7870/api`

### Current 4B Lab Compose Defaults

`../docker/docker-compose vLLM Qwen3.5-4B-lab.yaml` currently uses:

- `vllm/vllm-openai:latest`
- `--attention-backend ${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}`
- `--max-num-seqs 10`
- `--max-num-batched-tokens 10000`
- `--gpu-memory-utilization 0.4`
- `--max-model-len 10000`
- `--mm-processor-kwargs {"do_sample_frames":true,"fps":1.0}`
- `--no-enable-prefix-caching`

## Prompt Presets

The UI "Prompt Mode" dropdown includes:

- `Custom`
- `Search/Indexing`
- `Understanding/Summarization`
- `Benchmarking (Visible Chunk Summary)`
- `Tagging`
- `Classifier (Single Category)`
- `Video Type (One Word)`

`Benchmarking (Visible Chunk Summary)` is designed for chunk-level benchmark
comparisons and asks for a fixed output shape:

- Exactly 4 summary sentences
- Exactly 6 bullet points
- Exactly 8 keywords

## Local API Endpoints

All local API routes are under `/api`.

- `GET /health`
- `POST /run`
- `GET /runs`
- `GET /runs/{run_id}`

### Example: single run

```bash
curl -X POST http://127.0.0.1:7870/api/run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Describe the scene and list key objects.",
    "image_paths": ["/tmp/image_1.jpg", "/tmp/image_2.jpg"],
    "video_paths": ["/tmp/clip_1.mp4", "/tmp/clip_2.mp4"],
    "target_height": 480
  }'
```

## vLLM Endpoint Reference

vLLM supports multiple OpenAI-compatible endpoints. In this lab:

- Used directly: `/v1/chat/completions`, `/v1/models`
- Useful for extended experiments: `/v1/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/rerank`, `/pooling`, `/score`, `/tokenize`, `/detokenize`

Reference:

- https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#supported-apis

## Engine-Arg Tuning Guide

The lab compose file exposes key knobs for multimodal experiments:

- `--limit-mm-per-prompt` to permit many images + up to two videos.
- `--mm-processor-kwargs` for processor sampling behavior.
- `--media-io-kwargs` for video frame intake behavior.
- `--max-num-seqs` for request concurrency at engine level.
- `--gpu-memory-utilization` and `--max-model-len` for memory/performance tradeoffs.
- `--mm-processor-cache-type` and prefix caching flags for reuse/caching behavior.

Reference:

- https://docs.vllm.ai/en/latest/configuration/engine_args.html
- https://docs.vllm.ai/en/stable/features/multimodal_inputs.html


## Practical Speed Levers

Use these levers to improve video generation speed (measure each change in isolation):

1. Reduce `target_height` (for example 720 -> 480 -> 360).
2. Reduce video frame sampling (`video_sampling_fps` / `mm_processor_kwargs.fps`).
3. Cap output length (`max_tokens`, `max_completion_tokens`) for your demo prompt.
4. Use segmented video processing for long clips and increase `segment_workers` when GPU has headroom.
5. Increase engine throughput knobs (`--max-num-seqs`) carefully while watching OOM risk.
6. Keep cache enabled for production latency; disable cache for fair comparisons.

## Benchmark Semantics (Chunk-First)

Benchmark parallelism is chunk-first, not duplicate full-run fanout:

- `Chunk Parallel Requests to Test (CSV)` means how many video chunks are sent to vLLM concurrently.
- One benchmark repeat is one logical run (one final aggregated output), even when chunk parallel is > 1.
- If segmentation is off (or only one chunk is produced), chunk parallel has no effect.

Optional compare mode:

- `Include Non-Segmented Baseline` adds one non-segmented baseline per height.
- `segmented`: sweeps all chunk-parallel values with current segment settings.
- `non_segmented`: forces `segment_max_duration_s=0` and `segment_overlap_s=0`.

## Benchmark Prompt (Low-Variance)

For stable perf comparisons, use the one-word classifier prompt:

- UI preset: `Video Type (One Word)`
- Script flag: `--one-word-video-type-prompt`

Example:

```bash
python3 scripts/mm_lab_video_perf_suite.py \
  --video /path/to/test.mp4 \
  --one-word-video-type-prompt
```

### Graph Guide

- Graph 1: Latency vs Chunk Parallel Requests.
- Graph 2: Throughput vs Chunk Parallel Requests.
- Graph 3: Full Video Completion Time (ms) by config and segmentation mode.
- Graph 4: Stacked time split (`preprocess %` + `request %`) by combo.

Use native chart fullscreen from each graph toolbar. Hover tooltips stay enabled in fullscreen.

## Video Bug Note

As of **March 14, 2026**, vLLM PR `#33956` remains open:

- https://github.com/vllm-project/vllm/pull/33956

The compose profile now defaults to `do_sample_frames=true` (with `fps=1.0`) for
video sampling. The UI/API still provide a "safe video processor defaults"
toggle to force `do_sample_frames=false` when you need safer behavior.
