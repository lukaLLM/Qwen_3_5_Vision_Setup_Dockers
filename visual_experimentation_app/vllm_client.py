"""Execution engine for multimodal calls against an OpenAI-compatible vLLM server."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai import OpenAI

from visual_experimentation_app.config import LabSettings, get_settings
from visual_experimentation_app.media_preprocess import (
    PreparedMedia,
    SegmentClip,
    cleanup_paths,
    encode_file_to_data_url,
    extract_video_segments,
    prepare_media,
)
from visual_experimentation_app.payload_builder import (
    build_messages,
    coerce_text,
    extract_message_parts,
    merge_extra_body,
    normalize_base_url,
)
from visual_experimentation_app.schemas import RunRequest


@dataclass(frozen=True)
class RunExecution:
    """Execution payload returned by the run engine."""

    output_text: str
    preprocess_ms: float
    request_ms: float
    total_ms: float
    ttft_ms: float | None
    effective_params: dict[str, Any]
    media_metadata: dict[str, Any]
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting captured from API usage metadata."""

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def _effective_setting(value: str | None, fallback: str) -> str:
    selected = (value or "").strip()
    return selected if selected else fallback


def _effective_timeout(value: float | None, fallback: float) -> float:
    if value is None:
        return fallback
    return max(1.0, float(value))


_VIDEO_PROCESSOR_HINT = (
    "vLLM video processor rejected this request (Qwen3VLProcessor). "
    "Try lowering video sampling FPS, enabling Safe video sampling, "
    "reducing segment workers, or using shorter segment duration."
)


def _exception_text(exc: Exception) -> str:
    parts: list[str] = [str(exc)]

    body = getattr(exc, "body", None)
    if body:
        try:
            parts.append(json.dumps(body, ensure_ascii=True, default=str))
        except (TypeError, ValueError):
            parts.append(str(body))

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            response_text = getattr(response, "text", "")
            if response_text:
                parts.append(str(response_text))
        except Exception:  # noqa: BLE001
            pass

    return "\n".join(part for part in parts if part).lower()


def is_video_processor_error(exc: Exception) -> bool:
    """Return whether exception text matches known video processor failures."""
    text = _exception_text(exc)
    return (
        "failed to apply qwen3vlprocessor" in text
        or ("error in preprocessing prompt inputs" in text and "video" in text)
        or (
            "video_processing_utils" in text
            and "out of bounds" in text
            and "index" in text
        )
    )


def summarize_execution_error(exc: Exception) -> str:
    """Return concise, actionable run error text for UI/API responses."""
    if is_video_processor_error(exc):
        return _VIDEO_PROCESSOR_HINT

    raw_error = str(exc).strip() or exc.__class__.__name__
    if len(raw_error) <= 700:
        return raw_error
    return f"{raw_error[:700]}..."


def build_execution_error_details(exc: Exception) -> dict[str, Any]:
    """Build rich error details while keeping UI-facing message concise."""
    details: dict[str, Any] = {
        "error_type": exc.__class__.__name__,
        "is_video_processor_error": is_video_processor_error(exc),
        "raw_error": str(exc).strip() or repr(exc),
    }
    if details["is_video_processor_error"]:
        details["hint"] = _VIDEO_PROCESSOR_HINT
    return details


def _format_assistant_output(
    *,
    content: str,
    reasoning: str,
    include_reasoning: bool,
) -> str:
    clean_content = content.strip()
    clean_reasoning = reasoning.strip()
    if include_reasoning and clean_reasoning:
        if clean_content:
            return f"<think>\n{clean_reasoning}\n</think>\n\n{clean_content}"
        return f"<think>\n{clean_reasoning}\n</think>"
    if clean_content:
        return clean_content
    return clean_reasoning


def _segment_header(segment: SegmentClip, index: int, total: int) -> str:
    return f"### Segment {index}/{total} [{segment.start_s:.2f}s - {segment.end_s:.2f}s]"


def _model_cache_dir(model: str) -> Path:
    """Map Hugging Face model id to local cache directory name."""
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model.replace('/', '--')}"


def _load_generation_config_defaults(model: str) -> dict[str, Any]:
    """Best-effort read of local Hugging Face generation_config for this model."""
    model_dir = _model_cache_dir(model)
    if not model_dir.exists():
        return {
            "found": False,
            "source": "not_found",
            "message": "Model cache directory was not found locally.",
        }

    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        candidates = sorted(snapshots_dir.glob("*/generation_config.json"), reverse=True)
        if candidates:
            path = candidates[0]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                return {
                    "found": False,
                    "source": "parse_error",
                    "path": str(path),
                    "message": f"Found generation_config.json but failed to parse: {exc}",
                }

            sampling_keys = [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "max_new_tokens",
            ]
            sampling_values = {
                key: payload[key]
                for key in sampling_keys
                if key in payload and payload[key] is not None
            }
            return {
                "found": True,
                "source": "generation_config_json",
                "path": str(path),
                "sampling_values": sampling_values,
            }

    no_exist_dir = model_dir / ".no_exist"
    no_exist_markers = list(no_exist_dir.glob("*/generation_config.json"))
    if no_exist_markers:
        return {
            "found": False,
            "source": "missing_in_model_repo",
            "message": "Model repository appears to have no generation_config.json.",
        }

    return {
        "found": False,
        "source": "unknown",
        "message": (
            "Could not locate generation_config.json in local cache. "
            "vLLM will use model defaults if available, otherwise its internal defaults."
        ),
    }


def _chat_completion_kwargs(
    *,
    request: RunRequest,
    model: str,
    messages: list[dict[str, object]],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "extra_body": extra_body,
    }
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = request.max_completion_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.presence_penalty is not None:
        kwargs["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        kwargs["frequency_penalty"] = request.frequency_penalty
    return kwargs


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_usage_tokens(payload: object) -> TokenUsage:
    usage = getattr(payload, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=_int_or_none(getattr(usage, "prompt_tokens", None)),
        output_tokens=_int_or_none(getattr(usage, "completion_tokens", None)),
        total_tokens=_int_or_none(getattr(usage, "total_tokens", None)),
    )


def _sum_token_usage(usages: Iterable[TokenUsage]) -> TokenUsage:
    usage_list = list(usages)

    prompt_values = [
        value for value in (item.prompt_tokens for item in usage_list) if value is not None
    ]
    output_values = [
        value for value in (item.output_tokens for item in usage_list) if value is not None
    ]
    total_values = [
        value for value in (item.total_tokens for item in usage_list) if value is not None
    ]

    return TokenUsage(
        prompt_tokens=sum(prompt_values) if prompt_values else None,
        output_tokens=sum(output_values) if output_values else None,
        total_tokens=sum(total_values) if total_values else None,
    )


def _invoke_completion(
    *,
    client: OpenAI,
    request: RunRequest,
    model: str,
    messages: list[dict[str, object]],
    extra_body: dict[str, Any],
) -> tuple[str, float | None, TokenUsage]:
    kwargs = _chat_completion_kwargs(
        request=request,
        model=model,
        messages=messages,
        extra_body=extra_body,
    )

    if not request.measure_ttft:
        response = client.chat.completions.create(**kwargs)
        if not response.choices:
            return "", None, _extract_usage_tokens(response)
        content, reasoning = extract_message_parts(response.choices[0].message)
        return (
            _format_assistant_output(
                content=content,
                reasoning=reasoning,
                include_reasoning=request.show_reasoning,
            ),
            None,
            _extract_usage_tokens(response),
        )

    start = time.perf_counter()
    first_chunk_time: float | None = None
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    final_usage = TokenUsage()

    stream = client.chat.completions.create(
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )
    for event in stream:
        event_usage = _extract_usage_tokens(event)
        if (
            event_usage.prompt_tokens is not None
            or event_usage.output_tokens is not None
            or event_usage.total_tokens is not None
        ):
            final_usage = event_usage
        if not event.choices:
            continue
        delta = event.choices[0].delta
        content_delta = coerce_text(getattr(delta, "content", None))
        reasoning_delta = coerce_text(getattr(delta, "reasoning_content", None))
        if not reasoning_delta:
            reasoning_delta = coerce_text(getattr(delta, "reasoning", None))

        changed = False
        if content_delta:
            content_chunks.append(content_delta)
            changed = True
        if reasoning_delta:
            reasoning_chunks.append(reasoning_delta)
            changed = True
        if changed and first_chunk_time is None:
            first_chunk_time = time.perf_counter()

    output = _format_assistant_output(
        content="".join(content_chunks),
        reasoning="".join(reasoning_chunks),
        include_reasoning=request.show_reasoning,
    )
    ttft_ms = ((first_chunk_time - start) * 1000.0) if first_chunk_time is not None else None
    return output, ttft_ms, final_usage


def _build_client(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    extra_headers: dict[str, str],
) -> OpenAI:
    kwargs: dict[str, Any] = {
        "base_url": normalize_base_url(base_url),
        "api_key": api_key,
        "timeout": timeout_seconds,
    }
    if extra_headers:
        kwargs["default_headers"] = extra_headers
    return OpenAI(**kwargs)


def _build_segments(prepared: PreparedMedia, request: RunRequest) -> list[SegmentClip]:
    if len(prepared.video_paths) != 1:
        return []
    video_path = prepared.video_paths[0]
    if request.segment_max_duration_s <= 0:
        return [SegmentClip(path=video_path, start_s=0.0, end_s=0.0, is_temp=False)]
    return extract_video_segments(
        video_path=video_path,
        max_duration_s=request.segment_max_duration_s,
        overlap_s=request.segment_overlap_s,
    )


def _prepare_message_payloads(
    *,
    prompt: str,
    text_input: str | None,
    image_paths: Iterable[Path],
    video_paths: Iterable[Path],
    image_cache_uuids: list[str],
    video_cache_uuids: list[str],
    disable_caching: bool,
) -> list[dict[str, object]]:
    image_path_list = list(image_paths)
    video_path_list = list(video_paths)
    image_data_urls = [encode_file_to_data_url(path) for path in image_path_list]
    video_data_urls = [encode_file_to_data_url(path) for path in video_path_list]

    effective_image_uuids = image_cache_uuids
    effective_video_uuids = video_cache_uuids
    if disable_caching:
        effective_image_uuids = [f"nocache-img-{uuid4().hex}" for _ in image_path_list]
        effective_video_uuids = [f"nocache-vid-{uuid4().hex}" for _ in video_path_list]

    return build_messages(
        prompt=prompt,
        text_input=text_input,
        image_data_urls=image_data_urls,
        video_data_urls=video_data_urls,
        image_cache_uuids=effective_image_uuids,
        video_cache_uuids=effective_video_uuids,
    )


def _run_non_segmented(
    *,
    client: OpenAI,
    request: RunRequest,
    model: str,
    extra_body: dict[str, Any],
    prepared: PreparedMedia,
) -> tuple[str, float | None, TokenUsage]:
    messages = _prepare_message_payloads(
        prompt=request.prompt,
        text_input=request.text_input,
        image_paths=prepared.image_paths,
        video_paths=prepared.video_paths,
        image_cache_uuids=request.image_cache_uuids,
        video_cache_uuids=request.video_cache_uuids,
        disable_caching=request.disable_caching,
    )
    return _invoke_completion(
        client=client,
        request=request,
        model=model,
        messages=messages,
        extra_body=extra_body,
    )


def _run_segmented(
    *,
    client: OpenAI,
    request: RunRequest,
    model: str,
    extra_body: dict[str, Any],
    prepared: PreparedMedia,
    segments: list[SegmentClip],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
) -> tuple[str, float | None, TokenUsage]:
    if len(segments) == 1:
        messages = _prepare_message_payloads(
            prompt=request.prompt,
            text_input=request.text_input,
            image_paths=prepared.image_paths,
            video_paths=[segments[0].path],
            image_cache_uuids=request.image_cache_uuids,
            video_cache_uuids=request.video_cache_uuids,
            disable_caching=request.disable_caching,
        )
        return _invoke_completion(
            client=client,
            request=request,
            model=model,
            messages=messages,
            extra_body=extra_body,
        )

    worker_count = max(1, min(request.segment_workers, len(segments)))
    results: list[tuple[str, float | None, TokenUsage]] = [
        ("", None, TokenUsage()) for _ in segments
    ]

    def run_one(index: int, segment: SegmentClip) -> tuple[int, str, float | None, TokenUsage]:
        threaded_client = _build_client(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers=request.request_extra_headers,
        )
        messages = _prepare_message_payloads(
            prompt=request.prompt,
            text_input=request.text_input,
            image_paths=prepared.image_paths,
            video_paths=[segment.path],
            image_cache_uuids=request.image_cache_uuids,
            video_cache_uuids=[],
            disable_caching=request.disable_caching,
        )
        text, ttft, usage = _invoke_completion(
            client=threaded_client,
            request=request,
            model=model,
            messages=messages,
            extra_body=extra_body,
        )
        return index, text, ttft, usage

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(run_one, index, segment): index
            for index, segment in enumerate(segments)
        }
        for future in as_completed(futures):
            index, text, ttft, usage = future.result()
            results[index] = (text, ttft, usage)

    sections: list[str] = []
    ttft_values: list[float] = []
    usage_values: list[TokenUsage] = []
    for index, segment in enumerate(segments, start=1):
        text, ttft, usage = results[index - 1]
        section_body = text.strip() or "_No output._"
        sections.append(f"{_segment_header(segment, index, len(segments))}\n{section_body}")
        if ttft is not None:
            ttft_values.append(ttft)
        usage_values.append(usage)
    combined = "\n\n".join(sections)
    aggregate_ttft = min(ttft_values) if ttft_values else None
    aggregate_usage = _sum_token_usage(usage_values)
    return combined, aggregate_ttft, aggregate_usage


def execute_run(request: RunRequest) -> RunExecution:
    """Execute one run request and return output + timing details."""
    settings: LabSettings = get_settings()
    started_at = time.perf_counter()

    base_url = _effective_setting(request.base_url, settings.base_url)
    model = _effective_setting(request.model, settings.model)
    api_key = _effective_setting(request.api_key, settings.api_key)
    timeout_seconds = _effective_timeout(request.timeout_seconds, settings.timeout_seconds)

    preprocess_start = time.perf_counter()
    prepared = prepare_media(
        image_paths=request.image_paths,
        video_paths=request.video_paths,
        preprocess_images=request.preprocess_images,
        preprocess_video=request.preprocess_video,
        target_height=request.target_height,
        target_video_fps=request.target_video_fps,
    )
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    segments = _build_segments(prepared, request)
    segment_cleanup = [segment.path for segment in segments if segment.is_temp]

    request_start = time.perf_counter()
    cache_salt: str | None = None
    token_usage = TokenUsage()
    try:
        extra_body = merge_extra_body(
            user_extra_body=request.request_extra_body,
            include_video=bool(prepared.video_paths),
            safe_video_sampling=request.safe_video_sampling,
            video_sampling_fps=request.video_sampling_fps,
            thinking_mode=request.thinking_mode,
            top_k=request.top_k,
        )
        if request.disable_caching:
            cache_salt = f"nocache-{uuid4().hex}"
            # Isolate prefix/KV cache per request to reduce cross-run cache effects.
            extra_body["cache_salt"] = cache_salt

        client = _build_client(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers=request.request_extra_headers,
        )

        if not segments:
            output_text, ttft_ms, token_usage = _run_non_segmented(
                client=client,
                request=request,
                model=model,
                extra_body=extra_body,
                prepared=prepared,
            )
        else:
            output_text, ttft_ms, token_usage = _run_segmented(
                client=client,
                request=request,
                model=model,
                extra_body=extra_body,
                prepared=prepared,
                segments=segments,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
    finally:
        cleanup_paths(segment_cleanup + prepared.cleanup_paths)

    request_ms = (time.perf_counter() - request_start) * 1000.0
    total_ms = (time.perf_counter() - started_at) * 1000.0

    effective_params: dict[str, Any] = {
        "base_url": normalize_base_url(base_url),
        "model": model,
        "use_model_defaults": request.use_model_defaults,
        "timeout_seconds": timeout_seconds,
        "target_height": request.target_height,
        "target_video_fps": request.target_video_fps,
        "safe_video_sampling": request.safe_video_sampling,
        "video_sampling_fps": request.video_sampling_fps,
        "video_count": len(prepared.video_paths),
        "segment_max_duration_s": request.segment_max_duration_s,
        "segment_overlap_s": request.segment_overlap_s,
        "segment_workers": request.segment_workers,
        "segment_count": len(segments) if segments else 0,
        "disable_caching": request.disable_caching,
        "cache_salt": cache_salt,
        "extra_body": extra_body,
        "sent_generation_params": {
            "max_tokens": request.max_tokens,
            "max_completion_tokens": request.max_completion_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "thinking_mode": request.thinking_mode,
        },
        "omitted_for_model_defaults": (
            [
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "frequency_penalty",
            ]
            if request.use_model_defaults
            else []
        ),
    }
    if request.use_model_defaults:
        effective_params["model_defaults_info"] = _load_generation_config_defaults(model)

    return RunExecution(
        output_text=output_text,
        preprocess_ms=preprocess_ms,
        request_ms=request_ms,
        total_ms=total_ms,
        ttft_ms=ttft_ms,
        effective_params=effective_params,
        media_metadata=prepared.metadata,
        prompt_tokens=token_usage.prompt_tokens,
        output_tokens=token_usage.output_tokens,
        total_tokens=token_usage.total_tokens,
    )
