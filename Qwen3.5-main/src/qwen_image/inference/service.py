"""Inference orchestration with request-overrides and env defaults."""

from __future__ import annotations

from dataclasses import dataclass

from qwen_image.config import get_settings
from qwen_image.inference.client import InferenceCall, run_segmented


@dataclass(frozen=True)
class InferenceOverrides:
    """Per-request overrides that may replace env defaults."""

    base_url: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    api_key: str | None = None
    thinking_mode: str | None = None


def normalize_thinking_mode(thinking_mode: str | None, *, fallback: str) -> bool | None:
    """Convert on/off/auto strings into model enable_thinking bool/None."""
    selected = (thinking_mode or fallback).strip().lower()
    if selected == "on":
        return True
    if selected == "off":
        return False
    return None


def build_inference_call(
    *,
    video_path: str,
    prompt: str,
    overrides: InferenceOverrides,
) -> InferenceCall:
    """Merge explicit overrides with env defaults and return an inference request."""
    settings = get_settings().inference

    effective_max_tokens = max(
        1,
        int(overrides.max_tokens)
        if overrides.max_tokens is not None
        else settings.max_tokens,
    )
    effective_max_completion = max(
        1,
        int(overrides.max_completion_tokens)
        if overrides.max_completion_tokens is not None
        else int(overrides.max_tokens)
        if overrides.max_tokens is not None
        else settings.max_completion_tokens,
    )

    return InferenceCall(
        video_path=video_path,
        prompt=prompt,
        base_url=(overrides.base_url or settings.base_url).strip(),
        model=(overrides.model or settings.model).strip(),
        max_tokens=effective_max_tokens,
        max_completion_tokens=effective_max_completion,
        api_key=(overrides.api_key or settings.api_key).strip(),
        enable_thinking=normalize_thinking_mode(
            overrides.thinking_mode,
            fallback=settings.thinking_mode,
        ),
    )


def run_inference(
    *,
    video_path: str,
    prompt: str,
    overrides: InferenceOverrides,
) -> str:
    """Run segmented inference with defaults resolved from settings."""
    request = build_inference_call(video_path=video_path, prompt=prompt, overrides=overrides)
    return run_segmented(request)
