"""Low-level client wrappers around vLLM segmented inference calls."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass

from vllm_video_call import call_vllm_segmented, stream_vllm_segmented


@dataclass(frozen=True)
class InferenceCall:
    """Resolved inference call payload sent to segmented client."""

    video_path: str
    prompt: str
    base_url: str
    model: str
    max_tokens: int
    max_completion_tokens: int | None
    api_key: str
    enable_thinking: bool | None


def run_segmented(
    request: InferenceCall,
    *,
    preprocess_status_callback: Callable[[str], None] | None = None,
    segment_status_callback: Callable[[int, int, float, float], None] | None = None,
) -> str:
    """Run segmented inference and return final text."""
    return call_vllm_segmented(
        video_path=request.video_path,
        prompt=request.prompt,
        base_url=request.base_url,
        model=request.model,
        max_tokens=request.max_tokens,
        max_completion_tokens=request.max_completion_tokens,
        api_key=request.api_key,
        preprocess_status_callback=preprocess_status_callback,
        segment_status_callback=segment_status_callback,
        enable_thinking=request.enable_thinking,
    )


def stream_segmented(
    request: InferenceCall,
    *,
    preprocess_status_callback: Callable[[str], None] | None = None,
    segment_status_callback: Callable[[int, int, float, float], None] | None = None,
) -> Generator[str, None, None]:
    """Stream segmented inference output as markdown text."""
    return stream_vllm_segmented(
        video_path=request.video_path,
        prompt=request.prompt,
        base_url=request.base_url,
        model=request.model,
        max_tokens=request.max_tokens,
        max_completion_tokens=request.max_completion_tokens,
        api_key=request.api_key,
        preprocess_status_callback=preprocess_status_callback,
        segment_status_callback=segment_status_callback,
        enable_thinking=request.enable_thinking,
    )
