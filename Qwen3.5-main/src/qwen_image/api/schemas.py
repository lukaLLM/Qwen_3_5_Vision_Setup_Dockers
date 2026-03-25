"""Pydantic request schemas for OpenAI-style API routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """Request payload for OpenAI-style chat completions wrapper."""

    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    base_url: str | None = None
    api_key: str | None = None
    prompt_id: str | None = None
    prompt_name: str | None = None
    prompt_text: str | None = None
    video_path: str | None = None
    video_url: str | None = None
    thinking: str | None = None
