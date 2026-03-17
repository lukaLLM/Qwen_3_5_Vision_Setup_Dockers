"""Typed request/response schemas for the local MM lab API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ThinkingMode = Literal["auto", "on", "off"]
RunStatus = Literal["ok", "error"]
BenchmarkStatus = Literal["ok", "partial", "error"]


class RunRequest(BaseModel):
    """Input payload for a single multimodal experiment run."""

    model_config = {"extra": "forbid"}

    prompt: str = Field(..., min_length=1)
    text_input: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    video_paths: list[str] = Field(default_factory=list)
    video_path: str | None = None

    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)

    use_model_defaults: bool = False
    max_tokens: int | None = Field(default=81920, ge=1)
    max_completion_tokens: int | None = Field(default=81920, ge=1)
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int | None = Field(default=20, ge=1)
    presence_penalty: float | None = Field(default=1.5, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    thinking_mode: ThinkingMode = "auto"
    show_reasoning: bool = False
    measure_ttft: bool = True

    preprocess_images: bool = True
    preprocess_video: bool = True
    target_height: int = Field(default=480, ge=64, le=4096)
    target_video_fps: float | None = Field(default=1.0, gt=0.0, le=240.0)
    safe_video_sampling: bool = True
    video_sampling_fps: float | None = Field(default=None, gt=0.0, le=240.0)

    segment_max_duration_s: float = Field(default=0.0, ge=0.0, le=3600.0)
    segment_overlap_s: float = Field(default=0.0, ge=0.0, le=300.0)
    segment_workers: int = Field(default=1, ge=1, le=64)

    image_cache_uuids: list[str] = Field(default_factory=list)
    video_cache_uuids: list[str] = Field(default_factory=list)
    video_cache_uuid: str | None = None
    disable_caching: bool = False
    request_extra_body: dict[str, Any] = Field(default_factory=dict)
    request_extra_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prompt_and_segments(self) -> RunRequest:
        """Normalize prompt and enforce consistent segment settings."""
        prompt = self.prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty after trimming.")
        self.prompt = prompt

        normalized_video_paths = [
            str(path).strip() for path in self.video_paths if str(path).strip()
        ]
        legacy_video_path = (self.video_path or "").strip()
        if legacy_video_path and legacy_video_path not in normalized_video_paths:
            normalized_video_paths.insert(0, legacy_video_path)
        if len(normalized_video_paths) > 2:
            raise ValueError("At most 2 videos are supported per request.")
        self.video_paths = normalized_video_paths
        self.video_path = self.video_paths[0] if self.video_paths else None

        normalized_video_uuids = [
            str(uuid_value).strip()
            for uuid_value in self.video_cache_uuids
            if str(uuid_value).strip()
        ]
        legacy_video_uuid = (self.video_cache_uuid or "").strip()
        if legacy_video_uuid and legacy_video_uuid not in normalized_video_uuids:
            normalized_video_uuids.insert(0, legacy_video_uuid)
        self.video_cache_uuids = normalized_video_uuids
        self.video_cache_uuid = (
            self.video_cache_uuids[0] if self.video_cache_uuids else None
        )

        if self.segment_max_duration_s <= 0:
            self.segment_overlap_s = 0.0
        if self.segment_overlap_s >= self.segment_max_duration_s > 0:
            raise ValueError("segment_overlap_s must be less than segment_max_duration_s.")
        if self.segment_max_duration_s > 0 and len(self.video_paths) > 1:
            raise ValueError("Segmentation supports only one video at a time.")

        if self.use_model_defaults:
            self.max_tokens = None
            self.max_completion_tokens = None
            self.temperature = None
            self.top_p = None
            self.top_k = None
            self.presence_penalty = None
            self.frequency_penalty = None
        return self


class RunTiming(BaseModel):
    """Timing breakdown for one run."""

    preprocess_ms: float = 0.0
    request_ms: float = 0.0
    total_ms: float = 0.0
    ttft_ms: float | None = None


class RunResult(BaseModel):
    """Result envelope for one run."""

    run_id: str
    status: RunStatus
    created_at: str
    request: RunRequest
    output_text: str = ""
    error: str | None = None
    timings: RunTiming
    effective_params: dict[str, Any] = Field(default_factory=dict)
    media_metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRequest(BaseModel):
    """Input payload for benchmark sweeps."""

    model_config = {"extra": "forbid"}

    base_run: RunRequest
    repeats: int = Field(default=3, ge=1, le=200)
    warmup_runs: int = Field(default=0, ge=0, le=50)
    resolution_heights: list[int] = Field(default_factory=list)
    request_concurrency: list[int] = Field(default_factory=lambda: [1])
    segment_workers: list[int] = Field(default_factory=lambda: [1])
    continue_on_error: bool = True
    label: str | None = None


class BenchmarkRecord(BaseModel):
    """One benchmark trial record."""

    benchmark_id: str
    combo_key: str
    run_id: str
    repeat_index: int
    target_height: int
    request_concurrency: int
    segment_workers: int
    status: RunStatus
    preprocess_ms: float
    request_ms: float
    total_ms: float
    ttft_ms: float | None = None
    output_hash: str | None = None
    output_chars: int = 0
    error: str | None = None


class BenchmarkAggregate(BaseModel):
    """Aggregate latency stats for one sweep combination."""

    combo_key: str
    target_height: int
    request_concurrency: int
    segment_workers: int
    sample_count: int
    success_count: int
    p50_total_ms: float | None = None
    p95_total_ms: float | None = None
    min_total_ms: float | None = None
    max_total_ms: float | None = None
    avg_total_ms: float | None = None
    unique_output_count: int = 0
    output_consistency_ratio: float | None = None


class BenchmarkResult(BaseModel):
    """Benchmark response envelope."""

    benchmark_id: str
    status: BenchmarkStatus
    created_at: str
    request: BenchmarkRequest
    records: list[BenchmarkRecord] = Field(default_factory=list)
    aggregates: list[BenchmarkAggregate] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class RunHistoryItem(BaseModel):
    """Compact run summary used by history listing endpoints."""

    run_id: str
    created_at: str
    status: RunStatus
    model: str
    total_ms: float
    has_video: bool
    image_count: int
