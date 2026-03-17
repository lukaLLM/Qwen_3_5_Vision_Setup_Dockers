"""FastAPI routes and app factory for the local multimodal lab."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import cast
from uuid import uuid4

import gradio as gr
from fastapi import APIRouter, FastAPI, HTTPException

from visual_experimentation_app.benchmark_runner import run_benchmark
from visual_experimentation_app.config import get_settings
from visual_experimentation_app.result_store import (
    ensure_results_layout,
    list_run_history,
    load_run_result,
    save_benchmark_result,
    save_run_result,
)
from visual_experimentation_app.schemas import (
    BenchmarkRequest,
    BenchmarkResult,
    RunHistoryItem,
    RunRequest,
    RunResult,
    RunTiming,
)
from visual_experimentation_app.vllm_client import execute_run

router = APIRouter()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@router.get("/health")
def health() -> dict[str, object]:
    """Basic health endpoint for local tooling and smoke checks."""
    settings = get_settings()
    ensure_results_layout()
    return {
        "status": "ok",
        "api_prefix": settings.api_prefix,
        "ui_path": settings.ui_path,
        "vllm_base_url": settings.base_url,
        "default_model": settings.model,
        "results_dir": str(settings.results_dir),
    }


@router.post("/run", response_model=RunResult)
def run_once(request: RunRequest) -> RunResult:
    """Execute a single run and persist its result."""
    run_id = f"run_{uuid4().hex}"
    created_at = _utc_now_iso()
    started = perf_counter()

    try:
        execution = execute_run(request)
        result = RunResult(
            run_id=run_id,
            status="ok",
            created_at=created_at,
            request=request,
            output_text=execution.output_text,
            error=None,
            timings=RunTiming(
                preprocess_ms=execution.preprocess_ms,
                request_ms=execution.request_ms,
                total_ms=execution.total_ms,
                ttft_ms=execution.ttft_ms,
            ),
            effective_params=execution.effective_params,
            media_metadata=execution.media_metadata,
        )
    except Exception as exc:  # noqa: BLE001
        total_ms = (perf_counter() - started) * 1000.0
        result = RunResult(
            run_id=run_id,
            status="error",
            created_at=created_at,
            request=request,
            output_text="",
            error=str(exc),
            timings=RunTiming(
                preprocess_ms=0.0,
                request_ms=0.0,
                total_ms=total_ms,
                ttft_ms=None,
            ),
            effective_params={},
            media_metadata={},
        )

    save_run_result(result)
    return result


@router.post("/benchmark", response_model=BenchmarkResult)
def benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    """Run benchmark sweep combinations and persist benchmark artifacts."""
    benchmark_id = f"bench_{uuid4().hex}"
    created_at = _utc_now_iso()

    try:
        result = run_benchmark(request, benchmark_id=benchmark_id)
        result.created_at = created_at
    except Exception as exc:  # noqa: BLE001
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            status="error",
            created_at=created_at,
            request=request,
            records=[],
            aggregates=[],
            artifact_paths={"error": str(exc)},
        )
        paths = save_benchmark_result(result)
        result.artifact_paths = paths | {"error": str(exc)}
        return result

    paths = save_benchmark_result(result)
    result.artifact_paths = paths
    return result


@router.get("/runs", response_model=list[RunHistoryItem])
def runs(limit: int = 200) -> list[RunHistoryItem]:
    """List recent persisted run summaries."""
    clean_limit = max(1, min(limit, 1000))
    return list_run_history(limit=clean_limit)


@router.get("/runs/{run_id}", response_model=RunResult)
def run_detail(run_id: str) -> RunResult:
    """Fetch one persisted run by ID."""
    result = load_run_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return result


def create_app(*, include_ui: bool = True) -> FastAPI:
    """Create API app and optionally mount Gradio UI."""
    settings = get_settings()
    ensure_results_layout()

    app = FastAPI(title="Qwen3.5 MM Lab", version="0.1.0")
    app.include_router(router, prefix=settings.api_prefix)

    # Keep a JSON root when UI is disabled or mounted on a non-root path.
    if (not include_ui) or settings.ui_path != "/":

        @app.get("/")
        def root() -> dict[str, str]:
            return {
                "status": "ok",
                "api": settings.api_prefix,
                "ui": settings.ui_path if include_ui else "",
            }

    if not include_ui:
        return app

    from visual_experimentation_app.ui import build_ui_blocks, ui_css, ui_theme

    blocks = build_ui_blocks()
    return cast(
        FastAPI,
        gr.mount_gradio_app(
            app,
            blocks,
            path=settings.ui_path,
            theme=ui_theme(),
            css=ui_css(),
        ),
    )
