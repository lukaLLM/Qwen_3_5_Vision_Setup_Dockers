"""Persistence helpers for run and benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from visual_experimentation_app.config import get_settings
from visual_experimentation_app.schemas import BenchmarkResult, RunHistoryItem, RunResult


def _results_root() -> Path:
    return get_settings().results_dir


def _runs_dir() -> Path:
    return _results_root() / "runs"


def _benchmarks_dir() -> Path:
    return _results_root() / "benchmarks"


def _run_history_path() -> Path:
    return _results_root() / "run_history.jsonl"


def _benchmark_history_path() -> Path:
    return _results_root() / "benchmark_history.jsonl"


def ensure_results_layout() -> None:
    """Create local results directories if they do not yet exist."""
    _runs_dir().mkdir(parents=True, exist_ok=True)
    _benchmarks_dir().mkdir(parents=True, exist_ok=True)
    _results_root().mkdir(parents=True, exist_ok=True)


def save_run_result(result: RunResult) -> Path:
    """Persist one run result as JSON and append to run history."""
    ensure_results_layout()
    run_path = _runs_dir() / f"{result.run_id}.json"
    run_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    with _run_history_path().open("a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json())
        handle.write("\n")

    return run_path


def load_run_result(run_id: str) -> RunResult | None:
    """Load a run result by ID, if present."""
    run_path = _runs_dir() / f"{run_id}.json"
    if not run_path.exists():
        return None
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    return RunResult.model_validate(payload)


def _history_to_item(payload: dict[str, Any]) -> RunHistoryItem:
    request = payload.get("request", {})
    effective = payload.get("effective_params", {})
    timings = payload.get("timings", {})
    return RunHistoryItem(
        run_id=str(payload.get("run_id", "")),
        created_at=str(payload.get("created_at", "")),
        status=str(payload.get("status", "error")),  # type: ignore[arg-type]
        model=str(effective.get("model") or request.get("model") or ""),
        total_ms=float(timings.get("total_ms", 0.0)),
        has_video=bool(request.get("video_paths") or request.get("video_path")),
        image_count=len(request.get("image_paths") or []),
    )


def list_run_history(*, limit: int = 200) -> list[RunHistoryItem]:
    """Read run history entries in reverse chronological order."""
    history_path = _run_history_path()
    if not history_path.exists():
        return []

    lines = history_path.read_text(encoding="utf-8").splitlines()
    items: list[RunHistoryItem] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        payload = json.loads(line)
        items.append(_history_to_item(payload))
        if len(items) >= limit:
            break
    return items


def save_benchmark_result(result: BenchmarkResult) -> dict[str, str]:
    """Persist benchmark JSON + CSV artifacts and return their file paths."""
    ensure_results_layout()

    benchmark_json = _benchmarks_dir() / f"{result.benchmark_id}.json"
    benchmark_csv = _benchmarks_dir() / f"{result.benchmark_id}.csv"

    benchmark_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    csv_fields = [
        "benchmark_id",
        "combo_key",
        "run_id",
        "repeat_index",
        "target_height",
        "request_concurrency",
        "segment_workers",
        "status",
        "preprocess_ms",
        "request_ms",
        "total_ms",
        "ttft_ms",
        "output_hash",
        "output_chars",
        "error",
    ]
    with benchmark_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in result.records:
            writer.writerow(record.model_dump())

    with _benchmark_history_path().open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "benchmark_id": result.benchmark_id,
                    "created_at": result.created_at,
                    "status": result.status,
                    "record_count": len(result.records),
                    "json_path": str(benchmark_json),
                    "csv_path": str(benchmark_csv),
                }
            )
        )
        handle.write("\n")

    return {"json": str(benchmark_json), "csv": str(benchmark_csv)}
