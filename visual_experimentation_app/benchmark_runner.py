"""Benchmark orchestration for sweep-based multimodal experiments."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from itertools import product
from typing import Literal
from uuid import uuid4

from visual_experimentation_app.config import get_settings
from visual_experimentation_app.schemas import (
    BenchmarkAggregate,
    BenchmarkRecord,
    BenchmarkRequest,
    BenchmarkResult,
    RunRequest,
)
from visual_experimentation_app.vllm_client import RunExecution, execute_run

RunExecutor = Callable[[RunRequest], RunExecution]


def _sanitize_int_list(values: list[int], *, fallback: int, minimum: int = 1) -> list[int]:
    cleaned = sorted({max(minimum, int(value)) for value in values})
    if cleaned:
        return cleaned
    return [max(minimum, fallback)]


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile for empty values.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _aggregate(records: list[BenchmarkRecord]) -> BenchmarkAggregate:
    first = records[0]
    ok_records = [record for record in records if record.status == "ok"]
    successful = [record.total_ms for record in ok_records]
    sorted_success = sorted(successful)
    sample_count = len(records)
    success_count = len(sorted_success)

    if not sorted_success:
        return BenchmarkAggregate(
            combo_key=first.combo_key,
            target_height=first.target_height,
            request_concurrency=first.request_concurrency,
            segment_workers=first.segment_workers,
            sample_count=sample_count,
            success_count=0,
            unique_output_count=0,
            output_consistency_ratio=None,
        )

    output_values = [record.output_hash or "__empty_output__" for record in ok_records]
    output_counts = Counter(output_values)
    dominant_output_count = max(output_counts.values())

    return BenchmarkAggregate(
        combo_key=first.combo_key,
        target_height=first.target_height,
        request_concurrency=first.request_concurrency,
        segment_workers=first.segment_workers,
        sample_count=sample_count,
        success_count=success_count,
        p50_total_ms=_percentile(sorted_success, 0.50),
        p95_total_ms=_percentile(sorted_success, 0.95),
        min_total_ms=min(sorted_success),
        max_total_ms=max(sorted_success),
        avg_total_ms=statistics.fmean(sorted_success),
        unique_output_count=len(output_counts),
        output_consistency_ratio=dominant_output_count / success_count,
    )


def run_benchmark(
    request: BenchmarkRequest,
    *,
    benchmark_id: str,
    executor: RunExecutor | None = None,
) -> BenchmarkResult:
    """Execute sweep combinations and return benchmark records + aggregates."""
    run_executor = executor or execute_run
    settings = get_settings()

    heights = _sanitize_int_list(
        request.resolution_heights,
        fallback=request.base_run.target_height,
        minimum=64,
    )
    concurrency_values = _sanitize_int_list(
        request.request_concurrency,
        fallback=1,
        minimum=1,
    )
    segment_values = _sanitize_int_list(
        request.segment_workers,
        fallback=request.base_run.segment_workers,
        minimum=1,
    )

    all_records: list[BenchmarkRecord] = []

    for target_height, request_concurrency, segment_workers in product(
        heights,
        concurrency_values,
        segment_values,
    ):
        combo_key = f"h{target_height}-rc{request_concurrency}-sw{segment_workers}"

        combo_base = request.base_run.model_copy(deep=True)
        combo_base.target_height = target_height
        combo_base.segment_workers = segment_workers

        for _ in range(request.warmup_runs):
            run_executor(combo_base)

        max_workers = min(
            request_concurrency,
            settings.max_benchmark_workers,
            request.repeats,
        )
        max_workers = max(1, max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_executor, combo_base.model_copy(deep=True)): repeat_index
                for repeat_index in range(1, request.repeats + 1)
            }
            for future in as_completed(futures):
                repeat_index = futures[future]
                record = BenchmarkRecord(
                    benchmark_id=benchmark_id,
                    combo_key=combo_key,
                    run_id=f"run_{uuid4().hex}",
                    repeat_index=repeat_index,
                    target_height=target_height,
                    request_concurrency=request_concurrency,
                    segment_workers=segment_workers,
                    status="ok",
                    preprocess_ms=0.0,
                    request_ms=0.0,
                    total_ms=0.0,
                    ttft_ms=None,
                    error=None,
                )

                try:
                    execution = future.result()
                    normalized_output = execution.output_text.strip()
                    record.preprocess_ms = execution.preprocess_ms
                    record.request_ms = execution.request_ms
                    record.total_ms = execution.total_ms
                    record.ttft_ms = execution.ttft_ms
                    record.output_chars = len(normalized_output)
                    record.output_hash = (
                        sha256(normalized_output.encode("utf-8")).hexdigest()
                        if normalized_output
                        else None
                    )
                except Exception as exc:  # noqa: BLE001
                    record.status = "error"
                    record.error = str(exc)
                    if not request.continue_on_error:
                        raise RuntimeError(f"Benchmark run failed for {combo_key}: {exc}") from exc

                all_records.append(record)

    grouped: dict[str, list[BenchmarkRecord]] = {}
    for record in all_records:
        grouped.setdefault(record.combo_key, []).append(record)

    aggregates = [_aggregate(records) for records in grouped.values()]

    statuses = {record.status for record in all_records}
    status: Literal["ok", "partial", "error"]
    if statuses == {"ok"}:
        status = "ok"
    elif "ok" in statuses:
        status = "partial"
    else:
        status = "error"

    return BenchmarkResult(
        benchmark_id=benchmark_id,
        status=status,
        created_at="",
        request=request,
        records=all_records,
        aggregates=sorted(aggregates, key=lambda item: item.combo_key),
        artifact_paths={},
    )
