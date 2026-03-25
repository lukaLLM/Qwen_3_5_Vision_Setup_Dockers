"""Benchmark orchestration for sweep-based multimodal experiments."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable
from hashlib import sha256
from time import perf_counter, sleep
from typing import Literal
from uuid import uuid4

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


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _safe_ms_per_output_token(total_ms: float, output_tokens: int | None) -> float | None:
    if output_tokens is None or output_tokens <= 0:
        return None
    return total_ms / float(output_tokens)


def _safe_ms_per_100_output_tokens(total_ms: float, output_tokens: int | None) -> float | None:
    if output_tokens is None or output_tokens <= 0:
        return None
    return (100.0 * total_ms) / float(output_tokens)


def _aggregate(
    records: list[BenchmarkRecord],
    *,
    experiment_wall_time_ms: float | None,
) -> BenchmarkAggregate:
    first = records[0]
    ok_records = [record for record in records if record.status == "ok"]
    successful = [record.total_ms for record in ok_records]
    sorted_success = sorted(successful)
    sample_count = len(records)
    success_count = len(sorted_success)
    wall_seconds = (
        (experiment_wall_time_ms / 1000.0)
        if experiment_wall_time_ms is not None and experiment_wall_time_ms > 0
        else None
    )

    if not sorted_success:
        return BenchmarkAggregate(
            combo_key=first.combo_key,
            target_height=first.target_height,
            request_concurrency=first.request_concurrency,
            segment_workers=first.segment_workers,
            segmentation_mode=first.segmentation_mode,
            sample_count=sample_count,
            success_count=0,
            unique_output_count=0,
            output_consistency_ratio=None,
            experiment_wall_time_ms=experiment_wall_time_ms,
            total_output_tokens_across_all_parallel_requests=None,
            throughput_tokens_per_sec=None,
            throughput_requests_per_sec=(0.0 if wall_seconds is not None else None),
            token_metrics_coverage_ratio=None,
        )

    output_values = [record.output_hash or "__empty_output__" for record in ok_records]
    output_counts = Counter(output_values)
    dominant_output_count = max(output_counts.values())
    records_with_output_tokens = [
        record for record in ok_records if record.output_tokens is not None
    ]
    total_output_tokens = (
        sum(int(record.output_tokens or 0) for record in records_with_output_tokens)
        if records_with_output_tokens
        else None
    )
    token_coverage_ratio = len(records_with_output_tokens) / success_count
    throughput_requests_per_sec = (
        success_count / wall_seconds if wall_seconds is not None else None
    )
    throughput_tokens_per_sec = (
        total_output_tokens / wall_seconds
        if wall_seconds is not None and total_output_tokens is not None
        else None
    )

    return BenchmarkAggregate(
        combo_key=first.combo_key,
        target_height=first.target_height,
        request_concurrency=first.request_concurrency,
        segment_workers=first.segment_workers,
        segmentation_mode=first.segmentation_mode,
        sample_count=sample_count,
        success_count=success_count,
        p50_total_ms=_percentile(sorted_success, 0.50),
        p95_total_ms=_percentile(sorted_success, 0.95),
        min_total_ms=min(sorted_success),
        max_total_ms=max(sorted_success),
        avg_total_ms=statistics.fmean(sorted_success),
        unique_output_count=len(output_counts),
        output_consistency_ratio=dominant_output_count / success_count,
        experiment_wall_time_ms=experiment_wall_time_ms,
        total_output_tokens_across_all_parallel_requests=total_output_tokens,
        throughput_tokens_per_sec=throughput_tokens_per_sec,
        throughput_requests_per_sec=throughput_requests_per_sec,
        token_metrics_coverage_ratio=token_coverage_ratio,
    )


def run_benchmark(
    request: BenchmarkRequest,
    *,
    benchmark_id: str,
    executor: RunExecutor | None = None,
) -> BenchmarkResult:
    """Execute sweep combinations and return benchmark records + aggregates."""
    run_executor = executor or execute_run

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
    segmented_enabled = request.base_run.segment_max_duration_s > 0
    baseline_chunk_parallel = concurrency_values[0]

    all_records: list[BenchmarkRecord] = []
    combo_wall_times_ms: dict[str, float] = {}

    combos: list[tuple[int, int, str]] = []
    for target_height in heights:
        if segmented_enabled:
            for chunk_parallel in concurrency_values:
                combos.append((target_height, chunk_parallel, "segmented"))
            if request.include_non_segmented_baseline:
                # Non-segmented baseline is executed once per height because
                # chunk parallelism has no effect when segmentation is disabled.
                combos.append((target_height, baseline_chunk_parallel, "non_segmented"))
            continue

        # Base request already has segmentation off, so run one baseline only.
        combos.append((target_height, baseline_chunk_parallel, "non_segmented"))
    for combo_index, (target_height, chunk_parallel, segmentation_mode) in enumerate(combos):
        combo_key = f"m{segmentation_mode}-h{target_height}-rc{chunk_parallel}-sw{chunk_parallel}"

        combo_base = request.base_run.model_copy(deep=True)
        combo_base.target_height = target_height
        combo_base.segment_workers = chunk_parallel
        if segmentation_mode == "non_segmented":
            combo_base.segment_max_duration_s = 0.0
            combo_base.segment_overlap_s = 0.0

        for _ in range(request.warmup_runs):
            run_executor(combo_base)

        combo_started = perf_counter()
        for repeat_index in range(1, request.repeats + 1):
            record = BenchmarkRecord(
                benchmark_id=benchmark_id,
                combo_key=combo_key,
                run_id=f"run_{uuid4().hex}",
                repeat_index=repeat_index,
                target_height=target_height,
                request_concurrency=chunk_parallel,
                segment_workers=chunk_parallel,
                segmentation_mode=segmentation_mode,
                status="ok",
                preprocess_ms=0.0,
                request_ms=0.0,
                total_ms=0.0,
                ttft_ms=None,
                prompt_tokens=None,
                output_tokens=None,
                total_tokens=None,
                preprocess_pct=None,
                request_pct=None,
                ms_per_output_token=None,
                ms_per_100_output_tokens=None,
                error=None,
            )

            try:
                execution = run_executor(combo_base.model_copy(deep=True))
                normalized_output = execution.output_text.strip()
                record.preprocess_ms = execution.preprocess_ms
                record.request_ms = execution.request_ms
                record.total_ms = execution.total_ms
                record.ttft_ms = execution.ttft_ms
                record.prompt_tokens = execution.prompt_tokens
                record.output_tokens = execution.output_tokens
                record.total_tokens = execution.total_tokens
                record.preprocess_pct = _safe_ratio(record.preprocess_ms, record.total_ms)
                record.request_pct = _safe_ratio(record.request_ms, record.total_ms)
                record.ms_per_output_token = _safe_ms_per_output_token(
                    record.total_ms,
                    record.output_tokens,
                )
                record.ms_per_100_output_tokens = _safe_ms_per_100_output_tokens(
                    record.total_ms,
                    record.output_tokens,
                )
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
        combo_wall_times_ms[combo_key] = (perf_counter() - combo_started) * 1000.0
        if request.wait_between_combos_s > 0 and combo_index < (len(combos) - 1):
            sleep(request.wait_between_combos_s)

    grouped: dict[str, list[BenchmarkRecord]] = {}
    for record in all_records:
        grouped.setdefault(record.combo_key, []).append(record)

    aggregates = [
        _aggregate(
            records,
            experiment_wall_time_ms=combo_wall_times_ms.get(combo_key),
        )
        for combo_key, records in grouped.items()
    ]

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
