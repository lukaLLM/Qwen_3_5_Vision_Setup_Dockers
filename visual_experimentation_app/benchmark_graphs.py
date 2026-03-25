"""Benchmark graph data assembly helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from visual_experimentation_app.schemas import BenchmarkResult

GraphFrames = dict[str, pd.DataFrame]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _combo_label(
    *,
    target_height: int,
    segment_workers: int,
    request_concurrency: int,
    segmentation_mode: str,
) -> str:
    mode_tag = "seg" if segmentation_mode == "segmented" else "no-seg"
    return f"{mode_tag}-h{target_height}-cp{request_concurrency}"


def build_graph_frames(result: BenchmarkResult) -> GraphFrames:
    """Build canonical dataframes for benchmark visualizations."""
    latency_by_concurrency_rows: list[dict[str, object]] = []
    throughput_by_concurrency_rows: list[dict[str, object]] = []
    completion_time_rows: list[dict[str, object]] = []

    sorted_aggregates = sorted(
        result.aggregates,
        key=lambda item: (
            int(item.target_height),
            int(item.segment_workers),
            int(item.request_concurrency),
        ),
    )
    for aggregate in sorted_aggregates:
        target_height = int(aggregate.target_height)
        segment_workers = int(aggregate.segment_workers)
        request_concurrency = int(aggregate.request_concurrency)
        segmentation_mode = str(aggregate.segmentation_mode)
        latency_ms = float(aggregate.p50_total_ms) if aggregate.p50_total_ms is not None else None
        throughput_tokens_per_sec = (
            float(aggregate.throughput_tokens_per_sec)
            if aggregate.throughput_tokens_per_sec is not None
            else None
        )
        completion_time_ms = (
            float(aggregate.p50_total_ms)
            if aggregate.p50_total_ms is not None
            else (
                float(aggregate.avg_total_ms)
                if aggregate.avg_total_ms is not None
                else None
            )
        )

        mode_label = "Segmented" if segmentation_mode == "segmented" else "Non-Segmented"
        series_label = f"{target_height}px / {mode_label}"
        resolution_label = f"{target_height}px"
        combo_label = _combo_label(
            target_height=target_height,
            segment_workers=segment_workers,
            request_concurrency=request_concurrency,
            segmentation_mode=segmentation_mode,
        )

        if latency_ms is not None:
            latency_by_concurrency_rows.append(
                {
                    "request_concurrency": request_concurrency,
                    "target_height": target_height,
                    "segment_workers": segment_workers,
                    "segmentation_mode": segmentation_mode,
                    "resolution_label": resolution_label,
                    "series_label": series_label,
                    "combo_label": combo_label,
                    "latency_ms": latency_ms,
                }
            )

        if throughput_tokens_per_sec is not None:
            throughput_by_concurrency_rows.append(
                {
                    "request_concurrency": request_concurrency,
                    "target_height": target_height,
                    "segment_workers": segment_workers,
                    "segmentation_mode": segmentation_mode,
                    "resolution_label": resolution_label,
                    "series_label": series_label,
                    "combo_label": combo_label,
                    "throughput_tokens_per_sec": throughput_tokens_per_sec,
                }
            )

        if completion_time_ms is not None:
            completion_time_rows.append(
                {
                    "config_label": combo_label,
                    "combo_label": combo_label,
                    "target_height": target_height,
                    "segment_workers": segment_workers,
                    "segmentation_mode": segmentation_mode,
                    "request_concurrency": request_concurrency,
                    "resolution_label": resolution_label,
                    "completion_time_ms": completion_time_ms,
                }
            )

    time_split_rows: list[dict[str, object]] = []
    ok_records = [record for record in result.records if record.status == "ok"]
    combo_keys = sorted(
        {
            (
                int(record.target_height),
                int(record.segment_workers),
                int(record.request_concurrency),
                str(record.segmentation_mode),
            )
            for record in ok_records
        }
    )
    for target_height, segment_workers, request_concurrency, segmentation_mode in combo_keys:
        combo_records = [
            record
            for record in ok_records
            if int(record.target_height) == target_height
            and int(record.segment_workers) == segment_workers
            and int(record.request_concurrency) == request_concurrency
            and str(record.segmentation_mode) == segmentation_mode
        ]
        preprocess_values = [
            float(record.preprocess_pct)
            for record in combo_records
            if record.preprocess_pct is not None
        ]
        request_values = [
            float(record.request_pct)
            for record in combo_records
            if record.request_pct is not None
        ]
        combo_label = _combo_label(
            target_height=target_height,
            segment_workers=segment_workers,
            request_concurrency=request_concurrency,
            segmentation_mode=segmentation_mode,
        )
        if preprocess_values:
            time_split_rows.append(
                {
                    "combo_label": combo_label,
                    "target_height": target_height,
                    "segment_workers": segment_workers,
                    "segmentation_mode": segmentation_mode,
                    "request_concurrency": request_concurrency,
                    "stage": "preprocess",
                    "pct_value": (sum(preprocess_values) / len(preprocess_values)) * 100.0,
                }
            )
        if request_values:
            time_split_rows.append(
                {
                    "combo_label": combo_label,
                    "target_height": target_height,
                    "segment_workers": segment_workers,
                    "segmentation_mode": segmentation_mode,
                    "request_concurrency": request_concurrency,
                    "stage": "request",
                    "pct_value": (sum(request_values) / len(request_values)) * 100.0,
                }
            )

    latency_df = (
        pd.DataFrame(latency_by_concurrency_rows)
        if latency_by_concurrency_rows
        else _empty_frame(
            [
                "request_concurrency",
                "target_height",
                "segment_workers",
                "segmentation_mode",
                "resolution_label",
                "series_label",
                "combo_label",
                "latency_ms",
            ]
        )
    )
    throughput_df = (
        pd.DataFrame(throughput_by_concurrency_rows)
        if throughput_by_concurrency_rows
        else _empty_frame(
            [
                "request_concurrency",
                "target_height",
                "segment_workers",
                "segmentation_mode",
                "resolution_label",
                "series_label",
                "combo_label",
                "throughput_tokens_per_sec",
            ]
        )
    )
    completion_time_df = (
        pd.DataFrame(completion_time_rows)
        if completion_time_rows
        else _empty_frame(
            [
                "config_label",
                "combo_label",
                "target_height",
                "segment_workers",
                "segmentation_mode",
                "request_concurrency",
                "resolution_label",
                "completion_time_ms",
            ]
        )
    )
    time_split_df = (
        pd.DataFrame(time_split_rows)
        if time_split_rows
        else _empty_frame(
            [
                "combo_label",
                "target_height",
                "segment_workers",
                "segmentation_mode",
                "request_concurrency",
                "stage",
                "pct_value",
            ]
        )
    )

    return {
        "latency_by_concurrency": latency_df.sort_values(
            ["target_height", "segmentation_mode", "request_concurrency"]
        ),
        "throughput_by_concurrency": throughput_df.sort_values(
            ["target_height", "segmentation_mode", "request_concurrency"]
        ),
        "completion_time_ms": completion_time_df.sort_values(
            ["target_height", "segmentation_mode", "request_concurrency"]
        ),
        "time_split_stacked": time_split_df.sort_values(
            ["segmentation_mode", "combo_label", "stage"]
        ),
    }


def save_graph_csv_artifacts(
    *,
    result: BenchmarkResult,
    benchmark_dir: Path,
) -> dict[str, str]:
    """Persist graph dataframes as CSV artifacts and return their paths."""
    frames = build_graph_frames(result)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    path_specs = {
        "graph_latency_by_concurrency_csv": benchmark_dir
        / f"{result.benchmark_id}_graph_latency_by_concurrency.csv",
        "graph_throughput_by_concurrency_csv": benchmark_dir
        / f"{result.benchmark_id}_graph_throughput_by_concurrency.csv",
        "graph_completion_time_ms_csv": benchmark_dir
        / f"{result.benchmark_id}_graph_completion_time_ms.csv",
        "graph_time_split_stacked_csv": benchmark_dir
        / f"{result.benchmark_id}_graph_time_split_stacked.csv",
    }

    frames["latency_by_concurrency"].to_csv(
        path_specs["graph_latency_by_concurrency_csv"],
        index=False,
    )
    frames["throughput_by_concurrency"].to_csv(
        path_specs["graph_throughput_by_concurrency_csv"],
        index=False,
    )
    frames["completion_time_ms"].to_csv(
        path_specs["graph_completion_time_ms_csv"],
        index=False,
    )
    frames["time_split_stacked"].to_csv(
        path_specs["graph_time_split_stacked_csv"],
        index=False,
    )

    artifacts = {key: str(path) for key, path in path_specs.items()}
    # Backward-compatible alias for older consumers.
    artifacts["graph_completion_time_speedup_csv"] = artifacts["graph_completion_time_ms_csv"]
    return artifacts
