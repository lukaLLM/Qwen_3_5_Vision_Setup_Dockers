#!/usr/bin/env python3
"""Run a repeatable MM Lab video performance test suite for demo/reporting."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=body,
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_csv_ints(raw: str, *, minimum: int = 1) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        part = chunk.strip()
        if not part:
            continue
        values.append(max(minimum, int(part)))
    if not values:
        return [minimum]
    return sorted(set(values))


def _best_aggregate(aggregates: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in aggregates if item.get("p50_total_ms") is not None]
    if not valid:
        return None
    return min(valid, key=lambda item: float(item.get("p50_total_ms", 1e30)))


def _build_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_480p_fps2",
            "base_run_overrides": {
                "target_height": 480,
                "safe_video_sampling": False,
                "video_sampling_fps": 2.0,
                "disable_caching": False,
                "segment_max_duration_s": 0.0,
                "segment_overlap_s": 0.0,
            },
            "request_concurrency": [1],
            "segment_workers": [1],
        },
        {
            "name": "faster_360p_fps1",
            "base_run_overrides": {
                "target_height": 360,
                "safe_video_sampling": False,
                "video_sampling_fps": 1.0,
                "disable_caching": False,
                "segment_max_duration_s": 0.0,
                "segment_overlap_s": 0.0,
            },
            "request_concurrency": [1],
            "segment_workers": [1],
        },
        {
            "name": "segmented_parallel_480p",
            "base_run_overrides": {
                "target_height": 480,
                "safe_video_sampling": False,
                "video_sampling_fps": 2.0,
                "disable_caching": False,
                "segment_max_duration_s": 30.0,
                "segment_overlap_s": 2.0,
            },
            "request_concurrency": [1],
            "segment_workers": [1, 2],
        },
        {
            "name": "throughput_concurrency_480p",
            "base_run_overrides": {
                "target_height": 480,
                "safe_video_sampling": False,
                "video_sampling_fps": 2.0,
                "disable_caching": False,
                "segment_max_duration_s": 0.0,
                "segment_overlap_s": 0.0,
            },
            "request_concurrency": [1, 2],
            "segment_workers": [1],
        },
        {
            "name": "request_nocache_480p",
            "base_run_overrides": {
                "target_height": 480,
                "safe_video_sampling": False,
                "video_sampling_fps": 2.0,
                "disable_caching": True,
                "segment_max_duration_s": 0.0,
                "segment_overlap_s": 0.0,
            },
            "request_concurrency": [1],
            "segment_workers": [1],
        },
    ]


def main() -> int:
    """Run scenario benchmarks and write a JSON summary report."""
    parser = argparse.ArgumentParser(description="MM Lab video performance suite.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7870", help="MM Lab base URL.")
    parser.add_argument("--api-prefix", default="/api", help="API prefix (default: /api).")
    parser.add_argument("--video", required=True, help="Path to local test video.")
    parser.add_argument("--image", action="append", default=[], help="Optional image path(s).")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Model id.")
    parser.add_argument("--prompt", default="Summarize key events and important visual details.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per benchmark combo.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per combo.")
    parser.add_argument("--timeout-seconds", type=float, default=600.0, help="Request timeout.")
    parser.add_argument(
        "--report-dir",
        default="visual_experimentation_app/results/benchmarks",
        help="Directory for saving summary JSON report.",
    )
    parser.add_argument(
        "--extra-concurrency",
        default="",
        help="Optional extra request concurrency CSV to merge into throughput scenario.",
    )
    args = parser.parse_args()

    prefix = args.api_prefix.strip() or "/api"
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    if len(prefix) > 1:
        prefix = prefix.rstrip("/")

    health_url = f"{args.base_url.rstrip('/')}{prefix}/health"
    benchmark_url = f"{args.base_url.rstrip('/')}{prefix}/benchmark"

    try:
        health = _json_get(health_url)
    except urllib.error.URLError as exc:
        print(f"[mm-lab-perf] Health check failed: {exc}")
        return 1

    print("[mm-lab-perf] Health OK")
    print(json.dumps(health, indent=2))

    scenarios = _build_scenarios()
    if args.extra_concurrency.strip():
        extra = _parse_csv_ints(args.extra_concurrency)
        for scenario in scenarios:
            if scenario["name"] == "throughput_concurrency_480p":
                scenario["request_concurrency"] = sorted(
                    set(scenario["request_concurrency"] + extra)
                )

    report_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        base_run = {
            "prompt": args.prompt,
            "text_input": None,
            "image_paths": args.image,
            "video_path": args.video,
            "model": args.model,
            "timeout_seconds": args.timeout_seconds,
            "use_model_defaults": False,
            "max_tokens": 81920,
            "max_completion_tokens": 81920,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "frequency_penalty": 0.0,
            "thinking_mode": "auto",
            "show_reasoning": False,
            "measure_ttft": True,
            "preprocess_images": True,
            "preprocess_video": True,
            "request_extra_body": {
                "top_k": 20,
                "mm_processor_kwargs": {"fps": 2, "do_sample_frames": True},
            },
            "request_extra_headers": {},
        }
        base_run.update(scenario["base_run_overrides"])

        payload = {
            "base_run": base_run,
            "repeats": max(1, args.repeats),
            "warmup_runs": max(0, args.warmup),
            "resolution_heights": [base_run["target_height"]],
            "request_concurrency": scenario["request_concurrency"],
            "segment_workers": scenario["segment_workers"],
            "continue_on_error": True,
            "label": scenario["name"],
        }

        print(f"\n[mm-lab-perf] Running scenario: {scenario['name']}")
        try:
            result = _json_post(benchmark_url, payload)
        except urllib.error.URLError as exc:
            print(f"[mm-lab-perf] Scenario failed ({scenario['name']}): {exc}")
            report_rows.append({"scenario": scenario["name"], "status": "error", "error": str(exc)})
            continue

        aggregates = result.get("aggregates", [])
        best = _best_aggregate(aggregates if isinstance(aggregates, list) else [])
        row: dict[str, Any] = {
            "scenario": scenario["name"],
            "status": result.get("status", "unknown"),
            "benchmark_id": result.get("benchmark_id", ""),
        }
        if best:
            row.update(
                {
                    "combo_key": best.get("combo_key"),
                    "p50_total_ms": best.get("p50_total_ms"),
                    "p95_total_ms": best.get("p95_total_ms"),
                    "avg_total_ms": best.get("avg_total_ms"),
                    "output_consistency_ratio": best.get("output_consistency_ratio"),
                    "unique_output_count": best.get("unique_output_count"),
                }
            )
        report_rows.append(row)
        print(json.dumps(row, indent=2))

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "api_prefix": prefix,
        "model": args.model,
        "video": args.video,
        "images": args.image,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "rows": report_rows,
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"video_perf_suite_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[mm-lab-perf] Report written to: {report_path}")
    print("[mm-lab-perf] Scenario summary (lower p50 is better):")
    for row in report_rows:
        print(
            f"- {row.get('scenario')}: status={row.get('status')}, "
            f"p50={row.get('p50_total_ms')}, p95={row.get('p95_total_ms')}, "
            f"consistency={row.get('output_consistency_ratio')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
