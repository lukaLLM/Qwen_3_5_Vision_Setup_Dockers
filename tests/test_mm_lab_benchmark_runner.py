"""Tests for MM lab benchmark orchestration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.benchmark_runner import run_benchmark  # noqa: E402
from visual_experimentation_app.schemas import BenchmarkRequest, RunRequest  # noqa: E402
from visual_experimentation_app.vllm_client import RunExecution  # noqa: E402


class BenchmarkRunnerTest(unittest.TestCase):
    """Covers benchmark grid expansion and aggregate generation."""

    def test_run_benchmark_expands_grid(self) -> None:
        """Grid expansion should create expected run and aggregate counts."""
        base = RunRequest(prompt="hello")
        request = BenchmarkRequest(
            base_run=base,
            repeats=2,
            warmup_runs=0,
            resolution_heights=[360, 480],
            request_concurrency=[1, 2],
            segment_workers=[1],
            continue_on_error=True,
        )

        counter = {"index": 0}

        def fake_executor(_: RunRequest) -> RunExecution:
            counter["index"] += 1
            elapsed = float(counter["index"])
            return RunExecution(
                output_text="ok",
                preprocess_ms=10.0,
                request_ms=20.0,
                total_ms=elapsed,
                ttft_ms=5.0,
                effective_params={},
                media_metadata={},
            )

        result = run_benchmark(request, benchmark_id="bench_test", executor=fake_executor)
        self.assertEqual(result.status, "ok")
        # 2 heights * 2 concurrency * 1 segment_workers * 2 repeats.
        self.assertEqual(len(result.records), 8)
        self.assertEqual(len(result.aggregates), 4)
        for aggregate in result.aggregates:
            self.assertEqual(aggregate.unique_output_count, 1)
            self.assertEqual(aggregate.output_consistency_ratio, 1.0)

    def test_run_benchmark_respects_continue_on_error_false(self) -> None:
        """Runner should raise immediately when continue_on_error is disabled."""
        base = RunRequest(prompt="hello")
        request = BenchmarkRequest(
            base_run=base,
            repeats=1,
            warmup_runs=0,
            resolution_heights=[360],
            request_concurrency=[1],
            segment_workers=[1],
            continue_on_error=False,
        )

        def failing_executor(_: RunRequest) -> RunExecution:
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            run_benchmark(request, benchmark_id="bench_fail", executor=failing_executor)


if __name__ == "__main__":
    unittest.main()
