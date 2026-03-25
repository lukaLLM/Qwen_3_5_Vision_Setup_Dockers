"""Tests for MM lab benchmark orchestration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

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
            include_non_segmented_baseline=False,
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
                prompt_tokens=11,
                output_tokens=22,
                total_tokens=33,
            )

        result = run_benchmark(request, benchmark_id="bench_test", executor=fake_executor)
        self.assertEqual(result.status, "ok")
        # Base run has segmentation off -> one baseline per height.
        # 2 heights * 1 baseline combo * 2 repeats.
        self.assertEqual(len(result.records), 4)
        self.assertEqual(len(result.aggregates), 2)
        for record in result.records:
            self.assertEqual(record.prompt_tokens, 11)
            self.assertEqual(record.output_tokens, 22)
            self.assertEqual(record.total_tokens, 33)
            self.assertIsNotNone(record.preprocess_pct)
            self.assertIsNotNone(record.request_pct)
            self.assertIsNotNone(record.ms_per_output_token)
            self.assertIsNotNone(record.ms_per_100_output_tokens)
        for aggregate in result.aggregates:
            self.assertEqual(aggregate.unique_output_count, 1)
            self.assertEqual(aggregate.output_consistency_ratio, 1.0)
            self.assertIsNotNone(aggregate.experiment_wall_time_ms)
            self.assertIsNotNone(aggregate.throughput_requests_per_sec)
            self.assertIsNotNone(aggregate.total_output_tokens_across_all_parallel_requests)
            self.assertIsNotNone(aggregate.throughput_tokens_per_sec)
            self.assertEqual(aggregate.token_metrics_coverage_ratio, 1.0)

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

    def test_run_benchmark_token_coverage_ratio_handles_partial_usage(self) -> None:
        """Aggregate coverage ratio should reflect missing token usage rows."""
        base = RunRequest(prompt="hello")
        request = BenchmarkRequest(
            base_run=base,
            repeats=4,
            warmup_runs=0,
            resolution_heights=[480],
            request_concurrency=[2],
            segment_workers=[1],
            continue_on_error=True,
        )

        counter = {"index": 0}

        def fake_executor(_: RunRequest) -> RunExecution:
            counter["index"] += 1
            idx = counter["index"]
            has_usage = (idx % 2) == 0
            return RunExecution(
                output_text=f"run-{idx}",
                preprocess_ms=5.0,
                request_ms=10.0,
                total_ms=15.0,
                ttft_ms=2.0,
                effective_params={},
                media_metadata={},
                prompt_tokens=(10 if has_usage else None),
                output_tokens=(20 if has_usage else None),
                total_tokens=(30 if has_usage else None),
            )

        result = run_benchmark(request, benchmark_id="bench_partial_tokens", executor=fake_executor)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.aggregates), 1)
        aggregate = result.aggregates[0]
        self.assertEqual(aggregate.success_count, 4)
        self.assertEqual(aggregate.total_output_tokens_across_all_parallel_requests, 40)
        self.assertEqual(aggregate.token_metrics_coverage_ratio, 0.5)

    def test_run_benchmark_waits_between_combos(self) -> None:
        """Runner should sleep between combo groups when wait is configured."""
        base = RunRequest(prompt="hello")
        request = BenchmarkRequest(
            base_run=base,
            repeats=1,
            warmup_runs=0,
            resolution_heights=[360, 480],
            request_concurrency=[1],
            segment_workers=[1],
            wait_between_combos_s=0.25,
            continue_on_error=True,
        )

        def fake_executor(_: RunRequest) -> RunExecution:
            return RunExecution(
                output_text="ok",
                preprocess_ms=1.0,
                request_ms=1.0,
                total_ms=2.0,
                ttft_ms=0.5,
                effective_params={},
                media_metadata={},
                prompt_tokens=1,
                output_tokens=1,
                total_tokens=2,
            )

        with mock.patch("visual_experimentation_app.benchmark_runner.sleep") as mock_sleep:
            run_benchmark(request, benchmark_id="bench_wait", executor=fake_executor)

        self.assertEqual(mock_sleep.call_count, 1)
        mock_sleep.assert_called_with(0.25)

    def test_run_benchmark_ignores_segment_workers_sweep_in_chunk_mode(self) -> None:
        """Segment workers sweep should not multiply benchmark combos in chunk mode."""
        base = RunRequest(
            prompt="hello",
            video_paths=["/tmp/video.mp4"],
            segment_max_duration_s=30.0,
            segment_overlap_s=2.0,
        )
        request = BenchmarkRequest(
            base_run=base,
            repeats=1,
            warmup_runs=0,
            resolution_heights=[360],
            request_concurrency=[1, 2, 4],
            segment_workers=[1, 2, 8],
            include_non_segmented_baseline=False,
            continue_on_error=True,
        )

        call_count = {"value": 0}

        def fake_executor(_: RunRequest) -> RunExecution:
            call_count["value"] += 1
            return RunExecution(
                output_text="ok",
                preprocess_ms=1.0,
                request_ms=1.0,
                total_ms=2.0,
                ttft_ms=0.5,
                effective_params={},
                media_metadata={},
                prompt_tokens=1,
                output_tokens=1,
                total_tokens=2,
            )

        result = run_benchmark(
            request,
            benchmark_id="bench_ignore_segment_sweep",
            executor=fake_executor,
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.aggregates), 3)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(call_count["value"], 3)

    def test_run_benchmark_can_include_segmented_and_non_segmented_modes(self) -> None:
        """Benchmark can include both segmentation modes when requested."""
        base = RunRequest(
            prompt="hello",
            video_paths=["/tmp/video.mp4"],
            segment_max_duration_s=30.0,
            segment_overlap_s=2.0,
        )
        request = BenchmarkRequest(
            base_run=base,
            repeats=1,
            warmup_runs=0,
            resolution_heights=[480],
            request_concurrency=[1, 2],
            include_non_segmented_baseline=True,
            continue_on_error=True,
        )

        def fake_executor(_: RunRequest) -> RunExecution:
            return RunExecution(
                output_text="ok",
                preprocess_ms=1.0,
                request_ms=2.0,
                total_ms=3.0,
                ttft_ms=0.5,
                effective_params={},
                media_metadata={},
                prompt_tokens=1,
                output_tokens=1,
                total_tokens=2,
            )

        result = run_benchmark(request, benchmark_id="bench_modes", executor=fake_executor)
        self.assertEqual(result.status, "ok")
        # Segmented sweeps all chunk-parallel values, non-segmented baseline runs once.
        self.assertEqual(len(result.aggregates), 3)
        self.assertEqual(len(result.records), 3)
        modes = {record.segmentation_mode for record in result.records}
        self.assertEqual(modes, {"segmented", "non_segmented"})

    def test_run_benchmark_skips_duplicate_non_segmented_when_base_is_off(self) -> None:
        """If base run segmentation is off, only non-segmented mode should run."""
        base = RunRequest(prompt="hello", video_paths=["/tmp/video.mp4"], segment_max_duration_s=0.0)
        request = BenchmarkRequest(
            base_run=base,
            repeats=1,
            warmup_runs=0,
            resolution_heights=[480],
            request_concurrency=[1, 2],
            include_non_segmented_baseline=True,
            continue_on_error=True,
        )

        def fake_executor(_: RunRequest) -> RunExecution:
            return RunExecution(
                output_text="ok",
                preprocess_ms=1.0,
                request_ms=2.0,
                total_ms=3.0,
                ttft_ms=0.5,
                effective_params={},
                media_metadata={},
                prompt_tokens=1,
                output_tokens=1,
                total_tokens=2,
            )

        result = run_benchmark(request, benchmark_id="bench_modes_off", executor=fake_executor)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.aggregates), 1)
        self.assertEqual(len(result.records), 1)
        modes = {record.segmentation_mode for record in result.records}
        self.assertEqual(modes, {"non_segmented"})


if __name__ == "__main__":
    unittest.main()
