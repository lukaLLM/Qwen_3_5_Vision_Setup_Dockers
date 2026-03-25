"""Tests for MM lab benchmark artifact persistence."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.config import clear_settings_cache  # noqa: E402
from visual_experimentation_app.result_store import save_benchmark_result  # noqa: E402
from visual_experimentation_app.schemas import (  # noqa: E402
    BenchmarkAggregate,
    BenchmarkRecord,
    BenchmarkRequest,
    BenchmarkResult,
    RunRequest,
)


class ResultStoreBenchmarkArtifactsTest(unittest.TestCase):
    """Covers benchmark CSV + graph dataset artifact persistence."""

    def setUp(self) -> None:
        """Create isolated temp results directory per test."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(
            os.environ,
            {"MM_LAB_RESULTS_DIR": self.tmp_dir.name},
            clear=False,
        )
        self.env_patch.start()
        clear_settings_cache()

    def tearDown(self) -> None:
        """Release test resources and clear settings cache."""
        self.env_patch.stop()
        self.tmp_dir.cleanup()
        clear_settings_cache()

    def test_save_benchmark_result_writes_extended_csv_and_graph_csvs(self) -> None:
        """Persisted artifacts should include new metric columns and graph CSV files."""
        request = BenchmarkRequest(base_run=RunRequest(prompt="hello"), repeats=2)
        records = [
            BenchmarkRecord(
                benchmark_id="bench_test",
                combo_key="h480-rc1-sw1",
                run_id="run_1",
                repeat_index=1,
                target_height=480,
                request_concurrency=1,
                segment_workers=1,
                status="ok",
                preprocess_ms=100.0,
                request_ms=900.0,
                total_ms=1000.0,
                ttft_ms=50.0,
                output_hash="abc",
                output_chars=120,
                prompt_tokens=10,
                output_tokens=20,
                total_tokens=30,
                preprocess_pct=0.1,
                request_pct=0.9,
                ms_per_output_token=50.0,
                ms_per_100_output_tokens=5000.0,
                error=None,
            ),
            BenchmarkRecord(
                benchmark_id="bench_test",
                combo_key="h480-rc2-sw1",
                run_id="run_2",
                repeat_index=1,
                target_height=480,
                request_concurrency=2,
                segment_workers=1,
                status="ok",
                preprocess_ms=120.0,
                request_ms=980.0,
                total_ms=1100.0,
                ttft_ms=60.0,
                output_hash="def",
                output_chars=140,
                prompt_tokens=12,
                output_tokens=25,
                total_tokens=37,
                preprocess_pct=0.109090909,
                request_pct=0.890909091,
                ms_per_output_token=44.0,
                ms_per_100_output_tokens=4400.0,
                error=None,
            ),
        ]
        aggregates = [
            BenchmarkAggregate(
                combo_key="h480-rc1-sw1",
                target_height=480,
                request_concurrency=1,
                segment_workers=1,
                sample_count=1,
                success_count=1,
                p50_total_ms=1000.0,
                p95_total_ms=1000.0,
                min_total_ms=1000.0,
                max_total_ms=1000.0,
                avg_total_ms=1000.0,
                unique_output_count=1,
                output_consistency_ratio=1.0,
                experiment_wall_time_ms=1000.0,
                total_output_tokens_across_all_parallel_requests=20,
                throughput_tokens_per_sec=20.0,
                throughput_requests_per_sec=1.0,
                token_metrics_coverage_ratio=1.0,
            )
        ]

        result = BenchmarkResult(
            benchmark_id="bench_test",
            status="ok",
            created_at="2026-03-21T00:00:00+00:00",
            request=request,
            records=records,
            aggregates=aggregates,
            artifact_paths={},
        )

        paths = save_benchmark_result(result)

        self.assertIn("json", paths)
        self.assertIn("csv", paths)
        self.assertIn("graph_latency_by_concurrency_csv", paths)
        self.assertIn("graph_throughput_by_concurrency_csv", paths)
        self.assertIn("graph_completion_time_ms_csv", paths)
        self.assertIn("graph_completion_time_speedup_csv", paths)
        self.assertIn("graph_time_split_stacked_csv", paths)

        for path in paths.values():
            self.assertTrue(Path(path).exists(), msg=f"missing artifact: {path}")

        csv_header = Path(paths["csv"]).read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("prompt_tokens", csv_header)
        self.assertIn("ms_per_100_output_tokens", csv_header)
        completion_header = (
            Path(paths["graph_completion_time_ms_csv"])
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertIn("completion_time_ms", completion_header)


if __name__ == "__main__":
    unittest.main()
