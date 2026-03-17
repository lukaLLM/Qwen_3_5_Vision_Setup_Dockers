"""Tests for MM lab API routes."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.api import create_app  # noqa: E402
from visual_experimentation_app.config import clear_settings_cache  # noqa: E402
from visual_experimentation_app.schemas import (  # noqa: E402
    BenchmarkRequest,
    RunRequest,
)
from visual_experimentation_app.vllm_client import RunExecution  # noqa: E402


class MmLabApiTest(unittest.TestCase):
    """Covers core local API behavior."""

    def setUp(self) -> None:
        """Build isolated app client with temp results storage."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(
            os.environ,
            {"MM_LAB_RESULTS_DIR": self.tmp_dir.name},
            clear=False,
        )
        self.env_patch.start()
        clear_settings_cache()
        self.client = TestClient(create_app(include_ui=False))

    def tearDown(self) -> None:
        """Release temp resources and clear cached settings."""
        self.client.close()
        self.env_patch.stop()
        self.tmp_dir.cleanup()
        clear_settings_cache()

    def test_health(self) -> None:
        """Health endpoint returns local API metadata."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("/api", payload["api_prefix"])

    def test_run_success(self) -> None:
        """Run endpoint returns successful payload when execution succeeds."""
        with mock.patch(
            "visual_experimentation_app.api.execute_run",
            return_value=RunExecution(
                output_text="done",
                preprocess_ms=1.0,
                request_ms=2.0,
                total_ms=3.0,
                ttft_ms=0.5,
                effective_params={"model": "Qwen/Qwen3.5-4B"},
                media_metadata={},
            ),
        ):
            response = self.client.post("/api/run", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["output_text"], "done")

    def test_run_detail_404(self) -> None:
        """Run detail endpoint returns 404 for unknown run IDs."""
        response = self.client.get("/api/runs/not_found")
        self.assertEqual(response.status_code, 404)

    def test_benchmark_error_returns_error_status(self) -> None:
        """Benchmark endpoint returns status=error when runner raises."""
        benchmark_request = BenchmarkRequest(base_run=RunRequest(prompt="x"))
        with mock.patch(
            "visual_experimentation_app.api.run_benchmark",
            side_effect=RuntimeError("benchmark failed"),
        ):
            response = self.client.post(
                "/api/benchmark",
                json=benchmark_request.model_dump(),
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertIn("error", payload["artifact_paths"])


if __name__ == "__main__":
    unittest.main()
