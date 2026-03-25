"""Tests for MM lab vLLM client error helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.vllm_client import (  # noqa: E402
    TokenUsage,
    _extract_usage_tokens,
    _sum_token_usage,
    build_execution_error_details,
    is_video_processor_error,
    summarize_execution_error,
)


class _FakeExc(Exception):
    """Exception type with optional response/body attributes for tests."""

    def __init__(self, message: str, *, body: object | None = None) -> None:
        super().__init__(message)
        self.body = body


class VllmClientErrorHelpersTest(unittest.TestCase):
    """Covers concise error summarization and processor error detection."""

    def test_detects_qwen3vl_processor_error(self) -> None:
        """Known Qwen3VLProcessor errors should be classified as processor failures."""
        exc = _FakeExc("Failed to apply Qwen3VLProcessor on request payload")
        self.assertTrue(is_video_processor_error(exc))

    def test_summary_returns_actionable_hint_for_processor_error(self) -> None:
        """Processor failures should return concise actionable guidance."""
        exc = _FakeExc("Error code: 400 - Failed to apply Qwen3VLProcessor")
        summary = summarize_execution_error(exc)
        self.assertIn("Qwen3VLProcessor", summary)
        self.assertIn("Safe video sampling", summary)

    def test_build_details_keeps_raw_error_and_flag(self) -> None:
        """Detailed payload should preserve raw error text for diagnostics."""
        exc = _FakeExc(
            "Error code: 400",
            body={"error": {"message": "Failed to apply Qwen3VLProcessor"}},
        )
        details = build_execution_error_details(exc)
        self.assertEqual(details["error_type"], "_FakeExc")
        self.assertTrue(details["is_video_processor_error"])
        self.assertIn("Error code: 400", details["raw_error"])
        self.assertIn("hint", details)

    def test_summary_truncates_long_non_processor_errors(self) -> None:
        """Generic very long errors should be truncated for UI readability."""
        long_error = "x" * 1000
        exc = _FakeExc(long_error)
        summary = summarize_execution_error(exc)
        self.assertTrue(summary.endswith("..."))
        self.assertLess(len(summary), len(long_error))

    def test_extract_usage_tokens_from_non_stream_response(self) -> None:
        """Usage extraction should read prompt/completion/total token fields."""
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=111, completion_tokens=222, total_tokens=333)
        )
        usage = _extract_usage_tokens(response)
        self.assertEqual(usage.prompt_tokens, 111)
        self.assertEqual(usage.output_tokens, 222)
        self.assertEqual(usage.total_tokens, 333)

    def test_extract_usage_tokens_from_stream_chunk(self) -> None:
        """Stream chunks with usage payload should be parsed identically."""
        chunk = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
        usage = _extract_usage_tokens(chunk)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)

    def test_extract_usage_tokens_missing_usage_returns_none_fields(self) -> None:
        """Missing usage should map to a fully-null usage struct."""
        usage = _extract_usage_tokens(SimpleNamespace(usage=None))
        self.assertIsNone(usage.prompt_tokens)
        self.assertIsNone(usage.output_tokens)
        self.assertIsNone(usage.total_tokens)

    def test_sum_token_usage_supports_partial_segment_data(self) -> None:
        """Summation should accumulate available segment usage and ignore missing values."""
        usage = _sum_token_usage(
            [
                TokenUsage(prompt_tokens=5, output_tokens=10, total_tokens=15),
                TokenUsage(prompt_tokens=None, output_tokens=None, total_tokens=None),
                TokenUsage(prompt_tokens=7, output_tokens=14, total_tokens=21),
            ]
        )
        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.output_tokens, 24)
        self.assertEqual(usage.total_tokens, 36)


if __name__ == "__main__":
    unittest.main()
