"""Tests for MM lab request schema normalization and limits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.schemas import RunRequest  # noqa: E402


class RunRequestSchemaTest(unittest.TestCase):
    """Covers backward compatibility and two-video constraints."""

    def test_legacy_video_fields_are_normalized(self) -> None:
        """Legacy singular fields should still populate plural fields."""
        request = RunRequest(
            prompt="hello",
            video_path="/tmp/a.mp4",
            video_cache_uuid="vid-a",
        )
        self.assertEqual(request.video_paths, ["/tmp/a.mp4"])
        self.assertEqual(request.video_path, "/tmp/a.mp4")
        self.assertEqual(request.video_cache_uuids, ["vid-a"])
        self.assertEqual(request.video_cache_uuid, "vid-a")

    def test_supports_two_videos(self) -> None:
        """Schema should accept exactly two videos."""
        request = RunRequest(
            prompt="hello",
            video_paths=["/tmp/a.mp4", "/tmp/b.mp4"],
            video_cache_uuids=["vid-a", "vid-b"],
        )
        self.assertEqual(len(request.video_paths), 2)
        self.assertEqual(request.video_path, "/tmp/a.mp4")
        self.assertEqual(request.video_cache_uuid, "vid-a")

    def test_rejects_more_than_two_videos(self) -> None:
        """Schema should reject requests that include more than two videos."""
        with self.assertRaises(ValidationError):
            RunRequest(
                prompt="hello",
                video_paths=["/tmp/a.mp4", "/tmp/b.mp4", "/tmp/c.mp4"],
            )

    def test_rejects_segmentation_with_multiple_videos(self) -> None:
        """Segmentation is only supported when a single video is present."""
        with self.assertRaises(ValidationError):
            RunRequest(
                prompt="hello",
                video_paths=["/tmp/a.mp4", "/tmp/b.mp4"],
                segment_max_duration_s=10.0,
            )


if __name__ == "__main__":
    unittest.main()
