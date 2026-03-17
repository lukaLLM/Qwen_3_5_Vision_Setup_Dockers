"""Tests for MM lab media preprocessing helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.media_preprocess import build_segment_ranges, should_downscale  # noqa: E402


class MediaPreprocessHelpersTest(unittest.TestCase):
    """Covers pure segmentation and resize helper logic."""

    def test_should_downscale(self) -> None:
        """Downscale helper should only trigger when source exceeds target."""
        self.assertTrue(should_downscale(source_height=720, target_height=480))
        self.assertFalse(should_downscale(source_height=360, target_height=480))

    def test_build_segment_ranges_without_split(self) -> None:
        """Short clips should return a single [0, duration] segment."""
        ranges = build_segment_ranges(duration_s=12.0, max_duration_s=30.0, overlap_s=2.0)
        self.assertEqual(ranges, [(0.0, 12.0)])

    def test_build_segment_ranges_with_overlap(self) -> None:
        """Segment planner should emit overlapped ranges for long clips."""
        ranges = build_segment_ranges(duration_s=65.0, max_duration_s=30.0, overlap_s=2.0)
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0], (0.0, 32.0))
        self.assertEqual(ranges[1], (28.0, 62.0))
        self.assertEqual(ranges[2], (58.0, 65.0))


if __name__ == "__main__":
    unittest.main()
