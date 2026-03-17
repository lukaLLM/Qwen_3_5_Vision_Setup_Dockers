"""Tests for MM lab payload assembly helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.payload_builder import (  # noqa: E402
    build_messages,
    merge_extra_body,
    normalize_base_url,
    parse_json_object,
)


class PayloadBuilderTest(unittest.TestCase):
    """Covers URL normalization, JSON parsing, and multimodal message construction."""

    def test_normalize_base_url_appends_v1(self) -> None:
        """Base URL helper should normalize to `/v1` suffix."""
        self.assertEqual(normalize_base_url("http://localhost:8000"), "http://localhost:8000/v1")
        self.assertEqual(normalize_base_url("http://localhost:8000/v1"), "http://localhost:8000/v1")

    def test_build_messages_supports_images_videos_and_cache_uuids(self) -> None:
        """Message payload should include all multimodal items and UUID hints."""
        messages = build_messages(
            prompt="Describe scene",
            text_input="extra context",
            image_data_urls=["data:image/png;base64,a", "data:image/png;base64,b"],
            video_data_urls=["data:video/mp4;base64,c", "data:video/mp4;base64,d"],
            image_cache_uuids=["img-1", "img-2"],
            video_cache_uuids=["vid-1", "vid-2"],
        )
        content = messages[0]["content"]
        assert isinstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["uuid"], "img-1")
        self.assertEqual(content[3]["type"], "video_url")
        self.assertEqual(content[3]["video_url"]["uuid"], "vid-1")
        self.assertEqual(content[4]["type"], "video_url")
        self.assertEqual(content[4]["video_url"]["uuid"], "vid-2")

    def test_merge_extra_body_applies_safe_defaults(self) -> None:
        """Extra body merge should inject safe video and thinking defaults."""
        merged = merge_extra_body(
            user_extra_body={},
            include_video=True,
            safe_video_sampling=True,
            video_sampling_fps=2.0,
            thinking_mode="off",
            top_k=20,
        )
        self.assertEqual(merged["top_k"], 20)
        self.assertEqual(merged["mm_processor_kwargs"]["do_sample_frames"], False)
        self.assertEqual(merged["chat_template_kwargs"]["enable_thinking"], False)

    def test_parse_json_object_rejects_non_object(self) -> None:
        """JSON parser should reject payloads that are not object-shaped."""
        with self.assertRaises(ValueError):
            parse_json_object('["a"]', field_name="bad_json")


if __name__ == "__main__":
    unittest.main()
