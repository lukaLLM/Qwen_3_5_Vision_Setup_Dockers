"""Unit tests for classification parsing, rendering, and simple-mode inference flow."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from gui import (
    _default_thinking_mode,
    _classification_badge_css_class,
    _classification_prompt_items,
    _parse_classification_label,
    _split_classification_output,
    run_inference,
)


class ClassificationHelpersTest(unittest.TestCase):
    """Covers deterministic label parsing and color mapping."""

    def test_parse_valid_labels_for_each_prompt(self) -> None:
        """Each configured classification prompt accepts its declared labels."""
        self.assertEqual(_parse_classification_label("[Security] Burglary", "burglary"), "burglary")
        self.assertEqual(
            _parse_classification_label("[Safety] Warehouse", "helmet_off"),
            "helmet_off",
        )
        self.assertEqual(
            _parse_classification_label("[Security] Shoplifting", "shoplifting"),
            "shoplifting",
        )
        self.assertEqual(
            _parse_classification_label("[Security] Car break in", "car_break_in"),
            "car_break_in",
        )
        self.assertEqual(_parse_classification_label("[Safety] Fire", "fire"), "fire")
        self.assertEqual(
            _parse_classification_label("[Safety] Railroad tracks", "on_tracks"),
            "on_tracks",
        )

    def test_parse_rejects_non_exact_label_line(self) -> None:
        """Parser returns parse_error when first line is not an exact configured label."""
        self.assertEqual(
            _parse_classification_label("[Security] Shoplifting", "shoplifting detected"),
            "parse_error",
        )
        self.assertEqual(
            _parse_classification_label("[Safety] Fire", "normal"),
            "parse_error",
        )
        self.assertEqual(
            _parse_classification_label("[Unknown Prompt]", "normal"),
            "parse_error",
        )

    def test_split_classification_output(self) -> None:
        """Output splitter keeps first non-empty line as label and rest as explanation."""
        label, explanation = _split_classification_output(
            "\n\nshoplifting\nItem disappears under jacket. Hands hide the product."
        )
        self.assertEqual(label, "shoplifting")
        self.assertEqual(explanation, "Item disappears under jacket. Hands hide the product.")

        empty_label, empty_explanation = _split_classification_output("  \n\t  ")
        self.assertEqual(empty_label, "")
        self.assertEqual(empty_explanation, "")

    def test_badge_color_mapping(self) -> None:
        """Severity labels map to expected badge classes."""
        self.assertEqual(_classification_badge_css_class("burglary"), "classification-red")
        self.assertEqual(_classification_badge_css_class("shoplifting"), "classification-red")
        self.assertEqual(_classification_badge_css_class("helmet_off"), "classification-red")
        self.assertEqual(_classification_badge_css_class("car_break_in"), "classification-red")
        self.assertEqual(_classification_badge_css_class("fire"), "classification-red")
        self.assertEqual(_classification_badge_css_class("on_tracks"), "classification-red")
        self.assertEqual(_classification_badge_css_class("suspicious"), "classification-orange")
        self.assertEqual(_classification_badge_css_class("normal"), "classification-green")
        self.assertEqual(_classification_badge_css_class("no_break_in"), "classification-green")
        self.assertEqual(_classification_badge_css_class("no_fire"), "classification-green")
        self.assertEqual(_classification_badge_css_class("not_on_tracks"), "classification-green")

    def test_filter_only_classification_prompts(self) -> None:
        """Simple mode filter keeps only configured classification prompts."""
        prompts = [
            {"name": "[Overview] Something", "text": "x"},
            {"name": "[Security] Burglary", "text": "x"},
            {"name": "[Safety] Warehouse", "text": "x"},
            {"name": "[Security] Shoplifting", "text": "x"},
            {"name": "[Security] Car break in", "text": "x"},
            {"name": "[Safety] Fire", "text": "x"},
            {"name": "[Safety] Railroad tracks", "text": "x"},
        ]
        filtered = _classification_prompt_items(prompts)
        self.assertEqual(
            [item["name"] for item in filtered],
            [
                "[Security] Burglary",
                "[Safety] Warehouse",
                "[Security] Shoplifting",
                "[Security] Car break in",
                "[Safety] Fire",
                "[Safety] Railroad tracks",
            ],
        )

    def test_default_thinking_mode_from_env(self) -> None:
        """Thinking mode radio default follows VLLM_THINKING_MODE from environment."""
        with mock.patch.dict(os.environ, {"VLLM_THINKING_MODE": "off"}, clear=False):
            self.assertEqual(_default_thinking_mode(), "Off")
        with mock.patch.dict(os.environ, {"VLLM_THINKING_MODE": "on"}, clear=False):
            self.assertEqual(_default_thinking_mode(), "On")
        with mock.patch.dict(os.environ, {"VLLM_THINKING_MODE": "something_else"}, clear=False):
            self.assertEqual(_default_thinking_mode(), "Auto")


class SimpleClassificationFlowTest(unittest.TestCase):
    """Covers one-call simple-mode flow with first-line label parsing."""

    def setUp(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with open(path, "wb") as video_file:
            video_file.write(b"test")
        self.video_path = path

    def tearDown(self) -> None:
        if os.path.exists(self.video_path):
            os.unlink(self.video_path)

    def _run_simple(self, model_output: str) -> tuple[list[tuple[str, str, str]], mock.MagicMock]:
        with (
            mock.patch("gui.plan_video_segments", return_value=[(0.0, 0.0)]),
            mock.patch("gui.call_vllm_segmented", return_value=model_output) as mocked_call,
        ):
            updates = list(
                run_inference(
                    video=self.video_path,
                    selected_prompt_name="[Safety] Fire",
                    prompt="Classify this scene.",
                    base_url="http://localhost:8000/v1",
                    model="Qwen",
                    max_completion_tokens=128,
                    use_stream=False,
                    thinking_mode="Auto",
                    show_thinking=False,
                    debug_mode=False,
                    simple_mode=True,
                )
            )
        return updates, mocked_call

    def test_simple_mode_renders_label_and_explanation(self) -> None:
        """Simple mode uses one model call and renders badge plus explanation block."""
        updates, mocked_call = self._run_simple(
            "fire\nVisible flames near the rail line. This supports fire detection."
        )
        self.assertEqual(mocked_call.call_count, 1)
        sent_prompt = mocked_call.call_args.kwargs["prompt"]
        self.assertIn("Output exactly one label", sent_prompt)

        final_output, final_status, final_copy = updates[-1]
        self.assertEqual(final_status, "Completed.")
        self.assertIn("FIRE", final_output)
        self.assertIn("Why:</strong>", final_output)
        self.assertIn("Classification: fire", final_copy)

    def test_first_non_empty_line_is_used_as_label(self) -> None:
        """Parser ignores leading empty lines when deriving classification label."""
        updates, _ = self._run_simple(
            "\n\nfire\nFlames are visible. Heat source appears active."
        )
        final_output, _, final_copy = updates[-1]
        self.assertIn("FIRE", final_output)
        self.assertIn("Classification: fire", final_copy)

    def test_unknown_label_is_still_rendered(self) -> None:
        """Unknown first-line labels are still displayed and not blocked."""
        updates, _ = self._run_simple(
            "fire detected\nFlames are visible. The model used a non-canonical label."
        )
        final_output, final_status, final_copy = updates[-1]
        self.assertEqual(final_status, "Completed.")
        self.assertIn("FIRE DETECTED", final_output)
        self.assertIn("Classification: fire detected", final_copy)


if __name__ == "__main__":
    unittest.main()
