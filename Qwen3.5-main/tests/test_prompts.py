"""Tests for prompt ID migration, stability, and lookup behavior."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml  # type: ignore[import-untyped]

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_image.prompts import (  # noqa: E402
    find_prompt_by_id,
    load_prompts,
)


class PromptIdMigrationTest(unittest.TestCase):
    """Covers automatic prompt ID migration and duplicate rejection."""

    def _write_yaml(self, payload: list[dict[str, str]]) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_load_prompts_migrates_missing_ids_and_persists(self) -> None:
        """Legacy prompts without `id` are migrated once and saved back to disk."""
        path = self._write_yaml(
            [
                {"name": "Prompt A", "text": "Text A"},
                {"name": "Prompt B", "text": "Text B"},
            ]
        )
        try:
            first = load_prompts(path=Path(path))
            second = load_prompts(path=Path(path))
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        finally:
            os.unlink(path)

        self.assertTrue(all("id" in item for item in first))
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertTrue(all("id" in item for item in raw))

    def test_duplicate_explicit_prompt_ids_are_rejected(self) -> None:
        """Prompt file with duplicate IDs raises a ValueError."""
        path = self._write_yaml(
            [
                {"id": "dup_id", "name": "Prompt A", "text": "Text A"},
                {"id": "dup_id", "name": "Prompt B", "text": "Text B"},
            ]
        )
        try:
            with self.assertRaises(ValueError):
                load_prompts(path=Path(path))
        finally:
            os.unlink(path)

    def test_find_prompt_by_id(self) -> None:
        """Prompt lookup by ID returns the expected prompt record."""
        path = self._write_yaml(
            [
                {"id": "security_shoplifting", "name": "[Security] Shoplifting", "text": "Text"},
            ]
        )
        try:
            prompts = load_prompts(path=Path(path))
            found = find_prompt_by_id(prompts, "security_shoplifting")
        finally:
            os.unlink(path)

        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found["name"], "[Security] Shoplifting")


if __name__ == "__main__":
    unittest.main()
