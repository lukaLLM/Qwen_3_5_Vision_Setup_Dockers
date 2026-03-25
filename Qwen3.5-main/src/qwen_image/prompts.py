"""Prompt library persistence and classification metadata."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from qwen_image.config import get_settings

PromptItem = dict[str, str]
PromptList = list[PromptItem]

CLASSIFICATION_PROMPT_LABELS: dict[str, tuple[str, ...]] = {
    "[Security] Burglary": ("burglary", "suspicious", "normal"),
    "[Safety] Warehouse": ("helmet_off", "suspicious", "normal"),
    "[Security] Shoplifting": ("shoplifting", "suspicious", "normal"),
    "[Security] Car break in": ("car_break_in", "no_break_in"),
    "[Safety] Fire": ("fire", "no_fire"),
    "[Safety] Railroad tracks": ("on_tracks", "not_on_tracks"),
}

PROMPT_ID_RE = re.compile(r"[^a-z0-9_]+")


def prompts_path() -> Path:
    """Return active prompt file path."""
    return get_settings().paths.prompts_path


def normalize_prompt_id(value: str) -> str:
    """Normalize a prompt ID into lowercase underscore style."""
    cleaned = PROMPT_ID_RE.sub("_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def deterministic_prompt_id(name: str, text: str) -> str:
    """Generate a deterministic prompt ID from name/text for migration."""
    slug = normalize_prompt_id(name) or "prompt"
    digest = hashlib.sha1(f"{name}\n{text}".encode()).hexdigest()[:8]
    return f"{slug}_{digest}"


def _ensure_generated_unique(base_id: str, seen_ids: set[str]) -> str:
    candidate = base_id
    suffix = 2
    while candidate in seen_ids:
        candidate = f"{base_id}_{suffix}"
        suffix += 1
    return candidate


def _safe_prompt_items(data: Any) -> tuple[PromptList, bool]:
    prompts: PromptList = []
    migrated = False
    seen_ids: set[str] = set()

    if not isinstance(data, list):
        return prompts, migrated

    for entry in data:
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name", "")).strip()
        text = str(entry.get("text", "")).strip()
        if not name or not text:
            continue

        explicit_id_raw = str(entry.get("id", "")).strip()
        explicit_id = normalize_prompt_id(explicit_id_raw)
        if explicit_id and explicit_id != explicit_id_raw:
            migrated = True

        prompt_id = explicit_id if explicit_id else deterministic_prompt_id(name, text)
        if not explicit_id:
            migrated = True

        if prompt_id in seen_ids:
            if explicit_id:
                raise ValueError(f"Duplicate prompt id detected: {prompt_id}")
            prompt_id = _ensure_generated_unique(prompt_id, seen_ids)
            migrated = True

        seen_ids.add(prompt_id)
        prompts.append({"id": prompt_id, "name": name, "text": text})

    return prompts, migrated


def _normalize_for_save(prompts: PromptList) -> PromptList:
    normalized: PromptList = []
    seen_ids: set[str] = set()

    for entry in prompts:
        name = str(entry.get("name", "")).strip()
        text = str(entry.get("text", "")).strip()
        if not name or not text:
            continue

        raw_id = str(entry.get("id", "")).strip()
        prompt_id = normalize_prompt_id(raw_id) if raw_id else deterministic_prompt_id(name, text)
        if not prompt_id:
            prompt_id = deterministic_prompt_id(name, text)

        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id detected: {prompt_id}")
        seen_ids.add(prompt_id)

        normalized.append({"id": prompt_id, "name": name, "text": text})

    return normalized


def load_prompts(path: Path | None = None) -> PromptList:
    """Load prompt definitions from disk and auto-migrate missing IDs."""
    source = path or prompts_path()
    if not source.exists():
        return []

    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return []

    prompts, migrated = _safe_prompt_items(data)
    if migrated:
        save_prompts(prompts, path=source)
    return prompts


def save_prompts(prompts: PromptList, path: Path | None = None) -> None:
    """Persist prompt definitions to disk in YAML format."""
    target = path or prompts_path()
    normalized = _normalize_for_save(prompts)
    serialized = yaml.dump(
        normalized,
        Dumper=yaml.SafeDumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    target.write_text(serialized, encoding="utf-8")


def prompt_names(prompts: PromptList) -> list[str]:
    """Return prompt names in list order."""
    return [item["name"] for item in prompts]


def find_prompt(prompts: PromptList, name: str) -> PromptItem | None:
    """Find a prompt by exact name."""
    for item in prompts:
        if item["name"] == name:
            return item
    return None


def find_prompt_by_id(prompts: PromptList, prompt_id: str) -> PromptItem | None:
    """Find a prompt by exact normalized ID."""
    wanted = normalize_prompt_id(prompt_id)
    if not wanted:
        return None
    for item in prompts:
        if item.get("id") == wanted:
            return item
    return None


def split_classification_output(model_output: str) -> tuple[str, str]:
    """Split model output into first-line label and explanation."""
    lines = [line.strip() for line in model_output.splitlines() if line.strip()]
    if not lines:
        return "", ""
    label = lines[0]
    explanation = " ".join(lines[1:]).strip()
    return label, explanation


def parse_classification_label(prompt_name: str, model_output: str) -> str:
    """Parse and validate first-line classification label for a prompt."""
    labels = CLASSIFICATION_PROMPT_LABELS.get(prompt_name)
    if labels is None:
        return "parse_error"
    normalized = model_output.strip().lower()
    if normalized in labels:
        return normalized
    return "parse_error"


def classification_prompt_for_label(prompt_text: str, prompt_name: str) -> str:
    """Append strict output contract for classification prompts."""
    labels = CLASSIFICATION_PROMPT_LABELS.get(prompt_name)
    if labels is None:
        return prompt_text.strip()

    labels_text = " | ".join(labels)
    return (
        f"{prompt_text.strip()}\n\n"
        "Output contract:\n"
        f"- Output exactly one label and nothing else: {labels_text}"
    )


def prompt_records_for_api() -> list[dict[str, Any]]:
    """Return prompt records for API listing response."""
    records: list[dict[str, Any]] = []
    for item in load_prompts():
        records.append(
            {
                "id": item["id"],
                "name": item["name"],
                "text": item["text"],
                "labels": list(CLASSIFICATION_PROMPT_LABELS.get(item["name"], ())),
            }
        )
    return records
