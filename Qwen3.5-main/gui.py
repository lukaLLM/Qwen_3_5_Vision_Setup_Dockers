"""Gradio UI for segmented video QA against a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Generator
from html import escape
from pathlib import Path
from typing import Any, cast

import gradio as gr
import yaml  # type: ignore[import-untyped]

from vllm_video_call import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_SEGMENT_DURATION,
    DEFAULT_MODEL,
    DEFAULT_SEGMENT_OVERLAP,
    DEFAULT_TARGET_RES,
    DEFAULT_VIDEO_FPS,
    call_vllm_segmented,
    plan_video_segments,
    stream_vllm_segmented,
)

PROMPTS_PATH = Path(__file__).with_name("prompts.yaml")
LARGE_VIDEO_WARNING_MB = 120
DEFAULT_STREAM_OUTPUT = os.getenv("GUI_STREAM_OUTPUT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_DEBUG_MODE = os.getenv("GUI_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _default_thinking_mode() -> str:
    raw_mode = os.getenv("VLLM_THINKING_MODE", "auto").strip().lower()
    if raw_mode == "on":
        return "On"
    if raw_mode == "off":
        return "Off"
    return "Auto"


APP_THEME = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "sans-serif"],
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
  --bg-0: #0b111e;
  --bg-1: #131f35;
  --bg-2: #1f2f4e;
  --glass: rgba(255, 255, 255, 0.08);
  --glass-border: rgba(255, 255, 255, 0.2);
  --text-main: #ecf1ff;
  --text-muted: #a8b5d8;
  --accent: #4dd6b6;
  --accent-2: #62a6ff;
}

.gradio-container {
  background: radial-gradient(circle at 20% 10%, var(--bg-2), transparent 42%),
              radial-gradient(circle at 85% 5%, #2f4e80, transparent 35%),
              linear-gradient(140deg, var(--bg-0), var(--bg-1));
  min-height: 100vh;
  color: var(--text-main);
  font-family: 'Space Grotesk', sans-serif;
}

#app-title {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.25rem;
}

#app-subtitle {
  color: var(--text-muted);
  margin-top: 0;
}

.glass-panel {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  backdrop-filter: blur(16px);
  box-shadow: 0 14px 40px rgba(5, 12, 24, 0.45);
  padding: 14px;
}

#stream-panel {
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  min-height: 360px;
  background: rgba(5, 11, 22, 0.6);
  padding: 14px;
  overflow: auto;
}

.classification-wrap {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 240px;
  padding: 12px;
}

.classification-badge {
  border-radius: 999px;
  padding: 20px 34px;
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 700;
  letter-spacing: 0.05em;
  color: #ffffff;
  text-transform: uppercase;
  text-align: center;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
}

.classification-red {
  background: linear-gradient(120deg, #b42318, #ef4444);
}

.classification-orange {
  background: linear-gradient(120deg, #c2410c, #fb923c);
}

.classification-green {
  background: linear-gradient(120deg, #166534, #22c55e);
}

.classification-neutral {
  background: linear-gradient(120deg, #334155, #64748b);
}

.classification-details {
  margin: 14px auto 0 auto;
  width: min(780px, 100%);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  background: rgba(5, 11, 22, 0.65);
  padding: 12px 14px;
  color: var(--text-main);
  font-size: 1.22rem;
  line-height: 1.5;
}

.classification-details strong {
  color: var(--accent);
}

#mode-toggle-row {
  gap: 10px;
  margin-bottom: 6px;
}

#mode-toggle-row > div {
  min-width: 0;
}

#mode-toggle-row .gradio-checkbox {
  margin-bottom: 0 !important;
}

#simple-mode-toggle,
#simple-mode-toggle {
  border: 1px solid var(--glass-border);
  border-radius: 10px;
  background: rgba(5, 11, 22, 0.45);
  padding: 6px 10px;
}

button.primary {
  background: linear-gradient(120deg, var(--accent), var(--accent-2)) !important;
  border: 0 !important;
  color: #081327 !important;
  font-weight: 700 !important;
}

@media (max-width: 900px) {
  #stream-panel {
    min-height: 260px;
  }
}
"""


def _prompt_str_representer(dumper: Any, value: str) -> Any:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


yaml.add_representer(str, _prompt_str_representer, Dumper=yaml.SafeDumper)

PromptItem = dict[str, str]
PromptList = list[PromptItem]
PromptSelectionResult = tuple[str, str, str]
PromptMutationResult = tuple[PromptList, Any, str, str, str]
InferenceUpdate = tuple[str, str, str]
SimpleModeToggleResult = tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]

CLASSIFICATION_PROMPT_LABELS: dict[str, tuple[str, ...]] = {
    "[Security] Burglary": ("burglary", "suspicious", "normal"),
    "[Safety] Warehouse": ("helmet_off", "suspicious", "normal"),
    "[Security] Shoplifting": ("shoplifting", "suspicious", "normal"),
    "[Security] Car break in": ("car_break_in", "no_break_in"),
    "[Safety] Fire": ("fire", "no_fire"),
    "[Safety] Railroad tracks": ("on_tracks", "not_on_tracks"),
}
CLASSIFICATION_RED_LABELS = {
    "burglary",
    "shoplifting",
    "helmet_off",
    "car_break_in",
    "fire",
    "on_tracks",
    "parse_error",
}
CLASSIFICATION_ORANGE_LABELS = {"suspicious"}
CLASSIFICATION_GREEN_LABELS = {"normal", "no_break_in", "no_fire", "not_on_tracks"}
PROMPT_ID_RE = re.compile(r"[^a-z0-9_]+")


def _normalize_prompt_id(value: str) -> str:
    cleaned = PROMPT_ID_RE.sub("_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _deterministic_prompt_id(name: str, text: str) -> str:
    slug = _normalize_prompt_id(name) or "prompt"
    digest = hashlib.sha1(f"{name}\n{text}".encode()).hexdigest()[:8]
    return f"{slug}_{digest}"


def _next_unique_prompt_id(base_id: str, prompts: PromptList) -> str:
    used = {item.get("id", "") for item in prompts}
    candidate = base_id
    suffix = 2
    while candidate in used:
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
        raw_id = str(entry.get("id", "")).strip()
        normalized_id = _normalize_prompt_id(raw_id) if raw_id else ""
        if raw_id and normalized_id != raw_id:
            migrated = True

        prompt_id = normalized_id if normalized_id else _deterministic_prompt_id(name, text)
        if not normalized_id:
            migrated = True

        if prompt_id in seen_ids:
            if normalized_id:
                raise ValueError(f"Duplicate prompt id detected: {prompt_id}")
            prompt_id = _next_unique_prompt_id(prompt_id, prompts)
            migrated = True

        seen_ids.add(prompt_id)
        prompts.append({"id": prompt_id, "name": name, "text": text})
    return prompts, migrated


def load_prompts() -> PromptList:
    """Load prompt definitions from disk.

    Returns:
        A list of prompt items containing `name` and `text`.
    """
    if not PROMPTS_PATH.exists():
        return []
    try:
        data = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return []
    prompts, migrated = _safe_prompt_items(data)
    if migrated:
        save_prompts(prompts)
    return prompts


def save_prompts(prompts: PromptList) -> None:
    """Persist prompt definitions to disk in YAML format.

    Args:
        prompts: Prompt records to write.
    """
    normalized: PromptList = []
    seen_ids: set[str] = set()
    for item in prompts:
        name = str(item.get("name", "")).strip()
        text = str(item.get("text", "")).strip()
        if not name or not text:
            continue
        raw_id = str(item.get("id", "")).strip()
        prompt_id = _normalize_prompt_id(raw_id) if raw_id else _deterministic_prompt_id(name, text)
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id detected: {prompt_id}")
        seen_ids.add(prompt_id)
        normalized.append({"id": prompt_id, "name": name, "text": text})

    serialized = cast(
        str,
        yaml.dump(
            normalized,
            Dumper=yaml.SafeDumper,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        ),
    )
    PROMPTS_PATH.write_text(serialized, encoding="utf-8")


def prompt_names(prompts: PromptList) -> list[str]:
    """Return prompt names in list order.

    Args:
        prompts: Prompt records.

    Returns:
        The prompt names.
    """
    return [item["name"] for item in prompts]


def find_prompt(prompts: PromptList, name: str) -> PromptItem | None:
    """Find a prompt by exact name.

    Args:
        prompts: Prompt records.
        name: Prompt name to search for.

    Returns:
        The matching prompt, if found.
    """
    for item in prompts:
        if item["name"] == name:
            return item
    return None


def on_select_prompt(selected: str, prompts: PromptList) -> PromptSelectionResult:
    """Load prompt values into the edit fields.

    Args:
        selected: Name selected in the dropdown.
        prompts: Current prompt records.

    Returns:
        Tuple of prompt name, prompt text, and status message.
    """
    item = find_prompt(prompts, selected)
    if not item:
        return "", "", "_Prompt not found._"
    return item["name"], item["text"], f"Loaded prompt: `{item['name']}`"


def on_save_prompt(name: str, text: str, prompts: PromptList) -> PromptMutationResult:
    """Create a new prompt record.

    Args:
        name: Candidate prompt name.
        text: Candidate prompt body.
        prompts: Existing prompt records.

    Returns:
        Updated prompt state, dropdown update object, normalized name/text, and status.
    """
    clean_name = name.strip()
    clean_text = text.strip()

    if not clean_name:
        return prompts, gr.update(), name, text, "Prompt name is required."
    if not clean_text:
        return prompts, gr.update(), name, text, "Prompt text is required."

    lowered = {item["name"].lower() for item in prompts}
    if clean_name.lower() in lowered:
        return prompts, gr.update(), name, text, f"Prompt `{clean_name}` already exists."

    new_id = _next_unique_prompt_id(_deterministic_prompt_id(clean_name, clean_text), prompts)
    updated = [*prompts, {"id": new_id, "name": clean_name, "text": clean_text}]
    save_prompts(updated)
    return (
        updated,
        gr.update(choices=prompt_names(updated), value=clean_name),
        clean_name,
        clean_text,
        f"Saved new prompt: `{clean_name}`",
    )


def on_update_prompt(
    selected: str,
    name: str,
    text: str,
    prompts: PromptList,
) -> PromptMutationResult:
    """Update the selected prompt.

    Args:
        selected: Existing selected prompt name.
        name: New prompt name.
        text: New prompt text.
        prompts: Existing prompt records.

    Returns:
        Updated prompt state, dropdown update object, normalized name/text, and status.
    """
    if not selected:
        return prompts, gr.update(), name, text, "Select a prompt to update."

    clean_name = name.strip()
    clean_text = text.strip()
    if not clean_name:
        return prompts, gr.update(), name, text, "Prompt name is required."
    if not clean_text:
        return prompts, gr.update(), name, text, "Prompt text is required."

    index = next((i for i, item in enumerate(prompts) if item["name"] == selected), None)
    if index is None:
        return prompts, gr.update(), name, text, "Selected prompt no longer exists."

    for i, item in enumerate(prompts):
        if i != index and item["name"].lower() == clean_name.lower():
            return prompts, gr.update(), name, text, f"Prompt `{clean_name}` already exists."

    updated = list(prompts)
    existing_id = str(prompts[index].get("id", "")).strip()
    updated[index] = {
        "id": _normalize_prompt_id(existing_id) or _deterministic_prompt_id(clean_name, clean_text),
        "name": clean_name,
        "text": clean_text,
    }
    save_prompts(updated)
    return (
        updated,
        gr.update(choices=prompt_names(updated), value=clean_name),
        clean_name,
        clean_text,
        f"Updated prompt: `{clean_name}`",
    )


def on_delete_prompt(selected: str, prompts: PromptList) -> PromptMutationResult:
    """Delete the selected prompt.

    Args:
        selected: Prompt name to delete.
        prompts: Existing prompt records.

    Returns:
        Updated prompt state, dropdown update object, current name/text, and status.
    """
    if not selected:
        return prompts, gr.update(), "", "", "Select a prompt to delete."

    updated = [item for item in prompts if item["name"] != selected]
    save_prompts(updated)

    if not updated:
        return updated, gr.update(choices=[], value=None), "", "", f"Deleted prompt: `{selected}`"

    first = updated[0]
    return (
        updated,
        gr.update(choices=prompt_names(updated), value=first["name"]),
        first["name"],
        first["text"],
        f"Deleted prompt: `{selected}`",
    )


def refresh_prompts_from_disk() -> PromptMutationResult:
    """Reload prompt data from disk and refresh UI controls.

    Returns:
        Updated prompt state, dropdown update object, selected name/text, and status.
    """
    try:
        prompts = load_prompts()
    except ValueError as exc:
        return [], gr.update(choices=[], value=None), "", "", f"Prompt load error: {exc}"
    names = prompt_names(prompts)

    if not prompts:
        return (
            prompts,
            gr.update(choices=[], value=None),
            "",
            "",
            "No prompts found in prompts.yaml.",
        )

    first = prompts[0]
    return (
        prompts,
        gr.update(choices=names, value=first["name"]),
        first["name"],
        first["text"],
        f"Loaded {len(prompts)} prompts from prompts.yaml.",
    )


def _classification_prompt_items(prompts: PromptList) -> PromptList:
    return [item for item in prompts if item["name"] in CLASSIFICATION_PROMPT_LABELS]


def _select_prompt(prompts: PromptList, preferred_name: str | None) -> PromptItem | None:
    if preferred_name:
        selected = find_prompt(prompts, preferred_name)
        if selected is not None:
            return selected
    if not prompts:
        return None
    return prompts[0]


def on_toggle_simple_mode(
    enabled: bool,
    selected_prompt_name: str | None,
    prompts: PromptList,
) -> SimpleModeToggleResult:
    """Toggle simplified classification mode and refresh dependent controls.

    Args:
        enabled: Whether simple classification mode is enabled.
        selected_prompt_name: Current selected prompt name.
        prompts: Full prompt library from state.

    Returns:
        Component updates for prompt controls and advanced settings visibility.
    """
    target_prompts = _classification_prompt_items(prompts) if enabled else prompts
    target_names = prompt_names(target_prompts)
    target_item = _select_prompt(target_prompts, selected_prompt_name)
    target_name = target_item["name"] if target_item else ""
    target_text = target_item["text"] if target_item else ""
    target_dropdown_value: str | None = target_name if target_name else None
    show_advanced = not enabled
    prompt_message = (
        f"Simple mode enabled with {len(target_prompts)} classification prompts."
        if enabled
        else f"Loaded {len(target_prompts)} prompts from prompts.yaml."
    )

    return (
        gr.update(choices=target_names, value=target_dropdown_value),
        gr.update(value=target_name, visible=show_advanced),
        gr.update(value=target_text, visible=show_advanced),
        gr.update(value=prompt_message, visible=show_advanced),
        gr.update(visible=show_advanced),
        gr.update(visible=show_advanced),
        gr.update(visible=show_advanced),
        gr.update(visible=show_advanced),
        gr.update(visible=show_advanced),
        gr.update(visible=show_advanced),
        gr.update(visible=show_advanced),
    )


def _parse_classification_label(prompt_name: str, model_output: str) -> str:
    labels = CLASSIFICATION_PROMPT_LABELS.get(prompt_name)
    if labels is None:
        return "parse_error"
    normalized = model_output.strip().lower()
    if normalized in labels:
        return normalized
    return "parse_error"


def _classification_prompt_for_label(prompt_text: str, prompt_name: str) -> str:
    labels = CLASSIFICATION_PROMPT_LABELS.get(prompt_name)
    if labels is None:
        return prompt_text.strip()

    labels_text = " | ".join(labels)
    base = prompt_text.strip()
    return (
        f"{base}\n\n"
        "Output contract:\n"
        f"- Output exactly one label and nothing else: {labels_text}"
    )


def _classification_badge_css_class(label: str) -> str:
    if label in CLASSIFICATION_RED_LABELS:
        return "classification-red"
    if label in CLASSIFICATION_ORANGE_LABELS:
        return "classification-orange"
    if label in CLASSIFICATION_GREEN_LABELS:
        return "classification-green"
    return "classification-red"


def _classification_badge_markdown(label: str) -> str:
    display = label.replace("_", " ").upper()
    css_class = _classification_badge_css_class(label)
    return (
        "<div class='classification-wrap'>"
        f"<div class='classification-badge {css_class}'>{display}</div>"
        "</div>"
    )


def _classification_waiting_markdown() -> str:
    return (
        "<div class='classification-wrap'>"
        "<div class='classification-badge classification-neutral'>RUNNING</div>"
        "</div>"
    )


def _classification_with_arguments_markdown(
    badge_label: str,
    classification_text: str,
    explanation_text: str,
) -> str:
    display = classification_text.strip() or badge_label
    css_class = _classification_badge_css_class(badge_label)
    display_text = escape(display).replace("_", " ").upper()
    explanation = escape(explanation_text.strip() or "Not provided.")
    return (
        "<div class='classification-wrap'>"
        f"<div class='classification-badge {css_class}'>{display_text}</div>"
        "</div>"
        + "<div class='classification-details'>"
        + f"<p><strong>Why:</strong> {explanation}</p>"
        + "</div>"
    )


def _split_classification_output(model_output: str) -> tuple[str, str]:
    lines = [line.strip() for line in model_output.splitlines() if line.strip()]
    if not lines:
        return "", ""
    label_line = lines[0]
    explanation = "\n".join(lines[1:]).strip()
    return label_line, explanation


def _extract_video_path(video: Any) -> str:
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        for key in ("path", "name"):
            value = video.get(key)
            if isinstance(value, str):
                return value
    return ""


def _format_output_markdown(answer: str, video_name: str, warning: str, *, is_stream: bool) -> str:
    title = "Streaming Answer" if is_stream else "Answer"
    header = f"### {title}\n\n`{video_name}`\n"
    warning_line = f"\n> {warning}\n" if warning else ""
    body = answer if answer.strip() else "_Waiting for model output..._"
    return f"{header}{warning_line}\n{body}"


def _format_seconds(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def run_inference(
    video: Any,
    selected_prompt_name: str,
    prompt: str,
    base_url: str,
    model: str,
    max_completion_tokens: int,
    use_stream: bool,
    thinking_mode: str,
    show_thinking: bool,
    debug_mode: bool,
    simple_mode: bool,
) -> Generator[InferenceUpdate, None, None]:
    """Run segmented inference and stream progress updates for the UI.

    Args:
        video: Uploaded video payload from Gradio.
        selected_prompt_name: Selected prompt name from the dropdown.
        prompt: User prompt text.
        base_url: OpenAI-compatible vLLM URL.
        model: Model identifier.
        max_completion_tokens: Completion token cap.
        use_stream: Whether to stream output chunks.
        thinking_mode: Reasoning mode selector.
        show_thinking: Whether reasoning text should be included.
        debug_mode: Enable verbose terminal debug logs.
        simple_mode: Whether to render badge-only classification output.

    Yields:
        Tuples of markdown output, run status text, and accumulated plain text.
    """
    video_path = _extract_video_path(video)
    base_prompt = prompt.strip()
    clean_prompt = _classification_prompt_for_label(base_prompt, selected_prompt_name)
    effective_use_stream = use_stream and not simple_mode
    effective_show_thinking = show_thinking and not simple_mode

    def _debug(msg: str) -> None:
        if debug_mode:
            print(f"[gui-debug] {msg}", flush=True)

    if not video_path:
        yield "Upload a video file first.", "No video selected.", ""
        return
    if not clean_prompt:
        yield "Enter a prompt first.", "Prompt is empty.", ""
        return

    path = Path(video_path)
    if not path.exists():
        yield f"Video not found: `{video_path}`", "Missing video file.", ""
        return

    size_mb = path.stat().st_size / (1024 * 1024)
    warning = ""
    if size_mb > LARGE_VIDEO_WARNING_MB:
        warning = (
            f"Large clip ({size_mb:.1f} MB). Upload/encoding can be slow. "
            "Consider shorter clips for faster turnaround."
        )

    segment_ranges: list[tuple[float, float]] = [(0.0, 0.0)]
    try:
        segment_ranges = plan_video_segments(video_path)
    except Exception as exc:  # noqa: BLE001
        _debug(f"segment-plan-failed: {exc}")

    total_segments = len(segment_ranges)
    max_workers = max(1, int(DEFAULT_MAX_CONCURRENT))
    streamed = ""
    current_segment_status = ""
    clean_thinking_mode = thinking_mode.strip().lower()
    enable_thinking: bool | None = None
    if clean_thinking_mode == "on":
        enable_thinking = True
    elif clean_thinking_mode == "off":
        enable_thinking = False

    _debug(
        f"request video={path.name} size_mb={size_mb:.1f} model={model} "
        f"max_completion_tokens={int(max_completion_tokens)} stream={effective_use_stream} "
        f"thinking={thinking_mode} show_thinking={effective_show_thinking} "
        f"simple_mode={simple_mode}"
    )

    def _on_preprocess(msg: str) -> None:
        _debug(f"preprocess: {msg}")

    def _on_segment(index: int, total: int, start_s: float, end_s: float) -> None:
        nonlocal current_segment_status
        current_segment_status = (
            f"Segment {index}/{total} ({_format_seconds(start_s)}-{_format_seconds(end_s)})"
        )
        _debug(f"segment: {current_segment_status}")

    prep_status = f"Pre-processing video ({DEFAULT_TARGET_RES}p @ {DEFAULT_VIDEO_FPS:g}fps)..."
    if total_segments > 1:
        prep_status = (
            f"Pre-processing + segmenting into {total_segments} clips "
            f"({DEFAULT_MAX_SEGMENT_DURATION:g}s chunks, {DEFAULT_SEGMENT_OVERLAP:g}s overlap)..."
        )
    initial_output = (
        _classification_waiting_markdown()
        if simple_mode
        else _format_output_markdown(streamed, path.name, warning, is_stream=effective_use_stream)
    )
    yield (
        initial_output,
        prep_status,
        streamed,
    )

    try:
        def _call_segmented(prompt_text: str) -> str:
            return call_vllm_segmented(
                video_path=video_path,
                prompt=prompt_text,
                base_url=base_url,
                model=model,
                max_tokens=int(max_completion_tokens),
                max_completion_tokens=int(max_completion_tokens),
                max_workers=max_workers,
                preprocess_status_callback=_on_preprocess,
                segment_status_callback=_on_segment,
                enable_thinking=enable_thinking,
                include_reasoning=effective_show_thinking,
            )

        if effective_use_stream:
            for streamed in stream_vllm_segmented(
                video_path=video_path,
                prompt=clean_prompt,
                base_url=base_url,
                model=model,
                max_tokens=int(max_completion_tokens),
                max_completion_tokens=int(max_completion_tokens),
                preprocess_status_callback=_on_preprocess,
                segment_status_callback=_on_segment,
                enable_thinking=enable_thinking,
                include_reasoning=effective_show_thinking,
            ):
                run_status_text = "Streaming response..."
                if current_segment_status:
                    run_status_text = f"{current_segment_status} streaming..."
                elif total_segments > 1:
                    run_status_text = f"Streaming {total_segments} segments..."
                yield (
                    _format_output_markdown(streamed, path.name, warning, is_stream=True),
                    run_status_text,
                    streamed,
                )

            if not streamed:
                streamed = _call_segmented(clean_prompt)
        else:
            running_status = "Running classification..." if simple_mode else "Running inference..."
            if total_segments > 1:
                running_status = f"Running {total_segments} segments (concurrency={max_workers})..."
            running_output = (
                _classification_waiting_markdown()
                if simple_mode
                else _format_output_markdown(streamed, path.name, warning, is_stream=False)
            )
            yield (
                running_output,
                running_status,
                streamed,
            )

            streamed = _call_segmented(clean_prompt)

        _debug(f"response_chars={len(streamed)}")
        done_status = "Completed."
        if total_segments > 1:
            done_status = f"Completed {total_segments} segments."
        classification_text, explanation_text = _split_classification_output(streamed)
        if not classification_text:
            classification_text = "unknown"
        parsed_label = _parse_classification_label(selected_prompt_name, classification_text)
        badge_label = parsed_label if parsed_label != "parse_error" else classification_text.lower()
        final_output = (
            _classification_with_arguments_markdown(
                badge_label,
                classification_text,
                explanation_text,
            )
            if simple_mode
            else _format_output_markdown(
                streamed,
                path.name,
                warning,
                is_stream=effective_use_stream,
            )
        )
        copy_payload = streamed
        if simple_mode:
            copy_payload = (
                f"Classification: {classification_text or 'unknown'}\n"
                f"Explanation: {explanation_text or '(none)'}"
            )
        yield (
            final_output,
            done_status,
            copy_payload,
        )
    except Exception as exc:  # noqa: BLE001
        error_text = f"Request failed: {exc}"
        if "Number of samples" in error_text and "must be non-negative" in error_text:
            error_text = (
                f"{error_text}\n\n"
                "Video frame sampling failed due to broken duration/frame metadata. "
                "Try a short MP4 clip, or set `VLLM_DO_SAMPLE_FRAMES=0` in the client."
            )
        if simple_mode:
            yield _classification_badge_markdown("parse_error"), "Inference failed.", streamed
            return
        yield f"**{error_text}**", "Inference failed.", streamed


def copy_status_message(text: str) -> str:
    """Build copy feedback text for the status panel.

    Args:
        text: Text currently available for clipboard copy.

    Returns:
        A short status message for the UI.
    """
    if not text:
        return "Nothing to copy."
    return f"Copied {len(text)} characters."


def build_app() -> gr.Blocks:
    """Construct and return the Gradio application."""
    try:
        prompts = load_prompts()
    except ValueError:
        prompts = []
    names = prompt_names(prompts)
    default_name = names[0] if names else None
    default_text = prompts[0]["text"] if prompts else ""

    with gr.Blocks(title="vLLM Video-QA Studio") as app:
        gr.Markdown("## vLLM Video-QA Studio", elem_id="app-title")
        gr.Markdown(
            "Upload a clip, pick or manage a prompt, then run streaming multimodal QA.",
            elem_id="app-subtitle",
        )

        prompts_state = gr.State(prompts)
        streamed_text_state = gr.State("")

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes=["glass-panel"]):
                with gr.Row(elem_id="mode-toggle-row"):
                    simple_mode = gr.Checkbox(
                        label="Simple Classification Mode",
                        value=False,
                        elem_id="simple-mode-toggle",
                    )
                video_input = gr.Video(label="Video Upload + Preview")
                with gr.Accordion("Connection Settings", open=False) as connection_settings:
                    base_url = gr.Textbox(label="Base URL", value=DEFAULT_BASE_URL)
                    model = gr.Textbox(label="Model", value=DEFAULT_MODEL)
                    max_completion_tokens = gr.Slider(
                        minimum=8,
                        maximum=8192,
                        step=8,
                        value=DEFAULT_MAX_COMPLETION_TOKENS,
                        label="Max Completion Tokens",
                    )
                    use_stream = gr.Checkbox(label="Stream output", value=DEFAULT_STREAM_OUTPUT)
                    thinking_mode = gr.Radio(
                        choices=["Auto", "On", "Off"],
                        value=_default_thinking_mode(),
                        label="Thinking Mode",
                        info="Auto uses model default; Off is usually faster/cheaper.",
                    )
                    show_thinking = gr.Checkbox(
                        label="Show thinking in output",
                        value=False,
                    )
                    debug_mode = gr.Checkbox(
                        label="Debug logs (terminal)", value=DEFAULT_DEBUG_MODE
                    )
                settings_tip = gr.Markdown(
                    "Tip: Keep this app aligned with your vLLM startup flags, especially "
                    "`--limit-mm-per-prompt`."
                )

            with gr.Column(scale=7, elem_classes=["glass-panel"]):
                prompt_dropdown = gr.Dropdown(
                    choices=names,
                    value=default_name,
                    label="Prompt Library",
                    allow_custom_value=False,
                )
                prompt_name = gr.Textbox(
                    label="Prompt Name",
                    value=default_name or "",
                    placeholder="[Custom] Defensive transition breakdown",
                )
                prompt_text = gr.Textbox(label="Prompt Text", value=default_text, lines=8)

                with gr.Row():
                    reload_btn = gr.Button("Reload Prompts")
                    save_btn = gr.Button("Save")
                    update_btn = gr.Button("Update")
                    delete_btn = gr.Button("Delete")
                    run_btn = gr.Button("Run", variant="primary")

                prompt_status = gr.Markdown("_Ready._")
                output_md = gr.Markdown("_Model output will appear here._", elem_id="stream-panel")
                with gr.Row():
                    copy_btn = gr.Button("Copy")
                    run_status = gr.Markdown("Idle")

        prompt_dropdown.change(
            fn=on_select_prompt,
            inputs=[prompt_dropdown, prompts_state],
            outputs=[prompt_name, prompt_text, prompt_status],
        )

        simple_mode.change(
            fn=on_toggle_simple_mode,
            inputs=[simple_mode, prompt_dropdown, prompts_state],
            outputs=[
                prompt_dropdown,
                prompt_name,
                prompt_text,
                prompt_status,
                connection_settings,
                settings_tip,
                reload_btn,
                save_btn,
                update_btn,
                delete_btn,
                copy_btn,
            ],
        )

        reload_btn.click(
            fn=refresh_prompts_from_disk,
            outputs=[prompts_state, prompt_dropdown, prompt_name, prompt_text, prompt_status],
        )

        save_btn.click(
            fn=on_save_prompt,
            inputs=[prompt_name, prompt_text, prompts_state],
            outputs=[prompts_state, prompt_dropdown, prompt_name, prompt_text, prompt_status],
        )

        update_btn.click(
            fn=on_update_prompt,
            inputs=[prompt_dropdown, prompt_name, prompt_text, prompts_state],
            outputs=[prompts_state, prompt_dropdown, prompt_name, prompt_text, prompt_status],
        )

        delete_btn.click(
            fn=on_delete_prompt,
            inputs=[prompt_dropdown, prompts_state],
            outputs=[prompts_state, prompt_dropdown, prompt_name, prompt_text, prompt_status],
        )

        app.load(
            fn=refresh_prompts_from_disk,
            outputs=[prompts_state, prompt_dropdown, prompt_name, prompt_text, prompt_status],
        )

        run_btn.click(
            fn=run_inference,
            inputs=[
                video_input,
                prompt_dropdown,
                prompt_text,
                base_url,
                model,
                max_completion_tokens,
                use_stream,
                thinking_mode,
                show_thinking,
                debug_mode,
                simple_mode,
            ],
            outputs=[output_md, run_status, streamed_text_state],
        )

        copy_btn.click(
            fn=copy_status_message,
            inputs=[streamed_text_state],
            outputs=[run_status],
            js="""
            (text) => {
              if (text) {
                navigator.clipboard.writeText(text);
              }
              return [text];
            }
            """,
        )

    return cast(gr.Blocks, app)


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from qwen_image.server import run_server

    run_server()
