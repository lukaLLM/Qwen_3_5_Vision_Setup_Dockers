"""Gradio UI for local multimodal parameter experimentation."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

import gradio as gr
import pandas as pd

from visual_experimentation_app.benchmark_graphs import build_graph_frames
from visual_experimentation_app.benchmark_runner import run_benchmark
from visual_experimentation_app.config import get_settings
from visual_experimentation_app.payload_builder import parse_json_object
from visual_experimentation_app.result_store import (
    list_run_history,
    load_run_result,
    save_benchmark_result,
    save_run_result,
)
from visual_experimentation_app.schemas import (
    BenchmarkRequest,
    BenchmarkResult,
    RunRequest,
    RunResult,
    RunTiming,
)
from visual_experimentation_app.ui_presets import (
    DEFAULT_CUSTOM_PROMPT,
    DEFAULT_TAG_CATEGORIES,
    PROMPT_MODE_CHOICES,
    PROMPT_MODE_CLASSIFIER,
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_TAGGING,
    SEGMENTATION_PROFILE_CHOICES,
    SEGMENTATION_PROFILE_OFF,
    build_prompt_for_mode,
    segmentation_values_for_profile,
)
from visual_experimentation_app.vllm_client import (
    build_execution_error_details,
    execute_run,
    summarize_execution_error,
)

APP_THEME = gr.themes.Default()

CUSTOM_CSS = """
#single-run-status h1,
#single-run-status h2,
#single-run-status h3 {
  font-size: 1.35rem !important;
  line-height: 1.3 !important;
  margin: 0.25rem 0 0.35rem 0 !important;
}

#single-run-status p,
#single-run-status li {
  font-size: 1.05rem !important;
  line-height: 1.5 !important;
}
"""


def ui_theme() -> object:
    """Return the Gradio theme used by the MM lab."""
    return APP_THEME


def ui_css() -> str:
    """Return custom CSS for the MM lab Gradio app."""
    return CUSTOM_CSS


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _extract_paths(upload_value: Any) -> list[str]:
    if upload_value is None:
        return []
    if isinstance(upload_value, str):
        return [upload_value]
    if isinstance(upload_value, list):
        paths: list[str] = []
        for item in upload_value:
            paths.extend(_extract_paths(item))
        return paths
    if isinstance(upload_value, dict):
        for key in ("path", "name"):
            value = upload_value.get(key)
            if isinstance(value, str) and value.strip():
                return [value]
        return []

    name = getattr(upload_value, "name", None)
    if isinstance(name, str) and name.strip():
        return [name]
    return []


def _image_preview_value(upload_value: Any) -> list[str]:
    """Return image upload paths for inline preview."""
    return _extract_paths(upload_value)


def _video_preview_value(upload_value: Any) -> str | None:
    """Return first uploaded video path for inline preview."""
    paths = _extract_paths(upload_value)
    return paths[0] if paths else None


def _clean_text(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        return raw_value
    return str(raw_value)


def _csv_to_int_list(raw_value: Any, *, field_name: str) -> list[int]:
    cleaned = _clean_text(raw_value).strip()
    if not cleaned:
        return []
    values: list[int] = []
    for chunk in cleaned.split(","):
        part = chunk.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain comma-separated integers.") from exc
    return values


def _csv_to_str_list(raw_value: Any) -> list[str]:
    cleaned = _clean_text(raw_value)
    if not cleaned.strip():
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _apply_prompt_mode(mode: str, current_prompt: str, tag_categories_csv: str) -> str:
    """Return prompt text based on the selected UI preset mode."""
    return build_prompt_for_mode(
        mode=_clean_text(mode).strip(),
        current_prompt=_clean_text(current_prompt),
        tag_categories_csv=_clean_text(tag_categories_csv),
    )


def _refresh_prompt_for_tagging(
    mode: str,
    current_prompt: str,
    tag_categories_csv: str,
) -> str:
    """Refresh prompt when category-driven preset modes are active."""
    clean_mode = _clean_text(mode).strip()
    if clean_mode not in {PROMPT_MODE_TAGGING, PROMPT_MODE_CLASSIFIER}:
        return _clean_text(current_prompt)
    return build_prompt_for_mode(
        mode=clean_mode,
        current_prompt=_clean_text(current_prompt),
        tag_categories_csv=_clean_text(tag_categories_csv),
    )


def _apply_segmentation_profile(
    profile: str,
    current_duration: float,
    current_overlap: float,
) -> tuple[float, float]:
    """Apply segmentation defaults from the selected profile."""
    return segmentation_values_for_profile(
        profile=_clean_text(profile).strip(),
        current_duration=float(current_duration),
        current_overlap=float(current_overlap),
    )


def _build_run_request(
    *,
    prompt: str,
    text_input: str,
    image_upload: Any,
    video_upload: Any,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    use_model_defaults: bool,
    max_tokens: int,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    thinking_mode: str,
    show_reasoning: bool,
    measure_ttft: bool,
    preprocess_images: bool,
    preprocess_video: bool,
    target_height: int,
    target_video_fps: float,
    safe_video_sampling: bool,
    video_sampling_fps: float,
    segment_max_duration_s: float,
    segment_overlap_s: float,
    segment_workers: int,
    image_cache_uuids_text: str,
    video_cache_uuids_text: str,
    disable_caching: bool,
    extra_body_json: str,
    extra_headers_json: str,
) -> RunRequest:
    image_paths = _extract_paths(image_upload)
    video_paths = _extract_paths(video_upload)
    if len(video_paths) > 2:
        raise ValueError("Please upload at most 2 videos.")
    video_path = video_paths[0] if video_paths else None

    prompt_text = _clean_text(prompt).strip()
    text_input_value = _clean_text(text_input)
    base_url_value = _clean_text(base_url)
    model_value = _clean_text(model)
    api_key_value = _clean_text(api_key)
    image_cache_uuids_value = _clean_text(image_cache_uuids_text)
    video_cache_uuids_value = _clean_text(video_cache_uuids_text)
    extra_body_raw = _clean_text(extra_body_json)
    extra_headers_raw = _clean_text(extra_headers_json)

    extra_body = parse_json_object(extra_body_raw, field_name="Extra Body JSON")
    extra_headers = parse_json_object(extra_headers_raw, field_name="Extra Headers JSON")

    mode = thinking_mode.strip().lower()
    if mode not in {"auto", "on", "off"}:
        mode = "auto"
    thinking_mode_value = cast(Literal["auto", "on", "off"], mode)
    parsed_video_cache_uuids = _csv_to_str_list(video_cache_uuids_value)

    return RunRequest(
        prompt=prompt_text,
        text_input=text_input_value.strip() or None,
        image_paths=image_paths,
        video_paths=video_paths,
        video_path=video_path,
        base_url=base_url_value.strip() or None,
        model=model_value.strip() or None,
        api_key=api_key_value.strip() or None,
        timeout_seconds=timeout_seconds,
        use_model_defaults=use_model_defaults,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        thinking_mode=thinking_mode_value,
        show_reasoning=show_reasoning,
        measure_ttft=measure_ttft,
        preprocess_images=preprocess_images,
        preprocess_video=preprocess_video,
        target_height=target_height,
        target_video_fps=target_video_fps,
        safe_video_sampling=safe_video_sampling,
        video_sampling_fps=video_sampling_fps if not safe_video_sampling else None,
        segment_max_duration_s=segment_max_duration_s,
        segment_overlap_s=segment_overlap_s,
        segment_workers=segment_workers,
        image_cache_uuids=_csv_to_str_list(image_cache_uuids_value),
        video_cache_uuids=parsed_video_cache_uuids,
        video_cache_uuid=(
            parsed_video_cache_uuids[0]
            if parsed_video_cache_uuids
            else None
        ),
        disable_caching=disable_caching,
        request_extra_body=extra_body,
        request_extra_headers={str(key): str(value) for key, value in extra_headers.items()},
    )


def _execute_and_persist(run_request: RunRequest) -> RunResult:
    run_id = f"run_{uuid4().hex}"
    created_at = _utc_now_iso()
    started = perf_counter()

    try:
        execution = execute_run(run_request)
        result = RunResult(
            run_id=run_id,
            status="ok",
            created_at=created_at,
            request=run_request,
            output_text=execution.output_text,
            error=None,
            timings=RunTiming(
                preprocess_ms=execution.preprocess_ms,
                request_ms=execution.request_ms,
                total_ms=execution.total_ms,
                ttft_ms=execution.ttft_ms,
            ),
            effective_params=execution.effective_params,
            media_metadata=execution.media_metadata,
        )
    except Exception as exc:  # noqa: BLE001
        error_details = build_execution_error_details(exc)
        result = RunResult(
            run_id=run_id,
            status="error",
            created_at=created_at,
            request=run_request,
            output_text="",
            error=summarize_execution_error(exc),
            timings=RunTiming(
                preprocess_ms=0.0,
                request_ms=0.0,
                total_ms=(perf_counter() - started) * 1000.0,
                ttft_ms=None,
            ),
            effective_params={"error_details": error_details},
            media_metadata={},
        )

    save_run_result(result)
    return result


def _build_effective_request_markdown(result: RunResult) -> str:
    """Render a concise summary of effective request parameters."""
    effective = result.effective_params
    sent = effective.get("sent_generation_params", {})
    omitted = effective.get("omitted_for_model_defaults", [])
    defaults_info = effective.get("model_defaults_info", {})

    if not isinstance(sent, dict):
        sent = {}
    if not isinstance(omitted, list):
        omitted = []

    sent_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in sent.items())
    if not sent_lines:
        sent_lines = "- _none_"

    omitted_lines = "\n".join(f"- `{item}`" for item in omitted)
    if not omitted_lines:
        omitted_lines = "- _none_"

    defaults_md = ""
    if result.request.use_model_defaults:
        info_source = (
            defaults_info.get("source", "unknown")
            if isinstance(defaults_info, dict)
            else "unknown"
        )
        info_path = (
            defaults_info.get("path", "")
            if isinstance(defaults_info, dict)
            else ""
        )
        info_message = (
            defaults_info.get("message", "")
            if isinstance(defaults_info, dict)
            else ""
        )
        sampling_values = (
            defaults_info.get("sampling_values", {})
            if isinstance(defaults_info, dict)
            else {}
        )
        if isinstance(sampling_values, dict) and sampling_values:
            sampling_lines = "\n".join(
                f"- `{key}`: `{value}`" for key, value in sampling_values.items()
            )
        else:
            sampling_lines = "- _not discoverable from local generation_config_"

        path_line = f"- `generation_config_path`: `{info_path}`\n" if info_path else ""
        message_line = f"- note: {info_message}\n" if info_message else ""
        defaults_md = (
            "\n\n**Model/Server Defaults (Best Effort)**\n"
            f"- `source`: `{info_source}`\n"
            f"{path_line}"
            f"{message_line}"
            "\n**Resolved Sampling Defaults**\n"
            f"{sampling_lines}"
        )

    return (
        "### Effective Request\n"
        f"- `use_model_defaults`: `{result.request.use_model_defaults}`\n"
        f"- `disable_caching`: `{result.request.disable_caching}`\n"
        f"- `model`: `{effective.get('model', result.request.model or '')}`\n"
        f"- `base_url`: `{effective.get('base_url', result.request.base_url or '')}`\n\n"
        "**Sent Generation Params**\n"
        f"{sent_lines}\n\n"
        "**Omitted For Model Defaults**\n"
        f"{omitted_lines}"
        f"{defaults_md}"
    )


def _run_single(*args: Any) -> tuple[str, dict[str, Any], str, str]:
    try:
        request = _build_run_request(
            prompt=args[0],
            text_input=args[1],
            image_upload=args[2],
            video_upload=args[3],
            base_url=args[4],
            model=args[5],
            api_key=args[6],
            timeout_seconds=float(args[7]),
            use_model_defaults=bool(args[8]),
            max_tokens=int(args[9]),
            max_completion_tokens=int(args[10]),
            temperature=float(args[11]),
            top_p=float(args[12]),
            top_k=int(args[13]),
            presence_penalty=float(args[14]),
            frequency_penalty=float(args[15]),
            thinking_mode=args[16],
            show_reasoning=bool(args[17]),
            measure_ttft=bool(args[18]),
            preprocess_images=bool(args[19]),
            preprocess_video=bool(args[20]),
            target_height=int(args[21]),
            target_video_fps=float(args[22]),
            safe_video_sampling=bool(args[23]),
            video_sampling_fps=float(args[24]),
            segment_max_duration_s=float(args[25]),
            segment_overlap_s=float(args[26]),
            segment_workers=int(args[27]),
            image_cache_uuids_text=args[28],
            video_cache_uuids_text=args[29],
            disable_caching=bool(args[30]),
            extra_body_json=args[31],
            extra_headers_json=args[32],
        )
        chunk_parallel_requests = max(1, int(args[33]))
    except Exception as exc:  # noqa: BLE001
        return f"**Input error:** {exc}", {}, "_Invalid request._", "Input validation failed."

    request.segment_workers = chunk_parallel_requests
    result = _execute_and_persist(request)
    mode_text = "model-defaults mode" if result.request.use_model_defaults else "explicit mode"
    segment_count = int(result.effective_params.get("segment_count") or 0)
    segment_workers = int(result.effective_params.get("segment_workers") or request.segment_workers)
    segment_note = (
        "- Note: chunk parallel requests had no effect (segmentation OFF or only 1 chunk).\n"
        if segment_count <= 1 and chunk_parallel_requests > 1
        else ""
    )
    status_line = (
        (
            f"### Completed Time: `{result.timings.total_ms:.1f} ms`\n"
            f"- Run ID: `{result.run_id}`\n"
            f"- Mode: `{mode_text}`\n"
            f"- Chunk parallel requests: `{chunk_parallel_requests}`\n"
            f"- Segments: `{segment_count}` (effective chunk workers: `{segment_workers}`)\n"
            f"{segment_note}"
            "- See `Run Result JSON -> effective_params` for exact sent/omitted parameters."
        )
        if result.status == "ok"
        else f"### Run Failed\n- Run ID: `{result.run_id}`\n- Error: {result.error}"
    )
    output_markdown = result.output_text.strip() or "_No output text returned._"
    effective_request_md = _build_effective_request_markdown(result)
    return output_markdown, result.model_dump(), effective_request_md, status_line


def _empty_benchmark_graph_frames(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            columns=[
                "request_concurrency",
                "target_height",
                "segment_workers",
                "segmentation_mode",
                "resolution_label",
                "series_label",
                "combo_label",
                "latency_ms",
            ]
        ),
        pd.DataFrame(
            columns=[
                "request_concurrency",
                "target_height",
                "segment_workers",
                "segmentation_mode",
                "resolution_label",
                "series_label",
                "combo_label",
                "throughput_tokens_per_sec",
            ]
        ),
        pd.DataFrame(
            columns=[
                "config_label",
                "combo_label",
                "target_height",
                "segment_workers",
                "segmentation_mode",
                "request_concurrency",
                "resolution_label",
                "completion_time_ms",
            ]
        ),
        pd.DataFrame(
            columns=[
                "combo_label",
                "target_height",
                "segment_workers",
                "segmentation_mode",
                "request_concurrency",
                "stage",
                "pct_value",
            ]
        ),
    )


def _run_benchmark(
    *args: Any,
) -> tuple[
    str,
    dict[str, Any],
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    try:
        base_run = _build_run_request(
            prompt=args[0],
            text_input=args[1],
            image_upload=args[2],
            video_upload=args[3],
            base_url=args[4],
            model=args[5],
            api_key=args[6],
            timeout_seconds=float(args[7]),
            use_model_defaults=bool(args[8]),
            max_tokens=int(args[9]),
            max_completion_tokens=int(args[10]),
            temperature=float(args[11]),
            top_p=float(args[12]),
            top_k=int(args[13]),
            presence_penalty=float(args[14]),
            frequency_penalty=float(args[15]),
            thinking_mode=args[16],
            show_reasoning=bool(args[17]),
            measure_ttft=bool(args[18]),
            preprocess_images=bool(args[19]),
            preprocess_video=bool(args[20]),
            target_height=int(args[21]),
            target_video_fps=float(args[22]),
            safe_video_sampling=bool(args[23]),
            video_sampling_fps=float(args[24]),
            segment_max_duration_s=float(args[25]),
            segment_overlap_s=float(args[26]),
            segment_workers=int(args[27]),
            image_cache_uuids_text=args[28],
            video_cache_uuids_text=args[29],
            disable_caching=bool(args[30]),
            extra_body_json=args[31],
            extra_headers_json=args[32],
        )
        benchmark_request = BenchmarkRequest(
            base_run=base_run,
            repeats=int(args[33]),
            warmup_runs=int(args[34]),
            resolution_heights=_csv_to_int_list(
                args[35],
                field_name="Target Video Heights to Test",
            ),
            request_concurrency=_csv_to_int_list(
                args[36],
                field_name="Chunk Parallel Requests to Test",
            ),
            segment_workers=[base_run.segment_workers],
            wait_between_combos_s=float(args[37]),
            include_non_segmented_baseline=bool(args[38]),
            continue_on_error=bool(args[39]),
            label=args[40].strip() or None,
        )
    except Exception as exc:  # noqa: BLE001
        empty_graphs = _empty_benchmark_graph_frames()
        return (
            f"**Input error:** {exc}",
            {},
            "Benchmark input validation failed.",
            *empty_graphs,
        )

    benchmark_id = f"bench_{uuid4().hex}"
    created_at = _utc_now_iso()
    try:
        result = run_benchmark(benchmark_request, benchmark_id=benchmark_id)
        result.created_at = created_at
    except Exception as exc:  # noqa: BLE001
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            status="error",
            created_at=created_at,
            request=benchmark_request,
            records=[],
            aggregates=[],
            artifact_paths={"error": str(exc)},
        )
        paths = save_benchmark_result(result)
        result.artifact_paths = paths | {"error": str(exc)}
        empty_graphs = _empty_benchmark_graph_frames()
        return (
            f"**Benchmark failed:** {exc}",
            result.model_dump(),
            str(result.artifact_paths),
            *empty_graphs,
        )

    paths = save_benchmark_result(result)
    result.artifact_paths = paths
    graph_frames = build_graph_frames(result)

    success_count = sum(1 for item in result.records if item.status == "ok")
    summary = (
        f"Benchmark `{result.benchmark_id}` status: **{result.status}**\n\n"
        f"- Runs: {len(result.records)}\n"
        f"- Successful: {success_count}\n"
        f"- Aggregates: {len(result.aggregates)}"
    )
    return (
        summary,
        result.model_dump(),
        str(paths),
        graph_frames["latency_by_concurrency"],
        graph_frames["throughput_by_concurrency"],
        graph_frames["completion_time_ms"],
        graph_frames["time_split_stacked"],
    )


def _refresh_history() -> tuple[list[list[Any]], dict[str, Any]]:
    items = list_run_history(limit=200)
    rows = [
        [item.run_id, item.status, item.created_at, item.model, round(item.total_ms, 2)]
        for item in items
    ]
    dropdown = gr.update(
        choices=[item.run_id for item in items],
        value=items[0].run_id if items else None,
    )
    return rows, dropdown


def _load_history_detail(run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    result = load_run_result(run_id)
    if result is None:
        return {"error": f"Run not found: {run_id}"}
    return result.model_dump()


def build_ui_blocks() -> gr.Blocks:
    """Build and return MM lab Gradio Blocks."""
    settings = get_settings()

    with gr.Blocks(title="Qwen3.5 MM Lab") as app:
        gr.HTML(
            """
            <section id="mm-header">
              <p class="kicker">MM LAB</p>
              <h1>Qwen3.5 Multimodal Experiment Lab</h1>
              <p class="lead">
                Fast local iteration for multimodal prompts, generation controls, and
                reproducible benchmark runs.
              </p>
            </section>
            """
        )

        with gr.Tab("Single Run"):
            gr.Markdown(
                "Configure one request, run it, then inspect the effective payload and timings.",
                elem_classes=["tab-note"],
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, elem_classes=["panel"]):
                    prompt_mode = gr.Radio(
                        choices=PROMPT_MODE_CHOICES,
                        label="Prompt Mode",
                        value=PROMPT_MODE_CUSTOM,
                    )
                    tag_categories = gr.Textbox(
                        label="Tag Categories (CSV)",
                        value=DEFAULT_TAG_CATEGORIES,
                        info="Used by Tagging and Classifier presets.",
                    )
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=5,
                        value=DEFAULT_CUSTOM_PROMPT,
                    )
                    text_input = gr.Textbox(label="Additional Text Input (optional)", lines=3)
                    image_upload = gr.File(label="Images (multiple)", file_count="multiple")
                    image_preview = gr.Gallery(
                        label="Image Preview",
                        columns=4,
                        object_fit="contain",
                        height=220,
                    )
                    video_upload = gr.File(label="Videos (up to 2)", file_count="multiple")
                    video_preview = gr.Video(label="Video Preview")
                    run_button = gr.Button("Run", variant="primary")
                    run_status = gr.Markdown(
                        "Idle.",
                        elem_id="single-run-status",
                        elem_classes=["readable-output"],
                    )
                with gr.Column(scale=7, elem_classes=["panel"]):
                    with gr.Accordion("Connection + Generation", open=True):
                        base_url = gr.Textbox(label="vLLM Base URL", value=settings.base_url)
                        model = gr.Textbox(label="Model", value=settings.model)
                        api_key = gr.Textbox(
                            label="API Key",
                            value=settings.api_key,
                            type="password",
                        )
                        timeout_seconds = gr.Slider(
                            minimum=10,
                            maximum=600,
                            step=5,
                            value=settings.timeout_seconds,
                            label="Request Timeout (seconds)",
                        )
                        single_run_request_concurrency = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            value=1,
                            label="Chunk Parallel Requests",
                            info=(
                                "Number of chunks sent to vLLM at once "
                                "(when segmentation yields 2+ chunks)."
                            ),
                        )
                        use_model_defaults = gr.Checkbox(
                            label="Use model/server defaults (send minimal generation params)",
                            value=False,
                        )
                        max_tokens = gr.Slider(
                            minimum=1,
                            maximum=131072,
                            step=1,
                            value=1000,
                            label="max_tokens",
                        )
                        max_completion_tokens = gr.Slider(
                            minimum=1,
                            maximum=131072,
                            step=1,
                            value=1000,
                            label="max_completion_tokens",
                        )
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=2,
                            step=0.05,
                            value=1.0,
                            label="temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            step=0.05,
                            value=0.95,
                            label="top_p",
                        )
                        top_k = gr.Slider(minimum=1, maximum=200, step=1, value=20, label="top_k")
                        presence_penalty = gr.Slider(
                            minimum=-2,
                            maximum=2,
                            step=0.1,
                            value=1.5,
                            label="presence_penalty",
                        )
                        frequency_penalty = gr.Slider(
                            minimum=-2,
                            maximum=2,
                            step=0.1,
                            value=0.0,
                            label="frequency_penalty",
                        )
                        thinking_mode = gr.Radio(
                            choices=["auto", "on", "off"],
                            label="thinking_mode",
                            value="off",
                        )
                        show_reasoning = gr.Checkbox(label="Show reasoning output", value=False)
                        measure_ttft = gr.Checkbox(
                            label="Measure TTFT via streaming",
                            value=settings.default_measure_ttft,
                        )

                    with gr.Accordion("Preprocessing + Segmentation", open=False):
                        preprocess_images = gr.Checkbox(label="Preprocess images", value=True)
                        preprocess_video = gr.Checkbox(label="Preprocess video", value=True)
                        target_height = gr.Slider(
                            minimum=128,
                            maximum=1080,
                            step=16,
                            value=settings.default_target_height,
                            label="Target Max Height (px)",
                        )
                        target_video_fps = gr.Slider(
                            minimum=0.1,
                            maximum=30.0,
                            step=0.1,
                            value=settings.default_video_fps,
                            label="Target Video FPS",
                        )
                        safe_video_sampling = gr.Checkbox(
                            label="Safe video processor defaults (do_sample_frames=false)",
                            value=settings.default_safe_video_sampling,
                        )
                        video_sampling_fps = gr.Slider(
                            minimum=0.1,
                            maximum=30.0,
                            step=0.1,
                            value=settings.default_video_fps,
                            label="Video sampling fps when safe mode is OFF",
                        )
                        segmentation_profile = gr.Radio(
                            choices=SEGMENTATION_PROFILE_CHOICES,
                            label="Segmentation Profile",
                            value=SEGMENTATION_PROFILE_OFF,
                        )
                        gr.Markdown(
                            (
                                "Segment clips are re-encoded (`libx264`/`yuv420p`) for "
                                "processor metadata stability on long videos."
                            ),
                            elem_classes=["tab-note"],
                        )
                        segment_max_duration_s = gr.Slider(
                            minimum=0.0,
                            maximum=3600.0,
                            step=1.0,
                            value=0.0,
                            label="Segment max duration (seconds, 0 disables segmentation)",
                        )
                        segment_overlap_s = gr.Slider(
                            minimum=0.0,
                            maximum=300.0,
                            step=0.5,
                            value=0.0,
                            label="Segment overlap (seconds)",
                        )
                        segment_workers = gr.Number(
                            value=1,
                            precision=0,
                            visible=False,
                        )

                    with gr.Accordion("Advanced JSON", open=False):
                        image_cache_uuids = gr.Textbox(
                            label="Image cache UUIDs (comma-separated)",
                            placeholder="img-uuid-1,img-uuid-2",
                        )
                        video_cache_uuid = gr.Textbox(
                            label="Video cache UUIDs (comma-separated, optional)",
                            placeholder="vid-uuid-1,vid-uuid-2",
                        )
                        disable_caching = gr.Checkbox(
                            label=(
                                "Disable cache reuse for measurement "
                                "(cache_salt + random media UUIDs)"
                            ),
                            value=False,
                        )
                        extra_body_json = gr.Textbox(
                            label="Extra Body JSON",
                            lines=6,
                            value='{"top_k": 20}',
                        )
                        extra_headers_json = gr.Textbox(
                            label="Extra Headers JSON",
                            lines=4,
                            value="{}",
                        )

            with gr.Row(equal_height=True):
                with gr.Column(elem_classes=["panel"]):
                    run_output = gr.Markdown(
                        "Run output will appear here.",
                        elem_classes=["readable-output"],
                    )
                with gr.Column(elem_classes=["panel"]):
                    effective_request = gr.Markdown(
                        "Effective request details will appear here.",
                        elem_classes=["readable-output"],
                    )
            run_json = gr.JSON(label="Run Result JSON", elem_classes=["panel", "readable-output"])

        with gr.Tab("Benchmark"):
            gr.Markdown(
                "Sweep key variables to compare throughput, latency, and aggregate behavior.",
                elem_classes=["tab-note"],
            )
            gr.Markdown(
                (
                    "Benchmark reuses current Single Run settings as the base request "
                    "(including segmentation chunk seconds + overlap). "
                    "This tab sweeps target height and chunk parallel requests."
                ),
                elem_classes=["tab-note"],
            )
            benchmark_label = gr.Textbox(label="Benchmark label (optional)")
            repeats = gr.Slider(minimum=1, maximum=20, step=1, value=3, label="Repeats per combo")
            warmup_runs = gr.Slider(minimum=0, maximum=10, step=1, value=0, label="Warmup runs")
            resolution_heights = gr.Textbox(
                label="Target Video Heights to Test (px, CSV)",
                value="360,480,720",
                info="e.g. 360,480,720",
            )
            request_concurrency = gr.Textbox(
                label="Chunk Parallel Requests to Test (CSV)",
                value="1,2",
                info="Number of chunks sent to vLLM at once (e.g. 1,2,4).",
            )
            wait_between_combos_s = gr.Slider(
                minimum=0.0,
                maximum=60.0,
                step=0.5,
                value=0.0,
                label="Wait Between Benchmark Combos (seconds)",
                info="Pause after each combo group before starting the next one.",
            )
            include_non_segmented_baseline = gr.Checkbox(
                label="Include Non-Segmented Baseline",
                value=True,
                info=(
                    "Run one non-segmented baseline per height "
                    "(chunk-parallel sweep applies to segmented mode)."
                ),
            )
            continue_on_error = gr.Checkbox(label="Continue on run errors", value=True)
            benchmark_button = gr.Button("Run Benchmark", variant="primary")
            benchmark_summary = gr.Markdown(
                "No benchmark executed yet.",
                elem_classes=["readable-output"],
            )
            benchmark_json = gr.JSON(
                label="Benchmark Result JSON",
                elem_classes=["readable-output"],
            )
            benchmark_artifacts = gr.Markdown(
                "Artifacts will be listed here.",
                elem_classes=["readable-output"],
            )
            with gr.Row(equal_height=True):
                latency_plot = gr.LinePlot(
                    x="request_concurrency",
                    y="latency_ms",
                    color="series_label",
                    title="Graph 1: Latency vs Chunk Parallel Requests (split by resolution + mode)",
                    x_title="Chunk Parallel Requests",
                    y_title="Latency (ms)",
                    color_title="Series (Resolution / Mode)",
                    x_axis_format=".0f",
                    height=280,
                    caption="Use the chart toolbar (top-right) for Fullscreen and Export.",
                    tooltip=[
                        "combo_label",
                        "resolution_label",
                        "segmentation_mode",
                        "request_concurrency",
                        "latency_ms",
                    ],
                    buttons=["fullscreen", "export"],
                )
                throughput_plot = gr.LinePlot(
                    x="request_concurrency",
                    y="throughput_tokens_per_sec",
                    color="series_label",
                    title="Graph 2: Throughput vs Chunk Parallel Requests (split by resolution + mode)",
                    x_title="Chunk Parallel Requests",
                    y_title="Output Tokens / sec",
                    color_title="Series (Resolution / Mode)",
                    x_axis_format=".0f",
                    height=280,
                    caption="Use the chart toolbar (top-right) for Fullscreen and Export.",
                    tooltip=[
                        "combo_label",
                        "resolution_label",
                        "segmentation_mode",
                        "request_concurrency",
                        "throughput_tokens_per_sec",
                    ],
                    buttons=["fullscreen", "export"],
                )
            with gr.Row(equal_height=True):
                time_split_plot = gr.BarPlot(
                    x="config_label",
                    y="completion_time_ms",
                    color="segmentation_mode",
                    title="Graph 3: Full Video Completion Time (ms, segmented vs non-segmented)",
                    x_title="Config",
                    y_title="Completion Time (ms)",
                    color_title="Segmentation Mode",
                    height=280,
                    caption="Lower is better.",
                    tooltip=[
                        "combo_label",
                        "resolution_label",
                        "segmentation_mode",
                        "request_concurrency",
                        "completion_time_ms",
                    ],
                    buttons=["fullscreen", "export"],
                )
                tokens_scatter_plot = gr.BarPlot(
                    x="combo_label",
                    y="pct_value",
                    color="stage",
                    title="Graph 4: Stacked Time Split by Combo",
                    x_title="Combo (mode/height/chunk parallel requests)",
                    y_title="Percent of Total Time",
                    color_title="Stage",
                    height=280,
                    caption="Preprocess % + Request %",
                    tooltip=[
                        "combo_label",
                        "segmentation_mode",
                        "target_height",
                        "request_concurrency",
                        "pct_value",
                    ],
                    buttons=["fullscreen", "export"],
                )

        with gr.Tab("History"):
            gr.Markdown(
                "Browse prior runs and load full JSON details for quick side-by-side comparison.",
                elem_classes=["tab-note"],
            )
            refresh_history_btn = gr.Button("Refresh History")
            history_table = gr.Dataframe(
                headers=["run_id", "status", "created_at", "model", "total_ms"],
                datatype=["str", "str", "str", "str", "number"],
                interactive=False,
                elem_classes=["readable-output"],
            )
            history_run_id = gr.Dropdown(label="Select run_id", choices=[])
            load_detail_btn = gr.Button("Load Run Detail")
            history_json = gr.JSON(label="Run Detail", elem_classes=["readable-output"])

        run_inputs = [
            prompt,
            text_input,
            image_upload,
            video_upload,
            base_url,
            model,
            api_key,
            timeout_seconds,
            use_model_defaults,
            max_tokens,
            max_completion_tokens,
            temperature,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            thinking_mode,
            show_reasoning,
            measure_ttft,
            preprocess_images,
            preprocess_video,
            target_height,
            target_video_fps,
            safe_video_sampling,
            video_sampling_fps,
            segment_max_duration_s,
            segment_overlap_s,
            segment_workers,
            image_cache_uuids,
            video_cache_uuid,
            disable_caching,
            extra_body_json,
            extra_headers_json,
        ]
        run_inputs_single = run_inputs + [single_run_request_concurrency]

        run_button.click(
            fn=_run_single,
            inputs=run_inputs_single,
            outputs=[run_output, run_json, effective_request, run_status],
        )
        prompt_mode.change(
            fn=_apply_prompt_mode,
            inputs=[prompt_mode, prompt, tag_categories],
            outputs=[prompt],
        )
        tag_categories.change(
            fn=_refresh_prompt_for_tagging,
            inputs=[prompt_mode, prompt, tag_categories],
            outputs=[prompt],
        )
        segmentation_profile.change(
            fn=_apply_segmentation_profile,
            inputs=[segmentation_profile, segment_max_duration_s, segment_overlap_s],
            outputs=[segment_max_duration_s, segment_overlap_s],
        )
        image_upload.change(
            fn=_image_preview_value,
            inputs=[image_upload],
            outputs=[image_preview],
        )
        video_upload.change(
            fn=_video_preview_value,
            inputs=[video_upload],
            outputs=[video_preview],
        )

        benchmark_button.click(
            fn=_run_benchmark,
            inputs=run_inputs
            + [
                repeats,
                warmup_runs,
                resolution_heights,
                request_concurrency,
                wait_between_combos_s,
                include_non_segmented_baseline,
                continue_on_error,
                benchmark_label,
            ],
            outputs=[
                benchmark_summary,
                benchmark_json,
                benchmark_artifacts,
                latency_plot,
                throughput_plot,
                time_split_plot,
                tokens_scatter_plot,
            ],
        )

        refresh_history_btn.click(
            fn=_refresh_history,
            outputs=[history_table, history_run_id],
        )
        load_detail_btn.click(
            fn=_load_history_detail,
            inputs=[history_run_id],
            outputs=[history_json],
        )

        app.load(
            fn=_refresh_history,
            outputs=[history_table, history_run_id],
        )

    return cast(gr.Blocks, app)
