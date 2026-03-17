"""Gradio UI for local multimodal parameter experimentation."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

import gradio as gr

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
from visual_experimentation_app.vllm_client import execute_run

APP_THEME = gr.themes.Default()

CUSTOM_CSS = ""


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
        result = RunResult(
            run_id=run_id,
            status="error",
            created_at=created_at,
            request=run_request,
            output_text="",
            error=str(exc),
            timings=RunTiming(
                preprocess_ms=0.0,
                request_ms=0.0,
                total_ms=(perf_counter() - started) * 1000.0,
                ttft_ms=None,
            ),
            effective_params={},
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
    except Exception as exc:  # noqa: BLE001
        return f"**Input error:** {exc}", {}, "_Invalid request._", "Input validation failed."

    result = _execute_and_persist(request)
    mode_text = "model-defaults mode" if result.request.use_model_defaults else "explicit mode"
    status_line = (
        f"Run `{result.run_id}` completed in {result.timings.total_ms:.1f} ms ({mode_text}). "
        "See `Run Result JSON -> effective_params` for exact sent/omitted parameters."
        if result.status == "ok"
        else f"Run `{result.run_id}` failed: {result.error}"
    )
    output_markdown = result.output_text.strip() or "_No output text returned._"
    effective_request_md = _build_effective_request_markdown(result)
    return output_markdown, result.model_dump(), effective_request_md, status_line


def _run_benchmark(*args: Any) -> tuple[str, dict[str, Any], str]:
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
            resolution_heights=_csv_to_int_list(args[35], field_name="Resolution heights"),
            request_concurrency=_csv_to_int_list(args[36], field_name="Request concurrency"),
            segment_workers=_csv_to_int_list(args[37], field_name="Segment workers"),
            continue_on_error=bool(args[38]),
            label=args[39].strip() or None,
        )
    except Exception as exc:  # noqa: BLE001
        return f"**Input error:** {exc}", {}, "Benchmark input validation failed."

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
        return f"**Benchmark failed:** {exc}", result.model_dump(), str(result.artifact_paths)

    paths = save_benchmark_result(result)
    result.artifact_paths = paths

    success_count = sum(1 for item in result.records if item.status == "ok")
    summary = (
        f"Benchmark `{result.benchmark_id}` status: **{result.status}**\n\n"
        f"- Runs: {len(result.records)}\n"
        f"- Successful: {success_count}\n"
        f"- Aggregates: {len(result.aggregates)}"
    )
    return summary, result.model_dump(), str(paths)


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
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=5,
                        value="Describe what is happening.",
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
                    run_status = gr.Markdown("Idle.", elem_classes=["readable-output"])
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
                            value="auto",
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
                        segment_max_duration_s = gr.Slider(
                            minimum=0.0,
                            maximum=180.0,
                            step=1.0,
                            value=0.0,
                            label="Segment max duration (seconds, 0 disables segmentation)",
                        )
                        segment_overlap_s = gr.Slider(
                            minimum=0.0,
                            maximum=30.0,
                            step=0.5,
                            value=0.0,
                            label="Segment overlap (seconds)",
                        )
                        segment_workers = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            value=1,
                            label="Segment workers",
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
            benchmark_label = gr.Textbox(label="Benchmark label (optional)")
            repeats = gr.Slider(minimum=1, maximum=20, step=1, value=3, label="Repeats per combo")
            warmup_runs = gr.Slider(minimum=0, maximum=10, step=1, value=0, label="Warmup runs")
            resolution_heights = gr.Textbox(label="Resolution heights CSV", value="360,480,720")
            request_concurrency = gr.Textbox(label="Request concurrency CSV", value="1,2")
            segment_workers_sweep = gr.Textbox(label="Segment workers CSV", value="1,2")
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

        run_button.click(
            fn=_run_single,
            inputs=run_inputs,
            outputs=[run_output, run_json, effective_request, run_status],
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
                segment_workers_sweep,
                continue_on_error,
                benchmark_label,
            ],
            outputs=[benchmark_summary, benchmark_json, benchmark_artifacts],
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
