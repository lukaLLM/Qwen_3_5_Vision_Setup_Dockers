"""Video-to-text helpers for OpenAI-compatible vLLM chat completions."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import tempfile
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from openai import OpenAI


def _load_dotenv_defaults(dotenv_path: Path | None = None) -> None:
    path = dotenv_path or Path(__file__).with_name(".env")
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
            if "=" not in line:
                continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


_load_dotenv_defaults()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_COMPLETION_TOKENS = _env_int("VLLM_MAX_COMPLETION_TOKENS", DEFAULT_MAX_TOKENS)
DEFAULT_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
DEFAULT_VIDEO_FPS = _env_float("VLLM_VIDEO_FPS", 2.0)
DEFAULT_DO_SAMPLE_FRAMES = _env_bool("VLLM_DO_SAMPLE_FRAMES", True)
DEFAULT_AUTO_REENCODE_ON_ERROR = _env_bool("VLLM_AUTO_REENCODE_ON_ERROR", True)
DEFAULT_REENCODE_FPS = _env_float("VLLM_REENCODE_FPS", 2.0)
DEFAULT_PRESENCE_PENALTY = _env_float("VLLM_PRESENCE_PENALTY", 0.3)

DEFAULT_PREPROCESS = _env_bool("VLLM_PREPROCESS", True)
DEFAULT_TARGET_RES = max(64, _env_int("VLLM_TARGET_RES", 480))
DEFAULT_MAX_SEGMENT_DURATION = max(1.0, _env_float("VLLM_MAX_DURATION", 60.0))
DEFAULT_SEGMENT_OVERLAP = max(0.0, _env_float("VLLM_SEGMENT_OVERLAP", 3.0))
DEFAULT_MAX_CONCURRENT = max(1, _env_int("VLLM_MAX_CONCURRENT", 1))
DEFAULT_SEGMENT_DISABLE_SERVER_SAMPLING = _env_bool("VLLM_SEGMENT_DISABLE_SERVER_SAMPLING", True)


def get_runtime_video_settings() -> dict[str, object]:
    """Return effective runtime video-related settings."""
    return {
        "VLLM_TARGET_RES": DEFAULT_TARGET_RES,
        "VLLM_VIDEO_FPS": DEFAULT_VIDEO_FPS,
        "VLLM_MAX_DURATION": DEFAULT_MAX_SEGMENT_DURATION,
        "VLLM_SEGMENT_OVERLAP": DEFAULT_SEGMENT_OVERLAP,
        "VLLM_PREPROCESS": DEFAULT_PREPROCESS,
        "VLLM_DO_SAMPLE_FRAMES": DEFAULT_DO_SAMPLE_FRAMES,
        "VLLM_SEGMENT_DISABLE_SERVER_SAMPLING": DEFAULT_SEGMENT_DISABLE_SERVER_SAMPLING,
        "VLLM_MAX_CONCURRENT": DEFAULT_MAX_CONCURRENT,
    }


def print_runtime_video_settings(prefix: str = "[vllm-video]") -> None:
    """Print runtime video settings for debugging and support."""
    settings = get_runtime_video_settings()
    formatted = ", ".join(f"{key}={value}" for key, value in settings.items())
    print(f"{prefix} {formatted}", flush=True)


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return "http://localhost:8000/v1"
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _coerce_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                for key in ("text", "content", "reasoning_content", "reasoning", "output_text"):
                    text = item.get(key)
                    if isinstance(text, str):
                        parts.append(text)
                        break
        return "".join(parts)
    return str(content)


def _extract_message_text(message: object) -> str:
    content = _coerce_text(getattr(message, "content", None)).strip()
    if content:
        return content

    reasoning = _coerce_text(getattr(message, "reasoning_content", None)).strip()
    if reasoning:
        return reasoning

    reasoning = _coerce_text(getattr(message, "reasoning", None)).strip()
    if reasoning:
        return reasoning

    return ""


def _extract_message_parts(message: object) -> tuple[str, str]:
    content = _coerce_text(getattr(message, "content", None)).strip()
    reasoning = _coerce_text(getattr(message, "reasoning_content", None)).strip()
    if not reasoning:
        reasoning = _coerce_text(getattr(message, "reasoning", None)).strip()
    return content, reasoning


def _format_message_output(content: str, reasoning: str, *, include_reasoning: bool) -> str:
    clean_content = content.strip()
    clean_reasoning = reasoning.strip()
    if include_reasoning and clean_reasoning:
        if clean_content:
            return f"<think>\n{clean_reasoning}\n</think>\n\n{clean_content}"
        return f"<think>\n{clean_reasoning}\n</think>"
    if clean_content:
        return clean_content
    return clean_reasoning


def _safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        return


def _create_temp_mp4(prefix: str) -> Path:
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".mp4")
    os.close(fd)
    return Path(tmp_path)


def _parse_fps(raw_fps: object) -> float:
    if not isinstance(raw_fps, str):
        return 0.0
    value = raw_fps.strip()
    if not value or value in {"N/A", "0/0"}:
        return 0.0
    if "/" in value:
        num_str, den_str = value.split("/", maxsplit=1)
        try:
            num = float(num_str)
            den = float(den_str)
        except ValueError:
            return 0.0
        if den == 0:
            return 0.0
        return num / den
    try:
        return float(value)
    except ValueError:
        return 0.0


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _probe_video(video_path: str | Path) -> dict[str, object]:
    path = Path(video_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"Failed to probe video with ffprobe: {stderr}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse ffprobe output: {exc}") from exc

    streams = payload.get("streams", [])
    video_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if not isinstance(video_stream, dict):
        raise RuntimeError(f"No video stream found in {path}")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    codec = str(video_stream.get("codec_name") or "").lower()
    pix_fmt = str(video_stream.get("pix_fmt") or "").lower()
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))

    duration_s = 0.0
    format_obj = payload.get("format")
    if isinstance(format_obj, dict):
        try:
            duration_s = float(format_obj.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0
    if duration_s <= 0:
        try:
            duration_s = float(video_stream.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0

    return {
        "width": width,
        "height": height,
        "duration_s": max(duration_s, 0.0),
        "fps": max(fps, 0.0),
        "codec": codec,
        "pix_fmt": pix_fmt,
    }


def _format_timestamp(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _segment_ranges(
    duration_s: float, max_duration_s: float, overlap_s: float
) -> list[tuple[float, float]]:
    clean_duration = max(duration_s, 0.0)
    if clean_duration <= 0:
        return [(0.0, 0.0)]
    if clean_duration <= max_duration_s:
        return [(0.0, clean_duration)]

    clean_overlap = max(overlap_s, 0.0)
    ranges: list[tuple[float, float]] = []

    base_start = 0.0
    while base_start < clean_duration - 1e-6:
        base_end = min(clean_duration, base_start + max_duration_s)
        start = 0.0 if base_start <= 0 else max(0.0, base_start - clean_overlap)
        end = (
            clean_duration
            if base_end >= clean_duration
            else min(clean_duration, base_end + clean_overlap)
        )
        ranges.append((start, end))
        if base_end >= clean_duration:
            break
        base_start += max_duration_s
    return ranges


def plan_video_segments(video_path: str) -> list[tuple[float, float]]:
    """Plan segment boundaries for a video based on configured chunk size and overlap."""
    info = _probe_video(video_path)
    duration_s = _as_float(info.get("duration_s", 0.0))
    return _segment_ranges(duration_s, DEFAULT_MAX_SEGMENT_DURATION, DEFAULT_SEGMENT_OVERLAP)


def _extract_segment(source_path: Path, start_s: float, end_s: float) -> Path:
    target_path = _create_temp_mp4("vllm_segment_")
    duration = max(end_s - start_s, 0.05)

    copy_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        str(target_path),
    ]
    copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, check=False)
    if copy_result.returncode == 0:
        return target_path

    transcode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    transcode_result = subprocess.run(transcode_cmd, capture_output=True, text=True, check=False)
    if transcode_result.returncode != 0:
        _safe_unlink(target_path)
        stderr = (
            transcode_result.stderr.strip() or copy_result.stderr.strip() or "unknown ffmpeg error"
        )
        raise RuntimeError(f"Failed to extract segment with ffmpeg: {stderr}")

    return target_path


def _segment_video(
    video_path: str,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> list[tuple[str, float, float, bool]]:
    path = Path(video_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    ranges = plan_video_segments(str(path))
    if len(ranges) <= 1:
        start, end = ranges[0]
        return [(str(path), start, end, False)]

    if status_callback is not None:
        status_callback(
            "Segmenting video into "
            f"{len(ranges)} clips ({DEFAULT_MAX_SEGMENT_DURATION:g}s + "
            f"{DEFAULT_SEGMENT_OVERLAP:g}s overlap)."
        )

    segments: list[tuple[str, float, float, bool]] = []
    try:
        for index, (start_s, end_s) in enumerate(ranges, start=1):
            if status_callback is not None:
                status_callback(
                    f"Preparing segment {index}/{len(ranges)} "
                    f"({_format_timestamp(start_s)}-{_format_timestamp(end_s)})."
                )
            segment_path = _extract_segment(path, start_s, end_s)
            segments.append((str(segment_path), start_s, end_s, True))
    except Exception:
        for segment_file, _, _, is_temp in segments:
            if is_temp:
                _safe_unlink(segment_file)
        raise
    return segments


def _preprocess_video(
    video_path: str,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[str, bool]:
    path = Path(video_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not DEFAULT_PREPROCESS:
        return str(path), False

    info = _probe_video(path)
    codec = str(info.get("codec") or "")
    pix_fmt = str(info.get("pix_fmt") or "")
    height = _as_int(info.get("height"), 0)
    fps = _as_float(info.get("fps"), 0.0)
    duration_s = _as_float(info.get("duration_s"), 0.0)

    needs_reencode = (
        codec != "h264"
        or pix_fmt != "yuv420p"
        or height > DEFAULT_TARGET_RES
        or fps > (DEFAULT_VIDEO_FPS + 0.05)
    )
    if not needs_reencode:
        return str(path), False

    if status_callback is not None:
        status_callback(
            f"Pre-processing video to <= {DEFAULT_TARGET_RES}p @ {DEFAULT_VIDEO_FPS:g}fps."
        )

    target_path = _create_temp_mp4("vllm_preprocess_")

    vf_parts = [
        f"fps={DEFAULT_VIDEO_FPS:g}",
        f"scale=-2:{DEFAULT_TARGET_RES}:force_original_aspect_ratio=decrease",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "format=yuv420p",
    ]
    if duration_s > 0 and duration_s < 1.0:
        vf_parts.append("tpad=stop_mode=clone:stop_duration=1")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(vf_parts),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        _safe_unlink(target_path)
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Failed to pre-process video with ffmpeg: {stderr}")

    return str(target_path), True


def _video_to_data_url(
    video_path: str,
    *,
    preprocess_status_callback: Callable[[str], None] | None = None,
) -> str:
    path = Path(video_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    selected_path = str(path)
    selected_is_temp = False
    if DEFAULT_PREPROCESS:
        selected_path, selected_is_temp = _preprocess_video(
            video_path, status_callback=preprocess_status_callback
        )

    chosen_path = Path(selected_path)
    try:
        mime_type, _ = mimetypes.guess_type(chosen_path.name)
        if not mime_type:
            mime_type = "video/mp4"

        encoded = base64.b64encode(chosen_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    finally:
        if selected_is_temp:
            _safe_unlink(chosen_path)


def _build_messages(
    video_path: str,
    prompt: str,
    *,
    preprocess_status_callback: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": _video_to_data_url(
                            video_path, preprocess_status_callback=preprocess_status_callback
                        )
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _should_retry_with_reencode(exc: Exception) -> bool:
    text = _exception_text(exc)
    known_markers = (
        "number of samples",
        "must be non-negative",
        "temporal_factor",
        "failed to apply qwen3vlprocessor",
        "error in preprocessing prompt inputs",
        "400 bad request",
    )
    return any(marker in text for marker in known_markers)


def _is_frame_sampling_index_error(exc: Exception) -> bool:
    text = _exception_text(exc)
    return (
        "index" in text
        and "out of bounds for axis 0" in text
        and ("failed to apply qwen3vlprocessor" in text or "video_processing_utils" in text)
    )


def _should_retry_without_sampling(exc: Exception) -> bool:
    text = _exception_text(exc)
    if _is_frame_sampling_index_error(exc):
        return True
    return "failed to apply qwen3vlprocessor" in text or (
        "error in preprocessing prompt inputs" in text and "video" in text
    )


def _exception_text(exc: Exception) -> str:
    parts: list[str] = [str(exc)]

    body = getattr(exc, "body", None)
    if body:
        try:
            parts.append(json.dumps(body, ensure_ascii=True, default=str))
        except (TypeError, ValueError):
            parts.append(str(body))

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            response_text = getattr(response, "text", "")
            if response_text:
                parts.append(str(response_text))
        except Exception:
            pass

    return "\n".join(part for part in parts if part).lower()


def _reencode_video_for_vllm(video_path: str) -> str:
    path = Path(video_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    info = _probe_video(path)
    duration_s = _as_float(info.get("duration_s"), 0.0)

    target_path = _create_temp_mp4("vllm_reencode_")
    vf_parts = [
        f"fps={DEFAULT_REENCODE_FPS:g}",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "format=yuv420p",
    ]
    if duration_s > 0 and duration_s < 1.0:
        vf_parts.append("tpad=stop_mode=clone:stop_duration=1")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(vf_parts),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        _safe_unlink(target_path)
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Failed to re-encode video with ffmpeg: {stderr}")

    return str(target_path)


def _build_mm_processor_kwargs(
    *, do_sample_frames: bool | None = None, fps: float | None = None
) -> dict[str, object]:
    sampling_enabled = DEFAULT_DO_SAMPLE_FRAMES if do_sample_frames is None else do_sample_frames
    kwargs: dict[str, object] = {"do_sample_frames": sampling_enabled}
    if sampling_enabled:
        kwargs["fps"] = DEFAULT_VIDEO_FPS if fps is None else fps
    return kwargs


def _create_chat_completion(
    client: OpenAI,
    *,
    video_path: str,
    prompt: str,
    model: str,
    max_tokens: int,
    max_completion_tokens: int | None = None,
    stream: bool,
    preprocess_status_callback: Callable[[str], None] | None = None,
    mm_processor_kwargs: dict[str, object] | None = None,
    enable_thinking: bool | None = None,
) -> Any:
    effective_max_tokens = (
        max_completion_tokens if max_completion_tokens is not None else max_tokens
    )
    extra_body: dict[str, object] = {
        "top_k": 20,
        "mm_processor_kwargs": mm_processor_kwargs or _build_mm_processor_kwargs(),
    }
    if max_completion_tokens is not None:
        extra_body["max_completion_tokens"] = max_completion_tokens
    if enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    messages = cast(
        Any,
        _build_messages(video_path, prompt, preprocess_status_callback=preprocess_status_callback),
    )

    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=0,
        top_p=0.95,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        stream=stream,
        extra_body=extra_body,
    )


def call_vllm(
    video_path: str,
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_completion_tokens: int | None = None,
    api_key: str = DEFAULT_API_KEY,
    preprocess_status_callback: Callable[[str], None] | None = None,
    enable_thinking: bool | None = None,
    include_reasoning: bool = False,
    do_sample_frames: bool | None = None,
) -> str:
    """Send a video + prompt to a vLLM OpenAI-compatible endpoint."""

    def _run_once(path: str, *, mm_processor_kwargs: dict[str, object] | None = None) -> str:
        client = OpenAI(base_url=_normalize_base_url(base_url), api_key=api_key)
        effective_mm_processor_kwargs = mm_processor_kwargs or _build_mm_processor_kwargs(
            do_sample_frames=do_sample_frames
        )
        response = _create_chat_completion(
            client,
            video_path=path,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            stream=False,
            preprocess_status_callback=preprocess_status_callback,
            mm_processor_kwargs=effective_mm_processor_kwargs,
            enable_thinking=enable_thinking,
        )
        if not response.choices:
            return ""
        content, reasoning = _extract_message_parts(response.choices[0].message)
        return _format_message_output(content, reasoning, include_reasoning=include_reasoning)

    retry_without_sampling = False
    no_sampling_kwargs = _build_mm_processor_kwargs(do_sample_frames=False)
    try:
        return _run_once(video_path)
    except Exception as exc:
        latest_exc: Exception = exc
        if _should_retry_without_sampling(exc):
            retry_without_sampling = True
            if preprocess_status_callback is not None:
                preprocess_status_callback(
                    "Video processor sampling failed; retrying with do_sample_frames=false."
                )
            try:
                return _run_once(video_path, mm_processor_kwargs=no_sampling_kwargs)
            except Exception as exc_retry:
                latest_exc = exc_retry

        if not DEFAULT_AUTO_REENCODE_ON_ERROR or not _should_retry_with_reencode(latest_exc):
            if latest_exc is exc:
                raise
            raise latest_exc from exc

        repaired_path = _reencode_video_for_vllm(video_path)
        try:
            mm_kwargs = no_sampling_kwargs if retry_without_sampling else None
            return _run_once(repaired_path, mm_processor_kwargs=mm_kwargs)
        finally:
            _safe_unlink(repaired_path)


def stream_vllm(
    video_path: str,
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_completion_tokens: int | None = None,
    api_key: str = DEFAULT_API_KEY,
    preprocess_status_callback: Callable[[str], None] | None = None,
    enable_thinking: bool | None = None,
    include_reasoning: bool = False,
    do_sample_frames: bool | None = None,
) -> Generator[str, None, None]:
    """Yield incremental text from a streaming vLLM chat completion."""

    def _stream_once(
        path: str,
        *,
        mm_processor_kwargs: dict[str, object] | None = None,
    ) -> Generator[str, None, None]:
        client = OpenAI(base_url=_normalize_base_url(base_url), api_key=api_key)
        effective_mm_processor_kwargs = mm_processor_kwargs or _build_mm_processor_kwargs(
            do_sample_frames=do_sample_frames
        )
        stream = _create_chat_completion(
            client,
            video_path=path,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            stream=True,
            preprocess_status_callback=preprocess_status_callback,
            mm_processor_kwargs=effective_mm_processor_kwargs,
            enable_thinking=enable_thinking,
        )

        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        for event in stream:
            if not event.choices:
                continue
            choice = event.choices[0]
            delta_obj = choice.delta
            content_delta = _coerce_text(getattr(delta_obj, "content", None))
            reasoning_delta = _coerce_text(getattr(delta_obj, "reasoning_content", None))
            if not reasoning_delta:
                reasoning_delta = _coerce_text(getattr(delta_obj, "reasoning", None))

            changed = False
            if content_delta:
                content_chunks.append(content_delta)
                changed = True
            if include_reasoning and reasoning_delta:
                reasoning_chunks.append(reasoning_delta)
                changed = True
            if not changed:
                continue
            yield _format_message_output(
                "".join(content_chunks),
                "".join(reasoning_chunks),
                include_reasoning=include_reasoning,
            )

    retry_without_sampling = False
    no_sampling_kwargs = _build_mm_processor_kwargs(do_sample_frames=False)
    yielded_any = False
    try:
        for chunk in _stream_once(video_path):
            yielded_any = True
            yield chunk
        return
    except Exception as exc:
        latest_exc: Exception = exc
        if not yielded_any and _should_retry_without_sampling(exc):
            retry_without_sampling = True
            if preprocess_status_callback is not None:
                preprocess_status_callback(
                    "Video processor sampling failed; retrying with do_sample_frames=false."
                )
            try:
                for chunk in _stream_once(video_path, mm_processor_kwargs=no_sampling_kwargs):
                    yield chunk
                return
            except Exception as exc_retry:
                latest_exc = exc_retry

        if (
            yielded_any
            or not DEFAULT_AUTO_REENCODE_ON_ERROR
            or not _should_retry_with_reencode(latest_exc)
        ):
            if latest_exc is exc:
                raise
            raise latest_exc from exc

    repaired_path = _reencode_video_for_vllm(video_path)
    try:
        mm_kwargs = no_sampling_kwargs if retry_without_sampling else None
        for chunk in _stream_once(repaired_path, mm_processor_kwargs=mm_kwargs):
            yield chunk
    finally:
        _safe_unlink(repaired_path)


def _render_segmented_output(
    results: list[str], segments: list[tuple[str, float, float, bool]]
) -> str:
    if len(results) == 1:
        return results[0]

    sections: list[str] = []
    for index, ((_, start_s, end_s, _), text) in enumerate(
        zip(segments, results, strict=True), start=1
    ):
        header = f"**Segment {index} [{_format_timestamp(start_s)}-{_format_timestamp(end_s)}]**"
        body = text.strip() or "_No output._"
        sections.append(f"{header}\n{body}")
    return "\n\n".join(sections)


def call_vllm_segmented(
    video_path: str,
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_completion_tokens: int | None = None,
    api_key: str = DEFAULT_API_KEY,
    *,
    max_workers: int | None = None,
    preprocess_status_callback: Callable[[str], None] | None = None,
    segment_status_callback: Callable[[int, int, float, float], None] | None = None,
    enable_thinking: bool | None = None,
    include_reasoning: bool = False,
) -> str:
    """Run video QA with automatic segmentation and optional parallelism.

    Args:
        video_path: Local path to input video.
        prompt: User prompt text.
        base_url: OpenAI-compatible vLLM base URL.
        model: Served model name.
        max_tokens: Legacy max token cap.
        max_completion_tokens: Preferred generation cap.
        api_key: API key value for the OpenAI client.
        max_workers: Maximum number of concurrent segment requests.
        preprocess_status_callback: Optional callback for status lines.
        segment_status_callback: Optional callback with segment boundaries.
        enable_thinking: Optional model reasoning toggle.
        include_reasoning: Include reasoning content in output when available.

    Returns:
        Combined model output across segments.
    """
    segments = _segment_video(video_path, status_callback=preprocess_status_callback)
    total_segments = len(segments)
    segment_do_sample_frames = False if DEFAULT_SEGMENT_DISABLE_SERVER_SAMPLING else None
    if segment_do_sample_frames is False and preprocess_status_callback is not None:
        preprocess_status_callback(
            "Segment mode uses do_sample_frames=false for processor stability."
        )

    if total_segments == 1:
        segment_path, start_s, end_s, _ = segments[0]
        if segment_status_callback is not None:
            segment_status_callback(1, 1, start_s, end_s)
        return call_vllm(
            video_path=segment_path,
            prompt=prompt,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            api_key=api_key,
            preprocess_status_callback=preprocess_status_callback,
            enable_thinking=enable_thinking,
            include_reasoning=include_reasoning,
            do_sample_frames=segment_do_sample_frames,
        )

    workers = max(1, max_workers if max_workers is not None else DEFAULT_MAX_CONCURRENT)
    workers = min(workers, total_segments)
    if preprocess_status_callback is not None:
        preprocess_status_callback(f"Running {total_segments} segments with concurrency {workers}.")

    results: list[str] = [""] * total_segments
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, (segment_path, start_s, end_s, _) in enumerate(segments):
                if segment_status_callback is not None:
                    segment_status_callback(index + 1, total_segments, start_s, end_s)
                future = executor.submit(
                    call_vllm,
                    video_path=segment_path,
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    max_tokens=max_tokens,
                    max_completion_tokens=max_completion_tokens,
                    api_key=api_key,
                    preprocess_status_callback=preprocess_status_callback,
                    enable_thinking=enable_thinking,
                    include_reasoning=include_reasoning,
                    do_sample_frames=segment_do_sample_frames,
                )
                futures[future] = index

            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
    finally:
        for segment_path, _, _, is_temp in segments:
            if is_temp:
                _safe_unlink(segment_path)

    return _render_segmented_output(results, segments)


def stream_vllm_segmented(
    video_path: str,
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_completion_tokens: int | None = None,
    api_key: str = DEFAULT_API_KEY,
    *,
    preprocess_status_callback: Callable[[str], None] | None = None,
    segment_status_callback: Callable[[int, int, float, float], None] | None = None,
    enable_thinking: bool | None = None,
    include_reasoning: bool = False,
) -> Generator[str, None, None]:
    """Stream segmented model output as cumulative markdown text.

    Args:
        video_path: Local path to input video.
        prompt: User prompt text.
        base_url: OpenAI-compatible vLLM base URL.
        model: Served model name.
        max_tokens: Legacy max token cap.
        max_completion_tokens: Preferred generation cap.
        api_key: API key value for the OpenAI client.
        preprocess_status_callback: Optional callback for status lines.
        segment_status_callback: Optional callback with segment boundaries.
        enable_thinking: Optional model reasoning toggle.
        include_reasoning: Include reasoning content in output when available.

    Yields:
        Cumulative markdown output after each streamed chunk.
    """
    segments = _segment_video(video_path, status_callback=preprocess_status_callback)
    total_segments = len(segments)
    sections: list[str] = []
    segment_do_sample_frames = False if DEFAULT_SEGMENT_DISABLE_SERVER_SAMPLING else None
    if segment_do_sample_frames is False and preprocess_status_callback is not None:
        preprocess_status_callback(
            "Segment mode uses do_sample_frames=false for processor stability."
        )

    try:
        for index, (segment_path, start_s, end_s, _) in enumerate(segments, start=1):
            if segment_status_callback is not None:
                segment_status_callback(index, total_segments, start_s, end_s)

            header = (
                f"**Segment {index} [{_format_timestamp(start_s)}-{_format_timestamp(end_s)}]**"
            )
            streamed_text = ""
            yielded_any = False

            for streamed_text in stream_vllm(
                video_path=segment_path,
                prompt=prompt,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                api_key=api_key,
                preprocess_status_callback=preprocess_status_callback,
                enable_thinking=enable_thinking,
                include_reasoning=include_reasoning,
                do_sample_frames=segment_do_sample_frames,
            ):
                yielded_any = True
                body = streamed_text.strip() or "_Waiting for model output..._"
                current = f"{header}\n{body}"
                yield "\n\n".join([*sections, current])

            if not yielded_any:
                streamed_text = call_vllm(
                    video_path=segment_path,
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    max_tokens=max_tokens,
                    max_completion_tokens=max_completion_tokens,
                    api_key=api_key,
                    preprocess_status_callback=preprocess_status_callback,
                    enable_thinking=enable_thinking,
                    include_reasoning=include_reasoning,
                    do_sample_frames=segment_do_sample_frames,
                )

            body = streamed_text.strip() or "_No output._"
            sections.append(f"{header}\n{body}")
            yield "\n\n".join(sections)
    finally:
        for segment_path, _, _, is_temp in segments:
            if is_temp:
                _safe_unlink(segment_path)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run video QA against a local vLLM endpoint")
    parser.add_argument(
        "--video-path", "--video", default="3.mp4", help="Path to a local video file"
    )
    parser.add_argument(
        "--prompt", default="Summarize the video content.", help="Question/instruction"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM OpenAI API base URL")
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY, help="API key for OpenAI-compatible endpoint"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by vLLM")
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens"
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Preferred cap for generated tokens (OpenAI-compatible max_completion_tokens).",
    )
    parser.add_argument(
        "--thinking",
        choices=("auto", "on", "off"),
        default="auto",
        help="Reasoning mode for models that support thinking chat templates.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Include reasoning/thinking text in the returned output.",
    )
    parser.add_argument(
        "--no-segment",
        action="store_true",
        help="Disable automatic segmentation and send the full clip in one request.",
    )
    return parser


def main() -> None:
    """Run CLI entrypoint for local video QA."""
    print_runtime_video_settings(prefix="[vllm-video-cli]")
    parser = _build_cli_parser()
    args = parser.parse_args()
    enable_thinking: bool | None = None
    if args.thinking == "on":
        enable_thinking = True
    elif args.thinking == "off":
        enable_thinking = False

    if args.no_segment:
        answer = call_vllm(
            video_path=args.video_path,
            prompt=args.prompt,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            max_completion_tokens=args.max_completion_tokens,
            api_key=args.api_key,
            enable_thinking=enable_thinking,
            include_reasoning=args.show_thinking,
        )
    else:
        answer = call_vllm_segmented(
            video_path=args.video_path,
            prompt=args.prompt,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            max_completion_tokens=args.max_completion_tokens,
            api_key=args.api_key,
            enable_thinking=enable_thinking,
            include_reasoning=args.show_thinking,
        )
    print(answer)


if __name__ == "__main__":
    main()
