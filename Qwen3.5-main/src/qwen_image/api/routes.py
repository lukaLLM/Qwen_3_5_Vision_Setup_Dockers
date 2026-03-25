"""FastAPI routes for prompt discovery and video chat completions."""

from __future__ import annotations

import base64
import ipaddress
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from qwen_image.api.schemas import ChatCompletionRequest
from qwen_image.config import get_settings
from qwen_image.inference.client import run_segmented
from qwen_image.inference.service import InferenceOverrides, build_inference_call
from qwen_image.prompts import (
    CLASSIFICATION_PROMPT_LABELS,
    find_prompt,
    find_prompt_by_id,
    load_prompts,
    parse_classification_label,
    prompt_records_for_api,
    split_classification_output,
)

router = APIRouter(prefix="/v1")


def _load_prompts_or_500() -> list[dict[str, str]]:
    try:
        return load_prompts()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Prompt configuration error: {exc}") from exc


def _extract_user_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return [item for item in content if isinstance(item, dict)]
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
    return []


def _extract_text_from_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in _extract_user_content(messages):
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _extract_video_url(messages: list[dict[str, Any]]) -> str | None:
    for item in _extract_user_content(messages):
        if item.get("type") != "video_url":
            continue
        video_url = item.get("video_url")
        if isinstance(video_url, dict):
            url = video_url.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
    return None


def _decode_data_url_to_temp_file(data_url: str) -> Path:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid data URL format.")
    _, payload = data_url.split(",", 1)
    try:
        decoded = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid base64 video payload: {exc}") from exc

    tmp_path = Path(f"/tmp/qwen_api_{uuid.uuid4().hex}.mp4")
    tmp_path.write_bytes(decoded)
    return tmp_path


def _is_private_or_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True

    try:
        addr_infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resolve video_url host: {hostname}",
        ) from exc

    for info in addr_infos:
        ip_raw = info[4][0]
        ip_obj = ipaddress.ip_address(ip_raw)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            return True
    return False


def _download_remote_video_to_temp_file(url: str) -> Path:
    settings = get_settings().security
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid video_url host.")

    if settings.block_private_urls and _is_private_or_loopback(hostname):
        raise HTTPException(
            status_code=400,
            detail="Blocked private or loopback video_url host."
            " Set API_BLOCK_PRIVATE_URLS=0 only in trusted environments.",
        )

    timeout = settings.video_url_timeout_seconds
    max_bytes = settings.video_url_max_mb * 1024 * 1024
    tmp_path = Path(f"/tmp/qwen_api_{uuid.uuid4().hex}.mp4")

    request = urllib.request.Request(url, headers={"User-Agent": "qwen-image-api/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            out_file = tmp_path.open("wb")
            with out_file:
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "video_url payload too large "
                                f"({downloaded / (1024 * 1024):.1f} MB)."
                            ),
                        )
                    out_file.write(chunk)
    except HTTPException:
        with suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise
    except TimeoutError as exc:
        with suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise HTTPException(status_code=504, detail="video_url download timed out.") from exc
    except urllib.error.URLError as exc:
        with suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to fetch video_url: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        with suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise HTTPException(status_code=502, detail=f"video_url download failed: {exc}") from exc

    return tmp_path


def _resolve_video_path(request: ChatCompletionRequest) -> tuple[Path, bool]:
    if request.video_path:
        candidate = Path(request.video_path).expanduser()
        if candidate.exists():
            return candidate, False
        raise HTTPException(status_code=400, detail=f"video_path not found: {candidate}")

    url = request.video_url.strip() if request.video_url and request.video_url.strip() else None
    if not url:
        url = _extract_video_url(request.messages)

    if not url:
        raise HTTPException(
            status_code=400,
            detail=(
                "No video input found. Provide video_path, video_url, "
                "or messages[].content video_url."
            ),
        )

    lowered = url.lower()
    if lowered.startswith("data:"):
        return _decode_data_url_to_temp_file(url), True

    if lowered.startswith("http://") or lowered.startswith("https://"):
        return _download_remote_video_to_temp_file(url), True

    candidate = (
        Path(url[7:]).expanduser() if lowered.startswith("file://") else Path(url).expanduser()
    )
    if candidate.exists():
        return candidate, False
    raise HTTPException(status_code=400, detail=f"Unsupported/invalid video URL: {url}")


def _resolve_prompt(request: ChatCompletionRequest) -> tuple[str, str | None]:
    if request.prompt_text and request.prompt_text.strip():
        return request.prompt_text.strip(), request.prompt_name

    prompts = _load_prompts_or_500()

    if request.prompt_id and request.prompt_id.strip():
        prompt = find_prompt_by_id(prompts, request.prompt_id)
        if prompt is None:
            raise HTTPException(status_code=404, detail=f"Unknown prompt_id: {request.prompt_id}")
        return prompt["text"], prompt["name"]

    if request.prompt_name:
        prompt = find_prompt(prompts, request.prompt_name)
        if prompt is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown prompt_name: {request.prompt_name}",
            )
        return prompt["text"], request.prompt_name

    message_text = _extract_text_from_messages(request.messages)
    if message_text:
        return message_text, None

    raise HTTPException(
        status_code=400,
        detail=(
            "No prompt text found. Provide prompt_text, prompt_id, "
            "prompt_name, or messages text content."
        ),
    )


@router.get("/prompts")
def list_prompts() -> dict[str, Any]:
    """Return ready prompts with full text and optional classification labels."""
    _load_prompts_or_500()
    return {"object": "list", "data": prompt_records_for_api()}


@router.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    """Run segmented video inference with OpenAI-style request payload fields."""
    prompt_text, selected_prompt_name = _resolve_prompt(request)
    video_path, should_cleanup = _resolve_video_path(request)

    overrides = InferenceOverrides(
        base_url=request.base_url,
        model=request.model,
        max_tokens=request.max_tokens,
        max_completion_tokens=request.max_completion_tokens,
        api_key=request.api_key,
        thinking_mode=request.thinking,
    )

    try:
        inference_request = build_inference_call(
            video_path=str(video_path),
            prompt=prompt_text,
            overrides=overrides,
        )
        output_text = run_segmented(inference_request)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Inference failed: {exc}") from exc
    finally:
        if should_cleanup:
            with suppress(FileNotFoundError):
                os.unlink(video_path)

    label_line, explanation = split_classification_output(output_text)
    parsed_label: str | None = None
    if selected_prompt_name and selected_prompt_name in CLASSIFICATION_PROMPT_LABELS:
        candidate = parse_classification_label(selected_prompt_name, label_line)
        if candidate != "parse_error":
            parsed_label = candidate

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": inference_request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "classification": {
            "first_line": label_line,
            "label": parsed_label,
            "explanation": explanation,
        },
    }
