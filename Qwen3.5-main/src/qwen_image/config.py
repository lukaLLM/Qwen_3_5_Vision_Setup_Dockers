"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def load_dotenv_defaults(dotenv_path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from `.env` into process env if keys are unset."""
    default_path = Path(__file__).resolve().parents[2] / ".env"
    path = dotenv_path or default_path
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


def _normalize_ui_path(path: str) -> str:
    cleaned = path.strip() or "/ui"
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    if len(cleaned) > 1:
        cleaned = cleaned.rstrip("/")
    return cleaned


def _normalize_thinking_mode(mode: str) -> str:
    cleaned = mode.strip().lower()
    if cleaned in {"on", "off", "auto"}:
        return cleaned
    return "auto"


@dataclass(frozen=True)
class InferenceSettings:
    """Default inference/runtime values resolved from env."""

    base_url: str
    model: str
    api_key: str
    max_tokens: int
    max_completion_tokens: int
    video_fps: float
    do_sample_frames: bool
    auto_reencode_on_error: bool
    reencode_fps: float
    presence_penalty: float
    preprocess: bool
    target_res: int
    max_duration: float
    segment_overlap: float
    max_concurrent: int
    segment_disable_server_sampling: bool
    thinking_mode: str


@dataclass(frozen=True)
class GuiSettings:
    """GUI-specific runtime defaults."""

    stream_output: bool
    debug: bool


@dataclass(frozen=True)
class ServerSettings:
    """Unified server binding and mount-path settings."""

    host: str
    port: int
    ui_path: str


@dataclass(frozen=True)
class SecuritySettings:
    """Security and access-control settings for API/UI endpoints."""

    api_auth_token: str | None
    gui_local_only: bool
    video_url_timeout_seconds: float
    video_url_max_mb: int
    block_private_urls: bool


@dataclass(frozen=True)
class PathSettings:
    """Filesystem paths used by app components."""

    prompts_path: Path
    dotenv_path: Path


@dataclass(frozen=True)
class AppSettings:
    """Root settings object shared by API and UI layers."""

    inference: InferenceSettings
    gui: GuiSettings
    server: ServerSettings
    security: SecuritySettings
    paths: PathSettings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached app settings loaded from env and `.env` defaults."""
    load_dotenv_defaults()
    root = _project_root()

    max_tokens = max(1, _env_int("VLLM_MAX_TOKENS", 2048))
    max_completion_tokens = max(
        1, _env_int("VLLM_MAX_COMPLETION_TOKENS", max_tokens)
    )

    inference = InferenceSettings(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip()
        or "http://localhost:8000/v1",
        model=os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B").strip() or "Qwen/Qwen3.5-9B",
        api_key=os.getenv("VLLM_API_KEY", "EMPTY").strip() or "EMPTY",
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        video_fps=_env_float("VLLM_VIDEO_FPS", 2.0),
        do_sample_frames=_env_bool("VLLM_DO_SAMPLE_FRAMES", True),
        auto_reencode_on_error=_env_bool("VLLM_AUTO_REENCODE_ON_ERROR", True),
        reencode_fps=_env_float("VLLM_REENCODE_FPS", 2.0),
        presence_penalty=_env_float("VLLM_PRESENCE_PENALTY", 0.3),
        preprocess=_env_bool("VLLM_PREPROCESS", True),
        target_res=max(64, _env_int("VLLM_TARGET_RES", 480)),
        max_duration=max(1.0, _env_float("VLLM_MAX_DURATION", 60.0)),
        segment_overlap=max(0.0, _env_float("VLLM_SEGMENT_OVERLAP", 3.0)),
        max_concurrent=max(1, _env_int("VLLM_MAX_CONCURRENT", 1)),
        segment_disable_server_sampling=_env_bool(
            "VLLM_SEGMENT_DISABLE_SERVER_SAMPLING", True
        ),
        thinking_mode=_normalize_thinking_mode(os.getenv("VLLM_THINKING_MODE", "auto")),
    )

    gui = GuiSettings(
        stream_output=_env_bool("GUI_STREAM_OUTPUT", True),
        debug=_env_bool("GUI_DEBUG", False),
    )

    server = ServerSettings(
        host=os.getenv("APP_HOST", "0.0.0.0").strip() or "0.0.0.0",
        port=max(1, _env_int("APP_PORT", 7861)),
        ui_path=_normalize_ui_path(os.getenv("APP_UI_PATH", "/ui")),
    )
    raw_token = os.getenv("API_AUTH_TOKEN", "").strip()
    security = SecuritySettings(
        api_auth_token=raw_token or None,
        gui_local_only=_env_bool("GUI_LOCAL_ONLY", True),
        video_url_timeout_seconds=max(1.0, _env_float("API_VIDEO_URL_TIMEOUT_SECONDS", 30.0)),
        video_url_max_mb=max(1, _env_int("API_VIDEO_URL_MAX_MB", 200)),
        block_private_urls=_env_bool("API_BLOCK_PRIVATE_URLS", True),
    )

    prompt_env = os.getenv("PROMPTS_PATH", "").strip()
    prompts_path = Path(prompt_env).expanduser() if prompt_env else root / "prompts.yaml"
    dotenv_path = root / ".env"

    paths = PathSettings(prompts_path=prompts_path, dotenv_path=dotenv_path)
    return AppSettings(
        inference=inference,
        gui=gui,
        server=server,
        security=security,
        paths=paths,
    )


def clear_settings_cache() -> None:
    """Clear memoized settings for tests that mutate environment."""
    get_settings.cache_clear()
