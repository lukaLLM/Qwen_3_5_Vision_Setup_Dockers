"""Settings for the local multimodal experimentation lab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


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


def _normalize_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        return "http://127.0.0.1:8000/v1"
    if not base.endswith("/v1"):
        return f"{base}/v1"
    return base


def _normalize_path(value: str, default: str) -> str:
    cleaned = value.strip() if value.strip() else default
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    if len(cleaned) > 1:
        cleaned = cleaned.rstrip("/")
    return cleaned


def _load_dotenv_defaults(dotenv_path: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    path = dotenv_path or root / ".env"
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

        parsed = value.strip()
        if (parsed.startswith('"') and parsed.endswith('"')) or (
            parsed.startswith("'") and parsed.endswith("'")
        ):
            parsed = parsed[1:-1]
        os.environ[key] = parsed


@dataclass(frozen=True)
class LabSettings:
    """Environment-resolved runtime defaults for the MM lab."""

    host: str
    port: int
    ui_path: str
    api_prefix: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    results_dir: Path
    default_target_height: int
    default_video_fps: float
    default_safe_video_sampling: bool
    default_measure_ttft: bool
    max_benchmark_workers: int


@lru_cache(maxsize=1)
def get_settings() -> LabSettings:
    """Load and return cached MM lab settings."""
    _load_dotenv_defaults()
    root = Path(__file__).resolve().parents[1]

    base_url_env = (
        os.getenv("MM_LAB_VLLM_BASE_URL", "").strip()
        or os.getenv("VLLM_BASE_URL", "").strip()
        or "http://127.0.0.1:8000/v1"
    )
    model_env = (
        os.getenv("MM_LAB_VLLM_MODEL", "").strip()
        or os.getenv("VLLM_MODEL", "").strip()
        or "Qwen/Qwen3.5-4B"
    )
    api_key_env = (
        os.getenv("MM_LAB_VLLM_API_KEY", "").strip()
        or os.getenv("VLLM_API_KEY", "").strip()
        or "EMPTY"
    )

    results_dir_raw = os.getenv("MM_LAB_RESULTS_DIR", "").strip()
    if results_dir_raw:
        results_dir = Path(results_dir_raw).expanduser()
    else:
        results_dir = root / "visual_experimentation_app" / "results"

    return LabSettings(
        host=os.getenv("MM_LAB_HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=max(1, _env_int("MM_LAB_PORT", 7870)),
        ui_path=_normalize_path(os.getenv("MM_LAB_UI_PATH", "/"), "/"),
        api_prefix=_normalize_path(os.getenv("MM_LAB_API_PREFIX", "/api"), "/api"),
        base_url=_normalize_base_url(base_url_env),
        model=model_env,
        api_key=api_key_env,
        timeout_seconds=max(1.0, _env_float("MM_LAB_VLLM_TIMEOUT_SECONDS", 180.0)),
        results_dir=results_dir,
        default_target_height=max(
            64,
            _env_int(
                "MM_LAB_DEFAULT_TARGET_HEIGHT",
                _env_int("VLLM_TARGET_RES", 480),
            ),
        ),
        default_video_fps=max(
            0.1,
            _env_float(
                "MM_LAB_DEFAULT_VIDEO_FPS",
                _env_float("VLLM_VIDEO_FPS", 2.0),
            ),
        ),
        default_safe_video_sampling=_env_bool("MM_LAB_SAFE_VIDEO_SAMPLING", False),
        default_measure_ttft=_env_bool("MM_LAB_MEASURE_TTFT", True),
        max_benchmark_workers=max(1, _env_int("MM_LAB_MAX_BENCHMARK_WORKERS", 16)),
    )


def clear_settings_cache() -> None:
    """Clear memoized settings for tests that mutate the environment."""
    get_settings.cache_clear()
