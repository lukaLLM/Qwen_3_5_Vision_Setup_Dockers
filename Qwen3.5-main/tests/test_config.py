"""Tests for shared settings and fallback precedence."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_image.config import clear_settings_cache, get_settings  # noqa: E402
from qwen_image.inference.service import (  # noqa: E402
    InferenceOverrides,
    build_inference_call,
)


class ConfigPrecedenceTest(unittest.TestCase):
    """Ensures request/env fallback precedence is stable and explicit."""

    def tearDown(self) -> None:
        """Reset memoized settings after each test case."""
        clear_settings_cache()

    def test_env_defaults_are_loaded_when_request_is_missing(self) -> None:
        """Missing request values fall back to env-configured defaults."""
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_MODEL": "ENV_MODEL",
                "VLLM_BASE_URL": "http://env:8888/v1",
                "VLLM_MAX_COMPLETION_TOKENS": "444",
            },
            clear=False,
        ):
            clear_settings_cache()
            call = build_inference_call(
                video_path="/tmp/demo.mp4",
                prompt="hello",
                overrides=InferenceOverrides(),
            )

        self.assertEqual(call.model, "ENV_MODEL")
        self.assertEqual(call.base_url, "http://env:8888/v1")
        self.assertEqual(call.max_completion_tokens, 444)

    def test_request_values_override_env_defaults(self) -> None:
        """Explicit request values take priority over env defaults."""
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_MODEL": "ENV_MODEL",
                "VLLM_BASE_URL": "http://env:8888/v1",
                "VLLM_MAX_COMPLETION_TOKENS": "444",
            },
            clear=False,
        ):
            clear_settings_cache()
            call = build_inference_call(
                video_path="/tmp/demo.mp4",
                prompt="hello",
                overrides=InferenceOverrides(
                    model="REQ_MODEL",
                    base_url="http://req:9999/v1",
                    max_completion_tokens=123,
                ),
            )

        self.assertEqual(call.model, "REQ_MODEL")
        self.assertEqual(call.base_url, "http://req:9999/v1")
        self.assertEqual(call.max_completion_tokens, 123)

    def test_server_settings_read_new_app_env_vars(self) -> None:
        """Unified server host/port/UI-path are controlled by APP_* vars."""
        with mock.patch.dict(
            os.environ,
            {
                "APP_HOST": "127.0.0.1",
                "APP_PORT": "9001",
                "APP_UI_PATH": "dashboard",
            },
            clear=False,
        ):
            clear_settings_cache()
            settings = get_settings()

        self.assertEqual(settings.server.host, "127.0.0.1")
        self.assertEqual(settings.server.port, 9001)
        self.assertEqual(settings.server.ui_path, "/dashboard")

    def test_security_settings_read_auth_and_video_url_guards(self) -> None:
        """Security settings are loaded from new API/GUI env vars."""
        with mock.patch.dict(
            os.environ,
            {
                "API_AUTH_TOKEN": "abc123",
                "GUI_LOCAL_ONLY": "1",
                "API_VIDEO_URL_TIMEOUT_SECONDS": "45",
                "API_VIDEO_URL_MAX_MB": "123",
                "API_BLOCK_PRIVATE_URLS": "0",
            },
            clear=False,
        ):
            clear_settings_cache()
            settings = get_settings()

        self.assertEqual(settings.security.api_auth_token, "abc123")
        self.assertTrue(settings.security.gui_local_only)
        self.assertEqual(settings.security.video_url_timeout_seconds, 45.0)
        self.assertEqual(settings.security.video_url_max_mb, 123)
        self.assertFalse(settings.security.block_private_urls)


if __name__ == "__main__":
    unittest.main()
