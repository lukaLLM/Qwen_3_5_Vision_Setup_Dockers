"""Unit tests for unified app routing and OpenAI-style API behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from api_server import app
from qwen_image.config import clear_settings_cache


class ApiServerTest(unittest.TestCase):
    """Covers API endpoints, defaults resolution, auth, and UI access behavior."""

    def setUp(self) -> None:
        """Create a fresh test client and temporary mp4 file."""
        clear_settings_cache()
        self.client = TestClient(app)
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with open(path, "wb") as video_file:
            video_file.write(b"test")
        self.video_path = path

    def tearDown(self) -> None:
        """Close client and clean temporary resources."""
        clear_settings_cache()
        self.client.close()
        if os.path.exists(self.video_path):
            os.unlink(self.video_path)

    def test_list_prompts_includes_id(self) -> None:
        """Prompt listing endpoint returns prompt IDs, names, and texts."""
        response = self.client.get("/v1/prompts")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "list")
        self.assertTrue(any(item["name"] == "[Security] Shoplifting" for item in payload["data"]))
        self.assertTrue(all("id" in item for item in payload["data"]))
        self.assertTrue(all("text" in item for item in payload["data"]))

    def test_chat_completion_uses_prompt_name(self) -> None:
        """Inference endpoint resolves prompt text from prompt_name."""
        with mock.patch(
            "qwen_image.api.routes.run_segmented",
            return_value="fire\nVisible flames near the track. Fire behavior is clear.",
        ) as mocked_call:
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "prompt_name": "[Safety] Fire",
                    "video_path": self.video_path,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "ignore"}]}],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["classification"]["label"], "fire")
        sent_prompt = mocked_call.call_args.args[0].prompt
        self.assertIn("Classify this scene using exactly one label", sent_prompt)

    def test_chat_completion_uses_prompt_id(self) -> None:
        """Inference endpoint resolves prompt text from prompt_id."""
        prompts_response = self.client.get("/v1/prompts")
        self.assertEqual(prompts_response.status_code, 200)
        prompts = prompts_response.json()["data"]
        shoplifting = next(item for item in prompts if item["name"] == "[Security] Shoplifting")

        with mock.patch(
            "qwen_image.api.routes.run_segmented",
            return_value="shoplifting\nItem was concealed under jacket.",
        ) as mocked_call:
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "prompt_id": shoplifting["id"],
                    "video_path": self.video_path,
                },
            )

        self.assertEqual(response.status_code, 200)
        sent_prompt = mocked_call.call_args.args[0].prompt
        self.assertIn("Classify this scene", sent_prompt)

    def test_chat_completion_prompt_text_overrides_prompt_id_and_name(self) -> None:
        """Explicit prompt_text has highest precedence."""
        prompts_response = self.client.get("/v1/prompts")
        self.assertEqual(prompts_response.status_code, 200)
        prompts = prompts_response.json()["data"]
        burglary = next(item for item in prompts if item["name"] == "[Security] Burglary")

        with mock.patch(
            "qwen_image.api.routes.run_segmented",
            return_value="normal\nNothing suspicious is visible.",
        ) as mocked_call:
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "prompt_id": burglary["id"],
                    "prompt_name": "[Safety] Fire",
                    "prompt_text": "CUSTOM PROMPT TEXT",
                    "video_path": self.video_path,
                },
            )

        self.assertEqual(response.status_code, 200)
        sent_prompt = mocked_call.call_args.args[0].prompt
        self.assertEqual(sent_prompt, "CUSTOM PROMPT TEXT")

    def test_chat_completion_falls_back_to_messages_text(self) -> None:
        """When no prompt fields are given, endpoint uses user message text."""
        with mock.patch(
            "qwen_image.api.routes.run_segmented",
            return_value="normal\nNo danger is visible.",
        ) as mocked_call:
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "MESSAGE PROMPT"},
                                {"type": "video_url", "video_url": {"url": self.video_path}},
                            ],
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        sent_prompt = mocked_call.call_args.args[0].prompt
        self.assertEqual(sent_prompt, "MESSAGE PROMPT")

    def test_chat_completion_accepts_top_level_http_video_url(self) -> None:
        """Endpoint accepts top-level HTTP(S) video_url and downloads it."""
        with (
            mock.patch("qwen_image.api.routes._download_remote_video_to_temp_file") as mocked_dl,
            mock.patch(
                "qwen_image.api.routes.run_segmented",
                return_value="normal\nScene appears routine.",
            ) as mocked_call,
        ):
            temp_remote = Path(tempfile.mkstemp(suffix=".mp4")[1])
            temp_remote.write_bytes(b"video")
            mocked_dl.return_value = temp_remote
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "prompt_name": "[Security] Shoplifting",
                    "video_url": "https://example.com/demo.mp4",
                },
            )

        self.assertEqual(response.status_code, 200)
        mocked_dl.assert_called_once_with("https://example.com/demo.mp4")
        sent_video_path = mocked_call.call_args.args[0].video_path
        self.assertIn("/tmp/", sent_video_path)

    def test_video_url_private_host_is_blocked(self) -> None:
        """Private-network video URLs are blocked when protection is enabled."""
        with (
            mock.patch.dict(os.environ, {"API_BLOCK_PRIVATE_URLS": "1"}, clear=False),
            mock.patch("qwen_image.api.routes._is_private_or_loopback", return_value=True),
        ):
            clear_settings_cache()
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "prompt_name": "[Security] Shoplifting",
                    "video_url": "http://192.168.1.10/video.mp4",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Blocked private", response.json()["detail"])

    def test_chat_completion_uses_env_defaults_when_optional_fields_missing(self) -> None:
        """If optional request fields are missing, env defaults are applied."""
        with (
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_MODEL": "ENV_MODEL",
                    "VLLM_BASE_URL": "http://env-host:9999/v1",
                    "VLLM_MAX_COMPLETION_TOKENS": "321",
                    "VLLM_THINKING_MODE": "off",
                },
                clear=False,
            ),
            mock.patch(
                "qwen_image.api.routes.run_segmented",
                return_value="normal\nScene appears routine.",
            ) as mocked_call,
        ):
            clear_settings_cache()
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "prompt_name": "[Security] Shoplifting",
                    "video_path": self.video_path,
                },
            )

        self.assertEqual(response.status_code, 200)
        inference_request = mocked_call.call_args.args[0]
        self.assertEqual(inference_request.model, "ENV_MODEL")
        self.assertEqual(inference_request.base_url, "http://env-host:9999/v1")
        self.assertEqual(inference_request.max_completion_tokens, 321)
        self.assertFalse(inference_request.enable_thinking)

    def test_api_requires_token_when_configured(self) -> None:
        """When API_AUTH_TOKEN is set, /v1 requests require Bearer auth."""
        with mock.patch.dict(os.environ, {"API_AUTH_TOKEN": "secret-token"}, clear=False):
            clear_settings_cache()
            unauthorized = self.client.get("/v1/prompts")
            authorized = self.client.get(
                "/v1/prompts",
                headers={"Authorization": "Bearer secret-token"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)

    def test_api_remains_open_when_token_not_set(self) -> None:
        """Without API_AUTH_TOKEN configured, API stays open for local dev."""
        with mock.patch.dict(os.environ, {"API_AUTH_TOKEN": ""}, clear=False):
            clear_settings_cache()
            response = self.client.get("/v1/prompts")

        self.assertEqual(response.status_code, 200)

    def test_gui_is_local_only_by_default(self) -> None:
        """GUI rejects non-loopback clients when GUI_LOCAL_ONLY is enabled."""
        with mock.patch.dict(os.environ, {"GUI_LOCAL_ONLY": "1"}, clear=False):
            clear_settings_cache()
            blocked = self.client.get("/ui", headers={"X-Forwarded-For": "8.8.8.8"})
            allowed = self.client.get("/ui", headers={"X-Forwarded-For": "127.0.0.1"})

        self.assertEqual(blocked.status_code, 403)
        self.assertIn(allowed.status_code, {200, 301, 302, 307, 308})

    def test_missing_prompt_returns_400(self) -> None:
        """Prompt source is required across prompt_name/prompt_id/prompt_text/messages."""
        response = self.client.post(
            "/v1/chat/completions",
            json={"video_path": self.video_path},
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_video_returns_400(self) -> None:
        """Video source is required via video_path/video_url/messages video_url."""
        response = self.client.post(
            "/v1/chat/completions",
            json={"prompt_name": "[Safety] Fire"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
