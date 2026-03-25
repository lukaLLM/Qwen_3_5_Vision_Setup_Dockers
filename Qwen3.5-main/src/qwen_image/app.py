"""Unified FastAPI application factory (API + mounted Gradio UI)."""

from __future__ import annotations

import hmac
import ipaddress
from typing import Any

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from qwen_image.api import router as api_router
from qwen_image.config import get_settings
from qwen_image.ui import build_ui_blocks, ui_css, ui_theme


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",", maxsplit=1)[0].strip()
    if request.client is not None and request.client.host:
        return request.client.host
    return ""


def _is_loopback_client(address: str) -> bool:
    if not address:
        return False
    lowered = address.lower()
    if lowered in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def create_app() -> FastAPI:
    """Create the unified app exposing `/v1/*` API and mounted UI."""
    settings = get_settings()

    app = FastAPI(title="Qwen3.5 Video Classification API", version="2.0.0")

    @app.middleware("http")
    async def access_middleware(request: Request, call_next: Any) -> Any:
        path = request.url.path
        runtime_settings = get_settings()

        if path == "/v1" or path.startswith("/v1/"):
            expected_token = runtime_settings.security.api_auth_token
            if expected_token:
                auth_header = request.headers.get("authorization", "").strip()
                if not auth_header.lower().startswith("bearer "):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing bearer token in Authorization header."},
                    )
                provided_token = auth_header[7:].strip()
                if not hmac.compare_digest(provided_token, expected_token):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid bearer token."},
                    )

        if runtime_settings.security.gui_local_only:
            ui_path = runtime_settings.server.ui_path
            if (
                (path == ui_path or path.startswith(f"{ui_path}/"))
                and not _is_loopback_client(_client_ip(request))
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "GUI is restricted to local loopback clients."},
                )

        return await call_next(request)

    app.include_router(api_router)

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "status": "ok",
            "api": "/v1",
            "ui": settings.server.ui_path,
        }

    blocks = build_ui_blocks()
    return gr.mount_gradio_app(
        app,
        blocks,
        path=settings.server.ui_path,
        theme=ui_theme(),
        css=ui_css(),
    )
