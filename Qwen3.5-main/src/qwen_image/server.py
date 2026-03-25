"""Unified server entrypoint for API + UI."""

from __future__ import annotations

import uvicorn

from qwen_image.app import create_app
from qwen_image.config import get_settings
from qwen_image.ui import print_ui_startup_banner

app = create_app()


def run_server() -> None:
    """Run the unified application with env-driven host/port."""
    settings = get_settings()
    print_ui_startup_banner()
    print(
        (
            f"[server] API and UI running on http://{settings.server.host}:{settings.server.port} "
            f"(UI path: {settings.server.ui_path}, API prefix: /v1)"
        ),
        flush=True,
    )
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, reload=False)


if __name__ == "__main__":
    run_server()
