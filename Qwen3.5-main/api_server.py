"""Compatibility entrypoint for the unified API + UI server."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_image.server import app, run_server  # noqa: E402

__all__ = ["app"]

if __name__ == "__main__":
    run_server()
