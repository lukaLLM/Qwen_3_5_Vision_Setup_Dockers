"""Adapter that reuses the existing Gradio UI implementation."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from vllm_video_call import print_runtime_video_settings


def _resolve_legacy_gui_module() -> ModuleType:
    """Resolve the GUI module, supporting `python gui.py` execution mode."""
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, "build_app"):
        return main_mod
    return importlib.import_module("gui")


legacy_gui = _resolve_legacy_gui_module()


def build_ui_blocks() -> object:
    """Build and return the Gradio Blocks instance for UI mounting."""
    return legacy_gui.build_app().queue(default_concurrency_limit=4)


def ui_theme() -> object:
    """Return the UI theme object from legacy GUI module."""
    return legacy_gui.APP_THEME


def ui_css() -> str:
    """Return the UI CSS from legacy GUI module."""
    return legacy_gui.CUSTOM_CSS


def print_ui_startup_banner() -> None:
    """Print current UI runtime settings."""
    print_runtime_video_settings(prefix="[gui-startup]")
    print(
        (
            "[gui-startup] "
            f"GUI_STREAM_OUTPUT={legacy_gui.DEFAULT_STREAM_OUTPUT}, "
            f"GUI_DEBUG={legacy_gui.DEFAULT_DEBUG_MODE}"
        ),
        flush=True,
    )
