#!/usr/bin/env python3
"""Simple smoke check for the local MM lab API."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def _http_get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=data,
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    """Run a lightweight health + optional run smoke check against MM lab."""
    parser = argparse.ArgumentParser(description="Smoke test MM lab health and run endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7870", help="MM lab host URL.")
    parser.add_argument(
        "--api-prefix",
        default="/api",
        help="API prefix path (default: /api).",
    )
    parser.add_argument("--prompt", default="Describe the media inputs.", help="Prompt text.")
    parser.add_argument("--image", action="append", default=[], help="Image path (repeatable).")
    parser.add_argument("--video", default="", help="Video path.")
    parser.add_argument("--health-only", action="store_true", help="Only check /health endpoint.")
    args = parser.parse_args()

    prefix = args.api_prefix.strip() or "/api"
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    if len(prefix) > 1:
        prefix = prefix.rstrip("/")

    health_url = f"{args.base_url.rstrip('/')}{prefix}/health"
    run_url = f"{args.base_url.rstrip('/')}{prefix}/run"

    try:
        health = _http_get_json(health_url)
    except urllib.error.URLError as exc:
        print(f"[mm-lab-smoke] health check failed: {exc}")
        return 1

    print("[mm-lab-smoke] health:", json.dumps(health, indent=2))
    if args.health_only:
        return 0

    payload: dict[str, object] = {"prompt": args.prompt}
    if args.image:
        payload["image_paths"] = args.image
    if args.video.strip():
        payload["video_path"] = args.video.strip()

    try:
        run_result = _http_post_json(run_url, payload)
    except urllib.error.URLError as exc:
        print(f"[mm-lab-smoke] run request failed: {exc}")
        return 1

    print("[mm-lab-smoke] run:", json.dumps(run_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
