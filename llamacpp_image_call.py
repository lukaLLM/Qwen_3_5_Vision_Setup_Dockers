#!/usr/bin/env python3
import argparse
import base64
import mimetypes
import pathlib
import sys
from typing import Any

from openai import APIError, OpenAI


def file_to_data_url(path: pathlib.Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send an image prompt to local llama.cpp OpenAI-compatible API."
    )
    parser.add_argument("--image", default="5.jpeg", help="Path to image file.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="sk-no-key-required")
    parser.add_argument("--model", default="qwen35-122b-a10b-q4_k_m")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--reasoning-format",
        default="none",
        help="llama.cpp reasoning parser mode (use 'none' to bypass parser issues).",
    )
    parser.add_argument(
        "--parse-tool-calls",
        action="store_true",
        help="Enable llama.cpp tool-call parsing (off by default).",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print only assistant text instead of full JSON response.",
    )
    args = parser.parse_args()

    image_path = pathlib.Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    data_url = file_to_data_url(image_path)
    client = OpenAI(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
    )

    try:
        extra_body: dict[str, Any] = {
            "reasoning_format": args.reasoning_format,
            "parse_tool_calls": args.parse_tool_calls,
        }
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
    except APIError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    if args.text_only:
        message = completion.choices[0].message
        print(message.content or "")
    else:
        print(completion.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
