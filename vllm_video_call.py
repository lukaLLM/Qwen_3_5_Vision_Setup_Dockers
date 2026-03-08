#!/usr/bin/env python3
import argparse
import base64
import mimetypes
import pathlib
import sys

from openai import APIError, OpenAI


def file_to_data_url(path: pathlib.Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "video/mp4"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a video prompt to local vLLM OpenAI-compatible API."
    )
    parser.add_argument("--video", default="3.mp4", help="Path to video file.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="sk-no-key-required")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    parser.add_argument(
        "--prompt", default="What is the number of the play in center of the video ?"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print only assistant text instead of full JSON response.",
    )
    args = parser.parse_args()

    video_path = pathlib.Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    data_url = file_to_data_url(video_path)
    client = OpenAI(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
    )

    try:
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "video_url", "video_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=args.max_tokens,
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
