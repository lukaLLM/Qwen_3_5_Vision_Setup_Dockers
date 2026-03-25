#!/usr/bin/env python3
"""Evaluate classification prompts against videos in /home/onebonsai/Videos."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gui import _split_classification_output, find_prompt, load_prompts
from vllm_video_call import DEFAULT_BASE_URL, DEFAULT_MODEL, call_vllm_segmented

FOLDER_TO_PROMPT: dict[str, str] = {
    "bulgary": "[Security] Burglary",
    "shoplifiting": "[Security] Shoplifting",
    "train": "[Safety] Railroad tracks",
    "fire": "[Safety] Fire",
    "warehouse": "[Safety] Warehouse",
}


@dataclass(slots=True)
class EvalRow:
    """One evaluated clip row."""

    folder: str
    file: str
    expected: str
    predicted: str
    ok: bool


def _norm_token(value: str) -> str:
    lowered = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered)
    return cleaned.strip("_")


def _expected_label(folder: str, stem: str) -> str | None:
    token = _norm_token(stem)

    if folder == "bulgary":
        if "normal" in token:
            return "normal"
        if "sus" in token:
            return "suspicious"
        if "bulgary" in token or "burglary" in token:
            return "burglary"
        return None

    if folder == "shoplifiting":
        if "normal" in token:
            return "normal"
        if "sus" in token:
            return "suspicious"
        if "shoplifiting" in token or "shoplifting" in token:
            return "shoplifting"
        return None

    if folder == "train":
        if "no_danger" in token or "nodanger" in token or "safe" in token:
            return "not_on_tracks"
        if "danger" in token:
            return "on_tracks"
        return None

    if folder == "fire":
        if "no_fire" in token or "nofire" in token or "normal" in token or "safe" in token:
            return "no_fire"
        if "fire" in token:
            return "fire"
        return None

    if folder == "warehouse":
        if "no_helmet" in token or "nohelmet" in token or "helmet_off" in token:
            return "helmet_off"
        if "sus" in token:
            return "suspicious"
        if "normal" in token:
            return "normal"
        return None

    return None


def _normalize_predicted_label(prompt_name: str, first_line: str) -> str:
    token = _norm_token(first_line)

    if prompt_name == "[Security] Burglary":
        if "burglary" in token or "bulgary" in token:
            return "burglary"
        if "susp" in token or token == "sus":
            return "suspicious"
        if "normal" in token:
            return "normal"
        return token

    if prompt_name == "[Security] Shoplifting":
        if "shoplifting" in token or "shoplifiting" in token:
            return "shoplifting"
        if "susp" in token or token == "sus":
            return "suspicious"
        if "normal" in token:
            return "normal"
        return token

    if prompt_name == "[Safety] Railroad tracks":
        if "not_on_tracks" in token or "no_danger" in token or "safe" in token:
            return "not_on_tracks"
        if "on_tracks" in token or "danger" in token:
            return "on_tracks"
        return token

    if prompt_name == "[Safety] Fire":
        if "no_fire" in token or "nofire" in token:
            return "no_fire"
        if "fire" in token:
            return "fire"
        return token

    if prompt_name == "[Safety] Warehouse":
        if "helmet_off" in token or "no_helmet" in token or "nohelmet" in token:
            return "helmet_off"
        if "susp" in token or token == "sus":
            return "suspicious"
        if "normal" in token:
            return "normal"
        return token

    return token


def _iter_video_files(folder: Path) -> Iterable[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"}
        ]
    )


def _thinking_value(mode: str) -> bool | None:
    cleaned = mode.strip().lower()
    if cleaned == "on":
        return True
    if cleaned == "off":
        return False
    return None


def main() -> int:
    """Run evaluation across mapped video folders."""
    parser = argparse.ArgumentParser(description="Evaluate ready classification prompts on local videos.")
    parser.add_argument("--videos-root", default="/home/onebonsai/Videos")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--max-files-per-folder", type=int, default=None)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts()
    root = Path(args.videos_root).expanduser()
    if not root.exists():
        print(f"Videos root not found: {root}")
        return 1

    results: list[EvalRow] = []
    enable_thinking = _thinking_value(args.thinking)

    for folder_name, prompt_name in FOLDER_TO_PROMPT.items():
        folder = root / folder_name
        if not folder.exists():
            print(f"[skip] folder not found: {folder}")
            continue

        prompt_item = find_prompt(prompts, prompt_name)
        if prompt_item is None:
            print(f"[skip] prompt not found: {prompt_name}")
            continue

        processed = 0
        for video_path in _iter_video_files(folder):
            if args.max_files_per_folder is not None and processed >= args.max_files_per_folder:
                break
            expected = _expected_label(folder_name, video_path.stem)
            if expected is None:
                print(f"[skip] no expected label mapping for {video_path.name}")
                continue

            processed += 1
            if args.dry_run:
                predicted = "(dry-run)"
                ok = False
            else:
                try:
                    output = call_vllm_segmented(
                        video_path=str(video_path),
                        prompt=prompt_item["text"],
                        base_url=args.base_url,
                        model=args.model,
                        max_tokens=args.max_completion_tokens,
                        max_completion_tokens=args.max_completion_tokens,
                        enable_thinking=enable_thinking,
                    )
                except Exception as exc:  # noqa: BLE001
                    predicted = f"error:{exc}"
                    ok = False
                else:
                    label_line, _ = _split_classification_output(output)
                    predicted = _normalize_predicted_label(prompt_name, label_line)
                    ok = predicted == expected

            results.append(
                EvalRow(
                    folder=folder_name,
                    file=video_path.name,
                    expected=expected,
                    predicted=predicted,
                    ok=ok,
                )
            )

    if not results:
        print("No evaluation rows produced.")
        return 1

    print("folder,file,expected,predicted,ok")
    for row in results:
        print(f"{row.folder},{row.file},{row.expected},{row.predicted},{row.ok}")

    total = len(results)
    correct = sum(1 for row in results if row.ok)
    print(f"\nOverall: {correct}/{total} ({(100.0 * correct / total):.1f}%)")

    for folder_name in FOLDER_TO_PROMPT:
        folder_rows = [row for row in results if row.folder == folder_name]
        if not folder_rows:
            continue
        folder_correct = sum(1 for row in folder_rows if row.ok)
        print(
            f"{folder_name}: {folder_correct}/{len(folder_rows)} "
            f"({(100.0 * folder_correct / len(folder_rows)):.1f}%)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
