#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible chat/VLM API from the terminal.

Default settings target OpenAI GPT-5.5:

    export OPENAI_API_KEY="sk-..."
    python dexsafedagger_lab/distillation/tests/test_vlm_api.py

You can also put OPENAI_API_KEY=sk-... in a local .env file at the repo root.

For an image-capable VLM endpoint/model, pass an image and override the model or
base URL if needed:

    python dexsafedagger_lab/distillation/tests/test_vlm_api.py \
        --model YOUR_VLM_MODEL \
        --image /path/to/frame.png \
        --prompt "Describe the scene and say if teacher intervention is needed."
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise vision-language safety assistant for robot manipulation."
)
DEFAULT_USER_PROMPT = (
    "Answer briefly. If you can see an image, describe the scene and say whether "
    "a teacher policy should intervene. If there is no image, confirm the API is reachable."
)
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def is_placeholder_api_key(value: str | None) -> bool:
    cleaned = (value or "").strip().strip("'\"").lower()
    return cleaned in {"your_api_key", "your_key_here", "sk-...", "your_key*here"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call an OpenAI-compatible chat or VLM API and print the answer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=os.getenv("VLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-key-env",
        default=os.getenv("VLM_API_KEY_ENV", "OPENAI_API_KEY"),
        help="Environment variable containing the API key.",
    )
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--image",
        default=None,
        help="Optional local image path or http(s) image URL for VLM testing.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--thinking",
        choices=["default", "enabled", "disabled"],
        default="disabled",
        help="Provider-specific thinking mode; only sent for non-OpenAI-compatible DeepSeek runs.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full JSON response instead of just the assistant answer.",
    )
    return parser.parse_args()


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load KEY=VALUE pairs from a local .env file without overriding shell env."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and (key not in os.environ or is_placeholder_api_key(os.environ.get(key))):
            os.environ[key] = value


def image_to_data_url(image: str) -> str:
    if image.startswith(("http://", "https://")):
        return image

    path = Path(image).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": args.system},
    ]

    if args.image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(args.image)}},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": args.prompt})

    return messages


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    is_openai_gpt5 = args.base_url.rstrip("/").endswith("api.openai.com/v1") and args.model.startswith("gpt-5")
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": build_messages(args),
        "stream": False,
    }
    if is_openai_gpt5:
        payload["max_completion_tokens"] = args.max_tokens
    else:
        payload["temperature"] = args.temperature
        payload["max_tokens"] = args.max_tokens

    is_deepseek = "deepseek" in args.base_url.lower()
    if is_deepseek and args.thinking != "default":
        payload["thinking"] = {"type": args.thinking}

    return payload


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = f"HTTP {exc.code} from {url}\n{body}"
        if exc.code not in {401, 403}:
            message += (
                "\n\nIf you passed --image, make sure the selected endpoint/model supports "
                "image inputs."
            )
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned non-JSON response:\n{body}") from exc


def extract_answer(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return "<no choices returned>"

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False)


def main() -> int:
    load_env_file()
    args = parse_args()
    api_key = (os.environ.get(args.api_key_env) or "").strip()
    if not api_key:
        print(
            f"Missing API key. Set it first:\n\n"
            f"  export {args.api_key_env}='YOUR_API_KEY'\n",
            f"\nOr create {ENV_FILE} containing:\n\n"
            f"  {args.api_key_env}=YOUR_API_KEY\n",
            file=sys.stderr,
        )
        return 2
    if is_placeholder_api_key(api_key):
        print(
            f"The value in {args.api_key_env} still looks like a placeholder.\n\n"
            f"Edit {ENV_FILE} so it contains your real API key:\n\n"
            f"  {args.api_key_env}=sk-...\n",
            file=sys.stderr,
        )
        return 2

    payload = build_payload(args)

    print(f"Calling {args.base_url.rstrip('/')}/chat/completions")
    print(f"Model: {args.model}")
    print(f"Image: {args.image or 'none'}")
    print()

    try:
        response = post_chat_completion(
            base_url=args.base_url,
            api_key=api_key,
            payload=payload,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"API call failed:\n{exc}", file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(response, indent=2, ensure_ascii=False))
        return 0

    print("Answer:")
    print(extract_answer(response))

    usage = response.get("usage")
    if usage:
        print()
        print("Usage:")
        print(json.dumps(usage, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
