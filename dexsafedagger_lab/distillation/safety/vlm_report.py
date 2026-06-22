"""Reusable HTML reporting for VLM threshold-advisor calls."""

from __future__ import annotations

import copy
from datetime import datetime
import html
import json
from pathlib import Path
from typing import Any


def strip_visual_payloads(stats: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(stats)
    visual_samples = cleaned.get("visual_samples")
    if isinstance(visual_samples, list):
        for sample in visual_samples:
            if isinstance(sample, dict):
                sample.pop("image_data_url", None)
                sample["image_attached"] = True
    return cleaned


def extract_user_prompt(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if item.get("type") == "text"]
            return "\n\n".join(parts)
    return ""


def response_diagnostics(response: dict[str, Any] | None) -> dict[str, Any]:
    if not response:
        return {}
    choices = response.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    content = message.get("content", "")
    return {
        "id": response.get("id"),
        "model": response.get("model"),
        "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
        "content_type": type(content).__name__,
        "content_length": len(content) if isinstance(content, str) else None,
        "message_keys": sorted(message.keys()),
        "usage": response.get("usage"),
    }


def render_vlm_html_report(
    *,
    output_path: str | Path,
    title: str,
    base_info: dict[str, Any],
    payload: dict[str, Any],
    stats: dict[str, Any],
    reply_text: str,
    parsed_reply: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None = None,
    status: str = "unknown",
    error: str = "",
) -> None:
    output_path = Path(output_path)
    prompt = extract_user_prompt(payload)
    stats_no_images = strip_visual_payloads(stats)
    diagnostics = diagnostics or {}
    image_cards = []
    for idx, sample in enumerate(stats.get("visual_samples") or [], start=1):
        if not isinstance(sample, dict):
            continue
        metadata = {k: v for k, v in sample.items() if k != "image_data_url"}
        object_name = str(sample.get("object_name", "unknown_object"))
        image_cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(str(sample.get('image_data_url', '')))}" alt="visual sample {idx}">
              <h3>{html.escape(object_name)} sample {idx}</h3>
              <div class="sample-subtitle">
                env {html.escape(str(sample.get('env_id', '?')))} | {html.escape(str(sample.get('source', 'unknown')))}
              </div>
              <pre>{html.escape(json.dumps(metadata, indent=2, sort_keys=True))}</pre>
            </article>
            """
        )
    base_cards = []
    for key, value in base_info.items():
        base_cards.append(
            f"<div><strong>{html.escape(str(key))}</strong><br>{html.escape(str(value))}</div>"
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #18202a; }}
    header {{ padding: 22px 28px; background: #18202a; color: white; }}
    main {{ padding: 22px 28px; display: grid; gap: 22px; }}
    section {{ background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; }}
    h1, h2, h3 {{ margin-top: 0; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f0f2f5; padding: 12px; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #d9dee7; border-radius: 8px; padding: 12px; background: #fff; }}
    .card img {{ width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 6px; background: #d9dee7; }}
    .sample-subtitle {{ margin: -8px 0 10px; color: #586272; font-size: 13px; }}
    .kv {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .kv div {{ background: #f0f2f5; border-radius: 6px; padding: 10px; }}
    .status {{ display: inline-block; padding: 4px 8px; border-radius: 6px; background: #f0f2f5; color: #18202a; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div>{html.escape(datetime.now().isoformat(timespec="seconds"))}</div>
  </header>
  <main>
    <section>
      <h2>Call Status</h2>
      <div class="status">{html.escape(status)}</div>
      <pre>{html.escape(error or '<no error>')}</pre>
    </section>
    <section>
      <h2>Advisor Base Information</h2>
      <div class="kv">
        {''.join(base_cards)}
      </div>
    </section>
    <section>
      <h2>Prompt Sent To VLM</h2>
      <pre>{html.escape(prompt)}</pre>
    </section>
    <section>
      <h2>Structured Stats Without Image Payloads</h2>
      <pre>{html.escape(json.dumps(stats_no_images, indent=2, sort_keys=True))}</pre>
    </section>
    <section>
      <h2>Visual Samples</h2>
      <div class="grid">
        {''.join(image_cards)}
      </div>
    </section>
    <section>
      <h2>VLM Reply</h2>
      <pre>{html.escape(reply_text or '<no API call / no reply>')}</pre>
      <h3>Parsed JSON</h3>
      <pre>{html.escape(json.dumps(parsed_reply, indent=2, sort_keys=True) if parsed_reply else '<not parsed>')}</pre>
      <h3>Response Diagnostics</h3>
      <pre>{html.escape(json.dumps(diagnostics, indent=2, sort_keys=True) if diagnostics else '<none>')}</pre>
    </section>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
