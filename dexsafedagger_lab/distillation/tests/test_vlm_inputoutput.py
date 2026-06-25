#!/usr/bin/env python3
"""Visual smoke test for the VLM threshold-advisor input/output path.

This script does not launch Isaac Sim. It simulates one VLM advisor update at a
random training step, attaches balanced visual samples, writes an HTML
inspection UI, and prints the VLM reply in the terminal when API credentials are
available.

Example:
  python dexsafedagger_lab/distillation/tests/test_vlm_inputoutput.py

Optional:
  python dexsafedagger_lab/distillation/tests/test_vlm_inputoutput.py \
    --image-dir dexsafedagger_lab/distillation/runs/<run>/debug \
    --output-html /tmp/vlm_inputoutput.html
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import html
import json
import mimetypes
import os
from pathlib import Path
import random
import sys
import urllib.error
import urllib.request
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexsafedagger_lab.distillation.safety.vlm_threshold_advisor import (  # noqa: E402
    VLMThresholdAdvisor,
    _is_placeholder_api_key,
    _parse_json_answer,
)
from dexsafedagger_lab.distillation.safety.vlm_report import (  # noqa: E402
    render_vlm_html_report,
    response_diagnostics as report_response_diagnostics,
)


DEFAULT_DEBUG_IMAGE_DIR = (
    REPO_ROOT
    / "dexsafedagger_lab"
    / "distillation"
    / "runs"
    / "dexsafedagger-tg2-inspirehand-dexsafedaggerultra_20-17-11-01"
    / "debug"
)
DEFAULT_OUTPUT_HTML = Path("/tmp/dexsafedagger_vlm_inputoutput.html")
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and optionally call one simulated VLM advisor request.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_DEBUG_IMAGE_DIR)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument(
        "--object-names",
        default="object_0,object_1,object_2,object_3",
        help="Comma-separated object names represented in the simulated 32-env run.",
    )
    parser.add_argument("--envs-per-object", type=int, default=8)
    parser.add_argument("--samples-per-object", type=int, default=2)
    parser.add_argument("--base-l2-threshold", type=float, default=8.0)
    parser.add_argument("--base-success-threshold", type=float, default=0.2)
    parser.add_argument("--model", default=os.getenv("VLM_MODEL", None))
    parser.add_argument("--base-url", default=os.getenv("VLM_BASE_URL", None))
    parser.add_argument("--api-key-env", default=os.getenv("VLM_API_KEY_ENV", None))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Only build the payload and HTML UI; do not call the VLM API.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and (key not in os.environ or _is_placeholder_api_key(os.environ.get(key))):
            os.environ[key] = value


def image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def make_tiny_svg_data_url(label: str, hue: int) -> str:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240">
<rect width="320" height="240" fill="hsl({hue}, 65%, 22%)"/>
<circle cx="160" cy="112" r="56" fill="hsl({(hue + 80) % 360}, 70%, 58%)"/>
<rect x="88" y="156" width="144" height="28" rx="8" fill="hsl({(hue + 160) % 360}, 75%, 65%)"/>
<text x="160" y="214" text-anchor="middle" font-family="monospace" font-size="18" fill="white">{html.escape(label)}</text>
</svg>"""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def choose_image_data_urls(image_dir: Path, count: int, rng: random.Random) -> list[tuple[str, str]]:
    candidates = []
    if image_dir.is_dir():
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidates.extend(sorted(image_dir.glob(pattern)))
    if len(candidates) >= count:
        selected = rng.sample(candidates, count)
        return [(p.name, image_to_data_url(p)) for p in selected]

    fallback = []
    for idx in range(count):
        label = f"synthetic-{idx + 1}"
        fallback.append((label, make_tiny_svg_data_url(label, hue=(idx * 47) % 360)))
    return fallback


def parse_object_names(raw: str) -> list[str]:
    names = [item.strip() for item in str(raw).split(",") if item.strip()]
    return names or ["object_0"]


def build_balanced_sample_plan(
    *,
    object_names: list[str],
    envs_per_object: int,
    samples_per_object: int,
    max_samples: int,
) -> list[dict[str, Any]]:
    plan = []
    envs_per_object = max(1, int(envs_per_object))
    samples_per_object = max(1, int(samples_per_object))
    stride = max(1, envs_per_object // samples_per_object)
    for object_id, object_name in enumerate(object_names):
        for local_idx in range(samples_per_object):
            local_env_id = min(envs_per_object - 1, local_idx * stride)
            env_id = object_id * envs_per_object + local_env_id
            plan.append(
                {
                    "env_id": env_id,
                    "object_id": object_id,
                    "object_name": object_name,
                    "object_env_range": (
                        f"{object_id * envs_per_object}-"
                        f"{object_id * envs_per_object + envs_per_object - 1}"
                    ),
                    "object_local_sample_idx": local_idx,
                }
            )
            if len(plan) >= max_samples:
                return plan
    return plan


def tensor_stats(values: list[float]) -> dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {}

    def q(frac: float) -> float:
        idx = min(len(vals) - 1, max(0, round(frac * (len(vals) - 1))))
        return vals[idx]

    return {
        "mean": sum(vals) / len(vals),
        "min": vals[0],
        "max": vals[-1],
        "p10": q(0.1),
        "p50": q(0.5),
        "p90": q(0.9),
    }


def build_simulated_stats(
    *,
    images: list[tuple[str, str]],
    sample_plan: list[dict[str, Any]],
    rng: random.Random,
    l2_threshold: float,
    success_threshold: float,
) -> dict[str, Any]:
    step = rng.randrange(2_000, 100_000)
    frame = step * 32
    l2_values = [max(0.0, rng.gauss(7.0, 4.0)) for _ in range(32)]
    success_values = [max(0.0, min(1.0, rng.gauss(0.45, 0.2))) for _ in range(32)]
    intervention_rate = sum(v > l2_threshold for v in l2_values) / len(l2_values)
    unsafe_rate = rng.uniform(0.08, 0.25)
    reason_prop = {
        "object_out_of_bound": unsafe_rate * rng.uniform(0.25, 0.55),
        "hand_too_far": unsafe_rate * rng.uniform(0.00, 0.08),
        "harmful_collision": unsafe_rate * rng.uniform(0.30, 0.65),
        "palm_flipped": unsafe_rate * rng.uniform(0.05, 0.20),
    }
    total_reason = sum(reason_prop.values())
    if total_reason > 0:
        reason_prop = {k: v * unsafe_rate / total_reason for k, v in reason_prop.items()}

    source_cycle = ["warmstart_unsafe", "high_l2", "low_success", "unsafe_triggered"]
    visual_samples = []
    for idx, ((image_name, image_data_url), plan_item) in enumerate(zip(images, sample_plan)):
        l2 = max(0.0, rng.gauss(10.0 if idx % 2 else 5.5, 3.0))
        success = max(0.0, min(1.0, rng.gauss(0.35 if idx % 2 else 0.65, 0.18)))
        unsafe = bool(idx in {0, 3} or l2 > l2_threshold)
        visual_samples.append(
            {
                "image_data_url": image_data_url,
                "image_key": "img_left",
                "image_name": image_name,
                "source": source_cycle[idx % len(source_cycle)],
                "step": step - (len(images) - idx) * 20,
                "frame": frame,
                "teacher_student_l2": l2,
                "predictor_success": success,
                "unsafe": unsafe,
                "l2_threshold": l2_threshold,
                "success_threshold": success_threshold,
            }
            | plan_item
        )

    return {
        "step": step,
        "frame": frame,
        "sample_count": len(l2_values),
        "intervention_rate": intervention_rate,
        "l2_threshold": l2_threshold,
        "success_threshold": success_threshold,
        "l2": tensor_stats(l2_values),
        "success": tensor_stats(success_values),
        "unsafe_episode_rate": unsafe_rate,
        "unsafe_reason_prop": reason_prop,
        "visual_samples": visual_samples,
        "visual_buffer_size": 64,
        "visual_samples_attached": len(visual_samples),
    }


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
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    return json.loads(body)


def response_diagnostics(response: dict[str, Any]) -> dict[str, Any]:
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


def render_html(
    *,
    output_path: Path,
    advisor: VLMThresholdAdvisor,
    payload: dict[str, Any],
    stats: dict[str, Any],
    reply_text: str,
    parsed_reply: dict[str, Any] | None,
) -> None:
    prompt = extract_user_prompt(payload)
    stats_no_images = advisor._strip_visual_payloads(stats)
    image_cards = []
    for idx, sample in enumerate(stats.get("visual_samples") or [], start=1):
        metadata = {k: v for k, v in sample.items() if k != "image_data_url"}
        object_name = str(sample.get("object_name", "unknown_object"))
        image_cards.append(
            f"""
            <article class="card">
              <img src="{sample.get('image_data_url', '')}" alt="visual sample {idx}">
              <h3>{html.escape(object_name)} · sample {idx}</h3>
              <div class="sample-subtitle">
                env {html.escape(str(sample.get('env_id', '?')))} · {html.escape(str(sample.get('source', 'unknown')))}
              </div>
              <pre>{html.escape(json.dumps(metadata, indent=2, sort_keys=True))}</pre>
            </article>
            """
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DexSafeDagger VLM Advisor Input/Output</title>
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
  </style>
</head>
<body>
  <header>
    <h1>DexSafeDagger VLM Advisor Input/Output</h1>
    <div>{html.escape(datetime.now().isoformat(timespec="seconds"))}</div>
  </header>
  <main>
    <section>
      <h2>Advisor Base Information</h2>
      <div class="kv">
        <div><strong>model</strong><br>{html.escape(advisor.model)}</div>
        <div><strong>base_url</strong><br>{html.escape(advisor.base_url)}</div>
        <div><strong>env_file_loaded</strong><br>{html.escape(advisor.env_file_loaded or '<none>')}</div>
        <div><strong>mode</strong><br>{html.escape(advisor.mode)}</div>
        <div><strong>base_l2_threshold</strong><br>{advisor.base_l2_threshold}</div>
        <div><strong>base_success_threshold</strong><br>{advisor.base_success_threshold}</div>
        <div><strong>max_tokens</strong><br>{advisor.max_tokens}</div>
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
    </section>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    load_env_file(DEFAULT_ENV_FILE)
    args = parse_args()
    rng = random.Random(args.seed if args.seed is not None else random.randrange(1_000_000_000))
    object_names = parse_object_names(args.object_names)
    expected_samples = max(1, len(object_names) * max(1, int(args.samples_per_object)))
    sample_count = max(1, int(args.num_images or expected_samples))
    images = choose_image_data_urls(args.image_dir, sample_count, rng)
    sample_plan = build_balanced_sample_plan(
        object_names=object_names,
        envs_per_object=args.envs_per_object,
        samples_per_object=args.samples_per_object,
        max_samples=sample_count,
    )

    advisor_cfg = {
        "enabled": True,
        "mode": "shadow",
        "base_url": args.base_url,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "update_interval_steps": 10_000,
        "warmup_steps": 2_000,
        "min_samples": 1,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "visual_samples_per_update": sample_count,
        "visual_detail": "low",
    }
    advisor = VLMThresholdAdvisor(
        advisor_cfg,
        base_l2_threshold=args.base_l2_threshold,
        base_success_threshold=args.base_success_threshold,
        run_dir=str(REPO_ROOT),
        rank=0,
    )
    stats = build_simulated_stats(
        images=images[: len(sample_plan)],
        sample_plan=sample_plan,
        rng=rng,
        l2_threshold=advisor.current_l2_threshold,
        success_threshold=advisor.current_success_threshold,
    )
    payload = advisor._build_payload(stats)

    print("=== Simulated VLM Advisor Call ===")
    print(f"step: {stats['step']}")
    print(f"visual samples: {len(stats['visual_samples'])}")
    print(f"object balance: {len(object_names)} objects x {args.samples_per_object} samples")
    print(f"model: {advisor.model}")
    print(f"base_url: {advisor.base_url}")
    print(f"api_key_env: {advisor.api_key_env}")
    print(f"env_file_loaded: {advisor.env_file_loaded or DEFAULT_ENV_FILE}")

    reply_text = ""
    parsed_reply = None
    api_key = (os.environ.get(advisor.api_key_env) or "").strip()
    if args.no_api:
        reply_text = "API call skipped because --no-api was set."
        print(reply_text)
    elif not api_key or _is_placeholder_api_key(api_key):
        reply_text = f"API call skipped: missing or placeholder key in {advisor.api_key_env}."
        print(reply_text)
    else:
        print("Calling VLM API...")
        response = post_chat_completion(
            base_url=advisor.base_url,
            api_key=api_key,
            payload=payload,
            timeout=advisor.timeout,
        )
        reply_text = advisor._extract_answer(response)
        diagnostics = report_response_diagnostics(response)
        if not reply_text:
            reply_text = (
                "<empty assistant content>\n\n"
                "Response diagnostics:\n"
                f"{json.dumps(diagnostics, indent=2, sort_keys=True)}"
            )
        parsed_reply = _parse_json_answer(reply_text)
        print("\n=== Raw VLM Reply ===")
        print(reply_text)
        print("\n=== Response Diagnostics ===")
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
        print("\n=== Parsed VLM Reply ===")
        print(json.dumps(parsed_reply, indent=2, sort_keys=True) if parsed_reply else "<not parsed>")

    render_vlm_html_report(
        output_path=args.output_html,
        title="DexSafeDagger VLM Advisor Input/Output",
        base_info={
            "model": advisor.model,
            "base_url": advisor.base_url,
            "env_file_loaded": advisor.env_file_loaded or DEFAULT_ENV_FILE,
            "mode": advisor.mode,
            "base_l2_threshold": advisor.base_l2_threshold,
            "base_success_threshold": advisor.base_success_threshold,
            "max_tokens": advisor.max_tokens,
        },
        payload=payload,
        stats=stats,
        reply_text=reply_text,
        parsed_reply=parsed_reply,
        diagnostics=report_response_diagnostics(response) if "response" in locals() else None,
        status="skipped" if args.no_api else ("success" if parsed_reply else "failed"),
        error="" if parsed_reply or args.no_api else "could not parse JSON recommendation",
    )
    print(f"\nHTML UI written to: {args.output_html}")
    print("Open it in a browser to inspect the balanced images, prompt, stats, and reply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
