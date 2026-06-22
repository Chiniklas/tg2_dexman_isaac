"""VLM threshold advisor for DexSafeDagger arbitration.

The advisor does not decide teacher takeover directly. It recommends thresholds
for the existing teacher-student disagreement and predictor-risk gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import os
import urllib.error
import urllib.request
from typing import Any

from dexsafedagger_lab.distillation.safety.vlm_report import (
    render_vlm_html_report,
    response_diagnostics,
)


def _is_placeholder_api_key(value: str | None) -> bool:
    cleaned = (value or "").strip().strip("'\"").lower()
    return cleaned in {"your_api_key", "your_key_here", "sk-...", "your_key*here"}


def _load_env_file(path: str) -> None:
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and (key not in os.environ or _is_placeholder_api_key(os.environ.get(key))):
                os.environ[key] = value


def _find_env_file(start_dir: str | None) -> str | None:
    if not start_dir:
        return None
    current = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(current, ".env")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _parse_json_answer(answer: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(answer)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = answer.find("{")
    end = answer.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(answer[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class VLMThresholdAdvisorState:
    l2_threshold: float
    risk_threshold: float
    recommendation_count: int = 0
    attempt_count: int = 0
    failure_count: int = 0
    last_attempt_step: int = -1
    last_reason: str = ""
    last_error: str = ""
    last_confidence: float = 0.0


class VLMThresholdAdvisor:
    """Slow supervisory tuner for existing DexSafeDagger thresholds."""

    def __init__(
        self,
        cfg: dict[str, Any] | None,
        *,
        base_l2_threshold: float,
        base_risk_threshold: float,
        run_dir: str | None = None,
        rank: int = 0,
    ):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = str(cfg.get("mode", "shadow")).strip().lower()
        if self.mode not in {"shadow", "active"}:
            raise ValueError(f"vlm_threshold_advisor.mode must be 'shadow' or 'active', got {self.mode!r}.")
        self.rank = int(rank)
        self.base_l2_threshold = float(base_l2_threshold)
        self.base_risk_threshold = float(base_risk_threshold)
        self.state = VLMThresholdAdvisorState(
            l2_threshold=self.base_l2_threshold,
            risk_threshold=self.base_risk_threshold,
        )

        self.update_interval_steps = int(cfg.get("update_interval_steps", 1000))
        if self.update_interval_steps <= 0:
            self.update_interval_steps = 1000
        self.warmup_steps = int(cfg.get("warmup_steps", 0))
        self.min_samples = int(cfg.get("min_samples", 1))
        if self.min_samples <= 0:
            self.min_samples = 1
        self.smoothing = float(cfg.get("smoothing", 0.1))
        self.smoothing = min(1.0, max(0.0, self.smoothing))
        self.l2_min_scale = float(cfg.get("l2_min_scale", 0.5))
        self.l2_max_scale = float(cfg.get("l2_max_scale", 1.5))
        self.risk_min_scale = float(cfg.get("risk_min_scale", 0.5))
        self.risk_max_scale = float(cfg.get("risk_max_scale", 1.5))
        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.timeout = float(cfg.get("timeout", 90.0))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.visual_buffer_enabled = bool(cfg.get("visual_buffer_enabled", True))
        self.visual_buffer_size = int(cfg.get("visual_buffer_size", 64))
        if self.visual_buffer_size <= 0:
            self.visual_buffer_size = 64
        self.visual_capture_interval_steps = int(cfg.get("visual_capture_interval_steps", 20))
        if self.visual_capture_interval_steps <= 0:
            self.visual_capture_interval_steps = 20
        self.visual_captures_per_step = int(cfg.get("visual_captures_per_step", 2))
        if self.visual_captures_per_step <= 0:
            self.visual_captures_per_step = 1
        self.visual_samples_per_update = int(cfg.get("visual_samples_per_update", 8))
        if self.visual_samples_per_update <= 0:
            self.visual_samples_per_update = 1
        self.visual_image_key = str(cfg.get("visual_image_key", "img_left"))
        self.visual_max_edge = int(cfg.get("visual_max_edge", 256))
        if self.visual_max_edge <= 0:
            self.visual_max_edge = 256
        self.visual_jpeg_quality = int(cfg.get("visual_jpeg_quality", 70))
        self.visual_jpeg_quality = min(95, max(30, self.visual_jpeg_quality))
        self.visual_detail = str(cfg.get("visual_detail", "low"))
        self.base_url = str(cfg.get("base_url") or os.getenv("VLM_BASE_URL") or "https://api.openai.com/v1")
        self.model = str(cfg.get("model") or os.getenv("VLM_MODEL") or "gpt-5.5")
        self.api_key_env = str(
            cfg.get("api_key_env")
            or os.getenv("VLM_API_KEY_ENV")
            or ("OPENAI_API_KEY" if self.base_url.rstrip("/").endswith("api.openai.com/v1") else "VLM_API_KEY")
        )
        self.task = str(
            cfg.get(
                "task",
                "Tune teacher-student disagreement and predictor-risk thresholds for dexterous robot distillation.",
            )
        )
        self.run_dir = run_dir
        self.log_path = None
        self.report_enabled = bool(cfg.get("report_enabled", True))
        self.report_dir = str(
            cfg.get("report_dir")
            or (os.path.join(run_dir, "vlm_reports") if run_dir else "")
        )
        self.last_report_path = ""
        if run_dir:
            self.log_path = os.path.join(run_dir, "vlm_threshold_advisor.jsonl")

        env_file = cfg.get("env_file", None)
        if env_file is None:
            env_file = _find_env_file(run_dir)
        self.env_file_loaded = str(env_file) if env_file else ""
        if env_file:
            _load_env_file(str(env_file))

        self._last_update_step = -10**18

    @property
    def current_l2_threshold(self) -> float:
        return float(self.state.l2_threshold)

    @property
    def current_risk_threshold(self) -> float:
        return float(self.state.risk_threshold)

    def should_update(self, step: int, sample_count: int) -> bool:
        if not self.enabled:
            return False
        if int(step) < self.warmup_steps:
            return False
        if int(sample_count) < self.min_samples:
            return False
        return int(step) - int(self._last_update_step) >= self.update_interval_steps

    def status(self, *, step: int, sample_count: int) -> dict[str, Any]:
        next_step = max(self.warmup_steps, self._last_update_step + self.update_interval_steps)
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "step": int(step),
            "sample_count": int(sample_count),
            "min_samples": int(self.min_samples),
            "next_update_step": int(next_step),
            "ready": bool(self.should_update(step, sample_count)),
            "recommendation_count": int(self.state.recommendation_count),
            "attempt_count": int(self.state.attempt_count),
            "failure_count": int(self.state.failure_count),
            "last_attempt_step": int(self.state.last_attempt_step),
            "l2_threshold": float(self.current_l2_threshold),
            "risk_threshold": float(self.current_risk_threshold),
            "last_confidence": float(self.state.last_confidence),
            "last_error": str(self.state.last_error),
        }

    def maybe_update(self, *, step: int, stats: dict[str, Any]) -> dict[str, Any] | None:
        sample_count = int(stats.get("sample_count", 0))
        if not self.should_update(step, sample_count):
            return None
        self._last_update_step = int(step)
        self.state.attempt_count += 1
        self.state.last_attempt_step = int(step)
        self.last_report_path = ""
        if self.rank == 0:
            visual_count = len(stats.get("visual_samples") or [])
            print(
                "[VLMThresholdAdvisor] attempting recommendation "
                f"step={int(step)} model={self.model} samples={sample_count} visual_samples={visual_count}",
                flush=True,
            )
        recommendation = self._query(stats)
        if recommendation is None:
            self.state.failure_count += 1
            record = {
                "step": int(step),
                "mode": self.mode,
                "status": "failed",
                "stats": self._strip_visual_payloads(stats),
                "error": self.state.last_error,
                "report_path": self.last_report_path,
            }
            self._log(record)
            return None
        applied = self._apply_recommendation(recommendation)
        record = {
            "step": int(step),
            "mode": self.mode,
            "status": "success",
            "stats": self._strip_visual_payloads(stats),
            "recommendation": recommendation,
            "applied": applied,
            "report_path": self.last_report_path,
        }
        self._log(record)
        return record

    def _apply_recommendation(self, recommendation: dict[str, Any]) -> dict[str, Any]:
        rec_l2 = self._float_or_none(recommendation.get("l2_threshold"))
        rec_risk = self._float_or_none(recommendation.get("risk_threshold"))
        conf = self._float_or_none(recommendation.get("confidence"))
        self.state.last_confidence = 0.0 if conf is None else float(conf)
        self.state.last_reason = str(recommendation.get("reason", ""))
        self.state.recommendation_count += 1

        proposed_l2 = self.current_l2_threshold if rec_l2 is None else self._clamp_l2(rec_l2)
        proposed_risk = self.current_risk_threshold if rec_risk is None else self._clamp_risk(rec_risk)
        if self.mode == "active":
            self.state.l2_threshold = self._smooth(self.current_l2_threshold, proposed_l2)
            self.state.risk_threshold = self._smooth(self.current_risk_threshold, proposed_risk)
        return {
            "l2_threshold": float(self.state.l2_threshold),
            "risk_threshold": float(self.state.risk_threshold),
            "proposed_l2_threshold": float(proposed_l2),
            "proposed_risk_threshold": float(proposed_risk),
        }

    def _query(self, stats: dict[str, Any]) -> dict[str, Any] | None:
        api_key = (os.environ.get(self.api_key_env) or "").strip()
        if not api_key:
            self.state.last_error = f"missing API key env {self.api_key_env}"
            if self.rank == 0:
                print(
                    f"[VLMThresholdAdvisor] Missing API key env {self.api_key_env}; skipping recommendation.",
                    flush=True,
                )
            return None
        if _is_placeholder_api_key(api_key):
            self.state.last_error = f"{self.api_key_env} still looks like a placeholder"
            if self.rank == 0:
                print(f"[VLMThresholdAdvisor] {self.state.last_error}; skipping recommendation.", flush=True)
            return None
        payload = self._build_payload(stats)
        url = self.base_url.rstrip("/") + "/chat/completions"
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
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
            response_json = json.loads(body)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            self.state.last_error = f"HTTP {exc.code}: {detail or str(exc)}"
            if self.rank == 0:
                print(f"[VLMThresholdAdvisor] API call failed: {self.state.last_error}", flush=True)
            self._write_report(
                step=int(stats.get("step", self.state.last_attempt_step)),
                status="failed",
                stats=stats,
                payload=payload,
                reply_text="",
                parsed_reply=None,
                response_json=None,
                error=self.state.last_error,
            )
            return None
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            self.state.last_error = str(exc)
            if self.rank == 0:
                print(f"[VLMThresholdAdvisor] API call failed: {exc}", flush=True)
            self._write_report(
                step=int(stats.get("step", self.state.last_attempt_step)),
                status="failed",
                stats=stats,
                payload=payload,
                reply_text="",
                parsed_reply=None,
                response_json=None,
                error=self.state.last_error,
            )
            return None
        answer = self._extract_answer(response_json)
        parsed = _parse_json_answer(answer)
        if parsed is None:
            self.state.last_error = "could not parse JSON recommendation"
            if self.rank == 0:
                print("[VLMThresholdAdvisor] Could not parse JSON recommendation.", flush=True)
            reply_text = answer
            if not reply_text:
                reply_text = (
                    "<empty assistant content>\n\n"
                    "Response diagnostics:\n"
                    f"{json.dumps(response_diagnostics(response_json), indent=2, sort_keys=True)}"
                )
            self._write_report(
                step=int(stats.get("step", self.state.last_attempt_step)),
                status="failed",
                stats=stats,
                payload=payload,
                reply_text=reply_text,
                parsed_reply=None,
                response_json=response_json,
                error=self.state.last_error,
            )
            return None
        self.state.last_error = ""
        self._write_report(
            step=int(stats.get("step", self.state.last_attempt_step)),
            status="success",
            stats=stats,
            payload=payload,
            reply_text=answer,
            parsed_reply=parsed,
            response_json=response_json,
            error="",
        )
        return parsed

    def _build_payload(self, stats: dict[str, Any]) -> dict[str, Any]:
        visual_samples = stats.get("visual_samples") or []
        stats_for_prompt = self._strip_visual_payloads(stats)
        prompt = (
            "You are a conservative threshold advisor for DexSafeDagger robot distillation.\n"
            "Do not decide intervention directly. Recommend thresholds for the existing gates.\n\n"
            f"Task: {self.task}\n\n"
            "Existing arbitration:\n"
            "- intervene if teacher_student_l2 >= l2_threshold\n"
            "- or if predictor_risk >= risk_threshold\n\n"
            "Return only strict JSON with keys: l2_threshold, risk_threshold, confidence, reason.\n"
            "Keep changes conservative. Prefer small threshold shifts unless the recent window shows clear over- "
            "or under-intervention.\n\n"
            "Base thresholds and clamps:\n"
            f"- base_l2_threshold={self.base_l2_threshold}\n"
            f"- allowed_l2_range=[{self._clamp_l2(-float('inf'))}, {self._clamp_l2(float('inf'))}]\n"
            f"- base_risk_threshold={self.base_risk_threshold}\n"
            f"- allowed_risk_range=[{self._clamp_risk(-float('inf'))}, {self._clamp_risk(float('inf'))}]\n\n"
            "Current advisor thresholds:\n"
            f"- current_l2_threshold={self.current_l2_threshold}\n"
            f"- current_risk_threshold={self.current_risk_threshold}\n\n"
            "Recent rollout statistics:\n"
            f"{json.dumps(stats_for_prompt, ensure_ascii=False, sort_keys=True)}"
        )
        user_content: str | list[dict[str, Any]]
        if visual_samples:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for idx, sample in enumerate(visual_samples, start=1):
                image_url = sample.get("image_data_url")
                if not image_url:
                    continue
                metadata = {k: v for k, v in sample.items() if k != "image_data_url"}
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Visual sample {idx} metadata: "
                            f"{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}"
                        ),
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": self.visual_detail,
                        },
                    }
                )
            user_content = content
        else:
            user_content = prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You tune safety thresholds and return compact JSON only."},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }
        is_openai_gpt5 = self.base_url.rstrip("/").endswith("api.openai.com/v1") and self.model.startswith("gpt-5")
        if is_openai_gpt5:
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["temperature"] = self.temperature
            payload["max_tokens"] = self.max_tokens
        return payload

    def _extract_answer(self, response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return json.dumps(content, ensure_ascii=False)

    def _clamp_l2(self, value: float) -> float:
        low = self.base_l2_threshold * self.l2_min_scale
        high = self.base_l2_threshold * self.l2_max_scale
        return min(high, max(low, float(value)))

    def _clamp_risk(self, value: float) -> float:
        low = self.base_risk_threshold * self.risk_min_scale
        high = self.base_risk_threshold * self.risk_max_scale
        return min(high, max(low, float(value)))

    def _smooth(self, current: float, proposed: float) -> float:
        return (1.0 - self.smoothing) * float(current) + self.smoothing * float(proposed)

    def _float_or_none(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _strip_visual_payloads(self, stats: dict[str, Any]) -> dict[str, Any]:
        cleaned = copy.deepcopy(stats)
        visual_samples = cleaned.get("visual_samples")
        if isinstance(visual_samples, list):
            for sample in visual_samples:
                if isinstance(sample, dict):
                    sample.pop("image_data_url", None)
                    sample["image_attached"] = True
        return cleaned

    def _report_base_info(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "mode": self.mode,
            "base_l2_threshold": self.base_l2_threshold,
            "base_risk_threshold": self.base_risk_threshold,
            "current_l2_threshold": self.current_l2_threshold,
            "current_risk_threshold": self.current_risk_threshold,
            "max_tokens": self.max_tokens,
            "visual_samples_per_update": self.visual_samples_per_update,
            "env_file_loaded": self.env_file_loaded or "<none>",
        }

    def _write_report(
        self,
        *,
        step: int,
        status: str,
        stats: dict[str, Any],
        payload: dict[str, Any],
        reply_text: str,
        parsed_reply: dict[str, Any] | None,
        response_json: dict[str, Any] | None,
        error: str,
    ) -> None:
        if self.rank != 0 or not self.report_enabled or not self.report_dir:
            return
        attempt = max(1, int(self.state.attempt_count))
        safe_status = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(status))
        output_path = os.path.join(
            self.report_dir,
            f"step_{int(step):09d}_attempt_{attempt:04d}_{safe_status}.html",
        )
        try:
            render_vlm_html_report(
                output_path=output_path,
                title="DexSafeDagger VLM Advisor Report",
                base_info=self._report_base_info(),
                payload=payload,
                stats=stats,
                reply_text=reply_text,
                parsed_reply=parsed_reply,
                diagnostics=response_diagnostics(response_json),
                status=status,
                error=error,
            )
        except Exception as exc:
            print(f"[VLMThresholdAdvisor] report write failed: {exc}", flush=True)
            return
        self.last_report_path = output_path
        print(f"[VLMThresholdAdvisor] report: {output_path}", flush=True)

    def _log(self, record: dict[str, Any]) -> None:
        if self.rank != 0 or not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
