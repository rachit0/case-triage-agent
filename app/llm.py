"""Thin OpenAI-compatible chat client with the robustness the brief asks for:
exponential backoff with jitter, Retry-After support, JSON extraction/repair,
and a hard failure mode that the agent turns into UNSURE rather than a crash.

Why not native tool-calling? Free tiers differ wildly in tool-call support and
quality (Groq, OpenRouter's free models and Ollama are not consistent). Asking
for a single JSON action object per turn works everywhere and keeps the trace
readable. Trade-off documented in the README.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

import httpx

from . import config


class LLMError(RuntimeError):
    """Raised only after retries are exhausted."""


class LLMUnavailable(LLMError):
    """No key configured / offline mode - caller should use the fallback planner."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict[str, Any]:
    """Pull one JSON object out of a model response.

    Models on free tiers routinely wrap JSON in prose or code fences, or emit a
    trailing comma. We try, in order: whole string, fenced block, first balanced
    brace span, then a couple of cheap repairs.
    """
    if not text or not text.strip():
        raise LLMError("empty model response")
    candidates: list[str] = [text.strip()]

    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())

    start = text.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break

    for raw in candidates:
        for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
            try:
                parsed = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
    raise LLMError(f"no JSON object in model response: {text[:300]!r}")


class LLMClient:
    def __init__(self) -> None:
        self.base_url = config.LLM_BASE_URL.rstrip("/")
        self.api_key = config.LLM_API_KEY
        self.model = config.LLM_MODEL
        self.enabled = config.llm_enabled()
        self.calls = 0

    @property
    def mode(self) -> str:
        return f"live:{self.model}" if self.enabled else "offline-fallback"

    def complete_json(self, system: str, user: str,
                      temperature: float = 0.1) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (parsed_json, meta). Retries transient failures with backoff."""
        if not self.enabled:
            raise LLMUnavailable("no LLM_API_KEY configured (or LLM_OFFLINE set)")

        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            # Widely supported on OpenAI-compatible endpoints; harmless if ignored,
            # which is exactly why extract_json() still exists.
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"

        last_error = ""
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                with httpx.Client(timeout=config.LLM_TIMEOUT_S) as client:
                    resp = client.post(url, json=payload, headers=headers)
                self.calls += 1

                if resp.status_code in (429, 500, 502, 503, 504, 529):
                    last_error = f"http {resp.status_code}: {resp.text[:400]}"
                    # Groq's free tier meters tokens per MINUTE, and says exactly
                    # when the bucket refills. Guessing with exponential backoff
                    # under-waits a 60s window and burns the retry budget for
                    # nothing, so prefer what the provider tells us.
                    self._sleep(attempt,
                                resp.headers.get("Retry-After")
                                or resp.headers.get("x-ratelimit-reset-tokens")
                                or resp.headers.get("x-ratelimit-reset-requests"))
                    continue
                if resp.status_code >= 400:
                    raise LLMError(f"http {resp.status_code}: {resp.text[:300]}")

                content = resp.json()["choices"][0]["message"]["content"]
                parsed = extract_json(content)
                return parsed, {"attempts": attempt + 1, "raw_len": len(content or "")}

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._sleep(attempt, None)
            except LLMError as exc:
                # Malformed JSON is retryable: same prompt, new sample.
                last_error = str(exc)
                if attempt == config.LLM_MAX_RETRIES - 1:
                    break
                payload["temperature"] = min(0.6, payload["temperature"] + 0.2)
                self._sleep(attempt, None)
            except (KeyError, IndexError, ValueError) as exc:
                last_error = f"unexpected response shape: {type(exc).__name__}: {exc}"
                self._sleep(attempt, None)

        raise LLMError(f"exhausted {config.LLM_MAX_RETRIES} attempts; last error: {last_error}")

    @staticmethod
    def _parse_wait(value: str | None) -> float | None:
        """Seconds from a Retry-After or an x-ratelimit-reset-* header.

        Retry-After is plain seconds ("20"). Groq's reset headers use a compact
        duration instead ("105ms", "7.66s", "1m26.4s"), so parse both rather than
        silently falling through to a guess.
        """
        if not value:
            return None
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            pass
        m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?|(\d+(?:\.\d+)?)ms",
                         value, re.I)
        if not m:
            return None
        if m.group(3) is not None:
            return float(m.group(3)) / 1000.0
        mins, secs = m.group(1), m.group(2)
        if mins is None and secs is None:
            return None
        return float(mins or 0) * 60 + float(secs or 0)

    @classmethod
    def _sleep(cls, attempt: int, retry_after: str | None) -> None:
        wait = cls._parse_wait(retry_after)
        if wait is not None:
            # A tiny reset ("105ms") means the bucket is already nearly full and
            # the 429 was a burst; still pause briefly so we do not hot-loop.
            time.sleep(min(max(wait, 0.5), 65.0))
            return
        # Exponential backoff with full jitter, capped.
        time.sleep(min(16.0, (2 ** attempt)) * (0.5 + random.random() / 2))
