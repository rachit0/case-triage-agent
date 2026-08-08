"""Central configuration. Everything is overridable by environment variable so the
project runs on a fresh machine with zero edits to source."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = Path(os.getenv("CASES_CSV", ROOT / "data" / "support_cases.csv"))
DB_PATH = Path(os.getenv("TRIAGE_DB", ROOT / "data" / "triage.db"))

# --- LLM provider -----------------------------------------------------------
# We speak the OpenAI-compatible /chat/completions dialect, which Groq,
# OpenRouter, Together and a local Ollama all implement. Swapping provider is
# two environment variables, not a code change.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")

# If no API key is configured we fall back to a deterministic offline planner so
# that the whole system (API, gate, audit trail) is still demonstrable.
LLM_OFFLINE = os.getenv("LLM_OFFLINE", "").lower() in {"1", "true", "yes"}

LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))

# --- Agent bounds -----------------------------------------------------------
# MAX_TOOL_CALLS must stay STRICTLY BELOW MAX_AGENT_STEPS. Every tool call costs
# one step, so if the tool budget were the looser of the two it could never bind:
# the step wall would always hit first, with no step left to emit final_answer,
# and every thorough investigation would be forced to UNSURE. Keeping tools lower
# reserves (steps - tools) turns in which the loop can press for a verdict.
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "6"))


def llm_enabled() -> bool:
    return bool(LLM_API_KEY) and not LLM_OFFLINE
