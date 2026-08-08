"""Pydantic contracts.

Rule enforced throughout the codebase: *no control flow ever reads raw model
text*. The model emits JSON, it is parsed into one of these models, and only the
typed object is acted upon. Anything that fails validation becomes a retry and
then, if it keeps failing, an UNSURE verdict - never a crash and never a guess.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Label(str, Enum):
    DUPLICATE = "DUPLICATE"
    NOT_DUPLICATE = "NOT_DUPLICATE"
    UNSURE = "UNSURE"


class Decision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    OVERRIDE = "override"


class ReviewStatus(str, Enum):
    AWAITING_REVIEW = "awaiting_review"
    FINALISED = "finalised"


# --------------------------------------------------------------------------
# What the model is allowed to say back to us, each step.
# --------------------------------------------------------------------------
class ToolCallAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["call_tool"]
    thought: str = Field(default="", max_length=1200)
    tool: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1, max_length=64)
    observation: str = Field(min_length=1, max_length=600)


class Verdict(BaseModel):
    """The structured, validated recommendation. `label` is a strict enum;
    there is no free-text status field anywhere in the system."""
    model_config = ConfigDict(extra="forbid")

    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2000)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=12)

    @field_validator("rationale")
    @classmethod
    def _single_line(cls, v: str) -> str:
        return " ".join(v.split())


class FinalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["final_answer"]
    thought: str = Field(default="", max_length=1200)
    verdict: Verdict


# --------------------------------------------------------------------------
# Trace records - the append-only audit vocabulary.
# --------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    ts: str = Field(default_factory=_now)
    kind: str            # candidate_selected | model_step | tool_call | guard | verdict_drafted | human_decision
    detail: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# API request/response models.
# --------------------------------------------------------------------------
class HumanDecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)
    override_label: Label | None = None
    override_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("override_label")
    @classmethod
    def _label_ok(cls, v: Label | None) -> Label | None:
        return v


class InvestigationSummary(BaseModel):
    investigation_id: str
    pair_id: str
    case_a: str
    case_b: str
    account_a: str
    account_b: str
    subject_a: str
    subject_b: str
    cheap_score: float
    status: ReviewStatus
    draft_label: Label | None = None
    draft_confidence: float | None = None
    final_label: Label | None = None
    created_at: str


class InvestigationDetail(InvestigationSummary):
    candidate_reasons: list[str] = []
    verdict: Verdict | None = None
    steps_used: int = 0
    tool_calls_used: int = 0
    llm_mode: str = "unknown"
    injection_flags: list[dict[str, Any]] = []
    human_decision: dict[str, Any] | None = None
    trace: list[TraceEvent] = []
