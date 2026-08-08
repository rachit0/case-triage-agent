"""Part 3 - the approval flow.

Run:  uvicorn app.api:app --reload
Then: http://127.0.0.1:8000/docs

The human gate lives here and in the schema below it. There is no endpoint that
writes a final verdict; the only way an investigation leaves `awaiting_review`
is POST /investigations/{id}/decision, which inserts a row into the append-only
`human_decisions` table protected by a UNIQUE index.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from . import config, store
from .agent import investigate
from .candidates import Candidate, generate_candidates, select_for_investigation
from .data_loader import get_case
from .llm import LLMClient
from .schemas import (Decision, HumanDecisionIn, InvestigationDetail,
                      InvestigationSummary, ReviewStatus, Verdict)

_candidates_cache: list[Candidate] | None = None


def _candidates() -> list[Candidate]:
    global _candidates_cache
    if _candidates_cache is None:
        _candidates_cache = generate_candidates()
    return _candidates_cache


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.init_db()
    yield


app = FastAPI(
    title="Case Triage Agent",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "An agent investigates whether two support cases are duplicates and **stops**. "
        "Nothing is final until a human decides.\n\n"
        "**Demo path:** `POST /investigations/run` -> `GET /investigations` -> "
        "`GET /investigations/{id}` -> `POST /investigations/{id}/decision` -> `GET /audit`."
    ),
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    client = LLMClient()
    return {"ok": True, "llm_mode": client.mode, "model": config.LLM_MODEL,
            "base_url": config.LLM_BASE_URL,
            "max_agent_steps": config.MAX_AGENT_STEPS,
            "max_tool_calls": config.MAX_TOOL_CALLS,
            **store.stats()}


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------
@app.get("/candidates", tags=["candidates"],
         summary="Deterministic candidate pairs (Part 1 output)")
def list_candidates(limit: int = Query(25, ge=1, le=500),
                    min_score: float = Query(0.0, ge=0.0, le=1.0)) -> dict[str, Any]:
    cands = [c for c in _candidates() if c.cheap_score >= min_score]
    return {
        "total_candidates": len(_candidates()),
        "returned": min(limit, len(cands)),
        "items": [{
            "pair_id": c.pair_id, "case_a": c.case_a, "case_b": c.case_b,
            "cheap_score": c.cheap_score, "hours_apart": c.hours_apart,
            "reasons": c.reasons,
            "already_investigated": store.pair_exists(c.pair_id),
        } for c in cands[:limit]],
    }


# --------------------------------------------------------------------------
# Running investigations
# --------------------------------------------------------------------------
def _run_and_store(cand: Candidate) -> str:
    existing = store.pair_exists(cand.pair_id)
    if existing:
        return existing
    result = investigate(cand)
    return store.save_investigation(
        pair_id=cand.pair_id, case_a=cand.case_a, case_b=cand.case_b,
        cheap_score=cand.cheap_score, reasons=cand.reasons, verdict=result.verdict,
        trace=result.trace, steps_used=result.steps_used,
        tool_calls_used=result.tool_calls_used, llm_mode=result.llm_mode,
        injection_flags=result.injection_flags)


@app.post("/investigations/run", tags=["investigations"],
          summary="Investigate a capped, diversity-stratified batch of candidate pairs")
def run_batch(limit: int = Query(12, ge=1, le=30)) -> dict[str, Any]:
    """Seeds the review queue. Idempotent per pair: a pair already investigated
    is returned as-is rather than re-run, so repeated calls do not burn free-tier
    quota or fork the audit trail."""
    selected = select_for_investigation(_candidates(), limit=limit)
    created, reused = [], []
    for cand in selected:
        existing = store.pair_exists(cand.pair_id)
        (reused if existing else created).append(cand.pair_id)
        _run_and_store(cand)
    return {"requested": limit, "investigated": len(created), "already_present": len(reused),
            "new_pairs": created, "reused_pairs": reused,
            "next": "GET /investigations?status=awaiting_review"}


@app.post("/investigations/{pair_id}", tags=["investigations"],
          summary="Lazily investigate one specific candidate pair")
def run_one(pair_id: str) -> dict[str, Any]:
    cand = next((c for c in _candidates() if c.pair_id == pair_id), None)
    if cand is None:
        raise HTTPException(404, f"no candidate pair '{pair_id}'. "
                                 f"See GET /candidates for valid pair_ids.")
    inv_id = _run_and_store(cand)
    return {"investigation_id": inv_id, "pair_id": pair_id,
            "next": f"GET /investigations/{inv_id}"}


# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------
def _summary(entry: dict[str, Any]) -> InvestigationSummary:
    r, draft = entry["row"], entry["draft"]
    a, b = get_case(r["case_a"]), get_case(r["case_b"])
    return InvestigationSummary(
        investigation_id=r["id"], pair_id=r["pair_id"],
        case_a=r["case_a"], case_b=r["case_b"],
        account_a=a.account_name if a else "?", account_b=b.account_name if b else "?",
        subject_a=a.subject if a else "?", subject_b=b.subject if b else "?",
        cheap_score=r["cheap_score"], status=entry["status"],
        draft_label=draft.label, draft_confidence=draft.confidence,
        final_label=entry["final_label"], created_at=r["created_at"])


@app.get("/investigations", tags=["review"], response_model=list[InvestigationSummary],
         summary="List investigations (default: those awaiting a human decision)")
def list_investigations(status: ReviewStatus | None = ReviewStatus.AWAITING_REVIEW
                        ) -> list[InvestigationSummary]:
    return [_summary(e) for e in store.list_investigations(status)]


@app.get("/investigations/{investigation_id}", tags=["review"],
         response_model=InvestigationDetail,
         summary="One investigation with its recommendation and full evidence trail")
def get_investigation(investigation_id: str) -> InvestigationDetail:
    import json as _json

    row = store.get_investigation_row(investigation_id)
    if row is None:
        raise HTTPException(404, f"unknown investigation '{investigation_id}'")
    decision = store.get_decision(investigation_id)
    draft = Verdict.model_validate_json(row["draft_json"])
    entry = {"row": row,
             "status": ReviewStatus.FINALISED if decision else ReviewStatus.AWAITING_REVIEW,
             "draft": draft,
             "final_label": decision["final_label"] if decision else None}
    base = _summary(entry)
    return InvestigationDetail(
        **base.model_dump(),
        candidate_reasons=_json.loads(row["reasons_json"]),
        verdict=draft,
        steps_used=row["steps_used"], tool_calls_used=row["tool_calls_used"],
        llm_mode=row["llm_mode"],
        injection_flags=_json.loads(row["injection_json"]),
        human_decision=decision,
        trace=store.get_trace(investigation_id))


@app.get("/investigations/{investigation_id}/trace", tags=["audit"],
         summary="Just the append-only trace for one investigation")
def get_trace(investigation_id: str) -> dict[str, Any]:
    if store.get_investigation_row(investigation_id) is None:
        raise HTTPException(404, f"unknown investigation '{investigation_id}'")
    return {"investigation_id": investigation_id,
            "events": [e.model_dump() for e in store.get_trace(investigation_id)]}


# --------------------------------------------------------------------------
# THE HUMAN GATE
# --------------------------------------------------------------------------
@app.post("/investigations/{investigation_id}/decision", tags=["review"],
          summary="Record the human decision - the only way a verdict becomes final")
def decide(investigation_id: str,
           payload: HumanDecisionIn = Body(..., examples=[
               {"decision": "approve", "reviewer": "ankit", "note": "agrees with the evidence"},
               {"decision": "override", "reviewer": "ankit",
                "note": "different issue: 403 on reports vs a password lockout",
                "override_label": "NOT_DUPLICATE", "override_confidence": 0.9},
               {"decision": "reject", "reviewer": "ankit", "note": "evidence too thin, re-run"},
           ])) -> dict[str, Any]:
    if store.get_investigation_row(investigation_id) is None:
        raise HTTPException(404, f"unknown investigation '{investigation_id}'")

    if payload.decision is Decision.OVERRIDE and payload.override_label is None:
        raise HTTPException(422, "override requires 'override_label'")
    if payload.decision is not Decision.OVERRIDE and payload.override_label is not None:
        raise HTTPException(422, "'override_label' is only valid with decision='override'")
    if payload.decision in (Decision.REJECT, Decision.OVERRIDE) and not payload.note.strip():
        raise HTTPException(422, f"'{payload.decision.value}' requires a note explaining why")

    try:
        return {**store.record_decision(investigation_id, payload),
                "audit": f"GET /investigations/{investigation_id}/trace"}
    except store.DecisionAlreadyRecorded:
        existing = store.get_decision(investigation_id)
        raise HTTPException(409, {
            "error": "a human decision is already recorded for this investigation",
            "detail": "the decision log is append-only; decisions cannot be rewritten",
            "existing": existing})


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
@app.get("/audit", tags=["audit"],
         summary="The whole append-only log: tool calls, model steps, guards, decisions")
def audit(limit: int = Query(300, ge=1, le=2000)) -> dict[str, Any]:
    return {"stats": store.stats(), "events": store.audit_log(limit)}
