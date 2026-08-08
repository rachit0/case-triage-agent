"""Part 3 - SQLite persistence with an append-only audit trail.

Two design decisions worth defending:

1. Append-only is enforced by the DATABASE, not by convention. SQLite triggers
   raise on any UPDATE or DELETE against `trace_events` and `human_decisions`.
   A future careless endpoint - or someone at a sqlite3 prompt - cannot quietly
   rewrite history.

2. The human gate is a schema property, not a code path. There is no
   `final_label` column that code could set. An investigation is finalised if
   and only if a row exists in `human_decisions` for it, and a UNIQUE index
   allows exactly one such row. It is therefore impossible to finalise a verdict
   without a recorded human decision, no matter how the API is called.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .config import DB_PATH
from .schemas import Decision, HumanDecisionIn, Label, ReviewStatus, TraceEvent, Verdict

SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations (
    id              TEXT PRIMARY KEY,
    pair_id         TEXT NOT NULL UNIQUE,
    case_a          TEXT NOT NULL,
    case_b          TEXT NOT NULL,
    cheap_score     REAL NOT NULL,
    reasons_json    TEXT NOT NULL,
    draft_json      TEXT NOT NULL,
    steps_used      INTEGER NOT NULL,
    tool_calls_used INTEGER NOT NULL,
    llm_mode        TEXT NOT NULL,
    injection_json  TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trace_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT NOT NULL REFERENCES investigations(id),
    seq              INTEGER NOT NULL,
    ts               TEXT NOT NULL,
    kind             TEXT NOT NULL,
    detail_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_trace_inv ON trace_events(investigation_id, seq);

CREATE TABLE IF NOT EXISTS human_decisions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id   TEXT NOT NULL REFERENCES investigations(id),
    decision           TEXT NOT NULL,
    reviewer           TEXT NOT NULL,
    note               TEXT NOT NULL DEFAULT '',
    override_label     TEXT,
    override_confidence REAL,
    final_label        TEXT,
    decided_at         TEXT NOT NULL
);
-- One decision per investigation. This is the human gate, in the schema.
CREATE UNIQUE INDEX IF NOT EXISTS ux_decision_once ON human_decisions(investigation_id);

-- Append-only enforcement.
CREATE TRIGGER IF NOT EXISTS trace_no_update BEFORE UPDATE ON trace_events
BEGIN SELECT RAISE(ABORT, 'trace_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trace_no_delete BEFORE DELETE ON trace_events
BEGIN SELECT RAISE(ABORT, 'trace_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS decision_no_update BEFORE UPDATE ON human_decisions
BEGIN SELECT RAISE(ABORT, 'human_decisions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS decision_no_delete BEFORE DELETE ON human_decisions
BEGIN SELECT RAISE(ABORT, 'human_decisions is append-only'); END;
"""


class DecisionAlreadyRecorded(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------
def save_investigation(pair_id: str, case_a: str, case_b: str, cheap_score: float,
                       reasons: list[str], verdict: Verdict, trace: list[TraceEvent],
                       steps_used: int, tool_calls_used: int, llm_mode: str,
                       injection_flags: list[dict[str, Any]]) -> str:
    inv_id = f"inv_{uuid.uuid4().hex[:12]}"
    with connect() as conn:
        conn.execute(
            "INSERT INTO investigations (id, pair_id, case_a, case_b, cheap_score,"
            " reasons_json, draft_json, steps_used, tool_calls_used, llm_mode,"
            " injection_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (inv_id, pair_id, case_a, case_b, cheap_score, json.dumps(reasons),
             verdict.model_dump_json(), steps_used, tool_calls_used, llm_mode,
             json.dumps(injection_flags), _now()))
        conn.executemany(
            "INSERT INTO trace_events (investigation_id, seq, ts, kind, detail_json)"
            " VALUES (?,?,?,?,?)",
            [(inv_id, e.seq, e.ts, e.kind, json.dumps(e.detail, default=str))
             for e in trace])
    return inv_id


def record_decision(inv_id: str, payload: HumanDecisionIn) -> dict[str, Any]:
    """The one and only way an investigation becomes final."""
    inv = get_investigation_row(inv_id)
    if inv is None:
        raise KeyError(inv_id)

    draft = Verdict.model_validate_json(inv["draft_json"])
    if payload.decision is Decision.APPROVE:
        final_label: str | None = draft.label.value
    elif payload.decision is Decision.OVERRIDE:
        final_label = payload.override_label.value if payload.override_label else None
    else:  # reject - the recommendation is discarded, no verdict stands
        final_label = None

    decided_at = _now()
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO human_decisions (investigation_id, decision, reviewer, note,"
                " override_label, override_confidence, final_label, decided_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (inv_id, payload.decision.value, payload.reviewer, payload.note,
                 payload.override_label.value if payload.override_label else None,
                 payload.override_confidence, final_label, decided_at))
            # The decision itself is part of the trace.
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM trace_events"
                " WHERE investigation_id = ?", (inv_id,)).fetchone()["n"]
            conn.execute(
                "INSERT INTO trace_events (investigation_id, seq, ts, kind, detail_json)"
                " VALUES (?,?,?,?,?)",
                (inv_id, seq, decided_at, "human_decision", json.dumps({
                    "decision": payload.decision.value, "reviewer": payload.reviewer,
                    "note": payload.note,
                    "agent_draft_label": draft.label.value,
                    "agent_draft_confidence": draft.confidence,
                    "override_label": payload.override_label.value if payload.override_label else None,
                    "final_label": final_label})))
    except sqlite3.IntegrityError as exc:
        raise DecisionAlreadyRecorded(str(exc)) from exc

    return {"investigation_id": inv_id, "decision": payload.decision.value,
            "reviewer": payload.reviewer, "final_label": final_label,
            "decided_at": decided_at, "agent_draft_label": draft.label.value}


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------
def get_investigation_row(inv_id: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM investigations WHERE id = ?", (inv_id,)).fetchone()


def get_decision(inv_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM human_decisions WHERE investigation_id = ?",
                           (inv_id,)).fetchone()
    return dict(row) if row else None


def get_trace(inv_id: str) -> list[TraceEvent]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT seq, ts, kind, detail_json FROM trace_events"
            " WHERE investigation_id = ? ORDER BY seq", (inv_id,)).fetchall()
    return [TraceEvent(seq=r["seq"], ts=r["ts"], kind=r["kind"],
                       detail=json.loads(r["detail_json"])) for r in rows]


def list_investigations(status: ReviewStatus | None = None) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT i.*, d.decision AS d_decision, d.final_label AS d_final"
            " FROM investigations i"
            " LEFT JOIN human_decisions d ON d.investigation_id = i.id"
            " ORDER BY i.created_at DESC").fetchall()
    out = []
    for r in rows:
        finalised = r["d_decision"] is not None
        st = ReviewStatus.FINALISED if finalised else ReviewStatus.AWAITING_REVIEW
        if status is not None and st is not status:
            continue
        draft = Verdict.model_validate_json(r["draft_json"])
        out.append({"row": r, "status": st, "draft": draft,
                    "final_label": r["d_final"]})
    return out


def pair_exists(pair_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT id FROM investigations WHERE pair_id = ?",
                           (pair_id,)).fetchone()
    return row["id"] if row else None


def audit_log(limit: int = 500) -> list[dict[str, Any]]:
    """The whole append-only log, newest investigations first, in trace order."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT t.investigation_id, i.pair_id, t.seq, t.ts, t.kind, t.detail_json"
            " FROM trace_events t JOIN investigations i ON i.id = t.investigation_id"
            " ORDER BY t.id DESC LIMIT ?", (limit,)).fetchall()
    return [{"investigation_id": r["investigation_id"], "pair_id": r["pair_id"],
             "seq": r["seq"], "ts": r["ts"], "kind": r["kind"],
             "detail": json.loads(r["detail_json"])} for r in rows]


def stats() -> dict[str, Any]:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM investigations").fetchone()["c"]
        decided = conn.execute("SELECT COUNT(*) c FROM human_decisions").fetchone()["c"]
        by_draft = conn.execute(
            "SELECT json_extract(draft_json,'$.label') label, COUNT(*) c"
            " FROM investigations GROUP BY label").fetchall()
        by_final = conn.execute(
            "SELECT decision, COUNT(*) c FROM human_decisions GROUP BY decision").fetchall()
    return {"investigations": total, "awaiting_review": total - decided,
            "finalised": decided,
            "agent_draft_labels": {r["label"]: r["c"] for r in by_draft},
            "human_decisions": {r["decision"]: r["c"] for r in by_final}}


__all__ = ["init_db", "save_investigation", "record_decision", "get_investigation_row",
           "get_decision", "get_trace", "list_investigations", "pair_exists",
           "audit_log", "stats", "DecisionAlreadyRecorded", "Label"]
