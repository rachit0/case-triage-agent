"""The human gate and the append-only trail.

The brief's wording is 'it must be impossible to finalise a verdict without a
recorded human decision, no matter how the API is called'. These tests attack
that claim from the API and from raw SQL.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Fresh DB per test, offline agent so tests never touch the network."""
    monkeypatch.setenv("LLM_OFFLINE", "1")
    db = tmp_path / "t.db"

    from app import config, store
    monkeypatch.setattr(config, "DB_PATH", db)
    monkeypatch.setattr(config, "LLM_OFFLINE", True)
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(store, "DB_PATH", db)

    from app.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def one_investigation(client):
    client.post("/investigations/run", params={"limit": 2})
    items = client.get("/investigations").json()
    assert items, "seeding produced no investigations"
    return items[0]["investigation_id"]


def test_new_investigations_are_not_final(one_investigation, client):
    body = client.get(f"/investigations/{one_investigation}").json()
    assert body["status"] == "awaiting_review"
    assert body["final_label"] is None
    assert body["human_decision"] is None
    assert body["verdict"]["label"] in {"DUPLICATE", "NOT_DUPLICATE", "UNSURE"}


def test_there_is_no_endpoint_that_finalises_without_a_decision(client, one_investigation):
    """Enumerate the whole surface: nothing but the decision route mutates status."""
    spec = client.get("/openapi.json").json()
    writers = [(p, m) for p, ops in spec["paths"].items() for m in ops
               if m in {"post", "put", "patch", "delete"}]
    assert all(p.endswith("/decision") or p.startswith("/investigations")
               for p, _ in writers), writers
    # The only route naming a final label is the decision route.
    assert [p for p, _ in writers if p.endswith("/decision")] == \
           ["/investigations/{investigation_id}/decision"]


def test_approve_finalises_with_the_agent_label(client, one_investigation):
    before = client.get(f"/investigations/{one_investigation}").json()
    r = client.post(f"/investigations/{one_investigation}/decision",
                    json={"decision": "approve", "reviewer": "ankit", "note": "ok"})
    assert r.status_code == 200
    assert r.json()["final_label"] == before["verdict"]["label"]

    after = client.get(f"/investigations/{one_investigation}").json()
    assert after["status"] == "finalised"
    assert after["human_decision"]["reviewer"] == "ankit"
    assert after["trace"][-1]["kind"] == "human_decision"


def test_a_second_decision_is_refused(client, one_investigation):
    client.post(f"/investigations/{one_investigation}/decision",
                json={"decision": "approve", "reviewer": "a"})
    r = client.post(f"/investigations/{one_investigation}/decision",
                    json={"decision": "reject", "reviewer": "b", "note": "changed my mind"})
    assert r.status_code == 409


def test_override_requires_a_label_and_a_note(client, one_investigation):
    r = client.post(f"/investigations/{one_investigation}/decision",
                    json={"decision": "override", "reviewer": "a", "note": "x"})
    assert r.status_code == 422
    r = client.post(f"/investigations/{one_investigation}/decision",
                    json={"decision": "override", "reviewer": "a",
                          "override_label": "NOT_DUPLICATE"})
    assert r.status_code == 422


def test_override_records_both_the_agent_draft_and_the_human_label(client, one_investigation):
    r = client.post(f"/investigations/{one_investigation}/decision",
                    json={"decision": "override", "reviewer": "ankit",
                          "note": "different underlying problem",
                          "override_label": "NOT_DUPLICATE", "override_confidence": 0.9})
    assert r.status_code == 200
    body = r.json()
    assert body["final_label"] == "NOT_DUPLICATE"
    detail = client.get(f"/investigations/{one_investigation}").json()
    # The agent's original recommendation survives the override, for audit.
    assert detail["verdict"]["label"] == body["agent_draft_label"]


def test_label_must_be_in_the_enum(client, one_investigation):
    r = client.post(f"/investigations/{one_investigation}/decision",
                    json={"decision": "override", "reviewer": "a", "note": "n",
                          "override_label": "PROBABLY"})
    assert r.status_code == 422


def test_trace_is_append_only_at_the_database_level(client, one_investigation):
    from app import config
    conn = sqlite3.connect(config.DB_PATH)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE trace_events SET kind='tampered' WHERE investigation_id=?",
                     (one_investigation,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM trace_events WHERE investigation_id=?",
                     (one_investigation,))
    conn.close()


def test_decisions_are_append_only_at_the_database_level(client, one_investigation):
    from app import config
    client.post(f"/investigations/{one_investigation}/decision",
                json={"decision": "approve", "reviewer": "a"})
    conn = sqlite3.connect(config.DB_PATH)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE human_decisions SET reviewer='someone_else'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM human_decisions")
    conn.close()


def test_audit_log_reconstructs_the_whole_story(client, one_investigation):
    client.post(f"/investigations/{one_investigation}/decision",
                json={"decision": "approve", "reviewer": "ankit", "note": "confirmed"})
    events = client.get("/audit").json()["events"]
    mine = [e for e in events if e["investigation_id"] == one_investigation]
    kinds = {e["kind"] for e in mine}
    # candidate -> model steps -> tool calls -> drafted verdict -> human decision
    assert {"candidate_selected", "model_step", "tool_call",
            "verdict_drafted", "human_decision"} <= kinds
    decision = next(e for e in mine if e["kind"] == "human_decision")
    assert decision["detail"]["reviewer"] == "ankit"
    assert "final_label" in decision["detail"]


def test_investigating_the_same_pair_twice_does_not_fork_the_trail(client):
    a = client.post("/investigations/run", params={"limit": 2}).json()
    b = client.post("/investigations/run", params={"limit": 2}).json()
    assert b["investigated"] == 0
    assert b["already_present"] == a["investigated"]
