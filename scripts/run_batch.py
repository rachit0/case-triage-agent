"""Investigate a batch of candidate pairs from the CLI and persist them.

    python -m scripts.run_batch --limit 12

Equivalent to POST /investigations/run, but handy for seeding the queue before
demoing the API.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import store  # noqa: E402
from app.agent import investigate  # noqa: E402
from app.candidates import generate_candidates, select_for_investigation  # noqa: E402
from app.data_loader import get_case  # noqa: E402
from app.llm import LLMClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    store.init_db()
    client = LLMClient()
    print(f"LLM mode: {client.mode}\n")

    selected = select_for_investigation(generate_candidates(), limit=args.limit)
    for i, cand in enumerate(selected, 1):
        if (existing := store.pair_exists(cand.pair_id)):
            print(f"[{i:2}/{len(selected)}] {cand.pair_id:24} already investigated ({existing})")
            continue
        a, b = get_case(cand.case_a), get_case(cand.case_b)
        res = investigate(cand, client)
        inv_id = store.save_investigation(
            pair_id=cand.pair_id, case_a=cand.case_a, case_b=cand.case_b,
            cheap_score=cand.cheap_score, reasons=cand.reasons, verdict=res.verdict,
            trace=res.trace, steps_used=res.steps_used,
            tool_calls_used=res.tool_calls_used, llm_mode=res.llm_mode,
            injection_flags=res.injection_flags)
        tools = [e.detail["tool"] for e in res.trace if e.kind == "tool_call"]
        flag = "  [INJECTION FLAGGED]" if res.injection_flags else ""
        print(f"[{i:2}/{len(selected)}] {cand.pair_id:24} "
              f"{res.verdict.label.value:14} conf={res.verdict.confidence:.2f} "
              f"steps={res.steps_used} tools={tools}{flag}")
        print(f"          {a.account_name[:24]:24} | {a.subject[:38]}")
        print(f"          {b.account_name[:24]:24} | {b.subject[:38]}")
        print(f"          -> {res.verdict.rationale[:180]}")
        print(f"          inv={inv_id}\n")

    print("\n", store.stats())
    print("\nNow: uvicorn app.api:app --reload   and open http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    main()
