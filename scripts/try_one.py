"""Investigate a single pair and print the trace.  python -m scripts.try_one CS-A__CS-B"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import investigate  # noqa: E402
from app.candidates import generate_candidates, select_for_investigation  # noqa: E402


def main() -> None:
    cands = generate_candidates()
    if len(sys.argv) > 1:
        wanted = sys.argv[1]
        cand = next((c for c in cands if c.pair_id == wanted), None)
        if cand is None:
            print(f"no candidate {wanted}")
            return
    else:
        cand = select_for_investigation(cands, 12)[0]

    res = investigate(cand)
    print(f"pair={cand.pair_id}  mode={res.llm_mode}  "
          f"steps={res.steps_used}  tool_calls={res.tool_calls_used}\n")
    for ev in res.trace:
        detail = json.dumps(ev.detail, default=str)
        print(f"  #{ev.seq:2} {ev.kind:20} {detail[:200]}")
    print("\nVERDICT:", json.dumps(res.verdict.model_dump(), indent=2)[:1200])


if __name__ == "__main__":
    main()
