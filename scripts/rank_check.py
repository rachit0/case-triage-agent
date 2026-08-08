"""Where do the hand-read interesting pairs rank in the candidate list?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.candidates import generate_candidates  # noqa: E402

INTERESTING = {
    frozenset(("CS-64254", "CS-29126")): "reworded dup (excel/blank)",
    frozenset(("CS-47101", "CS-50425")): "reworded dup (data feed)",
    frozenset(("CS-42208", "CS-43728")): "reworded dup (webhook)",
    frozenset(("CS-34123", "CS-19742")): "reworded dup (throttling)",
    frozenset(("CS-53642", "CS-34906")): "reworded dup (portal login)",
    frozenset(("CS-60493", "CS-64238")): "reworded dup + PROMPT INJECTION",
    frozenset(("CS-68292", "CS-27367")): "NOT dup (403 vs lockout)",
    frozenset(("CS-13442", "CS-56388")): "NOT dup (follow-up)",
}


def main() -> None:
    cands = generate_candidates()
    found = {}
    for rank, c in enumerate(cands, 1):
        key = frozenset((c.case_a, c.case_b))
        if key in INTERESTING:
            found[key] = (rank, c)
    print(f"{len(cands)} candidates\n")
    for key, label in INTERESTING.items():
        if key in found:
            rank, c = found[key]
            print(f"  rank {rank:4}/{len(cands)}  score={c.cheap_score:.2f}  {c.pair_id:24} {label}")
        else:
            print(f"  MISSED (recall failure!)          {sorted(key)}          {label}")

    bands = {"hi >=0.85": 0, "mid 0.60-0.85": 0, "low 0.40-0.60": 0, "vlow <0.40": 0}
    for c in cands:
        if c.cheap_score >= 0.85:
            bands["hi >=0.85"] += 1
        elif c.cheap_score >= 0.60:
            bands["mid 0.60-0.85"] += 1
        elif c.cheap_score >= 0.40:
            bands["low 0.40-0.60"] += 1
        else:
            bands["vlow <0.40"] += 1
    print("\nscore distribution:", bands)


if __name__ == "__main__":
    main()
