"""Inspect the candidate-generation output.  python -m scripts.show_candidates"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.candidates import generate_candidates, select_for_investigation  # noqa: E402
from app.data_loader import get_case  # noqa: E402


def line(c) -> str:
    a, b = get_case(c.case_a), get_case(c.case_b)
    gap = f"{c.hours_apart:7.1f}h" if c.hours_apart is not None else "      ?"
    return (f"{c.cheap_score:.2f} {c.pair_id:22} {gap}  "
            f"{a.account_name[:20]:20}/{b.account_name[:20]:20} "
            f"{a.subject[:30]:30}|| {b.subject[:30]}")


def main() -> None:
    cands = generate_candidates()
    print(f"total candidate pairs: {len(cands)}  (from 269 cases = 36,046 possible)")
    print("\n--- top 15 by cheap score ---")
    for c in cands[:15]:
        print(line(c))

    sel = select_for_investigation(cands, limit=12)
    print(f"\n--- selected {len(sel)} for investigation (diversity-capped) ---")
    for c in sel:
        print(line(c))
        print(f"        reasons: {c.reasons}")


if __name__ == "__main__":
    main()
