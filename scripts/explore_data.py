"""Step 0: read the data before writing the agent.

Run:  python -m scripts.explore_data
Prints the messiness that shaped every downstream design decision.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data_loader import load_cases, normalise_account  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    cases = load_cases()
    print(f"rows: {len(cases)}   unique case_id: {len({c.case_id for c in cases})}")

    rule("1. account_name is dirty")
    # Read the raw CSV here on purpose: Case.account_name is already cleaned, so
    # counting whitespace damage on it would report zero and hide the problem.
    import csv as _csv
    from app.config import DATA_CSV
    with DATA_CSV.open("r", encoding="utf-8", newline="") as fh:
        raw_names = [r["account_name"] for r in _csv.DictReader(fh)]
    raw = Counter(raw_names)
    padded = [n for n in raw if n != n.strip()]
    upper = [n for n in raw if n.isupper()]
    print(f"distinct raw account strings : {len(raw)}")
    print(f"distinct normalised keys     : {len({normalise_account(n) for n in raw})}")
    print(f"leading/trailing whitespace  : {padded}")
    print(f"ALL CAPS                     : {upper}")

    # Near-miss account names = probable typos of a real account.
    keys = sorted({normalise_account(n) for n in raw})
    print("\nlikely typos (edit-distance-1 neighbours among account keys):")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if abs(len(a) - len(b)) <= 1 and _within_one(a, b):
                print(f"   {a!r:32} ~ {b!r}")

    rule("2. contact_email is dirty / missing")
    missing = [c for c in cases if not c.contact_email]
    caps = [c for c in cases if c.contact_email and c.contact_email != c.contact_email.lower()]
    print(f"missing email : {len(missing)} rows  e.g. {[c.case_id for c in missing[:5]]}")
    print(f"UPPERCASE     : {len(caps)} rows  e.g. {[c.case_id for c in caps[:5]]}")
    nickname = []
    for c in cases:
        if not c.contact_email:
            continue
        local = c.contact_email.split("@")[0].lower()
        first = c.contact_name.split(" ")[0].lower() if c.contact_name else ""
        if first and first not in local:
            nickname.append((c.case_id, c.contact_name, c.contact_email))
    print(f"name/email mismatch (nicknames): {nickname}")

    rule("3. description text is templated -> identical text != duplicate")
    by_body = defaultdict(list)
    for c in cases:
        by_body[c.body_clean].append(c.case_id)
    repeated = sorted(((len(v), k[:60]) for k, v in by_body.items() if len(v) > 1), reverse=True)
    print(f"{len(repeated)} description templates are reused; top 5:")
    for n, snippet in repeated[:5]:
        print(f"   {n:3} x  {snippet}...")
    print("\n=> Lexical similarity alone will produce hundreds of false positives.")

    rule("4. same account+contact, near-identical case, minutes apart (true dupes)")
    by_pair = defaultdict(list)
    for c in cases:
        if c.email_key:
            by_pair[(c.account_key, c.email_key)].append(c)
    hits = 0
    for (_acct, _mail), group in by_pair.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda c: c.created_at or c.case_id)
        for a, b in zip(group, group[1:]):
            if a.created_at and b.created_at:
                gap_h = (b.created_at - a.created_at).total_seconds() / 3600
                if gap_h <= 48:
                    hits += 1
                    if hits <= 12:
                        print(f"   {a.case_id} / {b.case_id}  gap={gap_h:5.1f}h  "
                              f"{a.account_name.strip()[:22]:22} | {a.subject[:34]:34} || {b.subject[:34]}")
    print(f"   ... {hits} same-account+contact pairs within 48h total")

    rule("5. reworded duplicates: same issue, low lexical overlap")
    print("   Found by reading, not by token overlap - these are why we need an LLM:")
    for pair in [("CS-64254", "CS-29126"), ("CS-47101", "CS-50425"),
                 ("CS-42208", "CS-43728"), ("CS-34123", "CS-19742"),
                 ("CS-53642", "CS-34906"), ("CS-60493", "CS-64238")]:
        idx = {c.case_id: c for c in cases}
        a, b = idx[pair[0]], idx[pair[1]]
        jac = len(a.subject_tokens & b.subject_tokens) / max(1, len(a.subject_tokens | b.subject_tokens))
        print(f"   {a.case_id}/{b.case_id} subject-jaccard={jac:.2f}  {a.subject!r} vs {b.subject!r}")

    rule("6. lookalikes that are NOT duplicates")
    print("   CS-68292 / CS-27367  Finwick Robotics, same contact, both 'Portal login'")
    print("     -> one is a 403 on the reports module, one is a locked-out password reset.")
    print("   CS-13442 / CS-56388  Ostara Energy, same contact, both about the data feed")
    print("     -> the second is an explicit FOLLOW-UP about missing backfill, not a re-report.")

    rule("7. SECURITY: prompt injection embedded in case text")
    inj = re.compile(r"(system note|ignore (all )?previous|do not flag|classify this|"
                     r"automated reviewers?|instructions?:)", re.I)
    for c in cases:
        if inj.search(c.description) or inj.search(c.subject):
            print(f"   {c.case_id}  {c.account_name.strip()}")
            print(f"      {c.description[:220]}")
    print("\n   => Case text is UNTRUSTED. It is fenced, never concatenated into the")
    print("      system prompt, and the model is told data cannot issue instructions.")


def _within_one(a: str, b: str) -> bool:
    """True if a and b differ by <=1 edit (insert/delete/substitute/transpose)."""
    if a == b:
        return False
    if len(a) == len(b):
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diff) == 1:
            return True
        if len(diff) == 2 and diff[1] == diff[0] + 1:  # transposition
            i = diff[0]
            return a[i] == b[i + 1] and a[i + 1] == b[i]
        return False
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(b)):
        if b[:i] + b[i + 1:] == a:
            return True
    return False


if __name__ == "__main__":
    main()
