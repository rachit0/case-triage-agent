"""Part 1 - deterministic candidate-pair generation.

This is a *blocking* stage, not a classifier. Its only job is to shrink
269*268/2 = 36,046 possible pairs down to a few hundred worth an agent's time,
while losing as few true duplicates as possible. Precision is the agent's job.

Three blocks, deliberately overlapping:
  A. same normalised account key      - the common case
  B. same contact email               - survives account-name typos
  C. fuzzy account key                - survives 'Ostara Eergy' when email is
                                        missing or differs
  D. high subject/body overlap across different accounts (capped) - not because
     we expect duplicates there, but because the agent must be *seen* to reject
     templated lookalikes. Without these, every pair we hand it is a duplicate
     and NOT_DUPLICATE never gets exercised.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations

from .data_loader import Case, load_cases

FUZZY_ACCOUNT_THRESHOLD = 0.88
CROSS_ACCOUNT_SUBJECT_THRESHOLD = 0.75
CROSS_ACCOUNT_CAP = 40


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class Candidate:
    """An unordered pair, stored in a stable order: earlier case first."""
    case_a: str
    case_b: str
    reasons: list[str] = field(default_factory=list)
    cheap_score: float = 0.0
    hours_apart: float | None = None

    @property
    def pair_id(self) -> str:
        return f"{self.case_a}__{self.case_b}"


def _ordered(a: Case, b: Case) -> tuple[Case, Case]:
    """Earlier case first so the later one reads as 'the possible duplicate'.
    Falls back to case_id when a timestamp failed to parse, which keeps the
    pair_id stable across runs."""
    if a.created_at and b.created_at:
        return (a, b) if a.created_at <= b.created_at else (b, a)
    return (a, b) if a.case_id <= b.case_id else (b, a)


def _score(a: Case, b: Case) -> tuple[float, list[str], float | None]:
    reasons: list[str] = []
    score = 0.0

    if a.account_key and a.account_key == b.account_key:
        score += 0.30
        reasons.append("same_account_key")
    elif a.account_key and b.account_key and ratio(a.account_key, b.account_key) >= FUZZY_ACCOUNT_THRESHOLD:
        score += 0.24
        reasons.append("fuzzy_account_match")

    if a.email_key and a.email_key == b.email_key:
        score += 0.25
        reasons.append("same_contact_email")
    elif a.contact_key and a.contact_key == b.contact_key:
        score += 0.12
        reasons.append("same_contact_name")

    subj = jaccard(a.subject_tokens, b.subject_tokens)
    body = jaccard(a.body_tokens, b.body_tokens)
    score += 0.25 * subj + 0.10 * body
    if subj >= 0.5:
        reasons.append(f"subject_overlap={subj:.2f}")
    if body >= 0.7:
        reasons.append(f"body_overlap={body:.2f}")

    gap_h: float | None = None
    if a.created_at and b.created_at:
        gap_h = abs((b.created_at - a.created_at).total_seconds()) / 3600
        if gap_h <= 1:
            score += 0.15
            reasons.append("within_1h")
        elif gap_h <= 48:
            score += 0.10
            reasons.append("within_48h")
        elif gap_h <= 24 * 14:
            score += 0.03

    return round(min(score, 1.0), 4), reasons, gap_h


def generate_candidates(cases: list[Case] | None = None) -> list[Candidate]:
    cases = cases if cases is not None else load_cases()

    by_account: dict[str, list[Case]] = defaultdict(list)
    by_email: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        if c.account_key:
            by_account[c.account_key].append(c)
        if c.email_key:
            by_email[c.email_key].append(c)

    raw_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add(a: Case, b: Case, block: str) -> None:
        if a.case_id == b.case_id:
            return
        first, second = _ordered(a, b)
        raw_pairs[(first.case_id, second.case_id)].add(block)

    # Block A - same account.
    for group in by_account.values():
        if len(group) > 1:
            for a, b in combinations(group, 2):
                add(a, b, "block:account")

    # Block B - same email (catches typo'd account names).
    for group in by_email.values():
        if len(group) > 1:
            for a, b in combinations(group, 2):
                add(a, b, "block:email")

    # Block C - fuzzy account keys. Compared key-to-key (44 keys), not
    # case-to-case, so this stays trivially cheap.
    keys = sorted(by_account)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ratio(ka, kb) >= FUZZY_ACCOUNT_THRESHOLD:
                for a in by_account[ka]:
                    for b in by_account[kb]:
                        add(a, b, "block:fuzzy_account")

    # Block D - cross-account lexical lookalikes, capped. Scored first, then the
    # strongest CROSS_ACCOUNT_CAP are kept, so this cannot blow up the workload.
    cross: list[tuple[float, Case, Case]] = []
    for a, b in combinations(cases, 2):
        if a.account_key == b.account_key:
            continue
        s = jaccard(a.subject_tokens, b.subject_tokens)
        if s >= CROSS_ACCOUNT_SUBJECT_THRESHOLD:
            cross.append((s + jaccard(a.body_tokens, b.body_tokens), a, b))
    cross.sort(key=lambda t: t[0], reverse=True)
    for _, a, b in cross[:CROSS_ACCOUNT_CAP]:
        add(a, b, "block:cross_account_lookalike")

    index = {c.case_id: c for c in cases}
    out: list[Candidate] = []
    for (id_a, id_b), blocks in raw_pairs.items():
        a, b = index[id_a], index[id_b]
        score, reasons, gap = _score(a, b)
        out.append(Candidate(id_a, id_b, sorted(blocks) + reasons, score, gap))

    out.sort(key=lambda c: (-c.cheap_score, c.pair_id))
    return out


BAND_QUOTAS = {"high": 0.25, "mid": 0.45, "low": 0.15, "lookalike": 0.15}


def _band(cand: Candidate) -> str:
    if "block:cross_account_lookalike" in cand.reasons and "same_account_key" not in cand.reasons:
        return "lookalike"
    if cand.cheap_score >= 0.85:
        return "high"
    if cand.cheap_score >= 0.60:
        return "mid"
    if cand.cheap_score >= 0.40:
        return "low"
    return "noise"


def select_for_investigation(candidates: list[Candidate], limit: int = 12,
                             max_per_account: int = 2) -> list[Candidate]:
    """Pick a *diverse* capped subset, stratified by cheap score.

    Taking the top-N by score alone hands the agent twelve near-identical
    same-account-same-minute pairs, all trivially DUPLICATE - which tells us
    nothing about whether the agent can reason. Measured on this dataset, every
    genuinely hard pair (reworded duplicates, and lookalikes that are not
    duplicates) scores 0.46-0.78, i.e. the *middle* band. So we sample bands
    rather than skimming the top, and reserve slots for cross-account
    lookalikes, so the queue exercises DUPLICATE, NOT_DUPLICATE and UNSURE.

    Pairs below 0.40 are dropped: 830 of 926 sit there and they are overwhelmingly
    unrelated cases that merely share a templated sentence.
    """
    from .data_loader import get_case

    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        band = _band(cand)
        if band != "noise":
            buckets[band].append(cand)

    quotas = {band: max(1, round(limit * share)) for band, share in BAND_QUOTAS.items()}
    picked: list[Candidate] = []
    seen: set[str] = set()
    per_account: dict[str, int] = defaultdict(int)

    def try_take(cand: Candidate, enforce_account_cap: bool = True) -> bool:
        if cand.pair_id in seen or len(picked) >= limit:
            return False
        case = get_case(cand.case_a)
        key = case.account_key if case else ""
        if enforce_account_cap and per_account[key] >= max_per_account:
            return False
        per_account[key] += 1
        seen.add(cand.pair_id)
        picked.append(cand)
        return True

    for band in ("high", "mid", "low", "lookalike"):
        taken = 0
        for cand in buckets.get(band, []):
            if taken >= quotas[band] or len(picked) >= limit:
                break
            # Lookalikes are cross-account by definition, so the per-account cap
            # would fight the quota that exists to include them.
            if try_take(cand, enforce_account_cap=(band != "lookalike")):
                taken += 1

    # Top up to `limit` from anything left, best score first.
    if len(picked) < limit:
        for cand in candidates:
            if len(picked) >= limit:
                break
            if _band(cand) != "noise":
                try_take(cand, enforce_account_cap=False)

    picked.sort(key=lambda c: (-c.cheap_score, c.pair_id))
    return picked
