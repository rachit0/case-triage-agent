"""Load and normalise the CRM export.

Design note: normalisation is *additive*. We never overwrite the raw value, we
add `*_norm` fields alongside it. The agent's evidence and the audit trail must
be able to quote what the CRM actually said, not a cleaned-up version of it.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from .config import DATA_CSV

# Words that carry no signal when comparing two support subjects. Kept small on
# purpose: an aggressive stoplist destroys recall in the blocking stage.
STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "for", "to", "and", "or", "is", "are",
    "not", "with", "our", "we", "us", "your", "you", "please", "from", "at",
    "this", "that", "it", "be", "as", "by", "has", "have", "was", "were",
}

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_TOKEN = re.compile(r"[a-z0-9]+")

# Sign-off noise that CRM email ingestion staples onto the body. Stripping it
# stops two unrelated cases from looking similar because both were sent from a
# phone.
_BOILERPLATE = [
    re.compile(r"sent from my mobile device\.?", re.I),
    re.compile(r"raised earlier by email as well,? no response yet\.?", re.I),
]


def _clean_text(value: str) -> str:
    """NFKC-fold, collapse whitespace, strip. Handles the '  Padded Account '
    and 'NORVIG TEXTILES' noise seen in the export."""
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", value)
    return _WS.sub(" ", value).strip()


def normalise_account(name: str) -> str:
    """Aggressive key used only for grouping: lowercase, alphanumerics only.
    'Ostara Energy', '  Ostara Energy ' and 'OSTARA ENERGY' collapse to one key.
    Typos like 'Ostara Eergy' deliberately do NOT collapse here - fuzzy matching
    handles those in candidates.py, so the exact key stays trustworthy."""
    return _NON_ALNUM.sub("", _clean_text(name).lower())


def normalise_email(email: str) -> str:
    return _clean_text(email).lower()


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(_clean_text(text).lower())
            if t not in STOPWORDS and len(t) > 2}


def strip_boilerplate(text: str) -> str:
    out = _clean_text(text)
    for pattern in _BOILERPLATE:
        out = pattern.sub("", out)
    return _WS.sub(" ", out).strip()


@dataclass(frozen=True)
class Case:
    case_id: str
    created_at_raw: str
    channel: str
    status: str
    priority: str
    account_name: str
    contact_name: str
    contact_email: str
    subject: str
    description: str

    created_at: datetime | None = None
    account_key: str = ""
    email_key: str = ""
    contact_key: str = ""
    subject_tokens: frozenset[str] = field(default_factory=frozenset)
    body_tokens: frozenset[str] = field(default_factory=frozenset)
    body_clean: str = ""

    @property
    def text(self) -> str:
        """Subject + description, boilerplate removed. Used for text similarity."""
        return f"{self.subject}. {self.body_clean}"


def _parse_dt(value: str) -> datetime | None:
    value = _clean_text(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _build(row: dict[str, str]) -> Case:
    get = lambda k: _clean_text(row.get(k, ""))  # noqa: E731
    subject = get("subject")
    description = get("description")
    body_clean = strip_boilerplate(description)
    contact = get("contact_name")
    return Case(
        case_id=get("case_id"),
        created_at_raw=get("created_at"),
        channel=get("channel"),
        status=get("status"),
        priority=get("priority"),
        account_name=get("account_name"),
        contact_name=contact,
        contact_email=get("contact_email"),
        subject=subject,
        description=description,
        created_at=_parse_dt(row.get("created_at", "")),
        account_key=normalise_account(get("account_name")),
        email_key=normalise_email(get("contact_email")),
        contact_key=_NON_ALNUM.sub("", contact.lower()),
        subject_tokens=frozenset(tokens(subject)),
        body_tokens=frozenset(tokens(body_clean)),
        body_clean=body_clean,
    )


def load_cases(path: Path | None = None) -> list[Case]:
    path = Path(path or DATA_CSV)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [_build(row) for row in csv.DictReader(fh) if _clean_text(row.get("case_id", ""))]


@lru_cache(maxsize=1)
def case_index() -> dict[str, Case]:
    """case_id -> Case. Cached; the CSV is read-only reference data."""
    return {c.case_id: c for c in load_cases()}


def get_case(case_id: str) -> Case | None:
    return case_index().get(case_id)
