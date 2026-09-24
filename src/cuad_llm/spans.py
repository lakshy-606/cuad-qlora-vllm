"""Text normalisation and mapping predicted quotes back onto the contract."""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def locate(quote: str, contract: str) -> tuple[int, int] | None:
    """Find a quote in the contract, tolerating whitespace and case differences."""
    i = contract.find(quote)
    if i != -1:
        return i, i + len(quote)
    words = quote.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, contract, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def merge_predictions(quotes: list[str], contract: str) -> list[str]:
    """Collapse per-chunk quotes into contract-level spans.

    Quotes found in the contract are merged where their character ranges overlap or touch, so a clause
    extracted in two halves from neighbouring chunks becomes one span again. Quotes that can't be
    located (paraphrases, hallucinations) are kept verbatim so they still count against the model.
    """
    ranges: list[tuple[int, int]] = []
    unlocated: list[str] = []
    for q in quotes:
        q = q.strip()
        if not q:
            continue
        loc = locate(q, contract)
        if loc is None:
            unlocated.append(q)
        else:
            ranges.append(loc)
    ranges.sort()
    merged: list[list[int]] = []
    for s, e in ranges:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    located = [contract[s:e] for s, e in merged]
    seen = set()
    extra = []
    for q in unlocated:
        if normalize(q) not in seen:
            seen.add(normalize(q))
            extra.append(q)
    return located + extra
