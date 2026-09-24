"""Turn contracts into model requests and model responses back into contract-level predictions.

Shared by every inference setup (zero-shot, RAG, fine-tuned) so they are scored identically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cuad_llm.chunking import chunk_text
from cuad_llm.data import Contract, Span
from cuad_llm.metrics import PairPrediction
from cuad_llm.prompts import build_messages, parse_answer
from cuad_llm.spans import merge_predictions

# Separates non-adjacent retrieved passages in a RAG excerpt.
EXCERPT_GAP = "\n\n[...]\n\n"


@dataclass
class Request:
    contract_id: str
    category: str
    messages: list[dict]


def full_context_requests(
    contracts: list[Contract], categories: dict[str, str], chunk_chars: int, overlap_chars: int
) -> list[Request]:
    """One request per (chunk, category). Categories vary fastest so each chunk's prompts are
    adjacent, which is what lets the server reuse the cached chunk prefix."""
    requests = []
    for contract in contracts:
        for chunk in chunk_text(contract.text, chunk_chars, overlap_chars):
            for category, description in categories.items():
                messages = build_messages(category, description, chunk.text)
                requests.append(Request(contract.contract_id, category, messages))
    return requests


def build_excerpt(text: str, ranges: list[tuple[int, int]]) -> str:
    return EXCERPT_GAP.join(text[s:e] for s, e in ranges)


def spans_retrieved(spans: list[Span], ranges: list[tuple[int, int]]) -> bool:
    """Every gold span lies fully inside one retrieved range."""
    return all(any(s <= sp.start and sp.end <= e for s, e in ranges) for sp in spans)


def assemble_predictions(
    contracts: list[Contract], categories: list[str], requests: list[Request], responses: list[str]
) -> tuple[list[PairPrediction], dict[str, int]]:
    """Parse responses and merge quotes per (contract, category).

    Every contract x category pair gets a prediction, even with no request (e.g. nothing retrieved).
    Unparseable responses count as "not present" and are tallied in the stats.
    """
    quotes: dict[tuple[str, str], list[str]] = defaultdict(list)
    parse_failures = 0
    for req, text in zip(requests, responses, strict=True):
        parsed = parse_answer(text)
        if parsed is None:
            parse_failures += 1
            continue
        for q in parsed:
            # A RAG answer may copy the gap marker; split so each side can still be located.
            quotes[(req.contract_id, req.category)].extend(
                part for part in q.split("[...]") if part.strip()
            )

    predictions = []
    for contract in contracts:
        for category in categories:
            pred = merge_predictions(quotes[(contract.contract_id, category)], contract.text)
            gold = [s.text for s in contract.labels[category]]
            predictions.append(PairPrediction(contract.contract_id, category, gold, pred))
    return predictions, {"requests": len(requests), "parse_failures": parse_failures}
