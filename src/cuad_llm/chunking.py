"""Split long contracts into overlapping windows and project gold spans onto them.

Full CUAD contracts run up to ~300k characters, far too long to fine-tune on. We train and infer on
windows, then merge per-window predictions back to the contract level for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from cuad_llm.data import Span

# Preferred places to end a window, best first.
_BREAKS = ("\n\n", "\n", ". ", "; ", " ")


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    text: str


def _last_break(text: str, lo: int, hi: int) -> int | None:
    """Position just after the last, best-ranked break in text[lo:hi], or None."""
    for sep in _BREAKS:
        i = text.rfind(sep, lo, hi)
        if i != -1:
            return i + len(sep)
    return None


def _first_break(text: str, lo: int, hi: int) -> int | None:
    """Position just after the first, best-ranked break in text[lo:hi], or None."""
    for sep in _BREAKS:
        i = text.find(sep, lo, hi)
        if i != -1 and i + len(sep) < hi:
            return i + len(sep)
    return None


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[Chunk]:
    if overlap_chars >= max_chars // 2:
        raise ValueError("overlap_chars must be less than half of max_chars")
    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # Prefer a natural break in the last 20% of the window.
            end = _last_break(text, start + int(max_chars * 0.8), end) or end
        chunks.append(Chunk(start, end, text[start:end]))
        if end >= n:
            break
        # Start the next window inside the overlap region, snapped to a break if one exists. Only
        # search the first half of the overlap: searching all of it finds the break the previous
        # window ended on, which shrinks the overlap to nothing.
        next_start = end - overlap_chars
        start = _first_break(text, next_start, next_start + overlap_chars // 2) or next_start
    return chunks


def spans_in_chunk(spans: list[Span], chunk: Chunk, all_chunks: list[Chunk]) -> list[str]:
    """Gold quotes a model should extract from this chunk.

    A span fully inside the chunk is kept whole. A span that crosses the chunk boundary is kept
    (clipped) only if no chunk contains it fully; otherwise the chunk that holds it whole owns it,
    which keeps sliver fragments out of the training targets.
    """
    quotes = []
    for span in spans:
        if chunk.start <= span.start and span.end <= chunk.end:
            quotes.append(span.text)
            continue
        s, e = max(span.start, chunk.start), min(span.end, chunk.end)
        if s >= e:
            continue
        if any(c.start <= span.start and span.end <= c.end for c in all_chunks):
            continue
        clipped = chunk.text[s - chunk.start : e - chunk.start].strip()
        if clipped:
            quotes.append(clipped)
    return quotes
