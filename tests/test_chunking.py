from itertools import pairwise

from cuad_llm.chunking import chunk_text, spans_in_chunk
from cuad_llm.data import Span

TEXT = "\n\n".join(f"Section {i}. " + "The parties agree to terms. " * 8 for i in range(40))


def test_chunks_cover_text_with_overlap():
    chunks = chunk_text(TEXT, max_chars=1000, overlap_chars=150)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(TEXT)
    for prev, nxt in pairwise(chunks):
        assert prev.end - nxt.start >= 150 // 2, "consecutive chunks must overlap"
        assert nxt.start > prev.start, "chunking must make progress"
    for c in chunks:
        assert c.text == TEXT[c.start : c.end]
        assert len(c.text) <= 1000


def test_short_text_is_one_chunk():
    (chunk,) = chunk_text("tiny contract", 1000, 100)
    assert chunk.text == "tiny contract"


def test_span_is_owned_by_chunk_that_fully_contains_it():
    chunks = chunk_text(TEXT, max_chars=1000, overlap_chars=150)
    # A short span straddling the end of chunk 0 but fully inside chunk 1's overlap.
    start = chunks[0].end - 20
    span = Span(TEXT[start : start + 40], start)
    assert span.end <= chunks[1].end and span.start >= chunks[1].start
    assert spans_in_chunk([span], chunks[0], chunks) == []
    assert spans_in_chunk([span], chunks[1], chunks) == [span.text]


def test_span_longer_than_any_chunk_is_clipped_into_each():
    chunks = chunk_text(TEXT, max_chars=1000, overlap_chars=150)
    span = Span(TEXT[100:2500], 100)
    pieces = [spans_in_chunk([span], c, chunks) for c in chunks]
    touched = [p for p in pieces if p]
    assert len(touched) >= 3
    assert all(piece[0] in span.text for piece in touched)
