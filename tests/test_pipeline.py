from cuad_llm.data import Contract, Span
from cuad_llm.pipeline import (
    EXCERPT_GAP,
    assemble_predictions,
    build_excerpt,
    full_context_requests,
    spans_retrieved,
)

TEXT = "Governed by New York law. Payment is due in 30 days. Either party may terminate."
CONTRACT = Contract(
    "c1",
    TEXT,
    {
        "Governing Law": [Span("Governed by New York law.", 0)],
        "Non-Compete": [],
    },
)
CATEGORIES = {"Governing Law": "Which law governs", "Non-Compete": "Restrictions on competing"}


def test_requests_group_categories_by_chunk():
    reqs = full_context_requests([CONTRACT], CATEGORIES, chunk_chars=40, overlap_chars=10)
    assert len(reqs) % len(CATEGORIES) == 0 and len(reqs) > len(CATEGORIES)
    # Consecutive requests for one chunk share everything up to the category line.
    a, b = reqs[0].messages[1]["content"], reqs[1].messages[1]["content"]
    prefix = a.split("Clause category:")[0]
    assert b.startswith(prefix) and len(prefix) > 20


def test_assemble_merges_chunk_quotes_and_counts_parse_failures():
    reqs = full_context_requests([CONTRACT], CATEGORIES, chunk_chars=40, overlap_chars=10)
    responses = []
    for r in reqs:
        if r.category == "Governing Law":
            responses.append('["Governed by New York", "New York law."]')
        else:
            responses.append("I cannot tell")  # unparseable
    preds, stats = assemble_predictions([CONTRACT], list(CATEGORIES), reqs, responses)
    by_cat = {p.category: p for p in preds}
    assert by_cat["Governing Law"].pred == ["Governed by New York law."]
    assert by_cat["Non-Compete"].pred == []
    assert stats["parse_failures"] == len(reqs) // 2


def test_assemble_splits_quotes_on_rag_gap_marker():
    excerpt = build_excerpt(TEXT, [(0, 25), (53, len(TEXT))])
    assert EXCERPT_GAP in excerpt
    reqs = full_context_requests([CONTRACT], {"Governing Law": ""}, 1000, 100)[:1]
    quote = excerpt.replace("\n", " ")
    preds, _ = assemble_predictions([CONTRACT], ["Governing Law"], reqs, [f'["{quote}"]'])
    assert preds[0].pred == ["Governed by New York law.", "Either party may terminate."]


def test_spans_retrieved():
    spans = CONTRACT.labels["Governing Law"]
    assert spans_retrieved(spans, [(0, 30)])
    assert not spans_retrieved(spans, [(5, 30)])
