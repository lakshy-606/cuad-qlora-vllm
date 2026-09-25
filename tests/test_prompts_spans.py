from cuad_llm.prompts import build_messages, parse_answer
from cuad_llm.spans import locate, merge_predictions


def test_parse_answer_variants():
    assert parse_answer('["a", "b"]') == ["a", "b"]
    assert parse_answer("[]") == []
    assert parse_answer('```json\n["x"]\n```') == ["x"]
    assert parse_answer('<think>hmm</think>\n["x"]') == ["x"]
    assert parse_answer("No related clause.") == []
    assert parse_answer('Here you go: ["x"]') == ["x"]
    assert parse_answer("the clause is about payment") is None
    assert parse_answer("[1, 2]") is None


def test_build_messages_round_trips_answer():
    msgs = build_messages("Governing Law", "Which law governs", "excerpt", ['He said "hi"'])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert parse_answer(msgs[-1]["content"]) == ['He said "hi"']


CONTRACT = "This Agreement is governed by the laws of\nNew York. Payment is due in 30 days."


def test_locate_tolerates_whitespace_and_case():
    assert locate("governed by the laws of New York", CONTRACT) is not None
    assert locate("GOVERNED BY THE LAWS", CONTRACT) is not None
    assert locate("governed by California law", CONTRACT) is None


def test_merge_joins_overlapping_pieces_and_keeps_unlocated():
    merged = merge_predictions(
        ["governed by the laws", "the laws of\nNew York.", "made up clause"], CONTRACT
    )
    assert merged == ["governed by the laws of\nNew York.", "made up clause"]
