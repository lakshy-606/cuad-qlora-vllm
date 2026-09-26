import json

from cuad_llm.data import load_squad_file

TEXT = "This Agreement is between Acme Corp. (Acme) and Beta LLC (Beta)."
QUESTION = 'Highlight the parts (if any) of this contract related to "{}" that should be reviewed by a lawyer. Details: {}'


def qa(qid, category, answers):
    return {
        "id": qid,
        "question": QUESTION.format(category, "desc"),
        "answers": [{"text": t, "answer_start": TEXT.index(t)} for t in answers],
    }


def write(tmp_path, qas):
    path = tmp_path / "cuad.json"
    doc = {"title": "c1", "paragraphs": [{"context": TEXT, "qas": qas}]}
    path.write_text(json.dumps({"data": [doc]}))
    return path


def test_separate_questions_for_one_category_are_all_kept(tmp_path):
    # CUAD's train file repeats a category once per answer; every answer must survive.
    path = write(
        tmp_path,
        [
            qa("q1", "Parties", ["Acme Corp."]),
            qa("q2", "Parties", ["Beta LLC"]),
            qa("q3", "Parties", ["Acme"]),
            qa("q4", "Governing Law", []),
        ],
    )
    (contract,), categories = load_squad_file(path)
    assert [s.text for s in contract.labels["Parties"]] == ["Acme Corp.", "Acme", "Beta LLC"]
    assert contract.labels["Governing Law"] == []
    assert set(categories) == {"Parties", "Governing Law"}


def test_duplicate_answers_are_dropped(tmp_path):
    path = write(tmp_path, [qa("q1", "Parties", ["Acme Corp.", "Acme Corp."])])
    (contract,), _ = load_squad_file(path)
    assert [s.text for s in contract.labels["Parties"]] == ["Acme Corp."]
