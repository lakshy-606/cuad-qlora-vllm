"""Load the CUAD v1 SQuAD-format release into one record per contract."""

from __future__ import annotations

import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

CUAD_ZIP_URL = "https://github.com/The-Atticus-Project/cuad/raw/main/data.zip"
TRAIN_FILE = "train_separate_questions.json"
TEST_FILE = "test.json"

_QUESTION_RE = re.compile(
    r'related to "(?P<category>.+?)" that should be reviewed by a lawyer\.'
    r"(?: Details: (?P<description>.*))?$",
    re.DOTALL,
)


@dataclass(frozen=True)
class Span:
    text: str
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


@dataclass
class Contract:
    contract_id: str
    text: str
    # Every category is a key; an empty list means the clause is absent from the contract.
    labels: dict[str, list[Span]]

    def to_json(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "text": self.text,
            "labels": {
                cat: [{"text": s.text, "start": s.start} for s in spans]
                for cat, spans in self.labels.items()
            },
        }

    @classmethod
    def from_json(cls, d: dict) -> Contract:
        return cls(
            contract_id=d["contract_id"],
            text=d["text"],
            labels={
                cat: [Span(s["text"], s["start"]) for s in spans]
                for cat, spans in d["labels"].items()
            },
        )


def download_cuad(raw_dir: Path) -> Path:
    """Download and extract the CUAD SQuAD-format files if they aren't already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / TRAIN_FILE).exists() and (raw_dir / TEST_FILE).exists():
        return raw_dir
    zip_path = raw_dir / "data.zip"
    if not zip_path.exists():
        print(f"Downloading {CUAD_ZIP_URL}")
        urllib.request.urlretrieve(CUAD_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return raw_dir


def parse_question(question: str) -> tuple[str, str]:
    m = _QUESTION_RE.search(question)
    if not m:
        raise ValueError(f"Unrecognised CUAD question format: {question[:120]!r}")
    return m["category"], (m["description"] or "").strip()


def load_squad_file(path: Path) -> tuple[list[Contract], dict[str, str]]:
    """Parse a CUAD SQuAD-format file.

    Returns the contracts and a mapping of category name -> category description.
    """
    raw = json.loads(Path(path).read_text())["data"]
    contracts: list[Contract] = []
    categories: dict[str, str] = {}
    for doc in raw:
        # CUAD stores each contract as a single paragraph.
        (para,) = doc["paragraphs"]
        text = para["context"]
        # The train file ("separate questions") repeats a category once per answer, one answer each;
        # the test file has one question per category with all answers. Accumulate across both.
        spans_by_cat: dict[str, dict[tuple[int, str], Span]] = {}
        for qa in para["qas"]:
            category, description = parse_question(qa["question"])
            categories.setdefault(category, description)
            spans = spans_by_cat.setdefault(category, {})
            for ans in qa["answers"]:
                span = Span(ans["text"], ans["answer_start"])
                if text[span.start : span.end] != span.text:
                    raise ValueError(f"Offset mismatch in {qa['id']}")
                spans[(span.start, span.text)] = span
        labels = {
            cat: sorted(spans.values(), key=lambda s: s.start)
            for cat, spans in spans_by_cat.items()
        }
        contracts.append(Contract(doc["title"], text, labels))
    return contracts, categories


def split_train_val(
    contracts: list[Contract], val_size: int, seed: int
) -> tuple[list[Contract], list[Contract]]:
    ids = sorted(c.contract_id for c in contracts)
    val_ids = set(random.Random(seed).sample(ids, val_size))
    train = [c for c in contracts if c.contract_id not in val_ids]
    val = [c for c in contracts if c.contract_id in val_ids]
    return train, val


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_contracts(path: Path) -> list[Contract]:
    return [Contract.from_json(d) for d in read_jsonl(path)]
