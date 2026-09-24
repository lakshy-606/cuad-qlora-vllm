"""Download CUAD, split it, and build chunk-level SFT data.

Outputs (in processed_dir):
  categories.json                   category -> description (41 entries)
  contracts_{train,val,test}.jsonl  one contract per line with gold spans for every category
  sft_{train,val}.jsonl             chat-format examples, one per (chunk, category)

Usage: uv run python scripts/prepare_data.py [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from cuad_llm.chunking import chunk_text, spans_in_chunk
from cuad_llm.data import (
    TEST_FILE,
    TRAIN_FILE,
    Contract,
    download_cuad,
    load_squad_file,
    split_train_val,
    write_jsonl,
)
from cuad_llm.prompts import build_messages

CHARS_PER_TOKEN = 4  # rough estimate for English legal text; only used for the size report


def build_sft_examples(
    contracts: list[Contract],
    categories: dict[str, str],
    chunk_chars: int,
    overlap_chars: int,
    negatives_per_positive: float,
    rng: random.Random,
) -> tuple[list[dict], dict[str, int]]:
    positives, negatives = [], []
    for contract in contracts:
        chunks = chunk_text(contract.text, chunk_chars, overlap_chars)
        for i, chunk in enumerate(chunks):
            for category, description in categories.items():
                quotes = spans_in_chunk(contract.labels[category], chunk, chunks)
                example = {
                    "contract_id": contract.contract_id,
                    "chunk_index": i,
                    "category": category,
                    "messages": build_messages(category, description, chunk.text, quotes),
                }
                (positives if quotes else negatives).append(example)

    n_neg = min(len(negatives), round(len(positives) * negatives_per_positive))
    examples = positives + rng.sample(negatives, n_neg)
    rng.shuffle(examples)
    stats = {
        "positive_chunk_pairs": len(positives),
        "negative_chunk_pairs_available": len(negatives),
        "negative_chunk_pairs_kept": n_neg,
        "examples": len(examples),
    }
    return examples, stats


def pair_stats(contracts: list[Contract]) -> str:
    pairs = [bool(spans) for c in contracts for spans in c.labels.values()]
    return f"{len(contracts)} contracts, {len(pairs)} pairs, {sum(pairs) / len(pairs):.1%} positive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    raw_dir = download_cuad(Path(cfg["raw_dir"]))
    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    train_all, categories = load_squad_file(raw_dir / TRAIN_FILE)
    test, test_categories = load_squad_file(raw_dir / TEST_FILE)
    assert set(categories) == set(test_categories), "train/test category mismatch"
    assert len(categories) == 41, f"expected 41 CUAD categories, got {len(categories)}"
    train, val = split_train_val(train_all, cfg["val_contracts"], cfg["seed"])

    (out / "categories.json").write_text(json.dumps(categories, indent=2))
    for name, split in [("train", train), ("val", val), ("test", test)]:
        write_jsonl(out / f"contracts_{name}.jsonl", (c.to_json() for c in split))
        print(f"{name:>5}: {pair_stats(split)}")

    rng = random.Random(cfg["seed"])
    for name, split in [("train", train), ("val", val)]:
        examples, stats = build_sft_examples(
            split,
            categories,
            cfg["chunk_chars"],
            cfg["overlap_chars"],
            cfg["negatives_per_positive"],
            rng,
        )
        write_jsonl(out / f"sft_{name}.jsonl", examples)
        chars = sum(len(m["content"]) for ex in examples for m in ex["messages"])
        print(
            f"sft_{name}: {stats['examples']} examples "
            f"({stats['positive_chunk_pairs']} positive, {stats['negative_chunk_pairs_kept']} of "
            f"{stats['negative_chunk_pairs_available']} negatives kept), "
            f"~{chars / CHARS_PER_TOKEN / 1e6:.1f}M tokens per epoch"
        )


if __name__ == "__main__":
    main()
