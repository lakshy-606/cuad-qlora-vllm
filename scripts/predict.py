"""Run a model over a CUAD split and write contract-level predictions.

Two modes:
  full  every (chunk, category) pair goes to the model; per-chunk quotes are merged per contract
  rag   hybrid BM25 + dense retrieval picks top-k passages per (contract, category); one request each

Outputs (in outputs/<run_name>/):
  responses.jsonl    raw response cache; rerunning resumes from it
  predictions.jsonl  one row per (contract, category) with gold and predicted quotes
  run_stats.json     request counts, token usage, parse failures, retrieval recall (rag)

Usage:
  uv run python scripts/predict.py --config configs/zeroshot_gpt4omini.yaml [--limit 3] [--dry-run]
  uv run python scripts/predict.py --config configs/finetuned_qwen3_4b.yaml \
      --split val --model checkpoint-400 --run-name ft_val_checkpoint-400
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

from cuad_llm.chunking import chunk_text
from cuad_llm.data import Contract, load_contracts, write_jsonl
from cuad_llm.inference import ModelConfig, run_chat
from cuad_llm.pipeline import (
    Request,
    assemble_predictions,
    build_excerpt,
    full_context_requests,
    spans_retrieved,
)
from cuad_llm.prompts import build_messages

CHARS_PER_TOKEN = 4  # rough estimate, only for the dry-run size report


def rag_requests(
    contracts: list[Contract], categories: dict[str, str], rag: dict
) -> tuple[list[Request], dict[str, float]]:
    # Imported here so full-context runs don't need the `rag` dependency group.
    from sentence_transformers import SentenceTransformer

    from cuad_llm.retrieval import HybridRetriever, merge_ranges

    # CPU by default: on a shared GPU, vLLM has already claimed nearly all the memory.
    embedder = SentenceTransformer(rag["embed_model"], device=rag.get("embed_device", "cpu"))
    queries = {cat: f"{cat}: {desc}" for cat, desc in categories.items()}
    query_vecs = embedder.encode([rag.get("query_prefix", "") + q for q in queries.values()])

    requests, hits, positives = [], 0, 0
    for n, contract in enumerate(contracts, 1):
        passages = chunk_text(contract.text, rag["passage_chars"], rag["passage_overlap"])
        vecs = embedder.encode([p.text for p in passages], batch_size=64)
        retriever = HybridRetriever([p.text for p in passages], vecs)
        for (category, query), qvec in zip(queries.items(), query_vecs):
            top = retriever.search(query, qvec, rag["top_k"])
            ranges = merge_ranges([passages[i] for i in top])
            excerpt = build_excerpt(contract.text, ranges)
            messages = build_messages(category, categories[category], excerpt)
            requests.append(Request(contract.contract_id, category, messages))
            if contract.labels[category]:
                positives += 1
                hits += spans_retrieved(contract.labels[category], ranges)
        if n % 10 == 0 or n == len(contracts):
            print(f"  retrieved for {n}/{len(contracts)} contracts")
    return requests, {"retrieval_recall": hits / positives if positives else float("nan")}


def main() -> None:
    load_dotenv()  # API keys from a git-ignored .env file, e.g. OPENAI_API_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="only the first N contracts (smoke test)")
    parser.add_argument("--split", help="override the config's split (e.g. val)")
    parser.add_argument("--model", help="override the model name (e.g. a vLLM LoRA adapter name)")
    parser.add_argument("--run-name", help="override the output run name")
    parser.add_argument(
        "--dry-run", action="store_true", help="build requests, don't call the model"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.split:
        cfg["split"] = args.split
    if args.model:
        cfg["model"]["model"] = args.model
    if args.run_name:
        cfg["run_name"] = args.run_name

    processed = Path(cfg.get("processed_dir", "data/processed"))
    categories = json.loads((processed / "categories.json").read_text())
    contracts = load_contracts(processed / f"contracts_{cfg['split']}.jsonl")
    limit = args.limit or cfg.get("limit_contracts")
    if limit:
        contracts = contracts[:limit]

    stats: dict = {"run_name": cfg["run_name"], "split": cfg["split"], "mode": cfg["mode"]}
    if cfg["mode"] == "full":
        full = cfg["full"]
        requests = full_context_requests(
            contracts, categories, full["chunk_chars"], full["overlap_chars"]
        )
    elif cfg["mode"] == "rag":
        requests, rag_stats = rag_requests(contracts, categories, cfg["rag"])
        stats.update(rag_stats)
    else:
        raise ValueError(f"unknown mode {cfg['mode']!r}")

    prompt_chars = sum(len(m["content"]) for r in requests for m in r.messages)
    print(
        f"{len(contracts)} contracts, {len(requests)} requests, "
        f"~{prompt_chars / CHARS_PER_TOKEN / 1e6:.1f}M prompt tokens"
    )
    if "retrieval_recall" in stats:
        print(f"retrieval recall (gold spans fully retrieved): {stats['retrieval_recall']:.1%}")
    if args.dry_run:
        return

    out = Path(cfg.get("output_dir", "outputs")) / cfg["run_name"]
    model_cfg = ModelConfig(**cfg["model"])
    t0 = time.perf_counter()
    completions = run_chat([r.messages for r in requests], model_cfg, out / "responses.jsonl")
    stats["wall_time_s"] = time.perf_counter() - t0

    predictions, parse_stats = assemble_predictions(
        contracts, list(categories), requests, [c.text for c in completions]
    )
    write_jsonl(out / "predictions.jsonl", (asdict(p) for p in predictions))

    stats.update(parse_stats)
    stats["model"] = asdict(model_cfg)
    stats["n_contracts"] = len(contracts)
    stats["prompt_tokens"] = sum(c.prompt_tokens for c in completions)
    stats["cached_prompt_tokens"] = sum(c.cached_tokens for c in completions)
    stats["completion_tokens"] = sum(c.completion_tokens for c in completions)
    stats["mean_latency_s"] = sum(c.latency_s for c in completions) / max(len(completions), 1)
    (out / "run_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"parse failures: {parse_stats['parse_failures']} / {parse_stats['requests']}")
    print(f"wrote {out / 'predictions.jsonl'}; score with scripts/evaluate.py {out}")


if __name__ == "__main__":
    main()
