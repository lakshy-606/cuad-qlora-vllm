"""Break down why a run loses coverage F1: false positives, and why found clauses aren't fully covered.

For every pair where the clause is present and the model predicted something but coverage failed,
the failure is classified by comparing character ranges in the contract:
  missed_span   one of several gold spans has no overlap with any prediction
  split         a gold span is covered by the union of predicted quotes, but by no single quote
  short         predictions overlap a gold span but leave part of it uncovered (boundary too tight)
  unlocated     a prediction couldn't be found in the contract (paraphrase or invention)

Usage: uv run python scripts/analyze_errors.py outputs/<run> [outputs/<run> ...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from cuad_llm.data import load_contracts, read_jsonl
from cuad_llm.metrics import fully_covers
from cuad_llm.spans import locate


def covered_fraction(start: int, end: int, ranges: list[tuple[int, int]]) -> float:
    covered = [False] * (end - start)
    for s, e in ranges:
        for i in range(max(s, start), min(e, end)):
            covered[i - start] = True
    return sum(covered) / max(end - start, 1)


def classify(gold_spans, preds: list[str], text: str) -> str:
    ranges = [r for r in (locate(p, text) for p in preds) if r]
    if len(ranges) < len(preds) and not ranges:
        return "unlocated"
    reasons = []
    for g in gold_spans:
        if any(s <= g.start and g.end <= e for s, e in ranges):
            continue  # this gold span is fully inside one quote
        frac = covered_fraction(g.start, g.end, ranges)
        if frac == 0:
            reasons.append("missed_span")
        elif frac >= 0.999:
            reasons.append("split")
        else:
            reasons.append("short")
    # Report the most informative reason when a pair has several failing gold spans.
    for r in ("short", "split", "missed_span"):
        if r in reasons:
            return r
    return "unlocated"


def analyze(run_dir: Path, processed: Path) -> None:
    stats = json.loads((run_dir / "run_stats.json").read_text())
    contracts = {
        c.contract_id: c for c in load_contracts(processed / f"contracts_{stats['split']}.jsonl")
    }
    rows = list(read_jsonl(run_dir / "predictions.jsonl"))

    kinds: Counter[str] = Counter()
    short_fracs: list[float] = []
    per_cat: dict[str, Counter[str]] = defaultdict(Counter)
    pred_len, gold_len = [], []
    for r in rows:
        c = contracts[r["contract_id"]]
        gold_spans = c.labels[r["category"]]
        cat = per_cat[r["category"]]
        if not gold_spans:
            cat["fp" if r["pred"] else "tn"] += 1
            continue
        if not r["pred"]:
            cat["fn_lazy"] += 1
            continue
        pred_len.append(sum(len(p) for p in r["pred"]))
        gold_len.append(sum(len(s.text) for s in gold_spans))
        if fully_covers(r["pred"], r["gold"]):
            cat["tp"] += 1
            continue
        kind = classify(gold_spans, r["pred"], c.text)
        kinds[kind] += 1
        cat["fn_partial"] += 1
        if kind == "short":
            ranges = [x for x in (locate(p, c.text) for p in r["pred"]) if x]
            short_fracs.append(
                min(
                    covered_fraction(g.start, g.end, ranges)
                    for g in gold_spans
                    if covered_fraction(g.start, g.end, ranges) > 0
                )
            )

    print(f"\n=== {run_dir.name} ===")
    total_partial = sum(kinds.values())
    print(f"Found but not fully covered: {total_partial}")
    for k, n in kinds.most_common():
        print(f"  {k:<12} {n:5d}  ({n / total_partial:.0%})")
    if short_fracs:
        short_fracs.sort()
        mid = short_fracs[len(short_fracs) // 2]
        near = sum(f >= 0.8 for f in short_fracs)
        print(f"  'short' pairs: median {mid:.0%} of the gold span covered; {near} cover >= 80%")
    pl, gl = sorted(pred_len), sorted(gold_len)
    print(
        f"Found clauses: median predicted chars {pl[len(pl) // 2]}, median gold chars {gl[len(gl) // 2]}"
    )

    print("Coverage F1 by category (worst 12):")
    scored = []
    for cat, n in per_cat.items():
        tp, fp, fn = n["tp"], n["fp"], n["fn_lazy"] + n["fn_partial"]
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        scored.append((f1, cat, tp, fp, n["fn_lazy"], n["fn_partial"]))
    for f1, cat, tp, fp, lazy, partial in sorted(scored)[:12]:
        print(
            f"  {f1:.2f}  {cat:<38} tp {tp:3d}  fp {fp:3d}  missed {lazy:3d}  partial {partial:3d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    for run_dir in args.runs:
        analyze(run_dir, args.processed_dir)


if __name__ == "__main__":
    main()
