"""Score one or more prediction runs and print them next to published ContractEval numbers.

Writes metrics.json into each run directory.

Usage: uv run python scripts/evaluate.py outputs/<run_a> [outputs/<run_b> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuad_llm.data import load_contracts, read_jsonl
from cuad_llm.metrics import PairPrediction, evaluate

# Coverage F1 on the CUAD test split, from ContractEval (arXiv:2508.03080).
PUBLISHED = {"GPT-4.1 (ContractEval)": 0.641, "Claude Sonnet 4 (ContractEval)": 0.523}

COLUMNS = [
    "coverage_f1",
    "coverage_precision",
    "coverage_recall",
    "detection_f1",
    "detection_precision",
    "jaccard",
    "token_f1",
    "laziness",
    "verbatim",
]
# Short headers so the table fits a notebook cell.
HEADERS = {
    "coverage_f1": "cov_f1",
    "coverage_precision": "cov_p",
    "coverage_recall": "cov_r",
    "detection_f1": "det_f1",
    "detection_precision": "det_p",
    "token_f1": "tok_f1",
}


def score_run(run_dir: Path, processed_dir: Path) -> dict:
    stats = json.loads((run_dir / "run_stats.json").read_text())
    predictions = [PairPrediction(**row) for row in read_jsonl(run_dir / "predictions.jsonl")]
    contracts = load_contracts(processed_dir / f"contracts_{stats['split']}.jsonl")
    texts = {c.contract_id: c.text for c in contracts}
    metrics = evaluate(predictions, texts)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    header = (
        f"{'run':<32}" + "".join(f"{HEADERS.get(c, c):>10}" for c in COLUMNS) + "   cov_f1 95% CI"
    )
    print(header)
    print("-" * len(header))
    for run_dir in args.runs:
        m = score_run(run_dir, args.processed_dir)
        row = f"{run_dir.name:<32}" + "".join(f"{m[c]:>10.3f}" for c in COLUMNS)
        print(row + f"   [{m['coverage_f1_ci_low']:.3f}, {m['coverage_f1_ci_high']:.3f}]")
    for name, f1 in PUBLISHED.items():
        print(f"{name:<32}{f1:>10.3f}")


if __name__ == "__main__":
    main()
