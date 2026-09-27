"""Paired bootstrap comparison of one run against others on the same test contracts.

Usage: uv run python scripts/compare_runs.py outputs/<run> outputs/<baseline> [outputs/<baseline> ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cuad_llm.data import read_jsonl
from cuad_llm.metrics import PairPrediction, paired_bootstrap


def load(run: Path) -> list[PairPrediction]:
    return [PairPrediction(**r) for r in read_jsonl(run / "predictions.jsonl")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("baselines", nargs="+", type=Path)
    parser.add_argument("--metric", default="coverage_f1")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    a = load(args.run)
    print(f"{args.run.name} minus baseline, {args.metric}, paired bootstrap over contracts:")
    for base in args.baselines:
        r = paired_bootstrap(a, load(base), args.metric, args.n_bootstrap)
        print(
            f"  vs {base.name:<36} {r['diff']:+.3f}  95% CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            f"  P(no improvement) = {r['p_le_zero']:.4f}"
        )


if __name__ == "__main__":
    main()
