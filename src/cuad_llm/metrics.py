"""Contract-level evaluation metrics.

Every metric is computed over (contract, category) pairs, the unit ContractEval (arXiv:2508.03080)
reports on: 102 test contracts x 41 categories = 4,182 pairs.

- coverage_*  ContractEval's headline F1/F2. TP: gold non-empty and the prediction fully covers every
              gold span. FN: gold non-empty and the prediction is empty or doesn't fully cover it.
              FP: gold empty but prediction non-empty. (Partial coverage is an FN only, as in the paper.)
- jaccard     ContractEval's output-effectiveness score: token-set Jaccard on gold-positive pairs.
- detection_* Did the model say the clause is present at all (binary P/R/F1)?
- token_f1    SQuAD-style bag-of-tokens F1 on gold-positive pairs.
- laziness    Share of gold-positive pairs where the model said "not present".
- verbatim    Share of predicted quotes that appear verbatim (modulo whitespace/case) in the contract.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from cuad_llm.spans import normalize

_TOKEN_RE = re.compile(r"\w+")


@dataclass
class PairPrediction:
    contract_id: str
    category: str
    gold: list[str]
    pred: list[str]


def _tokens(texts: list[str]) -> list[str]:
    return _TOKEN_RE.findall(" ".join(texts).lower())


def fully_covers(pred: list[str], gold: list[str]) -> bool:
    """Every gold span is contained in some predicted span (whitespace/case-insensitive)."""
    norm_pred = [normalize(p) for p in pred]
    return all(any(normalize(g) in p for p in norm_pred) for g in gold)


def jaccard(pred: list[str], gold: list[str]) -> float:
    a, b = set(_tokens(pred)), set(_tokens(gold))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def token_f1(pred: list[str], gold: list[str]) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def _prf(tp: int, fp: int, fn: int, beta: float = 1.0) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    f = (1 + b2) * precision * recall / denom if denom else 0.0
    return precision, recall, f


def _pair_counts(p: PairPrediction) -> dict[str, int]:
    """Per-pair counts, summed across pairs to get corpus-level metrics."""
    gold_pos, pred_pos = bool(p.gold), bool(p.pred)
    covered = gold_pos and pred_pos and fully_covers(p.pred, p.gold)
    return {
        "cov_tp": int(covered),
        "cov_fp": int(not gold_pos and pred_pos),
        "cov_fn": int(gold_pos and not covered),
        "det_tp": int(gold_pos and pred_pos),
        "det_fp": int(not gold_pos and pred_pos),
        "det_fn": int(gold_pos and not pred_pos),
    }


def _summarise(counts: np.ndarray, keys: list[str]) -> dict[str, float]:
    c = dict(zip(keys, counts.sum(axis=0)))
    cov_p, cov_r, cov_f1 = _prf(c["cov_tp"], c["cov_fp"], c["cov_fn"])
    _, _, cov_f2 = _prf(c["cov_tp"], c["cov_fp"], c["cov_fn"], beta=2.0)
    det_p, det_r, det_f1 = _prf(c["det_tp"], c["det_fp"], c["det_fn"])
    return {
        "coverage_precision": cov_p,
        "coverage_recall": cov_r,
        "coverage_f1": cov_f1,
        "coverage_f2": cov_f2,
        "detection_precision": det_p,
        "detection_recall": det_r,
        "detection_f1": det_f1,
    }


def evaluate(
    predictions: list[PairPrediction],
    contracts: dict[str, str],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute all metrics. `contracts` maps contract_id -> full text (for the verbatim rate).

    95% confidence intervals come from a bootstrap over contracts, not pairs, since the 41 pairs from
    one contract are correlated.
    """
    keys = list(_pair_counts(predictions[0]).keys())
    counts = np.array([[_pair_counts(p)[k] for k in keys] for p in predictions])
    results = _summarise(counts, keys)

    positives = [p for p in predictions if p.gold]
    results["jaccard"] = float(np.mean([jaccard(p.pred, p.gold) for p in positives]))
    results["token_f1"] = float(np.mean([token_f1(p.pred, p.gold) for p in positives]))
    results["laziness"] = float(np.mean([not p.pred for p in positives]))

    quotes = [(q, p.contract_id) for p in predictions for q in p.pred]
    if quotes:
        norm_contracts = {cid: normalize(t) for cid, t in contracts.items()}
        results["verbatim"] = float(
            np.mean([normalize(q) in norm_contracts[cid] for q, cid in quotes])
        )
    else:
        results["verbatim"] = float("nan")

    results["n_pairs"] = len(predictions)
    results["n_contracts"] = len({p.contract_id for p in predictions})

    if n_bootstrap:
        cids = sorted({p.contract_id for p in predictions})
        by_contract = {cid: [] for cid in cids}
        for i, p in enumerate(predictions):
            by_contract[p.contract_id].append(i)
        per_contract = np.array([counts[by_contract[cid]].sum(axis=0) for cid in cids])
        rng = np.random.default_rng(seed)
        samples = {"coverage_f1": [], "detection_f1": []}
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(cids), len(cids))
            s = _summarise(per_contract[idx], keys)
            for k in samples:
                samples[k].append(s[k])
        for k, vals in samples.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            results[f"{k}_ci_low"] = float(lo)
            results[f"{k}_ci_high"] = float(hi)

    return results
