import math

import pytest

from cuad_llm.metrics import PairPrediction, evaluate, fully_covers, jaccard, token_f1

CONTRACTS = {
    "c1": "Governed by New York law. Payment due in 30 days. Either party may terminate.",
    "c2": "Governed by Delaware law. No assignment without consent.",
}


def P(cid, cat, gold, pred):
    return PairPrediction(cid, cat, gold, pred)


def test_fully_covers():
    assert fully_covers(["Governed by New York law. Payment"], ["new york law"])
    assert not fully_covers(["New York"], ["Governed by New York law"])
    assert not fully_covers(["Governed by New York law"], ["New York law", "Payment due"])


def test_overlap_scores():
    assert jaccard(["a b"], ["b c"]) == pytest.approx(1 / 3)
    assert token_f1(["a b"], ["b c"]) == pytest.approx(0.5)
    assert token_f1([], ["a"]) == 0.0


def test_evaluate_counts_match_contracteval_definitions():
    preds = [
        # TP: gold fully covered.
        P("c1", "Governing Law", ["Governed by New York law"], ["Governed by New York law."]),
        # FN only (not FP): predicted something but didn't fully cover the gold span.
        P("c1", "Payment", ["Payment due in 30 days"], ["30 days"]),
        # FN: lazy "not present".
        P("c1", "Termination", ["Either party may terminate"], []),
        # FP: hallucinated clause where gold is empty; also not verbatim.
        P("c2", "Non-Compete", [], ["Seller shall not compete"]),
        # TN.
        P("c2", "Audit Rights", [], []),
    ]
    r = evaluate(preds, CONTRACTS, n_bootstrap=50)
    # Coverage: TP=1, FP=1, FN=2 -> P=0.5, R=1/3.
    assert r["coverage_precision"] == pytest.approx(0.5)
    assert r["coverage_recall"] == pytest.approx(1 / 3)
    assert r["coverage_f1"] == pytest.approx(0.4)
    # Detection: TP=2, FP=1, FN=1.
    assert r["detection_f1"] == pytest.approx(2 / 3)
    assert r["laziness"] == pytest.approx(1 / 3)
    # 3 predicted quotes, 2 of them verbatim.
    assert r["verbatim"] == pytest.approx(2 / 3)
    assert r["n_pairs"] == 5 and r["n_contracts"] == 2
    assert r["coverage_f1_ci_low"] <= r["coverage_f1"] <= r["coverage_f1_ci_high"]


def test_verbatim_is_nan_without_predictions():
    r = evaluate([P("c1", "X", ["Governed by"], [])], CONTRACTS, n_bootstrap=0)
    assert math.isnan(r["verbatim"])


def test_paired_bootstrap_detects_a_consistent_improvement():
    from cuad_llm.metrics import paired_bootstrap

    better, worse = [], []
    for i in range(30):
        cid = f"c{i}"
        gold = ["Governed by New York law"]
        better.append(P(cid, "Governing Law", gold, gold))
        # The weaker run misses the clause in every third contract.
        worse.append(P(cid, "Governing Law", gold, [] if i % 3 == 0 else gold))
    r = paired_bootstrap(better, worse, n_bootstrap=500)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_le_zero"] == 0.0
    same = paired_bootstrap(better, better, n_bootstrap=200)
    assert same["diff"] == 0.0 and same["p_le_zero"] == 1.0
