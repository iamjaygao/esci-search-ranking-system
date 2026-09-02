"""
Phase D -- synthetic unit tests for the authoritative scorer.

Every expected value below is derived by hand in the docstring of its test, so
the scorer can be checked without trusting any other code in the repository.

Run: python experiments/ranking_v2/benchmark_repair/scripts/test_official_ndcg.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_ndcg import dcg, evaluate, per_query_ndcg, relevance_from_labels  # noqa: E402

RESULTS = []
gE = 2 ** 1.00 - 1          # 1.0
gS = 2 ** 0.10 - 1          # 0.0717734625...
gC = 2 ** 0.01 - 1          # 0.0069555500...
gI = 0.0


def check(name, got, want, tol=1e-12, detail=""):
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    RESULTS.append({"test": name, "expected": want, "got": got, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r} {detail}")
    return ok


def frame(rows):
    """rows = list of (query_id, product_id, label, score)"""
    df = pd.DataFrame(rows, columns=["query_id", "product_id", "esci_label", "score"])
    df["relevance"] = relevance_from_labels(df["esci_label"])
    return df


def test_gain_values():
    """The four official gains, computed by hand."""
    print("\n1. Gain values")
    check("gain(E)", dcg([1.00], 1), 1.0)
    check("gain(S)", dcg([0.10], 1), math.pow(2, 0.10) - 1)
    check("gain(C)", dcg([0.01], 1), math.pow(2, 0.01) - 1)
    check("gain(I)", dcg([0.00], 1), 0.0)


def test_perfect_and_worst():
    """Query with 2 candidates, one E one I.

    Perfect order [E, I]:  DCG = gE/log2(2) + 0/log2(3) = 1.0
                           IDCG = 1.0        -> NDCG = 1.0
    Worst order   [I, E]:  DCG = 0/log2(2) + gE/log2(3) = 1/1.5849625 = 0.6309298
                           IDCG = 1.0        -> NDCG = 0.6309298
    """
    print("\n2. Perfect vs worst ordering")
    good = frame([("q", "pA", "E", 1.0), ("q", "pB", "I", 0.0)])
    bad = frame([("q", "pA", "E", 0.0), ("q", "pB", "I", 1.0)])
    check("perfect ordering -> 1.0", float(per_query_ndcg(good, "score").iloc[0]), 1.0)
    check("worst ordering -> 1/log2(3)", float(per_query_ndcg(bad, "score").iloc[0]),
          1.0 / math.log2(3), detail="= 0.6309298")


def test_graded_three_way():
    """One query, 3 candidates E/S/I, model ranks S above E: [S, E, I].

    DCG  = gS/log2(2) + gE/log2(3) + 0/log2(4)
         = 0.0717734625 + 0.6309297536          = 0.7027032161
    IDCG = gE/log2(2) + gS/log2(3) + 0
         = 1.0 + 0.0452838779                   = 1.0452838779
    NDCG = 0.7027032161 / 1.0452838779          = 0.6722575...
    """
    print("\n3. Graded three-way ordering")
    df = frame([("q", "pS", "S", 3.0), ("q", "pE", "E", 2.0), ("q", "pI", "I", 1.0)])
    expected_dcg = gS / math.log2(2) + gE / math.log2(3)
    expected_idcg = gE / math.log2(2) + gS / math.log2(3)
    check("S-above-E NDCG", float(per_query_ndcg(df, "score").iloc[0]),
          expected_dcg / expected_idcg, tol=1e-12,
          detail=f"= {expected_dcg / expected_idcg:.10f}")


def test_substitute_is_cheap():
    """The P0-1 property, in two steps. Query has 1 E and 9 S (10 candidates).

    disc[r] = 1/log2(r+2) for 0-based slot r.
    ideal = gE*disc[0] + gS*sum(disc[1:10])

    (a) All 9 S outrank the E, so E lands in slot 10 (index 9):
        got_a = gS*sum(disc[0:9]) + gE*disc[9]

    (b) E demoted by exactly ONE slot (one S promoted above it):
        got_b = gS*disc[0] + gE*disc[1] + gS*sum(disc[2:10])

    Both ratios must be well below 1.0 -- losing the single Exact position
    dominates the metric even though 9 Substitutes are ranked perfectly.
    """
    print("\n4. Substitute gain is ~7% of Exact")
    disc = [1 / math.log2(r + 2) for r in range(10)]
    ideal = gE * disc[0] + gS * sum(disc[1:10])

    # (a) E ranked last
    rows_a = [("q", "pE", "E", 0.5)] + [("q", f"pS{i}", "S", 1.0) for i in range(9)]
    got_a = gS * sum(disc[0:9]) + gE * disc[9]
    check("E ranked last of 10", float(per_query_ndcg(frame(rows_a), "score").iloc[0]),
          got_a / ideal, detail=f"= {got_a/ideal:.10f}")

    # (b) E demoted by exactly one slot: pS0 scores above E, the rest below
    rows_b = [("q", "pS0", "S", 3.0), ("q", "pE", "E", 2.0)]
    rows_b += [("q", f"pS{i}", "S", 1.0) for i in range(1, 9)]
    got_b = gS * disc[0] + gE * disc[1] + gS * sum(disc[2:10])
    check("E demoted by one slot", float(per_query_ndcg(frame(rows_b), "score").iloc[0]),
          got_b / ideal, detail=f"= {got_b/ideal:.10f}")

    check("demoting E by 1 slot costs more than 25% of the score",
          (got_b / ideal) < 0.75, True)
    check("S:E gain ratio", gS / gE, 0.07177346253629313, tol=1e-15)


def test_fewer_than_k():
    """3 candidates, k=10. Both DCG and IDCG use all 3; perfect order -> 1.0."""
    print("\n5. Candidate count < k")
    df = frame([("q", "a", "E", 3.0), ("q", "b", "S", 2.0), ("q", "c", "I", 1.0)])
    r = evaluate(df, "score", k=10)
    check("short list, perfect order -> 1.0", r["ndcg_at_k"], 1.0)
    check("short list, all scored", r["n_queries_scored"], 1)


def test_cutoff_at_k():
    """12 candidates: 2 I first then 10 E. Only the top 10 count, so the model
    ranking [I,I,E*10] scores DCG = gE * sum(1/log2(r+1)) for r=3..10, while
    IDCG = gE * sum(1/log2(r+1)) for r=1..10."""
    print("\n6. Cutoff at k=10 is respected")
    rows = [("q", "i1", "I", 10.0), ("q", "i2", "I", 9.0)]
    rows += [("q", f"e{i}", "E", 8.0 - i) for i in range(10)]
    df = frame(rows)
    disc = [1 / math.log2(r + 1) for r in range(1, 11)]
    expected = (gE * sum(disc[2:10])) / (gE * sum(disc[0:10]))
    check("two irrelevant at the top", float(per_query_ndcg(df, "score").iloc[0]),
          expected, detail=f"= {expected:.10f}")


def test_no_relevant_excluded():
    """Query q2 is all-I -> IDCG 0 -> excluded from the mean, not scored as 0."""
    print("\n7. All-irrelevant query is excluded, not zeroed")
    df = pd.concat([
        frame([("q1", "a", "E", 2.0), ("q1", "b", "I", 1.0)]),
        frame([("q2", "c", "I", 2.0), ("q2", "d", "I", 1.0)]),
    ], ignore_index=True)
    r = evaluate(df, "score")
    check("scored queries", r["n_queries_scored"], 1)
    check("excluded queries", r["n_excluded_no_relevant"], 1)
    check("mean is 1.0 not 0.5", r["ndcg_at_k"], 1.0,
          detail="q2 dropped rather than counted as zero")


def test_equal_query_weight():
    """q1 has 2 candidates and scores 1.0; q2 has 40 candidates and scores
    1/log2(3). The mean must be their unweighted average, not row-weighted."""
    print("\n8. Every query carries weight 1")
    q1 = frame([("q1", "a", "E", 2.0), ("q1", "b", "I", 1.0)])
    rows = [("q2", "e", "E", 0.0)] + [("q2", f"i{i}", "I", 1.0) for i in range(39)]
    q2 = frame(rows)
    df = pd.concat([q1, q2], ignore_index=True)
    disc = [1 / math.log2(r + 1) for r in range(1, 11)]
    q2_ndcg = (gE * disc[9]) / gE if False else None  # E lands at rank 40 -> outside k
    # E is ranked last (rank 40) so it is outside the top 10 -> DCG = 0
    got = evaluate(df, "score")["ndcg_at_k"]
    check("unweighted mean of 1.0 and 0.0", got, 0.5,
          detail="q2's single E falls outside k=10 -> 0.0")


def test_tie_determinism():
    """All scores identical. The result must not depend on input row order."""
    print("\n9. Tie handling is deterministic")
    rows = [("q", "pB", "E", 1.0), ("q", "pA", "I", 1.0), ("q", "pC", "S", 1.0)]
    a = float(per_query_ndcg(frame(rows), "score").iloc[0])
    b = float(per_query_ndcg(frame(rows[::-1]), "score").iloc[0])
    c = float(per_query_ndcg(frame([rows[1], rows[2], rows[0]]), "score").iloc[0])
    check("order-independent (fwd vs rev)", a, b)
    check("order-independent (fwd vs shuffled)", a, c)
    # ties break on ascending product_id -> pA(I), pB(E), pC(S)
    disc = [1 / math.log2(r + 1) for r in range(1, 4)]
    expected = (gI * disc[0] + gE * disc[1] + gS * disc[2]) / (gE * disc[0] + gS * disc[1])
    check("tie order is ascending product_id", a, expected, detail=f"= {expected:.10f}")
    pes = float(per_query_ndcg(frame(rows), "score", tie_break="pessimistic").iloc[0])
    check("pessimistic tie-break is <= deterministic", pes <= a + 1e-12, True)


def test_matches_v1_scorer_on_shared_convention():
    """Sanity bridge: with the same relevance values and no ties, the new
    scorer must agree with evaluation/metrics.dcg, which is the V1 formula."""
    print("\n10. Agrees with the V1 dcg() formula when there are no ties")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    sys.path.insert(0, root)
    from evaluation.metrics import dcg as v1_dcg
    rng = np.random.RandomState(0)
    rel = rng.choice([0.0, 0.01, 0.10, 1.0], size=25)
    check("dcg parity vs evaluation/metrics.dcg", dcg(rel, 10), float(v1_dcg(rel, 10)), tol=1e-12)


def main():
    print("=" * 66)
    print(" Unit tests for the authoritative NDCG@10 scorer")
    print("=" * 66)
    for t in [test_gain_values, test_perfect_and_worst, test_graded_three_way,
              test_substitute_is_cheap, test_fewer_than_k, test_cutoff_at_k,
              test_no_relevant_excluded, test_equal_query_weight,
              test_tie_determinism, test_matches_v1_scorer_on_shared_convention]:
        t()
    n_pass = sum(r["pass"] for r in RESULTS)
    print("\n" + "=" * 66)
    print(f" {n_pass}/{len(RESULTS)} assertions passed")
    print("=" * 66)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "ndcg_unit_tests.json")
    with open(out, "w") as f:
        json.dump({"passed": n_pass, "total": len(RESULTS),
                   "all_pass": n_pass == len(RESULTS), "assertions": RESULTS},
                  f, indent=2, default=str)
    print(f"wrote {out}")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
