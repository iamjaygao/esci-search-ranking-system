"""
Phase C -- hard integrity assertions on the frozen benchmark.

Fails loudly (non-zero exit) if any assertion fails. Writes
integrity_checks.json with one PASS/FAIL record per assertion.

Run: python experiments/ranking_v2/benchmark_repair/scripts/validate_pool.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from reranking.advanced_features import ALL_FEATURES  # noqa: E402
from official_ndcg import OFFICIAL_RELEVANCE  # noqa: E402

DATA = os.path.join(BASE, "data")
EXAMPLES = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")

CHECKS = []


def record(name, passed, expected=None, actual=None, note="", upstream=False):
    """`upstream=True` marks a defect inherited from the ESCI source data that
    this phase is forbidden to fix (constraint 3: do not change the split).
    Such a check is reported as FAIL_UPSTREAM: it is never silently repaired,
    but it does not gate the build, because the build did not cause it."""
    if passed:
        status = "PASS"
    else:
        status = "FAIL_UPSTREAM" if upstream else "FAIL"
    CHECKS.append({"check": name, "status": status,
                   "expected": expected, "actual": actual, "note": note})
    extra = f"  expected={expected!r} actual={actual!r}" if expected is not None else ""
    print(f"  [{status}] {name}{extra} {note}")
    return passed


def main():
    print("=" * 78)
    print(" Frozen benchmark integrity validation")
    print("=" * 78)

    ex = pd.read_parquet(EXAMPLES, columns=[
        "example_id", "query_id", "product_id", "product_locale", "esci_label", "split"])
    ex["query_id"] = ex["query_id"].astype(str)
    ex["product_id"] = ex["product_id"].astype(str)

    frames = {}
    for name in ["train", "dev", "test"]:
        frames[name] = pd.read_parquet(os.path.join(DATA, f"{name}_clean.parquet"))

    with open(os.path.join(DATA, "build_manifest.json")) as f:
        manifest = json.load(f)

    # ------------------------------------------------------------------
    # 1. Row-count conservation: no join multiplied anything
    # ------------------------------------------------------------------
    print("\n1. Row-count conservation (P0-2)")
    judged_train = int((ex["split"] == "train").sum())
    judged_test = int((ex["split"] == "test").sum())
    got_train_dev = len(frames["train"]) + len(frames["dev"])
    record("train+dev rows == ESCI train judged pairs",
           got_train_dev == judged_train, judged_train, got_train_dev)
    record("test rows == ESCI test judged pairs",
           len(frames["test"]) == judged_test, judged_test, len(frames["test"]))
    record("no split exceeds its judged-pair count",
           got_train_dev <= judged_train and len(frames["test"]) <= judged_test,
           True, True, note="join_output_rows > expected_rows would fail here")
    v1_test_rows = 682233
    record("test pool is smaller than the V1 inflated pool",
           len(frames["test"]) < v1_test_rows, f"< {v1_test_rows}", len(frames["test"]),
           note=f"V1 had {v1_test_rows - judged_test} duplicated rows "
                f"({100*(v1_test_rows-judged_test)/judged_test:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Key uniqueness
    # ------------------------------------------------------------------
    print("\n2. Key uniqueness")
    for name, df in frames.items():
        record(f"[{name}] example_id unique", df["example_id"].is_unique,
               True, bool(df["example_id"].is_unique))
        dup = int(df.duplicated(["query_id", "product_id", "product_locale"]).sum())
        record(f"[{name}] (query_id, product_id, product_locale) unique", dup == 0, 0, dup)
    all_ids = pd.concat([f["example_id"] for f in frames.values()])
    record("example_id unique ACROSS all three splits", all_ids.is_unique,
           True, bool(all_ids.is_unique))

    # ------------------------------------------------------------------
    # 3. Conflicting labels
    # ------------------------------------------------------------------
    print("\n3. Label consistency")
    for name, df in frames.items():
        g = df.groupby(["query_id", "product_id", "product_locale"])["esci_label"].nunique()
        n = int((g > 1).sum())
        record(f"[{name}] no conflicting labels for one logical example", n == 0, 0, n)
        # relevance must match the official mapping exactly
        expect = df["esci_label"].map(OFFICIAL_RELEVANCE).astype(float)
        bad = int((~np.isclose(df["relevance"].values, expect.values)).sum())
        record(f"[{name}] relevance column matches the official mapping", bad == 0, 0, bad)
        ordinal_ok = df.groupby("lgb_label")["esci_label"].nunique().eq(1).all()
        record(f"[{name}] lgb_label is a 1:1 encoding of esci_label", bool(ordinal_ok), True,
               bool(ordinal_ok))

    # ------------------------------------------------------------------
    # 4. Query-level split integrity
    # ------------------------------------------------------------------
    print("\n4. Query-level split integrity (P0-3 prerequisite)")
    qs = {n: set(f["query_id"].unique()) for n, f in frames.items()}
    # query_id 79706 ("piano") is assigned to BOTH train and test by ESCI itself
    # (3 train rows / 31 test rows, disjoint products). Constraint 3 forbids
    # changing the split, so it is inherited, quantified and flagged -- not dropped.
    UPSTREAM_CROSS_SPLIT_QIDS = {"79706"}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        ov = qs[a] & qs[b]
        is_upstream = bool(ov) and ov.issubset(UPSTREAM_CROSS_SPLIT_QIDS)
        record(f"{a}/{b} query overlap", len(ov) == 0, 0, len(ov),
               note="" if not ov else
                    f"query_ids {sorted(ov)} -- inherited from ESCI source, "
                    f"not introduced by this build",
               upstream=is_upstream)
    esci_train_q = set(ex[ex["split"] == "train"]["query_id"].unique())
    esci_test_q = set(ex[ex["split"] == "test"]["query_id"].unique())
    record("train+dev queries == ESCI train queries",
           (qs["train"] | qs["dev"]) == esci_train_q,
           len(esci_train_q), len(qs["train"] | qs["dev"]),
           note="the original split is preserved, only the internal 85/15 is now persisted")
    record("test queries == ESCI test queries", qs["test"] == esci_test_q,
           len(esci_test_q), len(qs["test"]))
    record("dev split matches the V1 trainers' internal split (84731/14953)",
           len(qs["train"]) == 84731 and len(qs["dev"]) == 14953,
           "84731/14953", f"{len(qs['train'])}/{len(qs['dev'])}")
    # the 1 known cross-split query_id from Phase 0 -- quantify its blast radius
    xq = qs["test"] & (qs["train"] | qs["dev"])
    record("cross-split query_ids beyond the known upstream one",
           xq.issubset(UPSTREAM_CROSS_SPLIT_QIDS), "subset of {'79706'}", sorted(xq),
           note="ESCI assigns 79706 to both splits at source. No example_id is "
                "duplicated (each row lives in exactly one split); the leak is a "
                "shared query STRING, affecting 1 of 30,969 test queries (0.003%).")
    if xq:
        n_tr = int(frames["train"]["query_id"].isin(xq).sum()
                   + frames["dev"]["query_id"].isin(xq).sum())
        n_te = int(frames["test"]["query_id"].isin(xq).sum())
        prod_tr = set(pd.concat([frames["train"], frames["dev"]])
                      .loc[lambda d: d["query_id"].isin(xq), "product_id"])
        prod_te = set(frames["test"].loc[frames["test"]["query_id"].isin(xq), "product_id"])
        record("cross-split query shares no product between train and test",
               len(prod_tr & prod_te) == 0, 0, len(prod_tr & prod_te),
               note=f"{n_tr} train/dev rows vs {n_te} test rows for query_id(s) {sorted(xq)}")

    # ------------------------------------------------------------------
    # 5. Features present, finite, complete
    # ------------------------------------------------------------------
    print("\n5. Feature completeness and sanity")
    for name, df in frames.items():
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        record(f"[{name}] all 17 features present", not missing, [], missing)
        X = df[ALL_FEATURES].to_numpy(dtype=float)
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        record(f"[{name}] no NaN in features", n_nan == 0, 0, n_nan)
        record(f"[{name}] no Inf in features", n_inf == 0, 0, n_inf)
        for c in ["bm25_score", "semantic_score", "relevance", "lgb_label"]:
            nn = int(df[c].isna().sum())
            record(f"[{name}] no NaN in {c}", nn == 0, 0, nn)
    record("feature list identical to reranking.advanced_features.ALL_FEATURES",
           manifest["feature_list"] == ALL_FEATURES, True,
           manifest["feature_list"] == ALL_FEATURES,
           note="no feature added, removed or reordered")

    # ------------------------------------------------------------------
    # 6. Query groups intact
    # ------------------------------------------------------------------
    print("\n6. Query group integrity")
    for name, df in frames.items():
        sizes = df.groupby("query_id", sort=False).size()
        record(f"[{name}] group sizes sum to row count",
               int(sizes.sum()) == len(df), len(df), int(sizes.sum()))
        record(f"[{name}] every query has >= 1 candidate", int(sizes.min()) >= 1,
               ">=1", int(sizes.min()))
        # rows must be contiguous by query_id for LightGBM grouping
        contiguous = df["query_id"].ne(df["query_id"].shift()).cumsum().nunique() == df["query_id"].nunique()
        record(f"[{name}] rows are contiguous by query_id", bool(contiguous), True, bool(contiguous))
        no_rel = int((df.groupby("query_id")["relevance"].max() <= 0).sum())
        record(f"[{name}] queries with no relevant candidate (excluded by the scorer)",
               True, None, no_rel, note="informational")

    # ------------------------------------------------------------------
    # 7. Distributions (informational, recorded not asserted)
    # ------------------------------------------------------------------
    print("\n7. Distributions (recorded)")
    dist = {}
    for name, df in frames.items():
        sizes = df.groupby("query_id").size()
        lab = df["esci_label"].value_counts(normalize=True)
        dist[name] = {
            "rows": int(len(df)), "queries": int(df["query_id"].nunique()),
            "candidates_per_query": {
                "mean": float(sizes.mean()), "std": float(sizes.std()),
                "min": int(sizes.min()), "p5": float(sizes.quantile(.05)),
                "p25": float(sizes.quantile(.25)), "median": float(sizes.median()),
                "p75": float(sizes.quantile(.75)), "p95": float(sizes.quantile(.95)),
                "max": int(sizes.max()),
            },
            "label_pct": {k: round(100 * float(lab.get(k, 0)), 4) for k in ["E", "S", "C", "I"]},
            "locale_queries": {k: int(v) for k, v in
                               df.drop_duplicates("query_id")["product_locale"].value_counts().items()},
            "queries_lt_10_candidates": int((sizes < 10).sum()),
            "degenerate_queries_single_relevance": int(
                (df.groupby("query_id")["relevance"].nunique() <= 1).sum()),
        }
        print(f"  {name}: {dist[name]['rows']} rows / {dist[name]['queries']} queries, "
              f"labels {dist[name]['label_pct']}, locales {dist[name]['locale_queries']}")

    n_fail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    n_upstream = sum(1 for c in CHECKS if c["status"] == "FAIL_UPSTREAM")
    n_pass = sum(1 for c in CHECKS if c["status"] == "PASS")
    out = {
        "total_checks": len(CHECKS),
        "passed": n_pass,
        "failed_build_induced": n_fail,
        "failed_upstream_inherited": n_upstream,
        "build_is_valid": n_fail == 0,
        "status_legend": {
            "PASS": "assertion holds",
            "FAIL": "defect introduced by this build -- gates the benchmark",
            "FAIL_UPSTREAM": "defect inherited from the ESCI source data that constraint 3 "
                             "(do not change the split) forbids fixing here; documented and "
                             "quantified, never silently repaired",
        },
        "checks": CHECKS,
        "distributions": dist,
    }
    with open(os.path.join(BASE, "integrity_checks.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f" {n_pass} PASS / {n_fail} FAIL (build-induced) / "
          f"{n_upstream} FAIL_UPSTREAM (inherited, documented)")
    print("=" * 78)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
