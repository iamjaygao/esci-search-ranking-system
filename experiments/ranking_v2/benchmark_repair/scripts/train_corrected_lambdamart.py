"""
Phase F -- retrain LambdaMART with the corrected gain convention ONLY.

Everything except `label_gain` and the (now clean) training rows is copied
verbatim from scripts/train_lambdamart.py:

    objective="lambdarank", metric="ndcg", n_estimators=1000,
    learning_rate=0.05, num_leaves=31, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, n_jobs=-1
    early stopping: lgb.early_stopping(50), eval_at=[10]
    integer labels: E=3, S=2, C=1, I=0

No hyperparameter is tuned. Three variants are trained so the effect of the
POOL change and the effect of the GAIN change can be attributed separately:

  v1gain_cleanpool  label_gain=[0, 1, 3, 7]                     -- pool effect only
  exact_cleanpool   label_gain=[0, 2**.01-1, 2**.1-1, 1.0]      -- AUTHORITATIVE
  linear_cleanpool  label_gain=[0, 0.01, 0.10, 1.0]             -- the brief's literal spec

`exact_cleanpool` is authoritative because LightGBM uses label_gain[label]
directly as the DCG numerator, so only 2**relevance-1 makes LightGBM's internal
lambdarank objective numerically identical to evaluation/metrics.dcg and to
scripts/official_ndcg.py.

Never writes outside experiments/ranking_v2/benchmark_repair/.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from reranking.advanced_features import ALL_FEATURES  # noqa: E402
from official_ndcg import evaluate, per_query_ndcg  # noqa: E402

DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

# --- verbatim from scripts/train_lambdamart.py ---
FIXED_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

VARIANTS = {
    "v1gain_cleanpool": [0.0, 1.0, 3.0, 7.0],
    "exact_cleanpool": [0.0, float(2 ** 0.01 - 1), float(2 ** 0.10 - 1), 1.0],
    "linear_cleanpool": [0.0, 0.01, 0.10, 1.0],
}
AUTHORITATIVE = "exact_cleanpool"


def groups(df):
    """Per-query row counts, in the frame's existing (contiguous) order."""
    g = df.groupby("query_id", sort=False).size().to_numpy()
    assert g.sum() == len(df), "group/row count mismatch"
    return g


def main():
    os.makedirs(MODELS, exist_ok=True)

    tr = pd.read_parquet(os.path.join(DATA, "train_clean.parquet"))
    dv = pd.read_parquet(os.path.join(DATA, "dev_clean.parquet"))
    te = pd.read_parquet(os.path.join(DATA, "test_clean.parquet"))
    print(f"train {len(tr)} rows / {tr['query_id'].nunique()} queries")
    print(f"dev   {len(dv)} rows / {dv['query_id'].nunique()} queries")
    print(f"test  {len(te)} rows / {te['query_id'].nunique()} queries")

    Xtr, ytr, gtr = tr[ALL_FEATURES].values, tr["lgb_label"].values, groups(tr)
    Xdv, ydv, gdv = dv[ALL_FEATURES].values, dv["lgb_label"].values, groups(dv)

    dev_no_rel = int((dv.groupby("query_id")["relevance"].max() <= 0).sum())
    test_no_rel = int((te.groupby("query_id")["relevance"].max() <= 0).sum())
    print(f"dev queries with no relevant candidate: {dev_no_rel}; test: {test_no_rel}")

    results = {}
    for name, gain in VARIANTS.items():
        print("\n" + "=" * 70)
        print(f" Training variant: {name}   label_gain={gain}")
        print("=" * 70)
        t0 = time.time()
        model = lgb.LGBMRanker(label_gain=gain, **FIXED_PARAMS)
        model.fit(
            Xtr, ytr, group=gtr,
            eval_set=[(Xdv, ydv)], eval_group=[gdv], eval_at=[10],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        elapsed = time.time() - t0

        booster = model.booster_
        path = os.path.join(MODELS, f"corrected_lambdamart_{name}.txt")
        booster.save_model(path)

        lgb_dev_ndcg = float(model.best_score_["valid_0"]["ndcg@10"])

        dv2 = dv.copy()
        dv2["score"] = booster.predict(Xdv)
        te2 = te.copy()
        te2["score"] = booster.predict(te[ALL_FEATURES].values)
        dev_eval = evaluate(dv2, "score", k=10)
        test_eval = evaluate(te2, "score", k=10)

        imp_gain = booster.feature_importance("gain")
        imp_split = booster.feature_importance("split")
        total = float(imp_gain.sum())

        results[name] = {
            "label_gain": gain,
            "model_path": os.path.relpath(path, ROOT),
            "best_iteration": int(model.best_iteration_),
            "num_trees_in_saved_model": int(booster.num_trees()),
            "early_stopping_triggered": bool(booster.num_trees() < FIXED_PARAMS["n_estimators"]),
            "train_seconds": round(elapsed, 2),
            "lightgbm_internal_dev_ndcg_at_10": lgb_dev_ndcg,
            "official_scorer_dev_ndcg_at_10": dev_eval["ndcg_at_k"],
            "lightgbm_vs_official_dev_abs_diff": abs(lgb_dev_ndcg - dev_eval["ndcg_at_k"]),
            "dev": dev_eval,
            "test": test_eval,
            "feature_importance_gain_pct": {
                f: round(100.0 * float(g) / total, 4)
                for f, g in sorted(zip(ALL_FEATURES, imp_gain), key=lambda x: -x[1])
            },
            "feature_split_count": {f: int(s) for f, s in zip(ALL_FEATURES, imp_split)},
        }
        print(f"  best_iteration={model.best_iteration_}  trees={booster.num_trees()}  "
              f"({elapsed:.1f}s)")
        print(f"  LightGBM internal dev ndcg@10 = {lgb_dev_ndcg:.6f}")
        print(f"  official scorer  dev ndcg@10 = {dev_eval['ndcg_at_k']:.6f}  "
              f"(|diff| = {abs(lgb_dev_ndcg - dev_eval['ndcg_at_k']):.2e})")
        print(f"  official scorer TEST ndcg@10 = {test_eval['ndcg_at_k']:.6f}")

        # persist per-query test scores for the authoritative variant
        if name == AUTHORITATIVE:
            te2[["query_id", "example_id", "product_id", "score"]].to_parquet(
                os.path.join(MODELS, "corrected_lambdamart_test_scores.parquet"), index=False)

    # ------------------------------------------------------------------
    meta = {
        "authoritative_variant": AUTHORITATIVE,
        "fixed_hyperparameters": FIXED_PARAMS,
        "hyperparameters_source": "scripts/train_lambdamart.py (copied verbatim, nothing tuned)",
        "integer_label_map": {"E": 3, "S": 2, "C": 1, "I": 0},
        "dev_queries_with_no_relevant_candidate": dev_no_rel,
        "test_queries_with_no_relevant_candidate": test_no_rel,
        "objective_metric_alignment_check": {
            "claim": "LightGBM's internal ndcg@10 equals scripts/official_ndcg.py on dev "
                     "when label_gain == 2**relevance - 1",
            "abs_diff_exact_variant": results[AUTHORITATIVE]["lightgbm_vs_official_dev_abs_diff"],
            "abs_diff_linear_variant": results["linear_cleanpool"]["lightgbm_vs_official_dev_abs_diff"],
            "abs_diff_v1gain_variant": results["v1gain_cleanpool"]["lightgbm_vs_official_dev_abs_diff"],
        },
        "variants": results,
    }
    with open(os.path.join(MODELS, "training_results.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(" Summary (official scorer, clean frozen pool)")
    print("=" * 70)
    for n, r in results.items():
        star = "  <-- AUTHORITATIVE" if n == AUTHORITATIVE else ""
        print(f"  {n:20s} dev={r['dev']['ndcg_at_k']:.4f}  test={r['test']['ndcg_at_k']:.4f}  "
              f"trees={r['num_trees_in_saved_model']:4d}{star}")
    print("\nAlignment |LightGBM internal dev ndcg - official scorer dev ndcg|:")
    for n, r in results.items():
        print(f"  {n:20s} {r['lightgbm_vs_official_dev_abs_diff']:.3e}")


if __name__ == "__main__":
    main()
