"""
Phases E, G, H, I -- evaluate every baseline on the ONE frozen clean pool with
the ONE authoritative scorer, then produce the comparison tables.

Models (all inference-only except the corrected LambdaMART from Phase F):
  BM25          -- the bm25_score column already on the pool
  Two-Tower     -- the semantic_score column already on the pool
  MLP           -- output/best_advanced_reranker.pth, frozen weights + frozen
                   normalisation stats
  LambdaMART V1 -- output/lambdamart_model.txt, frozen (reference row only)
  LambdaMART    -- models/corrected_lambdamart_exact_cleanpool.txt (authoritative)

Bootstrap: query-level paired percentile bootstrap, n=10,000, seed=42 -- the
implementation audited in experiments/audit/task3_per_query_bootstrap.py.

Writes: baseline_comparison.csv, historical_results.csv, locale_results.csv,
        paired_comparisons.json, per_query_scores.parquet, mlp_fidelity.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
# NOTE: lightgbm is imported lazily inside main(). Importing it BEFORE torch
# segfaults on this macOS box (duplicate libomp initialisation); this is the
# same import order that experiments/ranking_v2/audit/scripts/00_build_caches.py
# already uses successfully.

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from reranking.advanced_features import ALL_FEATURES  # noqa: E402
from reranking.advanced_model import AdvancedDeepReranker  # noqa: E402
from official_ndcg import evaluate, per_query_ndcg  # noqa: E402

DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

N_BOOTSTRAP = 10000
SEED = 42

MODEL_COLS = {
    "BM25": "bm25_score",
    "Two-Tower": "semantic_score",
    "MLP": "mlp_score",
    "LambdaMART": "lambdamart_score",
}
REFERENCE_COLS = {"LambdaMART (V1 model, V1 gain)": "lambdamart_v1_score"}


def paired_bootstrap(a, b, name_a, name_b):
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).values
    n = len(d)
    rng = np.random.RandomState(SEED)
    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        boot[i] = d[rng.randint(0, n, size=n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    return {
        "comparison": f"{name_a} - {name_b}",
        "n_queries": int(n),
        f"mean_ndcg10_{name_a}": float(a.loc[common].mean()),
        f"mean_ndcg10_{name_b}": float(b.loc[common].mean()),
        "mean_paired_delta": float(d.mean()),
        "median_paired_delta": float(np.median(d)),
        "ci_95_low": float(lo),
        "ci_95_high": float(hi),
        "p_value_two_sided": p,
        "p_value_resolution_floor": 1.0 / N_BOOTSTRAP,
        "significant_at_0.05": bool(p < 0.05),
        "win": int((d > 1e-12).sum()),
        "loss": int((d < -1e-12).sum()),
        "tie": int((np.abs(d) <= 1e-12).sum()),
        "win_pct": round(100.0 * (d > 1e-12).mean(), 3),
        "loss_pct": round(100.0 * (d < -1e-12).mean(), 3),
        "tie_pct": round(100.0 * (np.abs(d) <= 1e-12).mean(), 3),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
    }


def main():
    import lightgbm as lgb  # see import-order note at the top of this file

    te = pd.read_parquet(os.path.join(DATA, "test_clean.parquet"))
    print(f"Frozen test pool: {len(te)} rows / {te['query_id'].nunique()} queries")

    # ---------------- E3: MLP, frozen weights ----------------
    print("\n[E3] MLP inference (frozen checkpoint + frozen normalisation stats)")
    with open(os.path.join(ROOT, "output/advanced_normalization_stats.json")) as f:
        norm = json.load(f)
    assert norm["features"] == ALL_FEATURES, "MLP feature order differs from the pool"
    mean, std = np.array(norm["mean"]), np.array(norm["std"])
    Xn = (te[ALL_FEATURES].values - mean) / std
    mlp = AdvancedDeepReranker(input_dim=len(ALL_FEATURES))
    mlp.load_state_dict(torch.load(os.path.join(ROOT, "output/best_advanced_reranker.pth"),
                                   map_location="cpu", weights_only=True))
    mlp.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xn), 100000):
            preds.append(mlp(torch.tensor(Xn[i:i + 100000], dtype=torch.float32)).squeeze(-1).numpy())
    te["mlp_score"] = np.concatenate(preds)

    # ---------------- LambdaMART: V1 (reference) and corrected ----------------
    print("[ref] LambdaMART V1 inference (frozen output/lambdamart_model.txt)")
    v1 = lgb.Booster(model_file=os.path.join(ROOT, "output/lambdamart_model.txt"))
    te["lambdamart_v1_score"] = v1.predict(te[ALL_FEATURES].values)

    print("[F] LambdaMART corrected inference")
    corr_path = os.path.join(MODELS, "corrected_lambdamart_exact_cleanpool.txt")
    corr = lgb.Booster(model_file=corr_path)
    te["lambdamart_score"] = corr.predict(te[ALL_FEATURES].values)

    # ------------------------------------------------------------------
    # MLP fidelity: how much did recomputing features on the clean pool
    # shift the frozen MLP's inputs vs the V1 pool it was scored on before?
    # ------------------------------------------------------------------
    fidelity = {"status": "V1 cache unavailable"}
    v1_cache = os.path.join(ROOT, "experiments/ranking_v2/audit/_cache/test_scored.parquet")
    if os.path.exists(v1_cache):
        print("\n[integrity] comparing clean-pool features against the V1 pool")
        old = pd.read_parquet(v1_cache, columns=["query_id", "product_id", "product_locale",
                                                 "mlp_score"] + ALL_FEATURES)
        old = old.drop_duplicates(["query_id", "product_id", "product_locale"], keep="first")
        m = te.merge(old, on=["query_id", "product_id", "product_locale"],
                     how="inner", suffixes=("", "_v1"))
        fidelity = {"matched_rows": int(len(m)), "clean_pool_rows": int(len(te)), "per_feature": {}}
        for f in ALL_FEATURES:
            a, b = m[f].astype(float), m[f + "_v1"].astype(float)
            diff = (a - b).abs()
            fidelity["per_feature"][f] = {
                "rows_changed": int((diff > 1e-9).sum()),
                "pct_changed": round(100.0 * float((diff > 1e-9).mean()), 4),
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
            }
        sd = (m["mlp_score"] - m["mlp_score_v1"]).abs()
        fidelity["mlp_score_abs_diff"] = {
            "rows_changed": int((sd > 1e-9).sum()),
            "pct_changed": round(100.0 * float((sd > 1e-9).mean()), 4),
            "max": float(sd.max()), "mean": float(sd.mean()),
        }
        fidelity["interpretation"] = (
            "The frozen MLP's inputs move only where within-frame imputation medians or the "
            "top-20-BM25 dominant-category vote changed as a result of removing the 44,217 "
            "duplicate rows. Features that do not depend on within-frame aggregates "
            "(bm25_score, semantic_score, word_overlap, query_*, brand_match, color_match) "
            "must be bit-identical.")
        with open(os.path.join(BASE, "mlp_fidelity.json"), "w") as f:
            json.dump(fidelity, f, indent=2, default=str)
        changed = {k: v["pct_changed"] for k, v in fidelity["per_feature"].items()
                   if v["pct_changed"] > 0}
        print(f"  features that moved at all: {changed if changed else 'none'}")
        print(f"  mlp_score changed on {fidelity['mlp_score_abs_diff']['pct_changed']}% of rows "
              f"(max |diff| {fidelity['mlp_score_abs_diff']['max']:.3e})")

    # ------------------------------------------------------------------
    # Phase E + G: authoritative table
    # ------------------------------------------------------------------
    print("\n[E/G] scoring every model with the authoritative scorer")
    all_cols = {**MODEL_COLS, **REFERENCE_COLS}
    pq = {}
    rows = []
    for label, col in all_cols.items():
        r = evaluate(te, col, k=10)
        pq[label] = per_query_ndcg(te, col, k=10)
        status = "authoritative" if label in MODEL_COLS else "reference (not a V2 baseline)"
        rows.append({
            "model": label,
            "ndcg_at_10": round(r["ndcg_at_k"], 6),
            "candidate_pool": "clean frozen (test_clean.parquet)",
            "pool_rows": r["n_rows"],
            "queries_scored": r["n_queries_scored"],
            "queries_excluded_no_relevant": r["n_excluded_no_relevant"],
            "gain_convention": "official (E=1.00, S=0.10, C=0.01, I=0.00; gain=2**rel-1)",
            "scorer": "scripts/official_ndcg.py",
            "tie_break": "deterministic (ascending product_id)",
            "trained_or_frozen": ("frozen checkpoint" if label != "LambdaMART"
                                  else "retrained, corrected label_gain only"),
            "status": status,
            "ndcg_at_10_median": round(r["ndcg_at_k_median"], 6),
            "ndcg_at_10_std": round(r["ndcg_at_k_std"], 6),
        })
        print(f"  {label:32s} NDCG@10 = {r['ndcg_at_k']:.6f}  (n={r['n_queries_scored']})")

    bc = pd.DataFrame(rows)
    bc.to_csv(os.path.join(BASE, "baseline_comparison.csv"), index=False)

    # non-degenerate subset, reported alongside (Phase 0 P2-A)
    nun = te.groupby("query_id")["relevance"].nunique()
    nondeg = set(nun[nun > 1].index)
    nd_rows = [{"model": lab, "subset": "non_degenerate_queries_only",
                "n_queries": int(len(pq[lab][pq[lab].index.isin(nondeg)])),
                "ndcg_at_10": round(float(pq[lab][pq[lab].index.isin(nondeg)].mean()), 6)}
               for lab in all_cols]
    pd.DataFrame(nd_rows).to_csv(os.path.join(BASE, "nondegenerate_subset.csv"), index=False)

    # ------------------------------------------------------------------
    # Phase H: locale breakdown
    # ------------------------------------------------------------------
    print("\n[H] locale breakdown")
    qloc = te.drop_duplicates("query_id").set_index("query_id")["product_locale"]
    lrows = []
    for label in all_cols:
        s = pq[label]
        loc = qloc.reindex(s.index)
        for lc in ["us", "jp", "es"]:
            sub = s[loc == lc]
            lrows.append({"model": label, "locale": lc, "query_count": int(len(sub)),
                          "ndcg_at_10": round(float(sub.mean()), 6)})
        lrows.append({"model": label, "locale": "overall", "query_count": int(len(s)),
                      "ndcg_at_10": round(float(s.mean()), 6)})
    lr = pd.DataFrame(lrows)
    lr.to_csv(os.path.join(BASE, "locale_results.csv"), index=False)
    print(lr.pivot(index="model", columns="locale", values="ndcg_at_10").to_string())

    # ------------------------------------------------------------------
    # Phase I: paired comparisons
    # ------------------------------------------------------------------
    print("\n[I] paired bootstrap (n=10,000, seed=42)")
    comps = [("LambdaMART", "MLP"), ("LambdaMART", "BM25"), ("LambdaMART", "Two-Tower"),
             ("MLP", "BM25"), ("MLP", "Two-Tower"), ("Two-Tower", "BM25"),
             ("LambdaMART", "LambdaMART (V1 model, V1 gain)"),
             # like-for-like capacity test: BOTH models frozen from V1, both scored
             # on the clean pool. The "LambdaMART - MLP" row above is confounded
             # because LambdaMART was retrained on the repaired benchmark and the
             # MLP was not.
             ("LambdaMART (V1 model, V1 gain)", "MLP")]
    results = []
    for a, b in comps:
        r = paired_bootstrap(pq[a], pq[b], a, b)
        results.append(r)
        print(f"  {r['comparison']:48s} d={r['mean_paired_delta']:+.5f} "
              f"CI=[{r['ci_95_low']:+.5f},{r['ci_95_high']:+.5f}] p={r['p_value_two_sided']:.4f} "
              f"W/L/T={r['win']}/{r['loss']}/{r['tie']}")
    with open(os.path.join(BASE, "paired_comparisons.json"), "w") as f:
        json.dump({
            "methodology": ("Query-level paired percentile bootstrap over per-query NDCG@10 "
                            "deltas, n_bootstrap=10000, seed=42 -- the implementation audited "
                            "in experiments/audit/task3_per_query_bootstrap.py."),
            "pool": "clean frozen test pool, 638,016 rows / 30,969 queries",
            "scorer": "scripts/official_ndcg.py, official gain convention",
            "comparisons": results,
        }, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # per-query dump + legacy table
    # ------------------------------------------------------------------
    pqdf = pd.DataFrame({lab: pq[lab] for lab in all_cols})
    pqdf.index.name = "query_id"
    pqdf = pqdf.join(qloc.rename("locale"))
    pqdf = pqdf.join(te.groupby("query_id").size().rename("candidate_count"))
    pqdf.reset_index().to_parquet(os.path.join(BASE, "per_query_scores.parquet"), index=False)

    hist = pd.DataFrame([
        {"model": "BM25", "legacy_ndcg_at_10": 0.8188130,
         "legacy_pool": "bm25_scores_test.csv INNER JOIN ESCI labels (638,016 rows, no products join)",
         "legacy_gain_convention": "2**rel-1 with rel={E:1.0,S:0.1,C:0.01,I:0.0}",
         "comparable_to_new_table": "NO", "reason": "different candidate pool from the model rows"},
        {"model": "Two-Tower", "legacy_ndcg_at_10": 0.8267168,
         "legacy_pool": "two_tower_scores_test.csv INNER JOIN ESCI labels (638,016 rows)",
         "legacy_gain_convention": "2**rel-1 with rel={E:1.0,S:0.1,C:0.01,I:0.0}",
         "comparable_to_new_table": "NO", "reason": "different candidate pool from the model rows"},
        {"model": "MLP (17-feature)", "legacy_ndcg_at_10": 0.8457793,
         "legacy_pool": "extract_test_advanced_features (682,233 rows, 6.93% duplicated)",
         "legacy_gain_convention": "2**rel-1 with rel={E:1.0,S:0.1,C:0.01,I:0.0}",
         "comparable_to_new_table": "NO", "reason": "duplicate-inflated pool"},
        {"model": "LambdaMART (17-feature)", "legacy_ndcg_at_10": 0.8464228,
         "legacy_pool": "extract_test_advanced_features (682,233 rows, 6.93% duplicated)",
         "legacy_gain_convention": "trained on label_gain [0,1,3,7], scored on 2**rel-1",
         "comparable_to_new_table": "NO",
         "reason": "duplicate-inflated pool AND objective/metric mismatch"},
        {"model": "BM25 (on the LambdaMART pool)", "legacy_ndcg_at_10": 0.8155164,
         "legacy_pool": "extract_test_advanced_features (682,233 rows)",
         "legacy_gain_convention": "2**rel-1 with rel={E:1.0,S:0.1,C:0.01,I:0.0}",
         "comparable_to_new_table": "NO", "reason": "duplicate-inflated pool"},
    ])
    hist.insert(0, "STATUS", "LEGACY / NOT DIRECTLY COMPARABLE")
    hist.to_csv(os.path.join(BASE, "historical_results.csv"), index=False)
    print("\nWrote baseline_comparison.csv, historical_results.csv, locale_results.csv, "
          "paired_comparisons.json, nondegenerate_subset.csv, per_query_scores.parquet")


if __name__ == "__main__":
    main()
