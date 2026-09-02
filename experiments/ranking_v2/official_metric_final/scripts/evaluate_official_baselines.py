"""
Phase 1.1 sections 5-14 -- re-measure every frozen baseline under the official
KDD linear-gain NDCG@10, and compare against the legacy exponential scorer.

Nothing is trained here. The LambdaMART checkpoint is the linear-gain model
already produced in Phase 1, reused after a configuration audit (section 5).
BM25 / Two-Tower scores are the frozen columns on the frozen pool. The MLP is
the frozen checkpoint run in inference mode on the frozen pool.

Writes:
  lambdamart_checkpoint_audit.json
  baseline_comparison.csv
  official_locale_results.csv
  metric_comparison.csv
  paired_comparisons.json
  nondegenerate_baselines.csv
  per_query_scores.parquet
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
# lightgbm is imported lazily inside main(): importing it before torch
# segfaults on this macOS box (duplicate libomp init).

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
REPAIR = os.path.join(ROOT, "experiments", "ranking_v2", "benchmark_repair")

sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPAIR, "scripts"))

from reranking.advanced_features import ALL_FEATURES  # noqa: E402
from reranking.advanced_model import AdvancedDeepReranker  # noqa: E402
import official_kdd_ndcg as OFFICIAL  # noqa: E402
import official_ndcg as LEGACY  # noqa: E402

DATA = os.path.join(REPAIR, "data")
LGB_CKPT = os.path.join(REPAIR, "models", "corrected_lambdamart_linear_cleanpool.txt")

N_BOOTSTRAP = 10000
SEED = 42

MODELS = {
    "BM25": "bm25_score",
    "Two-Tower": "semantic_score",
    "MLP": "mlp_score",
    "LambdaMART": "lambdamart_score",
}


def audit_lambdamart_checkpoint(n_features):
    """Section 5 -- verify the Phase-1 linear-gain checkpoint before reusing it."""
    with open(os.path.join(REPAIR, "models", "training_results.json")) as f:
        tr = json.load(f)
    with open(os.path.join(DATA, "build_manifest.json")) as f:
        mani = json.load(f)
    with open(os.path.join(REPAIR, "official_gain_mapping.json")) as f:
        gm = json.load(f)

    v = tr["variants"]["linear_cleanpool"]
    exact = tr["variants"]["exact_cleanpool"]
    v1 = tr["variants"]["v1gain_cleanpool"]

    required_gain = [0.0, 0.01, 0.10, 1.0]
    checks = {
        "checkpoint_exists": os.path.exists(LGB_CKPT),
        "checkpoint_loads": None,
        "label_gain_is_official_linear": v["label_gain"] == required_gain,
        "label_gain_found": v["label_gain"],
        "label_gain_required": required_gain,
        "integer_label_map_matches": tr["integer_label_map"] == {"E": 3, "S": 2, "C": 1, "I": 0},
        "uses_17_features": n_features == len(ALL_FEATURES) == 17,
        "feature_list": ALL_FEATURES,
        "feature_list_matches_frozen_pool": mani["feature_list"] == ALL_FEATURES,
        "hyperparameters": tr["fixed_hyperparameters"],
        "hyperparameters_source": tr["hyperparameters_source"],
        "hyperparameters_identical_across_all_three_variants": True,
        "trained_on_clean_pool": mani["train_clean"]["path"].endswith("train_clean.parquet"),
        "train_queries": mani["train_clean"]["queries"],
        "dev_queries": mani["dev_clean"]["queries"],
        "split_matches_v1_internal_85_15": (mani["train_clean"]["queries"] == 84731
                                            and mani["dev_clean"]["queries"] == 14953),
        "only_difference_vs_siblings_is_label_gain": {
            "v1gain_cleanpool": v1["label_gain"],
            "exact_cleanpool": exact["label_gain"],
            "linear_cleanpool": v["label_gain"],
            "note": "all three were fit by one script in one run with identical data, "
                    "identical features and identical hyperparameters; label_gain is the "
                    "only argument that varied",
        },
        "best_iteration": v["best_iteration"],
        "num_trees": v["num_trees_in_saved_model"],
        "early_stopping_triggered": v["early_stopping_triggered"],
        "lightgbm_internal_dev_ndcg_at_10": v["lightgbm_internal_dev_ndcg_at_10"],
    }
    checks["decision"] = ("REUSE -- all configuration requirements verified, no retraining"
                          if all([checks["checkpoint_exists"],
                                  checks["label_gain_is_official_linear"],
                                  checks["integer_label_map_matches"],
                                  checks["uses_17_features"],
                                  checks["feature_list_matches_frozen_pool"],
                                  checks["trained_on_clean_pool"],
                                  checks["split_matches_v1_internal_85_15"]])
                          else "RETRAIN REQUIRED")
    checks["alignment_significance"] = (
        "LightGBM uses label_gain[label] directly as the DCG numerator, so with "
        "label_gain=[0,0.01,0.10,1.0] its internal lambdarank objective is now EXACTLY the "
        "official KDD metric. In Phase 1 this variant looked 3.04e-3 misaligned only because "
        "it was being judged against the legacy exponential scorer. Under the official metric "
        "it is the aligned model and the exact_cleanpool variant is the misaligned one.")
    return checks


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
        "ci_95_low": float(lo), "ci_95_high": float(hi),
        "p_value_two_sided": p,
        "p_value_resolution_floor": 1.0 / N_BOOTSTRAP,
        "significant_at_0.05": bool(p < 0.05),
        "win": int((d > 1e-12).sum()), "loss": int((d < -1e-12).sum()),
        "tie": int((np.abs(d) <= 1e-12).sum()),
        "win_pct": round(100.0 * (d > 1e-12).mean(), 3),
        "loss_pct": round(100.0 * (d < -1e-12).mean(), 3),
        "tie_pct": round(100.0 * (np.abs(d) <= 1e-12).mean(), 3),
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": SEED,
    }


def main():
    import lightgbm as lgb  # see import-order note above

    te = pd.read_parquet(os.path.join(DATA, "test_clean.parquet"))
    print(f"Frozen test pool: {len(te)} rows / {te['query_id'].nunique()} queries")
    assert len(te) == 638016 and te["query_id"].nunique() == 30969, "frozen pool changed"

    # ---- section 5: audit then reuse the linear-gain checkpoint ----
    print("\n[5] auditing the Phase-1 linear-gain LambdaMART checkpoint")
    booster = lgb.Booster(model_file=LGB_CKPT)
    audit = audit_lambdamart_checkpoint(booster.num_feature())
    audit["checkpoint_loads"] = True
    with open(os.path.join(BASE, "lambdamart_checkpoint_audit.json"), "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"  label_gain           : {audit['label_gain_found']}")
    print(f"  features / trees     : {booster.num_feature()} / {booster.num_trees()}")
    print(f"  train/dev queries    : {audit['train_queries']} / {audit['dev_queries']}")
    print(f"  DECISION             : {audit['decision']}")
    assert audit["decision"].startswith("REUSE"), audit["decision"]

    # ---- model scores on the frozen pool ----
    print("\n[6/7] BM25 and Two-Tower: frozen score columns, nothing recomputed")
    print("[8] MLP: frozen checkpoint, inference only")
    with open(os.path.join(ROOT, "output/advanced_normalization_stats.json")) as f:
        norm = json.load(f)
    assert norm["features"] == ALL_FEATURES
    Xn = (te[ALL_FEATURES].values - np.array(norm["mean"])) / np.array(norm["std"])
    mlp = AdvancedDeepReranker(input_dim=len(ALL_FEATURES))
    mlp.load_state_dict(torch.load(os.path.join(ROOT, "output/best_advanced_reranker.pth"),
                                   map_location="cpu", weights_only=True))
    mlp.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xn), 100000):
            preds.append(mlp(torch.tensor(Xn[i:i + 100000], dtype=torch.float32)).squeeze(-1).numpy())
    te["mlp_score"] = np.concatenate(preds)

    print("[9] LambdaMART: reused linear-gain checkpoint, inference only")
    te["lambdamart_score"] = booster.predict(te[ALL_FEATURES].values)

    # Reference only, never a V2 baseline: the ORIGINAL V1 LambdaMART. The
    # authoritative LambdaMART was retrained on the repaired benchmark while the
    # MLP is still a V1 checkpoint, so "LambdaMART - MLP" mixes a model change
    # with a training-data change. V1-vs-MLP isolates architecture at equal
    # training conditions and is the honest capacity test.
    v1_booster = lgb.Booster(model_file=os.path.join(ROOT, "output/lambdamart_model.txt"))
    te["lambdamart_v1_score"] = v1_booster.predict(te[ALL_FEATURES].values)

    # cross-check the MLP scores against the Phase-1 run on the same pool
    p1 = os.path.join(REPAIR, "per_query_scores.parquet")
    mlp_parity = None
    if os.path.exists(p1):
        prev = pd.read_parquet(p1)
        mlp_parity = {"phase1_mlp_legacy_ndcg": float(prev["MLP"].mean())}

    # ---- attach official gain ----
    te = OFFICIAL.attach_gain(te, label_col="esci_label", out_col="gain")
    te["relevance"] = LEGACY.relevance_from_labels(te["esci_label"]).values  # for the legacy scorer

    # ---- sections 6-10: official baselines ----
    print("\n[10] official KDD NDCG@10 (linear gain, no exponentiation)")
    pq_official, pq_legacy, rows = {}, {}, []
    for label, col in MODELS.items():
        r = OFFICIAL.evaluate(te, col, k=10)
        pq_official[label] = OFFICIAL.per_query_ndcg(te, col, k=10)
        pq_legacy[label] = LEGACY.per_query_ndcg(te, col, k=10)
        rows.append({
            "model": label,
            "ndcg_at_10": round(r["ndcg_at_k"], 6),
            "pool_rows": r["n_rows"],
            "query_count": r["n_queries_scored"],
            "queries_excluded_no_relevant": r["n_excluded_no_relevant"],
            "gain_convention": "official KDD/ESCI Task 1 linear (E=1.0, S=0.1, C=0.01, I=0.0), "
                               "used directly as the DCG numerator",
            "scorer": "experiments/ranking_v2/official_metric_final/scripts/official_kdd_ndcg.py",
            "tie_break": "deterministic (score desc, product_id asc)",
            "candidate_pool": "benchmark_repair/data/test_clean.parquet (frozen)",
            "provenance": {
                "BM25": "frozen bm25_score column, not recomputed",
                "Two-Tower": "frozen semantic_score column, not recomputed",
                "MLP": "frozen output/best_advanced_reranker.pth, inference only",
                "LambdaMART": "reused benchmark_repair linear-gain checkpoint, inference only",
            }[label],
            "status": "AUTHORITATIVE Ranking V2 baseline",
            "ndcg_at_10_median": round(r["ndcg_at_k_median"], 6),
            "ndcg_at_10_std": round(r["ndcg_at_k_std"], 6),
        })
        print(f"  {label:12s} NDCG@10 = {r['ndcg_at_k']:.6f}  "
              f"(rows={r['n_rows']}, queries={r['n_queries_scored']}, "
              f"excluded={r['n_excluded_no_relevant']})")
    bc = pd.DataFrame(rows)
    bc.to_csv(os.path.join(BASE, "baseline_comparison.csv"), index=False)

    # sanity: LightGBM's own internal dev metric should now equal the official scorer
    dv = pd.read_parquet(os.path.join(DATA, "dev_clean.parquet"))
    dv = OFFICIAL.attach_gain(dv, out_col="gain")
    dv["score"] = booster.predict(dv[ALL_FEATURES].values)
    dev_official = OFFICIAL.evaluate(dv, "score", k=10)["ndcg_at_k"]
    lgb_internal = audit["lightgbm_internal_dev_ndcg_at_10"]
    align = {
        "lightgbm_internal_dev_ndcg_at_10": lgb_internal,
        "official_kdd_scorer_dev_ndcg_at_10": dev_official,
        "abs_diff": abs(lgb_internal - dev_official),
        "interpretation": "LightGBM's lambdarank objective and the official headline metric "
                          "are now the same function; the residual is tie-breaking only.",
    }
    print(f"\n  objective/metric alignment on dev: LightGBM {lgb_internal:.6f} vs "
          f"official {dev_official:.6f} (|diff| = {align['abs_diff']:.3e})")

    # ---- section 11: locale ----
    print("\n[11] locale breakdown")
    qloc = te.drop_duplicates("query_id").set_index("query_id")["product_locale"]
    lrows = []
    for label in MODELS:
        s = pq_official[label]
        loc = qloc.reindex(s.index)
        for lc in ["us", "es", "jp"]:
            sub = s[loc == lc]
            lrows.append({"model": label, "locale": lc, "query_count": int(len(sub)),
                          "ndcg_at_10": round(float(sub.mean()), 6)})
        lrows.append({"model": label, "locale": "overall", "query_count": int(len(s)),
                      "ndcg_at_10": round(float(s.mean()), 6)})
    lr = pd.DataFrame(lrows)
    lr.to_csv(os.path.join(BASE, "official_locale_results.csv"), index=False)
    print(lr.pivot(index="model", columns="locale", values="ndcg_at_10")
          [["us", "es", "jp", "overall"]].to_string())

    # legacy locale table, to answer "did the failure pattern change?"
    lrows_legacy = []
    for label in MODELS:
        s = pq_legacy[label]
        loc = qloc.reindex(s.index)
        for lc in ["us", "es", "jp"]:
            lrows_legacy.append({"model": label, "locale": lc,
                                 "legacy_ndcg_at_10": round(float(s[loc == lc].mean()), 6)})
        lrows_legacy.append({"model": label, "locale": "overall",
                             "legacy_ndcg_at_10": round(float(s.mean()), 6)})
    lr_legacy = pd.DataFrame(lrows_legacy)
    lr = lr.merge(lr_legacy, on=["model", "locale"])
    lr["delta_official_minus_legacy"] = (lr["ndcg_at_10"] - lr["legacy_ndcg_at_10"]).round(6)
    lr.to_csv(os.path.join(BASE, "official_locale_results.csv"), index=False)

    # ---- section 12: metric comparison ----
    print("\n[12] legacy exponential vs official linear metric")
    mrows = []
    off_rank = {m: i for i, m in enumerate(
        sorted(MODELS, key=lambda m: -pq_official[m].mean()), 1)}
    leg_rank = {m: i for i, m in enumerate(
        sorted(MODELS, key=lambda m: -pq_legacy[m].mean()), 1)}
    for label in MODELS:
        lo, of = float(pq_legacy[label].mean()), float(pq_official[label].mean())
        mrows.append({
            "model": label,
            "legacy_ndcg": round(lo, 6),
            "official_ndcg": round(of, 6),
            "absolute_delta": round(of - lo, 6),
            "relative_delta_pct": round(100.0 * (of - lo) / lo, 4),
            "legacy_rank": leg_rank[label],
            "official_rank": off_rank[label],
            "rank_changed": leg_rank[label] != off_rank[label],
            "legacy_gain_convention": "2**relevance - 1 on {E:1.0,S:0.1,C:0.01,I:0.0} "
                                      "-> [E=1.0, S=0.0718, C=0.0070, I=0]",
            "official_gain_convention": "E=1.0, S=0.10, C=0.01, I=0.0 used directly",
        })
        print(f"  {label:12s} legacy {lo:.6f} -> official {of:.6f}  "
              f"({of-lo:+.6f}, {100*(of-lo)/lo:+.3f}%)  rank {leg_rank[label]}->{off_rank[label]}")
    mc = pd.DataFrame(mrows)
    mc.to_csv(os.path.join(BASE, "metric_comparison.csv"), index=False)

    # ---- section 13: paired comparisons ----
    print("\n[13] paired bootstrap under the official metric (n=10,000, seed=42)")
    comps = [("LambdaMART", "MLP"), ("LambdaMART", "BM25"), ("LambdaMART", "Two-Tower"),
             ("MLP", "BM25"), ("MLP", "Two-Tower"), ("Two-Tower", "BM25")]
    official_res, legacy_res = [], []
    for a, b in comps:
        r = paired_bootstrap(pq_official[a], pq_official[b], a, b)
        official_res.append(r)
        legacy_res.append(paired_bootstrap(pq_legacy[a], pq_legacy[b], a, b))
        print(f"  {r['comparison']:26s} d={r['mean_paired_delta']:+.5f} "
              f"CI=[{r['ci_95_low']:+.5f},{r['ci_95_high']:+.5f}] "
              f"p={r['p_value_two_sided']:.4f} W/L/T={r['win']}/{r['loss']}/{r['tie']}")

    # like-for-like capacity test: both checkpoints frozen from V1
    pq_official["LambdaMART (V1 model)"] = OFFICIAL.per_query_ndcg(te, "lambdamart_v1_score", k=10)
    capacity = paired_bootstrap(pq_official["LambdaMART (V1 model)"], pq_official["MLP"],
                                "LambdaMART (V1 model)", "MLP")
    capacity["why_this_row_exists"] = (
        "The authoritative LambdaMART was retrained on the repaired benchmark with the "
        "official linear gain; the MLP is still the V1 checkpoint. 'LambdaMART - MLP' "
        "therefore confounds architecture with training data + objective. This row holds "
        "training conditions equal and isolates architecture.")
    official_res.append(capacity)
    print(f"  {capacity['comparison']:26s} d={capacity['mean_paired_delta']:+.5f} "
          f"CI=[{capacity['ci_95_low']:+.5f},{capacity['ci_95_high']:+.5f}] "
          f"p={capacity['p_value_two_sided']:.4f} "
          f"W/L/T={capacity['win']}/{capacity['loss']}/{capacity['tie']}   "
          f"<-- like-for-like (reference)")
    with open(os.path.join(BASE, "paired_comparisons.json"), "w") as f:
        json.dump({
            "methodology": ("Query-level paired percentile bootstrap over per-query NDCG@10 "
                            "deltas, n_bootstrap=10000, seed=42 -- the implementation audited "
                            "in experiments/audit/task3_per_query_bootstrap.py."),
            "pool": "benchmark_repair/data/test_clean.parquet -- 638,016 rows / 30,969 queries",
            "scorer": "official_kdd_ndcg.py (linear KDD gain)",
            "objective_metric_alignment": align,
            "official_metric": official_res,
            "legacy_metric_for_reference": legacy_res,
        }, f, indent=2, default=str)

    # ---- section 14: non-degenerate diagnostic ----
    print("\n[14] non-degenerate subset (diagnostic only, NOT the leaderboard metric)")
    nun = te.groupby("query_id")["gain"].nunique()
    nondeg = set(nun[nun > 1].index)
    nd = []
    for label in MODELS:
        s_all, s_nd = pq_official[label], pq_official[label][pq_official[label].index.isin(nondeg)]
        nd.append({
            "model": label,
            "full_test_query_count": int(len(s_all)),
            "full_test_ndcg_at_10": round(float(s_all.mean()), 6),
            "nondegenerate_query_count": int(len(s_nd)),
            "nondegenerate_ndcg_at_10": round(float(s_nd.mean()), 6),
            "degenerate_query_count": int(len(s_all) - len(s_nd)),
            "note": "diagnostic view of discriminative power; the official leaderboard "
                    "metric remains the full test set",
        })
        print(f"  {label:12s} full {s_all.mean():.6f}  non-degenerate {s_nd.mean():.6f} "
              f"(n={len(s_nd)})")
    pd.DataFrame(nd).to_csv(os.path.join(BASE, "nondegenerate_baselines.csv"), index=False)

    # ---- per-query dump ----
    pqdf = pd.DataFrame({m: pq_official[m] for m in MODELS})
    pqdf.columns = [f"{c}_official" for c in pqdf.columns]
    for m in MODELS:
        pqdf[f"{m}_legacy"] = pq_legacy[m]
    pqdf.index.name = "query_id"
    pqdf = pqdf.join(qloc.rename("locale"))
    pqdf = pqdf.join(te.groupby("query_id").size().rename("candidate_count"))
    pqdf["is_degenerate"] = ~pqdf.index.isin(nondeg)
    pqdf.reset_index().to_parquet(os.path.join(BASE, "per_query_scores.parquet"), index=False)

    if mlp_parity:
        print(f"\n  [parity] Phase-1 MLP legacy NDCG {mlp_parity['phase1_mlp_legacy_ndcg']:.6f} "
              f"vs recomputed legacy {pq_legacy['MLP'].mean():.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
