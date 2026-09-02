"""
Phases F, G, H, I -- model capacity comparison, per-query failure audit,
query-slice audit, and paired bootstrap.

Everything is computed on two pools so the reader can see how much the
candidate-pool defect matters:

  POOL_OFFICIAL : evaluation/evaluate_advanced.extract_test_advanced_features
                  output exactly as the repo's eval scripts use it
                  (682,233 rows, contains 44,217 duplicate rows).
  POOL_CLEAN    : the same frame restricted to rows whose product_locale
                  matches the ESCI judgment's locale, then de-duplicated on
                  (query_id, product_id). This equals the 638,016 judged pairs.

Bootstrap methodology is the same query-level paired percentile bootstrap
already used in experiments/audit/task3_per_query_bootstrap.py
(n_bootstrap=10000, seed=42).

No retraining. No modification of any pre-existing file.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from evaluation.metrics import dcg  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "ranking_v2", "audit")
CACHE = os.path.join(OUT, "_cache")

N_BOOTSTRAP = 10000
SEED = 42
FRACTIONAL = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

MODELS = {
    "bm25": "bm25_raw",
    "two_tower": "tt_raw",
    "mlp": "mlp_score",
    "lambdamart": "lambdamart_score",
}


def per_query_ndcg(df, score_col, rel_col="rel", k=10):
    out = {}
    for qid, g in df.groupby("query_id", sort=False):
        sg = g.sort_values(by=score_col, ascending=False)
        rel = sg[rel_col].values
        idcg = dcg(sorted(rel, reverse=True), k)
        if idcg > 0:
            out[qid] = dcg(rel, k) / idcg
    return pd.Series(out)


def top_k_sets(df, score_col, k=10):
    out = {}
    for qid, g in df.groupby("query_id", sort=False):
        out[qid] = tuple(g.sort_values(by=score_col, ascending=False)["product_id"].head(k))
    return out


def paired_bootstrap(a, b, name_a, name_b):
    """Query-level paired percentile bootstrap over per-query NDCG deltas."""
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
        f"{name_a}_mean_ndcg10": float(a.loc[common].mean()),
        f"{name_b}_mean_ndcg10": float(b.loc[common].mean()),
        "mean_paired_delta": float(d.mean()),
        "median_paired_delta": float(np.median(d)),
        "ci_95_low": float(lo),
        "ci_95_high": float(hi),
        "p_value_two_sided": p,
        "p_value_note": "percentile bootstrap; resolution floor is 1/10000 = 0.0001",
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
    print("Loading cached test frame ...")
    df = pd.read_parquet(os.path.join(CACHE, "test_scored.parquet"))
    df["rel"] = df["esci_label"].map(FRACTIONAL).fillna(0.0)

    ex = pd.read_parquet(os.path.join(
        ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"),
        columns=["query_id", "product_id", "product_locale", "split"])
    ex["query_id"] = ex["query_id"].astype(str)
    ex["product_id"] = ex["product_id"].astype(str)
    judged = ex[ex["split"] == "test"][["query_id", "product_id", "product_locale"]].drop_duplicates()
    judged = judged.rename(columns={"product_locale": "judged_locale"})

    # ---------------- POOL_CLEAN ----------------
    dfc = df.merge(judged, on=["query_id", "product_id"], how="left")
    locale_ok = dfc["product_locale"] == dfc["judged_locale"]
    clean = dfc[locale_ok].drop_duplicates(subset=["query_id", "product_id"], keep="first").copy()

    pool_info = {
        "POOL_OFFICIAL_rows": int(len(df)),
        "POOL_OFFICIAL_queries": int(df["query_id"].nunique()),
        "POOL_CLEAN_rows": int(len(clean)),
        "POOL_CLEAN_queries": int(clean["query_id"].nunique()),
        "judged_pairs_in_esci_test": int(len(judged)),
        "clean_pool_equals_judged_pairs": int(len(clean)) == int(len(judged)),
        "clean_pool_construction": (
            "rows of the official pool whose products-table locale matches the ESCI "
            "judgment locale, de-duplicated on (query_id, product_id). Model scores are "
            "the ones the saved checkpoints actually produced -- no re-scoring."),
    }
    print(json.dumps(pool_info, indent=1))

    # ---------------- per-query NDCG on both pools ----------------
    pq = {}
    for pool_name, frame in [("official", df), ("clean", clean)]:
        pq[pool_name] = {}
        for m, col in MODELS.items():
            pq[pool_name][m] = per_query_ndcg(frame, col, "rel", 10)
            print(f"  [{pool_name}] {m:12s} NDCG@10 = {pq[pool_name][m].mean():.4f} "
                  f"(n={len(pq[pool_name][m])})")

    # ---------------- model_comparison.csv ----------------
    rows = []
    for pool_name in ["official", "clean"]:
        for m in MODELS:
            s = pq[pool_name][m]
            rows.append({
                "pool": pool_name,
                "model": m,
                "n_queries": int(len(s)),
                "ndcg_at_10": float(s.mean()),
                "ndcg_at_10_median": float(s.median()),
                "ndcg_at_10_std": float(s.std()),
            })
    mc = pd.DataFrame(rows)

    # pairwise deltas appended to the same file
    pair_rows = []
    for pool_name in ["official", "clean"]:
        P = pq[pool_name]
        for a, b in [("lambdamart", "bm25"), ("lambdamart", "mlp"),
                     ("lambdamart", "two_tower"), ("mlp", "bm25"),
                     ("two_tower", "bm25")]:
            common = P[a].index.intersection(P[b].index)
            d = (P[a].loc[common] - P[b].loc[common])
            pair_rows.append({
                "pool": pool_name, "comparison": f"{a} - {b}",
                "n_queries": int(len(d)),
                "mean_delta": float(d.mean()), "median_delta": float(d.median()),
                "win": int((d > 1e-12).sum()), "loss": int((d < -1e-12).sum()),
                "tie": int((d.abs() <= 1e-12).sum()),
                "win_pct": round(100.0 * (d > 1e-12).mean(), 3),
                "loss_pct": round(100.0 * (d < -1e-12).mean(), 3),
                "tie_pct": round(100.0 * (d.abs() <= 1e-12).mean(), 3),
                "mean_delta_on_wins": float(d[d > 1e-12].mean()) if (d > 1e-12).any() else 0.0,
                "mean_delta_on_losses": float(d[d < -1e-12].mean()) if (d < -1e-12).any() else 0.0,
            })
    pairs_df = pd.DataFrame(pair_rows)

    # ---------------- F1: LambdaMART vs MLP top-10 agreement ----------------
    print("\nF1: top-10 agreement LambdaMART vs MLP (clean pool) ...")
    t_lm = top_k_sets(clean, "lambdamart_score", 10)
    t_mlp = top_k_sets(clean, "mlp_score", 10)
    t_bm = top_k_sets(clean, "bm25_raw", 10)
    jac, exact_same_order, same_set = [], 0, 0
    jac_bm = []
    for qid in t_lm:
        A, B = set(t_lm[qid]), set(t_mlp[qid])
        jac.append(len(A & B) / max(len(A | B), 1))
        if t_lm[qid] == t_mlp[qid]:
            exact_same_order += 1
        if A == B:
            same_set += 1
        C = set(t_bm[qid])
        jac_bm.append(len(A & C) / max(len(A | C), 1))
    nq = len(t_lm)
    f1 = {
        "n_queries": nq,
        "mean_top10_jaccard_lambdamart_vs_mlp": float(np.mean(jac)),
        "median_top10_jaccard_lambdamart_vs_mlp": float(np.median(jac)),
        "pct_queries_identical_top10_set": round(100.0 * same_set / nq, 3),
        "pct_queries_identical_top10_ordering": round(100.0 * exact_same_order / nq, 3),
        "mean_top10_jaccard_lambdamart_vs_bm25": float(np.mean(jac_bm)),
        "note": "Jaccard computed over top-10 product_id sets on POOL_CLEAN.",
    }
    print(json.dumps(f1, indent=1))

    # ---------------- G: per-query failure audit (clean pool) ----------------
    print("\nG: per-query failure audit ...")
    P = pq["clean"]
    grp = clean.groupby("query_id")
    lc = clean.groupby(["query_id", "esci_label"]).size().unstack(fill_value=0)
    for c in ["E", "S", "C", "I"]:
        if c not in lc.columns:
            lc[c] = 0
    meta = pd.DataFrame({
        "query": grp["query"].first(),
        "candidate_count": grp.size(),
        "query_length": grp["query_length"].first(),
        "query_mean_idf": grp["query_mean_idf"].first(),
        "locale": grp["product_locale"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else ""),
    }).join(lc[["E", "S", "C", "I"]].rename(
        columns={"E": "E_count", "S": "S_count", "C": "C_count", "I": "I_count"}))
    meta["relevant_count"] = meta["E_count"] + meta["S_count"] + meta["C_count"]

    pqdf = pd.DataFrame({
        "bm25_ndcg": P["bm25"], "tt_ndcg": P["two_tower"],
        "mlp_ndcg": P["mlp"], "lambdamart_ndcg": P["lambdamart"],
    }).join(meta, how="inner")
    pqdf["delta_vs_bm25"] = pqdf["lambdamart_ndcg"] - pqdf["bm25_ndcg"]
    pqdf["delta_vs_tt"] = pqdf["lambdamart_ndcg"] - pqdf["tt_ndcg"]
    pqdf["delta_vs_mlp"] = pqdf["lambdamart_ndcg"] - pqdf["mlp_ndcg"]
    pqdf = pqdf.reset_index().rename(columns={"index": "query_id"})

    # top-3 relevant products per query, for context
    top_rel = (clean[clean["esci_label"].isin(["E", "S"])]
               .sort_values(["query_id", "esci_label"])
               .groupby("query_id")
               .apply(lambda g: " | ".join(
                   f"{r.esci_label}:{str(r.product_title)[:60]}" for r in g.head(3).itertuples()),
                   include_groups=False))
    pqdf["top_relevant_products"] = pqdf["query_id"].map(top_rel).fillna("")

    order = ["query_id", "query", "locale", "candidate_count", "relevant_count",
             "E_count", "S_count", "C_count", "I_count", "query_length", "query_mean_idf",
             "bm25_ndcg", "tt_ndcg", "mlp_ndcg", "lambdamart_ndcg",
             "delta_vs_bm25", "delta_vs_tt", "delta_vs_mlp", "top_relevant_products"]
    pqdf = pqdf[order]
    pqdf.to_csv(os.path.join(OUT, "per_query_ndcg_clean_pool.csv"), index=False)

    non_deg = pqdf[pqdf["lambdamart_ndcg"] < 1.0]
    non_deg.nsmallest(200, "lambdamart_ndcg").to_csv(
        os.path.join(OUT, "worst_queries.csv"), index=False)
    pqdf.nsmallest(200, "delta_vs_bm25").to_csv(
        os.path.join(OUT, "largest_losses_vs_bm25.csv"), index=False)
    pqdf.nlargest(200, "delta_vs_bm25").to_csv(
        os.path.join(OUT, "largest_gains_vs_bm25.csv"), index=False)
    pqdf.nsmallest(200, "delta_vs_tt").to_csv(
        os.path.join(OUT, "largest_losses_vs_two_tower.csv"), index=False)

    # "semantic_score brings the largest benefit" -- proxy: TT beats BM25 most
    pqdf["tt_minus_bm25"] = pqdf["tt_ndcg"] - pqdf["bm25_ndcg"]
    pqdf.nlargest(200, "tt_minus_bm25").to_csv(
        os.path.join(OUT, "largest_semantic_gains.csv"), index=False)

    # ---------------- H: query slices ----------------
    print("\nH: query slice audit (reusing output/query_slices/query_level_metrics.csv rules) ...")
    slice_path = os.path.join(ROOT, "output/query_slices/query_level_metrics.csv")
    qlm = pd.read_csv(slice_path, dtype={"query_id": str})
    slice_cols = {
        "Brand": "brand_explicit",
        "Model/SKU": "model_like",
        "Exact Attribute": "has_exact_attribute",
        "Size": "has_size",
        "Color": "has_color",
        "Capacity/Storage": "has_capacity",
        "Quantity": "has_quantity",
        "Dimensions": "has_dimensions",
        "Lexical Specific (top-25% IDF)": "lexical_specific",
        "Semantic Intent": "semantic_intent",
        "Other": "other",
    }
    j = pqdf.merge(qlm[["query_id"] + list(slice_cols.values())], on="query_id", how="inner")
    print(f"  slice-annotated queries matched: {len(j)} (us-locale only, by construction)")

    srows = []

    def add_slice(name, sub, definition):
        if len(sub) == 0:
            return
        srows.append({
            "slice": name,
            "definition": definition,
            "query_count": int(len(sub)),
            "pct_of_us_test": round(100.0 * len(sub) / len(j), 3),
            "bm25_ndcg_at_10": float(sub["bm25_ndcg"].mean()),
            "two_tower_ndcg_at_10": float(sub["tt_ndcg"].mean()),
            "mlp_ndcg_at_10": float(sub["mlp_ndcg"].mean()),
            "lambdamart_ndcg_at_10": float(sub["lambdamart_ndcg"].mean()),
            "lambdamart_minus_bm25": float(sub["delta_vs_bm25"].mean()),
            "lambdamart_minus_two_tower": float(sub["delta_vs_tt"].mean()),
            "lambdamart_minus_mlp": float(sub["delta_vs_mlp"].mean()),
            "mean_candidate_count": float(sub["candidate_count"].mean()),
            "mean_query_length": float(sub["query_length"].mean()),
            "low_sample_size_lt_300": bool(len(sub) < 300),
        })

    add_slice("ALL (us locale)", j, "all us-locale test queries with slice annotations")
    for name, col in slice_cols.items():
        add_slice(name, j[j[col].astype(str).str.lower() == "true"],
                  f"pre-existing rule: {col} == True (scripts/build_query_slices.py)")
    # query-length slices (defined here, simple and interpretable)
    add_slice("Short query (1-2 tokens)", j[j["query_length"] <= 2], "query_length <= 2")
    add_slice("Medium query (3-4 tokens)", j[(j["query_length"] >= 3) & (j["query_length"] <= 4)],
              "3 <= query_length <= 4")
    add_slice("Long query (>=5 tokens)", j[j["query_length"] >= 5], "query_length >= 5")
    add_slice("Numeric-heavy query", j[j["query"].astype(str).str.contains(r"\d", regex=True, na=False)],
              "query contains at least one digit")
    add_slice("Non-degenerate (has >1 distinct label)",
              j[~((j["E_count"] > 0) & (j["S_count"] == 0) & (j["C_count"] == 0) & (j["I_count"] == 0))],
              "excludes all-E queries where NDCG@10 == 1.0 for any ranking")
    add_slice("Small candidate set (<10)", j[j["candidate_count"] < 10], "candidate_count < 10")
    add_slice("Large candidate set (>=40)", j[j["candidate_count"] >= 40], "candidate_count >= 40")

    sdf = pd.DataFrame(srows)
    sdf.to_csv(os.path.join(OUT, "query_slice_results.csv"), index=False)
    print(sdf[["slice", "query_count", "bm25_ndcg_at_10", "two_tower_ndcg_at_10",
               "lambdamart_ndcg_at_10", "lambdamart_minus_bm25",
               "lambdamart_minus_two_tower"]].to_string(index=False))

    # ---------------- I: bootstrap ----------------
    print("\nI: paired bootstrap ...")
    boot = {"methodology": (
        "Query-level paired percentile bootstrap over per-query NDCG@10 deltas, "
        "n_bootstrap=10000, seed=42 -- the same procedure as the pre-existing "
        "experiments/audit/task3_per_query_bootstrap.py."),
        "pools": pool_info, "results": {}}
    for pool_name in ["official", "clean"]:
        Pp = pq[pool_name]
        res = []
        for a, b in [("lambdamart", "bm25"), ("lambdamart", "mlp"),
                     ("lambdamart", "two_tower"), ("mlp", "bm25"),
                     ("mlp", "two_tower"), ("two_tower", "bm25")]:
            r = paired_bootstrap(Pp[a], Pp[b], a, b)
            res.append(r)
            print(f"  [{pool_name}] {r['comparison']:26s} "
                  f"delta={r['mean_paired_delta']:+.5f} "
                  f"CI=[{r['ci_95_low']:+.5f},{r['ci_95_high']:+.5f}] "
                  f"p={r['p_value_two_sided']:.4f} "
                  f"W/L/T={r['win']}/{r['loss']}/{r['tie']}")
        boot["results"][pool_name] = res
    with open(os.path.join(OUT, "bootstrap_results.json"), "w") as f:
        json.dump(boot, f, indent=2)

    # ---------------- write model_comparison.csv ----------------
    mc.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    pairs_df.to_csv(os.path.join(OUT, "model_comparison_pairwise.csv"), index=False)

    with open(os.path.join(OUT, "model_capacity_audit.json"), "w") as f:
        json.dump({"pool_info": pool_info, "F1_top10_agreement": f1,
                   "headline": mc.to_dict("records")}, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
