"""
Phase C -- Evaluation protocol audit.

Answers, with numbers, the 8 questions in the brief, quantifies the
candidate-pool row inflation introduced by the two locale-unaware joins in
evaluation/evaluate_advanced.extract_test_advanced_features, and reconciles
the 0.8188 vs 0.8155 BM25 discrepancy.

Reuses evaluation/metrics.dcg verbatim. Writes only into the audit dir.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from evaluation.metrics import dcg  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "ranking_v2", "audit")
CACHE = os.path.join(OUT, "_cache")

FRACTIONAL = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
ORDINAL = {"E": 3, "S": 2, "C": 1, "I": 0}


def per_query_ndcg(df, score_col, rel_col, k=10):
    """Identical algorithm to evaluation/metrics.ndcg_at_k, but returns the
    per-query series instead of the mean."""
    out = {}
    for qid, g in df.groupby("query_id", sort=False):
        sg = g.sort_values(by=score_col, ascending=False)
        rel = sg[rel_col].values
        idcg = dcg(sorted(rel, reverse=True), k)
        if idcg > 0:
            out[qid] = dcg(rel, k) / idcg
    return pd.Series(out)


def main():
    rep = {}
    df = pd.read_parquet(os.path.join(CACHE, "test_scored.parquet"))
    ex = pd.read_parquet(os.path.join(
        ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"))
    ex["query_id"] = ex["query_id"].astype(str)
    ex["product_id"] = ex["product_id"].astype(str)
    ex_test = ex[ex["split"] == "test"]

    # ------------------------------------------------------------------
    # C0. Candidate-pool integrity: how many rows are duplicates?
    # ------------------------------------------------------------------
    n_rows = len(df)
    n_pairs = len(df[["query_id", "product_id"]].drop_duplicates())
    truth_pairs = len(ex_test[["query_id", "product_id"]].drop_duplicates())

    dup_counts = df.groupby(["query_id", "product_id"]).size()
    pr = pd.read_parquet(os.path.join(
        ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"),
        columns=["product_id", "product_locale"])
    pr["product_id"] = pr["product_id"].astype(str)
    esci_s = pd.read_parquet(os.path.join(
        ROOT, "esci-data/esci-s_dataset/esci_s_products.parquet"))
    id_col = "asin" if "asin" in esci_s.columns else esci_s.columns[0]
    esci_s[id_col] = esci_s[id_col].astype(str)

    rep["C0_candidate_pool_integrity"] = {
        "official_test_pool_rows": int(n_rows),
        "unique_query_product_pairs_in_pool": int(n_pairs),
        "true_judged_query_product_pairs_in_esci_test": int(truth_pairs),
        "duplicated_rows": int(n_rows - n_pairs),
        "duplication_rate_pct": round(100.0 * (n_rows - n_pairs) / n_pairs, 4),
        "max_copies_of_a_single_pair": int(dup_counts.max()),
        "pairs_appearing_more_than_once": int((dup_counts > 1).sum()),
        "queries_containing_at_least_one_duplicated_pair": int(
            dup_counts[dup_counts > 1].reset_index()["query_id"].nunique()),
        "cause_1_products_table_joined_on_product_id_only": {
            "product_ids_present_in_multiple_locales": int(
                (pr.groupby("product_id")["product_locale"].nunique() > 1).sum()),
            "code_ref": "evaluation/evaluate_advanced.py:62  pd.merge(df, df_pr, on='product_id')",
        },
        "cause_2_esci_s_table_joined_on_product_id_only": {
            "esci_s_rows": int(len(esci_s)),
            "esci_s_unique_ids": int(esci_s[id_col].nunique()),
            "esci_s_duplicate_id_rows": int(len(esci_s) - esci_s[id_col].nunique()),
            "code_ref": "evaluation/evaluate_advanced.py:42  pd.merge(df_pr, df_esci_s[...], on='product_id')",
        },
        "effect_on_ndcg": (
            "A duplicated (query, product) row is counted twice in both the ranked list "
            "and the ideal list, so NDCG@10 is computed over a candidate set that does "
            "not match the ESCI judgment set. Affected queries get a distorted (usually "
            "inflated, because duplicated relevant items fill adjacent top-10 slots) score."
        ),
    }

    # ------------------------------------------------------------------
    # C1-C8. Protocol questions
    # ------------------------------------------------------------------
    cpq = df.groupby("query_id").size()
    rel_std = df["relevance_standard"]
    idcg_pos = df.groupby("query_id")["relevance_standard"].max()

    # tie structure of each scorer
    tie_stats = {}
    for col in ["bm25_raw", "tt_raw", "lambdamart_score", "mlp_score"]:
        n_unique = df.groupby("query_id")[col].nunique()
        grp_size = df.groupby(["query_id", col])[col].transform("size")
        tie_stats[col] = {
            "rows_total": int(len(df)),
            "rows_sharing_their_score_with_another_row_in_the_same_query": int((grp_size > 1).sum()),
            "pct_rows_tied": round(100.0 * (grp_size > 1).sum() / len(df), 3),
            "queries_where_all_scores_distinct": int((n_unique == cpq).sum()),
            "mean_distinct_scores_per_query": float(n_unique.mean()),
            "mean_candidates_per_query": float(cpq.mean()),
        }

    rep["C_protocol_answers"] = {
        "C1_grouped_by_query": (
            "YES. evaluation/metrics.ndcg_at_k iterates df.groupby('query_id'), computes "
            "NDCG@10 per query and returns the unweighted arithmetic mean."),
        "C2_relevance_gain": (
            "The eval gain is 2**relevance - 1 with relevance = {E:1.0, S:0.1, C:0.01, I:0.0} "
            "-> effective gains [E=1.0, S=0.07177, C=0.00696, I=0.0]. This is NOT the ESCI "
            "standard [7,3,1,0], and NOT the [0,1,3,7] label_gain LambdaMART is trained on."),
        "C3_fewer_than_10_candidates": {
            "handling": "dcg() slices [:k]; shorter lists just use every candidate. Both DCG "
                        "and IDCG use the same short list, so NDCG is well-defined and is "
                        "frequently exactly 1.0.",
            "test_queries_with_lt_10_candidates": int((cpq < 10).sum()),
            "pct": round(100.0 * (cpq < 10).sum() / len(cpq), 3),
        },
        "C4_ties": {
            "handling": "pandas sort_values(ascending=False) is a stable mergesort-free quicksort "
                        "by default ('quicksort'); tied scores are therefore broken by an "
                        "UNSPECIFIED order that depends on row order in the frame, not by label. "
                        "No explicit tie-breaking rule and no expected-NDCG-over-ties correction.",
            "per_scorer": tie_stats,
        },
        "C5_equal_query_weight": "YES -- np.mean over per-query NDCG, so every query counts equally.",
        "C6_queries_filtered_out": {
            "rule": "ndcg_at_k skips any query whose IDCG@10 == 0 (i.e. every candidate is I / "
                    "relevance 0). Those queries are dropped from the mean entirely.",
            "test_queries_in_pool": int(len(cpq)),
            "test_queries_with_idcg_zero_dropped": int((idcg_pos <= 0).sum()),
            "test_queries_scored": int((idcg_pos > 0).sum()),
        },
        "C7_same_query_set": None,   # filled below
        "C8_same_candidate_set": None,
    }

    # ------------------------------------------------------------------
    # Degenerate queries: NDCG is 1.0 for ANY ranking
    # ------------------------------------------------------------------
    def degenerate_mask(frame, rel_col):
        # a query is degenerate at k=10 if every ranking yields NDCG=1.0
        nun = frame.groupby("query_id")[rel_col].nunique()
        return nun <= 1

    deg_std = degenerate_mask(df, "relevance_standard")
    # also: queries where the top-10 is forced (candidates <= 10 AND at most one
    # distinct positive level) -- covered by nunique<=1; plus queries where all
    # non-zero-gain items already fit in top 10 regardless of order
    rep["C_degenerate_queries"] = {
        "definition": "every candidate carries the same relevance value -> NDCG@10 == 1.0 for any ranking",
        "count": int(deg_std.sum()),
        "pct_of_scored_queries": round(100.0 * deg_std.sum() / len(deg_std), 3),
        "impact": (
            "These queries contribute a constant 1.0 to every model's mean NDCG@10 and "
            "carry zero discriminative signal, compressing all reported model gaps."),
    }

    # ------------------------------------------------------------------
    # Gain-convention sensitivity: same rankings, three gain conventions
    # ------------------------------------------------------------------
    print("Computing NDCG under 3 gain conventions ...")
    df = df.copy()
    df["rel_fractional"] = df["esci_label"].map(FRACTIONAL).fillna(0.0)
    df["rel_ordinal"] = df["esci_label"].map(ORDINAL).fillna(0).astype(float)
    # binary E-vs-rest
    df["rel_binary"] = (df["esci_label"] == "E").astype(float)

    scorers = {
        "bm25": "bm25_raw",
        "two_tower": "tt_raw",
        "mlp_17feature": "mlp_score",
        "lambdamart_17feature": "lambdamart_score",
    }
    conventions = {
        "official_repo_fractional_2^r-1": "rel_fractional",
        "esci_standard_ordinal_2^r-1_[0,1,3,7]": "rel_ordinal",
        "binary_E_vs_rest": "rel_binary",
    }

    rows = []
    per_query_store = {}
    for cname, rcol in conventions.items():
        for sname, scol in scorers.items():
            s = per_query_ndcg(df, scol, rcol, k=10)
            rows.append({
                "gain_convention": cname,
                "model": sname,
                "n_queries_scored": int(len(s)),
                "ndcg_at_10": float(s.mean()),
            })
            if cname == "official_repo_fractional_2^r-1":
                per_query_store[sname] = s
            print(f"  {cname:38s} {sname:22s} {s.mean():.4f}  (n={len(s)})")
    gain_df = pd.DataFrame(rows)
    gain_df.to_csv(os.path.join(OUT, "eval_gain_convention_sensitivity.csv"), index=False)

    # non-degenerate-only view under the official convention
    nondeg_ids = set(deg_std[~deg_std].index)
    nd_rows = []
    for sname, s in per_query_store.items():
        sub = s[s.index.isin(nondeg_ids)]
        nd_rows.append({
            "gain_convention": "official_repo_fractional_2^r-1",
            "subset": "non_degenerate_queries_only",
            "model": sname,
            "n_queries_scored": int(len(sub)),
            "ndcg_at_10": float(sub.mean()),
        })
    pd.DataFrame(nd_rows).to_csv(os.path.join(OUT, "eval_nondegenerate_subset.csv"), index=False)
    rep["C_nondegenerate_subset"] = nd_rows

    # ------------------------------------------------------------------
    # C7 / C8: are all four models on the same query & candidate set?
    # ------------------------------------------------------------------
    qsets = {k: set(v.index) for k, v in per_query_store.items()}
    all_same_q = all(qsets["bm25"] == v for v in qsets.values())
    rep["C_protocol_answers"]["C7_same_query_set"] = {
        "verdict": "YES on this recomputation" if all_same_q else "NO",
        "n_queries": {k: len(v) for k, v in qsets.items()},
        "caveat": (
            "In this audit all four scorers were evaluated on one shared frame, so they "
            "are trivially aligned. The REPO's own scripts are NOT aligned: "
            "evaluate_lambdamart.py / evaluate_advanced.py use the inflated "
            "extract_test_advanced_features pool (682,233 rows), while "
            "scripts/build_query_slices.py scores BM25/Two-Tower on the clean "
            "bm25_scores_test.csv x ESCI-labels pool (638,016 rows) and additionally "
            "filters to product_locale=='us'."),
    }
    rep["C_protocol_answers"]["C8_same_candidate_set"] = rep["C_protocol_answers"]["C7_same_query_set"]

    # ------------------------------------------------------------------
    # Reconcile 0.8188 vs 0.8155 BM25
    # ------------------------------------------------------------------
    print("Reconciling BM25 baselines ...")
    bm = pd.read_csv(os.path.join(ROOT, "output/bm25_scores_test.csv"))
    bm.columns = ["query_id", "product_id", "bm25_score"]
    bm["query_id"] = bm["query_id"].astype(str)
    bm["product_id"] = bm["product_id"].astype(str)
    truth = ex_test[["query_id", "product_id", "esci_label", "product_locale"]].drop_duplicates()
    clean = bm.merge(truth, on=["query_id", "product_id"], how="inner")
    clean["rel_fractional"] = clean["esci_label"].map(FRACTIONAL).fillna(0.0)

    s_clean = per_query_ndcg(clean, "bm25_score", "rel_fractional", 10)
    s_clean_us = per_query_ndcg(clean[clean["product_locale"] == "us"], "bm25_score", "rel_fractional", 10)
    s_infl = per_query_store["bm25"]

    tt = pd.read_csv(os.path.join(ROOT, "output/two_tower_scores_test.csv"))
    tt.columns = ["query_id", "product_id", "tt_score"]
    tt["query_id"] = tt["query_id"].astype(str)
    tt["product_id"] = tt["product_id"].astype(str)
    cleantt = tt.merge(truth, on=["query_id", "product_id"], how="inner")
    cleantt["rel_fractional"] = cleantt["esci_label"].map(FRACTIONAL).fillna(0.0)
    s_tt_clean = per_query_ndcg(cleantt, "tt_score", "rel_fractional", 10)

    rep["C_bm25_baseline_reconciliation"] = {
        "0.8188_all_locale_clean_pool": {
            "value": float(s_clean.mean()), "n_queries": int(len(s_clean)), "rows": int(len(clean)),
            "definition": "bm25_scores_test.csv INNER JOIN ESCI test labels, no products-table join",
            "matches_recorded_0.8188": abs(s_clean.mean() - 0.8188130279308162) < 1e-6,
        },
        "0.8155_inflated_pool_used_by_lambdamart_eval": {
            "value": float(s_infl.mean()), "n_queries": int(len(s_infl)), "rows": int(n_rows),
            "definition": "extract_test_advanced_features pool (duplicated rows from the two "
                          "locale-unaware joins)",
        },
        "us_locale_only_clean_pool": {
            "value": float(s_clean_us.mean()), "n_queries": int(len(s_clean_us)),
            "matches_recorded_0.8434": abs(s_clean_us.mean() - 0.8433611370662611) < 1e-6,
        },
        "two_tower_all_locale_clean_pool": {
            "value": float(s_tt_clean.mean()), "n_queries": int(len(s_tt_clean)),
            "matches_recorded_0.8267": abs(s_tt_clean.mean() - 0.8267167793233932) < 1e-6,
        },
        "root_cause": (
            "The gap is entirely candidate-pool construction, not the metric. The 0.8188 "
            "number is on the clean judged pool; 0.8155 is on the duplicate-inflated pool "
            "that the LambdaMART/MLP evaluation uses. All headline model numbers "
            "(0.8458 MLP, 0.8464 LambdaMART) live on the inflated pool, so the commonly "
            "quoted 'LambdaMART 0.8464 vs BM25 0.8188' comparison mixes two pools."),
    }

    with open(os.path.join(OUT, "evaluation_audit.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)

    # dump per-query official-convention NDCG for later phases
    pq = pd.DataFrame(per_query_store)
    pq.index.name = "query_id"
    pq.reset_index().to_parquet(os.path.join(CACHE, "per_query_ndcg_official.parquet"), index=False)
    print("\nWrote evaluation_audit.json, eval_gain_convention_sensitivity.csv, eval_nondegenerate_subset.csv")


if __name__ == "__main__":
    main()
