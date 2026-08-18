"""
Top-10 Overlap & Rank Agreement Analysis: BM25 vs 17-feature MLP vs LambdaMART
on the same ESCI fixed-candidate test benchmark.

No retraining. Reuses the existing extract_test_advanced_features() feature
extraction (shared by evaluate_advanced.py and evaluate_lambdamart.py) ONCE,
so BM25/MLP/LambdaMART are guaranteed to see identical query_ids and identical
candidate products per query. MLP and LambdaMART scores come from loading the
existing trained checkpoints (best_advanced_reranker.pth, lambdamart_model.txt)
and running inference only -- no gradient step, no fitting.

Usage:
    python scripts/analyze_topk_overlap.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PRODUCTS_PATH, ROOT_DIR
from evaluation.evaluate_advanced import extract_test_advanced_features
from reranking.advanced_model import AdvancedDeepReranker

OUT_DIR = f"{ROOT_DIR}/output/ranking_overlap"
TOPN = 10
MIN_CANDIDATES = 10
MIN_SHARED_FOR_SPEARMAN = 3
MLP_NDCG_AT_10 = 0.8458
LGB_NDCG_AT_10 = 0.8464


def load_scores():
    """Loads the shared candidate-pool dataframe once and attaches BM25 /
    MLP / LambdaMART scores as columns on that SAME dataframe -- guaranteeing
    identical query_ids, candidate products, dtypes, and no drift between
    the three "models" being compared."""
    print("=== Loading shared test candidate pool (no retraining) ===")
    with open(f"{ROOT_DIR}/output/advanced_normalization_stats.json") as f:
        stats = json.load(f)
    feature_cols = stats["features"]
    train_mean, train_std = np.array(stats["mean"]), np.array(stats["std"])
    idf_map = stats.get("idf_map", {})

    with open(f"{ROOT_DIR}/output/lambdamart_features.json") as f:
        lgb_meta = json.load(f)
    lgb_feature_cols = lgb_meta["features"]
    assert lgb_feature_cols == feature_cols, "MLP and LambdaMART were not trained on the same feature list/order"

    df = extract_test_advanced_features(idf_map)
    df = df.dropna(subset=feature_cols)
    df['query_id'] = df['query_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)
    print(f"Shared candidate pool: {len(df)} rows, {df['query_id'].nunique()} queries")

    # DISCOVERED, PRE-EXISTING ISSUE (not introduced today, not fixed today):
    # extract_test_advanced_features() merges candidates with the products table
    # on product_id ONLY (not product_id+product_locale). ~11,567 product_ids
    # exist in more than one locale in the catalog, so those rows fan out into
    # duplicates here. This affects the official evaluate_advanced.py /
    # evaluate_lambdamart.py NDCG@10 numbers too, since they call the exact same
    # function -- it is out of scope to fix the shared function today (that
    # would touch the official benchmark's feature engineering, which the task
    # explicitly says not to do). For THIS diagnostic only, dedupe locally so
    # Top-10 membership is well-defined.
    dup = df.duplicated(subset=['query_id', 'product_id']).sum()
    if dup:
        print(f"NOTE: {dup} duplicate (query_id, product_id) rows found (pre-existing, from the "
              f"product_id-only merge fanning out across the ~11,567 multi-locale product_ids in the "
              f"catalog). Deduping locally (keep first) for this diagnostic; not modifying the shared "
              f"extract_test_advanced_features() function used by the official benchmark.")
        df = df.drop_duplicates(subset=['query_id', 'product_id'], keep='first')

    features_raw = df[feature_cols].values
    features_norm = (features_raw - train_mean) / train_std

    print("Running MLP inference (loading existing checkpoint, no training)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = AdvancedDeepReranker(input_dim=len(feature_cols)).to(device)
    mlp.load_state_dict(torch.load(f"{ROOT_DIR}/output/best_advanced_reranker.pth", map_location=device, weights_only=True))
    mlp.eval()
    with torch.no_grad():
        x = torch.tensor(features_norm, dtype=torch.float32).to(device)
        df['mlp_score'] = mlp(x).cpu().squeeze().numpy()

    print("Running LambdaMART inference (loading existing checkpoint, no training)...")
    booster = lgb.Booster(model_file=f"{ROOT_DIR}/output/lambdamart_model.txt")
    df['lgb_score'] = booster.predict(features_raw)

    df = df.rename(columns={'bm25_score': 'bm25_score_col'})  # avoid clobbering during later ops; keep original name too
    df['bm25_score_col'] = df['bm25_score_col'].astype(float)

    return df[['query_id', 'product_id', 'esci_label', 'bm25_score_col', 'mlp_score', 'lgb_score']].rename(
        columns={'bm25_score_col': 'bm25_score'}
    )


def build_top10(df, score_col):
    """Returns {query_id: [(product_id, rank, score, label), ...]} Top-10 per
    query, restricted to queries with >= MIN_CANDIDATES total candidates."""
    lookup = {}
    for qid, group in df.groupby('query_id'):
        if len(group) < MIN_CANDIDATES:
            continue
        top = group.sort_values(score_col, ascending=False).head(TOPN)
        lookup[qid] = list(zip(top['product_id'], range(1, len(top) + 1), top[score_col], top['esci_label']))
    return lookup


def pair_overlap(top_a, top_b):
    """Per-query overlap/jaccard/spearman/position-shift/membership-change
    for one (model_a, model_b) pair. Only queries present in both."""
    rows = []
    all_shifts = []
    promoted_labels, dropped_labels = [], []
    for qid in set(top_a) & set(top_b):
        a_items = top_a[qid]
        b_items = top_b[qid]
        a_ids = [x[0] for x in a_items]
        b_ids = [x[0] for x in b_items]
        a_rank = {x[0]: x[1] for x in a_items}
        b_rank = {x[0]: x[1] for x in b_items}
        a_set, b_set = set(a_ids), set(b_ids)
        shared = a_set & b_set
        union = a_set | b_set

        overlap_count = len(shared)
        overlap_rate = overlap_count / TOPN
        jaccard = overlap_count / len(union) if union else np.nan

        if len(shared) >= MIN_SHARED_FOR_SPEARMAN:
            ranks_a = [a_rank[pid] for pid in shared]
            ranks_b = [b_rank[pid] for pid in shared]
            rho, _ = spearmanr(ranks_a, ranks_b)
            shifts = [abs(a_rank[pid] - b_rank[pid]) for pid in shared]
            all_shifts.extend(shifts)
            mean_shift = float(np.mean(shifts))
        else:
            rho = np.nan
            mean_shift = np.nan

        promoted = b_set - a_set  # in B (top10), not in A (top10)
        dropped = a_set - b_set   # in A (top10), not in B (top10)
        b_labels = {x[0]: x[3] for x in b_items}
        a_labels = {x[0]: x[3] for x in a_items}
        for pid in promoted:
            promoted_labels.append(b_labels[pid])
        for pid in dropped:
            dropped_labels.append(a_labels[pid])

        rows.append({
            "query_id": qid, "overlap_count": overlap_count, "overlap_rate": overlap_rate,
            "jaccard": jaccard, "spearman_rho": rho, "mean_rank_shift": mean_shift,
            "shared_count": len(shared), "n_promoted": len(promoted), "n_dropped": len(dropped),
        })
    return pd.DataFrame(rows), all_shifts, promoted_labels, dropped_labels


def summarize_pair(name, df_pair, all_shifts):
    n = len(df_pair)
    summary = {
        "pair": name, "n_queries": n,
        "mean_overlap_count": df_pair['overlap_count'].mean(), "median_overlap_count": df_pair['overlap_count'].median(),
        "p25_overlap_count": df_pair['overlap_count'].quantile(0.25), "p75_overlap_count": df_pair['overlap_count'].quantile(0.75),
        "min_overlap_count": df_pair['overlap_count'].min(), "max_overlap_count": df_pair['overlap_count'].max(),
        "mean_overlap_rate": df_pair['overlap_rate'].mean(),
        "mean_jaccard": df_pair['jaccard'].mean(), "median_jaccard": df_pair['jaccard'].median(),
        "pct_exact_same_top10": 100 * (df_pair['overlap_count'] == 10).mean(),
        "pct_overlap_ge_8": 100 * (df_pair['overlap_count'] >= 8).mean(),
        "pct_overlap_le_5": 100 * (df_pair['overlap_count'] <= 5).mean(),
    }
    valid_rho = df_pair['spearman_rho'].dropna()
    summary["n_queries_with_spearman"] = len(valid_rho)
    summary["mean_spearman"] = valid_rho.mean() if len(valid_rho) else np.nan
    summary["median_spearman"] = valid_rho.median() if len(valid_rho) else np.nan
    summary["pct_negative_spearman"] = 100 * (valid_rho < 0).mean() if len(valid_rho) else np.nan
    summary["pct_spearman_gt_0.8"] = 100 * (valid_rho > 0.8).mean() if len(valid_rho) else np.nan
    if all_shifts:
        arr = np.array(all_shifts)
        summary["mean_abs_rank_shift"] = float(arr.mean())
        summary["median_abs_rank_shift"] = float(np.median(arr))
        summary["p90_abs_rank_shift"] = float(np.percentile(arr, 90))
    else:
        summary["mean_abs_rank_shift"] = summary["median_abs_rank_shift"] = summary["p90_abs_rank_shift"] = np.nan
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_scores()

    print("\n=== Sanity checks ===")
    checks = {}
    n_before = df['query_id'].nunique()
    qualifying = df.groupby('query_id').size()
    qualifying = set(qualifying[qualifying >= MIN_CANDIDATES].index)
    print(f"  Queries with >= {MIN_CANDIDATES} candidates: {len(qualifying)} / {n_before} total")

    bm25_top10 = build_top10(df, 'bm25_score')
    mlp_top10 = build_top10(df, 'mlp_score')
    lgb_top10 = build_top10(df, 'lgb_score')

    checks['same_query_set'] = set(bm25_top10) == set(mlp_top10) == set(lgb_top10) == qualifying
    checks['topn_le_10'] = all(len(v) <= 10 for v in list(bm25_top10.values()) + list(mlp_top10.values()) + list(lgb_top10.values()))
    checks['no_dup_within_top10'] = all(
        len(set(x[0] for x in v)) == len(v) for v in list(bm25_top10.values()) + list(mlp_top10.values()) + list(lgb_top10.values())
    )
    checks['no_retraining'] = True  # only torch.load / lgb.Booster(model_file=...) used; zero .fit()/.backward() calls in this script
    for name, ok in checks.items():
        print(f"  {name}: {'OK' if ok else 'FAIL'}")
        assert ok, f"Sanity check failed: {name}"

    # Save Top-10 predictions (all three models) for transparency
    top10_records = []
    for model_name, lookup in [("BM25", bm25_top10), ("MLP", mlp_top10), ("LambdaMART", lgb_top10)]:
        for qid, items in lookup.items():
            for pid, rank, score, label in items:
                top10_records.append({"query_id": qid, "model": model_name, "product_id": pid,
                                       "rank": rank, "score": score, "esci_label": label})
    top10_df = pd.DataFrame(top10_records)
    top10_df.to_parquet(f"{OUT_DIR}/top10_predictions.parquet", index=False)

    print("\n=== Computing pairwise overlap ===")
    pairs = {
        "BM25 vs MLP": (bm25_top10, mlp_top10),
        "BM25 vs LambdaMART": (bm25_top10, lgb_top10),
        "MLP vs LambdaMART": (mlp_top10, lgb_top10),
    }
    pair_dfs, summaries, shift_records, membership_records = {}, [], [], []
    for name, (a, b) in pairs.items():
        df_pair, shifts, promoted_labels, dropped_labels = pair_overlap(a, b)
        df_pair['pair'] = name
        pair_dfs[name] = df_pair
        summaries.append(summarize_pair(name, df_pair, shifts))

        for lbl in promoted_labels:
            membership_records.append({"pair": name, "direction": "promoted_into_top10", "esci_label": lbl})
        for lbl in dropped_labels:
            membership_records.append({"pair": name, "direction": "dropped_from_top10", "esci_label": lbl})

    query_level_overlap = pd.concat(pair_dfs.values(), ignore_index=True)
    query_level_overlap.to_parquet(f"{OUT_DIR}/query_level_overlap.parquet", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(f"{OUT_DIR}/top10_overlap_summary.csv", index=False)

    md_lines = ["| Pair | Mean overlap /10 | Median | Mean Jaccard | % exact same Top10 | % overlap >=8 |",
                "|---|---:|---:|---:|---:|---:|"]
    for _, r in summary_df.iterrows():
        md_lines.append(f"| {r['pair']} | {r['mean_overlap_count']:.2f} | {r['median_overlap_count']:.0f} | "
                         f"{r['mean_jaccard']:.3f} | {r['pct_exact_same_top10']:.1f}% | {r['pct_overlap_ge_8']:.1f}% |")
    with open(f"{OUT_DIR}/top10_overlap_summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))

    rank_agreement_cols = ["pair", "n_queries", "n_queries_with_spearman", "mean_spearman", "median_spearman",
                            "pct_negative_spearman", "pct_spearman_gt_0.8",
                            "mean_abs_rank_shift", "median_abs_rank_shift", "p90_abs_rank_shift"]
    summary_df[rank_agreement_cols].to_csv(f"{OUT_DIR}/rank_agreement_summary.csv", index=False)
    print("\n=== Rank agreement (shared Top-10 items only) ===")
    print(summary_df[rank_agreement_cols].to_string(index=False))

    membership_df = pd.DataFrame(membership_records)
    label_dist = membership_df.groupby(['pair', 'direction', 'esci_label']).size().rename('count').reset_index()
    label_dist['pct_within_group'] = label_dist.groupby(['pair', 'direction'])['count'].transform(lambda x: 100 * x / x.sum())
    label_dist.to_csv(f"{OUT_DIR}/membership_change_label_distribution.csv", index=False)
    print("\n=== Membership-change label distribution ===")
    print(label_dist.to_string(index=False))

    # ============ Representative examples ============
    print("\n=== Building representative examples ===")
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_pr['product_id'] = df_pr['product_id'].astype(str)
    title_lookup = df_pr.drop_duplicates('product_id').set_index('product_id')['product_title']

    from config import EXAMPLES_PATH
    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    query_text_lookup = df_ex[df_ex['split'] == 'test'][['query_id', 'query']].drop_duplicates('query_id').set_index('query_id')['query']

    examples = []
    rng = np.random.RandomState(42)

    # Case A: BM25 vs MLP overlap <= 5
    caseA = pair_dfs["BM25 vs MLP"][pair_dfs["BM25 vs MLP"]['overlap_count'] <= 5]
    caseA_qids = rng.choice(caseA['query_id'].unique(), size=min(10, caseA['query_id'].nunique()), replace=False) if len(caseA) else []

    # Case B: MLP vs LambdaMART overlap >= 9 but rank order differs (spearman < 0.5 or mean_rank_shift notable)
    caseB_pool = pair_dfs["MLP vs LambdaMART"][(pair_dfs["MLP vs LambdaMART"]['overlap_count'] >= 9) &
                                                 (pair_dfs["MLP vs LambdaMART"]['spearman_rho'] < 0.9)]
    caseB_qids = rng.choice(caseB_pool['query_id'].unique(), size=min(10, caseB_pool['query_id'].nunique()), replace=False) if len(caseB_pool) else []

    # Case C: MLP promoted an E item and dropped a C/I item (BM25 -> MLP)
    caseC_qids = []
    for qid in set(bm25_top10) & set(mlp_top10):
        a_ids = {x[0]: x[3] for x in bm25_top10[qid]}
        b_ids = {x[0]: x[3] for x in mlp_top10[qid]}
        promoted_e = [pid for pid in (set(b_ids) - set(a_ids)) if b_ids[pid] == 'E']
        dropped_ci = [pid for pid in (set(a_ids) - set(b_ids)) if a_ids[pid] in ('C', 'I')]
        if promoted_e and dropped_ci:
            caseC_qids.append(qid)
    caseC_qids = rng.choice(caseC_qids, size=min(10, len(caseC_qids)), replace=False) if caseC_qids else []

    def dump_case(case_name, qids, model_lookups):
        for qid in qids:
            qtext = query_text_lookup.get(qid, "")
            for model_name, lookup in model_lookups:
                if qid not in lookup:
                    continue
                for pid, rank, score, label in lookup[qid]:
                    examples.append({
                        "case": case_name, "query_id": qid, "query": qtext, "model": model_name,
                        "product_id": pid, "rank": rank, "score": score, "esci_label": label,
                        "product_title": title_lookup.get(pid, ""),
                    })

    dump_case("A_bm25_vs_mlp_low_overlap", caseA_qids, [("BM25", bm25_top10), ("MLP", mlp_top10)])
    dump_case("B_mlp_vs_lgb_high_overlap_reordered", caseB_qids, [("MLP", mlp_top10), ("LambdaMART", lgb_top10)])
    dump_case("C_mlp_promotes_E_drops_CI", caseC_qids, [("BM25", bm25_top10), ("MLP", mlp_top10)])

    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(f"{OUT_DIR}/representative_examples.csv", index=False)
    print(f"Saved {OUT_DIR}/representative_examples.csv "
          f"(Case A: {len(caseA_qids)} queries, Case B: {len(caseB_qids)} queries, Case C: {len(caseC_qids)} queries)")

    # ============ Final answers ============
    print("\n=== Final answers ===")
    for _, r in summary_df.iterrows():
        print(f"  {r['pair']}: mean overlap {r['mean_overlap_count']:.2f}/10, mean Jaccard {r['mean_jaccard']:.3f}, "
              f"mean Spearman {r['mean_spearman']:.3f} (n={r['n_queries_with_spearman']})")
    print(f"\n  NDCG@10 context (unchanged, for reference only): MLP={MLP_NDCG_AT_10}, LambdaMART={LGB_NDCG_AT_10}")
    print("  Top-10 overlap is a diagnostic of ranking behavior, NOT a performance metric; NDCG@10 remains the relevance-quality metric.")

    return summary_df


if __name__ == "__main__":
    main()
