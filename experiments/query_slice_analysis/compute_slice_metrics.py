"""
Steps 3-6 -- Per-slice Recall@100 metrics, win/loss/tie counts, bootstrap CI,
and worst/best examples. Consumes query_assignments.csv and
_per_query_recall.csv produced by the previous two scripts in this
directory. No retraining, no modification of existing files.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from config import PRODUCTS_PATH

OUT_DIR = f"{ROOT_DIR}/experiments/query_slice_analysis"
RETRIEVAL_DIR = f"{ROOT_DIR}/output/full_retrieval"
LOCALE = "us"
WIN_EPS = 1e-9
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

SLICE_COLS = ['is_sku_model', 'is_storage_numeric', 'is_brand_heavy', 'is_strong_lexical', 'is_low_lexical_overlap']
SLICE_LABELS = {
    'is_sku_model': 'SKU / model-like',
    'is_storage_numeric': 'Storage / numeric-spec',
    'is_brand_heavy': 'Brand-heavy',
    'is_strong_lexical': 'Strong lexical (vs. relevant-title overlap)',
    'is_low_lexical_overlap': 'Low lexical-overlap',
}


def size_flag(n):
    if n < 30:
        return "TOO SMALL"
    if n < 100:
        return "exploratory"
    return "stable"


def bootstrap_ci(deltas, seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP):
    if len(deltas) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    n = len(deltas)
    means = np.empty(n_boot)
    arr = np.asarray(deltas)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[b] = arr[idx].mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def slice_row(name, sub):
    n = len(sub)
    bm25_r = sub['bm25_recall100'].mean()
    tt_r = sub['tt_recall100'].mean()
    delta = sub['delta_tt_minus_bm25']
    wins_tt = int((delta > WIN_EPS).sum())
    wins_bm25 = int((delta < -WIN_EPS).sum())
    ties = int((delta.abs() <= WIN_EPS).sum())
    ci_low, ci_high = bootstrap_ci(delta.values)
    return {
        "slice": name,
        "n_queries": n,
        "size_flag": size_flag(n),
        "bm25_recall100": bm25_r,
        "tt_recall100": tt_r,
        "delta_tt_minus_bm25": tt_r - bm25_r,
        "bm25_wins": wins_bm25,
        "tt_wins": wins_tt,
        "ties": ties,
        "mean_query_delta": float(delta.mean()),
        "median_query_delta": float(delta.median()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bm25_recall10": sub['bm25_recall10'].mean(),
        "tt_recall10": sub['tt_recall10'].mean(),
        "bm25_mrr10": sub['bm25_rr10'].mean(),
        "tt_mrr10": sub['tt_rr10'].mean(),
    }


def main():
    assign = pd.read_csv(f"{OUT_DIR}/query_assignments.csv", dtype={'query_id': str})
    per_query = pd.read_csv(f"{OUT_DIR}/_per_query_recall.csv", dtype={'query_id': str})
    df = per_query.merge(assign, on='query_id', how='inner')
    assert len(df) == len(per_query) == 5000, f"unexpected row count after merge: {len(df)}"

    rows = [slice_row("Overall", df)]
    for col in SLICE_COLS:
        sub = df[df[col]]
        rows.append(slice_row(SLICE_LABELS[col], sub))

    metrics_df = pd.DataFrame(rows)
    ordered_cols = ["slice", "n_queries", "size_flag", "bm25_recall100", "tt_recall100",
                     "delta_tt_minus_bm25", "bm25_wins", "tt_wins", "ties",
                     "mean_query_delta", "median_query_delta", "ci_low", "ci_high",
                     "bm25_recall10", "tt_recall10", "bm25_mrr10", "tt_mrr10"]
    metrics_df = metrics_df[ordered_cols]
    metrics_df.to_csv(f"{OUT_DIR}/slice_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    # ============ Step 6: worst/best examples per slice ============
    with open(f"{RETRIEVAL_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)
    bm25_df = pd.read_parquet(f"{RETRIEVAL_DIR}/bm25_retrieval.parquet")
    tt_df = pd.read_parquet(f"{RETRIEVAL_DIR}/two_tower_retrieval.parquet")
    for d in (bm25_df, tt_df):
        d['query_id'] = d['query_id'].astype(str)
        d['product_id'] = d['product_id'].astype(str)

    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_pr['product_id'] = df_pr['product_id'].astype(str)
    df_pr_us = df_pr[df_pr['product_locale'] == LOCALE].drop_duplicates('product_id')
    title_lookup = df_pr_us.set_index('product_id')['product_title'].to_dict()

    df_ex_labels = {}  # (query_id) -> {product_id: esci_label} built lazily from ground truth only (broad/exact sets)

    def top10_ids(sub_df, qid, score_col):
        g = sub_df[sub_df['query_id'] == qid].sort_values('rank').head(10)
        return list(g['product_id'])

    def relevant_labels(qid):
        gt = ground_truth[qid]
        exact = set(gt['relevant_exact'])
        broad = set(gt['relevant_broad'])
        return [f"{pid}:{'E' if pid in exact else 'S/C'}" for pid in broad]

    example_rows = []
    for col in SLICE_COLS:
        sub = df[df[col]]
        if len(sub) == 0:
            continue
        sub_sorted = sub.sort_values('delta_tt_minus_bm25')
        bm25_gg_tt = sub_sorted.head(10)  # most negative delta = BM25 >> TT
        tt_gg_bm25 = sub_sorted.tail(10).sort_values('delta_tt_minus_bm25', ascending=False)  # most positive = TT >> BM25

        for group_name, group_df in [("BM25>>TT", bm25_gg_tt), ("TT>>BM25", tt_gg_bm25)]:
            for _, r in group_df.iterrows():
                qid = r['query_id']
                bm25_top = top10_ids(bm25_df, qid, 'bm25_score')
                tt_top = top10_ids(tt_df, qid, 'semantic_score')
                example_rows.append({
                    "slice": SLICE_LABELS[col],
                    "direction": group_name,
                    "query_id": qid,
                    "query": r['query'],
                    "bm25_recall100": r['bm25_recall100'],
                    "tt_recall100": r['tt_recall100'],
                    "delta_tt_minus_bm25": r['delta_tt_minus_bm25'],
                    "relevant_items": "; ".join(relevant_labels(qid)),
                    "bm25_top10_ids": "; ".join(bm25_top),
                    "tt_top10_ids": "; ".join(tt_top),
                    "bm25_top3_titles": "; ".join(title_lookup.get(pid, "") for pid in bm25_top[:3]),
                    "tt_top3_titles": "; ".join(title_lookup.get(pid, "") for pid in tt_top[:3]),
                })

    examples_df = pd.DataFrame(example_rows)
    examples_df.to_csv(f"{OUT_DIR}/examples.csv", index=False)
    print(f"\nSaved {OUT_DIR}/examples.csv ({len(examples_df)} rows)")
    print(f"Saved {OUT_DIR}/slice_metrics.csv")


if __name__ == "__main__":
    main()
