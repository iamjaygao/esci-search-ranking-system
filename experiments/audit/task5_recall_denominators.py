"""
Task 5 -- Recall@100 under three relevance-denominator definitions (E only,
E+S, E+S+C). Reuses the existing eval_query_ids.json (seed=42, 5000 us-locale
test queries, unchanged) and the existing bm25_retrieval.parquet /
two_tower_retrieval.parquet full-catalog retrieval outputs. No retraining, no
resampling of queries. RRF fusion reuses rrf_fuse_query/recall_at_k/
build_rank_lookup imported verbatim from scripts/evaluate_full_retrieval.py
(k=60, unchanged).
"""
import os
import sys
import json

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from config import EXAMPLES_PATH
from scripts.evaluate_full_retrieval import build_rank_lookup, recall_at_k, rrf_recall, RRF_K

RETRIEVAL_DIR = f"{ROOT_DIR}/output/full_retrieval"
OUT_DIR = f"{ROOT_DIR}/experiments/audit"
LOCALE = "us"

DEFINITIONS = {
    "E only": {"E"},
    "E + S": {"E", "S"},
    "E + S + C": {"E", "S", "C"},
}


def main():
    with open(f"{RETRIEVAL_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)
    query_ids = [str(q) for q in eval_meta["query_ids"]]
    assert eval_meta["seed"] == 42 and eval_meta["locale"] == LOCALE
    print(f"Reusing existing eval query sample: {len(query_ids)} queries, seed={eval_meta['seed']}")

    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)
    df_eval = df_ex[(df_ex['split'] == 'test') & (df_ex['product_locale'] == LOCALE) &
                     (df_ex['query_id'].isin(set(query_ids)))][['query_id', 'product_id', 'esci_label']].drop_duplicates()

    bm25_df = pd.read_parquet(f"{RETRIEVAL_DIR}/bm25_retrieval.parquet")
    tt_df = pd.read_parquet(f"{RETRIEVAL_DIR}/two_tower_retrieval.parquet")
    bm25_df['query_id'] = bm25_df['query_id'].astype(str)
    tt_df['query_id'] = tt_df['query_id'].astype(str)
    bm25_df['product_id'] = bm25_df['product_id'].astype(str)
    tt_df['product_id'] = tt_df['product_id'].astype(str)

    bm25_lookup = build_rank_lookup(bm25_df)
    tt_lookup = build_rank_lookup(tt_df)

    rows = []
    for def_name, labels in DEFINITIONS.items():
        ground_truth = {}
        for qid, group in df_eval.groupby('query_id'):
            rel = set(group[group['esci_label'].isin(labels)]['product_id'])
            ground_truth[qid] = {"relevant_set": sorted(rel)}
        # queries in eval set with zero relevant items under this definition still need an entry
        for qid in query_ids:
            if qid not in ground_truth:
                ground_truth[qid] = {"relevant_set": []}

        n_relevant_per_query = [len(v["relevant_set"]) for v in ground_truth.values()]
        bm25_recall, n_bm25 = recall_at_k(bm25_lookup, ground_truth, "relevant_set", 100)
        tt_recall, n_tt = recall_at_k(tt_lookup, ground_truth, "relevant_set", 100)
        rrf_recall_val, _ = rrf_recall(bm25_lookup, tt_lookup, ground_truth, "relevant_set", top_n=100)

        rows.append({
            "relevant_denominator": def_name,
            "bm25_recall_at_100": bm25_recall,
            "two_tower_recall_at_100": tt_recall,
            "rrf_recall_at_100": rrf_recall_val,
            "n_queries_with_relevant_items": n_bm25,
            "mean_relevant_items_per_query": float(np.mean(n_relevant_per_query)),
        })
        print(f"[{def_name}] BM25={bm25_recall:.4f} TwoTower={tt_recall:.4f} RRF(k={RRF_K})={rrf_recall_val:.4f} "
              f"(n_queries_with_gt={n_bm25}, mean_relevant/query={np.mean(n_relevant_per_query):.2f})")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(f"{OUT_DIR}/recall_denominators.csv", index=False)
    print(f"\nSaved {OUT_DIR}/recall_denominators.csv")


if __name__ == "__main__":
    main()
