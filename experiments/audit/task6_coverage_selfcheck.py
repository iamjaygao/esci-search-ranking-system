"""
Task 6 -- Retrieval coverage decomposition self-consistency check (BM25-only /
Two-Tower-only / both / both-miss / union) at K=100, E+S+C relevance, using
the exact same query universe and relevance definition as Task 5's "E + S + C"
row. Recomputed from output/full_retrieval/{bm25,two_tower}_retrieval.parquet
+ raw ESCI labels -- not copied from the pre-existing
output/full_retrieval/retriever_overlap_summary.json. Reports whatever
algebraic relationships actually hold between macro (per-query-averaged)
Recall@100 and micro (item-instance-counted) coverage percentages, rather
than forcing them to match.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from config import EXAMPLES_PATH
from scripts.evaluate_full_retrieval import build_rank_lookup, recall_at_k, rrf_recall

RETRIEVAL_DIR = f"{ROOT_DIR}/output/full_retrieval"
OUT_DIR = f"{ROOT_DIR}/experiments/audit"
LOCALE = "us"
K = 100
BROAD_LABELS = {"E", "S", "C"}


def main():
    with open(f"{RETRIEVAL_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)
    query_ids = [str(q) for q in eval_meta["query_ids"]]

    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)
    df_eval = df_ex[(df_ex['split'] == 'test') & (df_ex['product_locale'] == LOCALE) &
                     (df_ex['query_id'].isin(set(query_ids)))][['query_id', 'product_id', 'esci_label']].drop_duplicates()

    ground_truth = {}
    for qid, group in df_eval.groupby('query_id'):
        rel = set(group[group['esci_label'].isin(BROAD_LABELS)]['product_id'])
        ground_truth[qid] = {"relevant_set": sorted(rel)}
    for qid in query_ids:
        if qid not in ground_truth:
            ground_truth[qid] = {"relevant_set": []}

    bm25_df = pd.read_parquet(f"{RETRIEVAL_DIR}/bm25_retrieval.parquet")
    tt_df = pd.read_parquet(f"{RETRIEVAL_DIR}/two_tower_retrieval.parquet")
    bm25_df['query_id'] = bm25_df['query_id'].astype(str)
    tt_df['query_id'] = tt_df['query_id'].astype(str)
    bm25_df['product_id'] = bm25_df['product_id'].astype(str)
    tt_df['product_id'] = tt_df['product_id'].astype(str)
    bm25_lookup = build_rank_lookup(bm25_df)
    tt_lookup = build_rank_lookup(tt_df)

    # ---- Macro: per-query-averaged Recall@100 (same function as Task 5 / evaluate_full_retrieval.py) ----
    bm25_recall_macro, n_q = recall_at_k(bm25_lookup, ground_truth, "relevant_set", K)
    tt_recall_macro, _ = recall_at_k(tt_lookup, ground_truth, "relevant_set", K)
    rrf_recall_macro, _ = rrf_recall(bm25_lookup, tt_lookup, ground_truth, "relevant_set", top_n=K)

    # ---- Micro: item-instance-counted coverage decomposition at K=100 ----
    both = bm25_only = tt_only = missed = 0
    for qid, gt in ground_truth.items():
        rel_set = gt["relevant_set"]
        bset = {pid for pid, r in bm25_lookup.get(qid, {}).items() if r <= K}
        tset = {pid for pid, r in tt_lookup.get(qid, {}).items() if r <= K}
        for pid in rel_set:
            in_b, in_t = pid in bset, pid in tset
            if in_b and in_t:
                both += 1
            elif in_b:
                bm25_only += 1
            elif in_t:
                tt_only += 1
            else:
                missed += 1

    total = both + bm25_only + tt_only + missed
    micro = {
        "both": both, "bm25_only": bm25_only, "tt_only": tt_only, "both_miss": missed,
        "total_relevant_item_instances": total,
        "pct_both": 100 * both / total, "pct_bm25_only": 100 * bm25_only / total,
        "pct_tt_only": 100 * tt_only / total, "pct_both_miss": 100 * missed / total,
        "pct_union": 100 * (both + bm25_only + tt_only) / total,
    }

    # ---- Algebraic identity checks (report actual numbers, don't force agreement) ----
    micro_bm25_recall = (both + bm25_only) / total
    micro_tt_recall = (both + tt_only) / total
    micro_union_recall = (both + bm25_only + tt_only) / total
    checks = {
        "macro_bm25_recall_at_100": bm25_recall_macro,
        "micro_bm25_recall_at_100 (both+bm25_only)/total": micro_bm25_recall,
        "macro_vs_micro_bm25_diff": bm25_recall_macro - micro_bm25_recall,
        "macro_tt_recall_at_100": tt_recall_macro,
        "micro_tt_recall_at_100 (both+tt_only)/total": micro_tt_recall,
        "macro_vs_micro_tt_diff": tt_recall_macro - micro_tt_recall,
        "macro_rrf_recall_at_100": rrf_recall_macro,
        "micro_union_recall (both+bm25_only+tt_only)/total": micro_union_recall,
        "sum_bm25only_ttonly_both_bothmiss_pct": (
            micro["pct_bm25_only"] + micro["pct_tt_only"] + micro["pct_both"] + micro["pct_both_miss"]
        ),
        "note": (
            "macro_* = mean of per-query recall (each query weighted equally, matching Recall@100 as "
            "reported elsewhere in this repo / Task 5). micro_* = counts of individual (query, relevant "
            "product) instances pooled across all queries, then divided by total instances (queries with "
            "many relevant items dominate). These are different averaging schemes and are not expected to "
            "be numerically identical; both are reported here rather than reconciled."
        ),
    }

    result = {"k": K, "relevance": "broad (E+S+C)", "n_queries": len(query_ids),
              "macro_recall": {"bm25": bm25_recall_macro, "two_tower": tt_recall_macro, "rrf": rrf_recall_macro},
              "micro_coverage": micro, "consistency_checks": checks}

    with open(f"{OUT_DIR}/retrieval_coverage.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
