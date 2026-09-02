"""
Compute per-query Recall@100 (and Recall@10 / per-query reciprocal rank) for
BM25 and Two-Tower on the existing 5,000-query full-catalog retrieval eval
set. No per-query file for this pipeline exists yet in the repo (see
AUDIT.md Section 6) -- this reuses the exact hit-counting logic of
recall_at_k() in scripts/evaluate_full_retrieval.py (imported, not
reimplemented) but keeps the per-query values instead of only the mean.
"""
import os
import sys
import json

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from scripts.evaluate_full_retrieval import build_rank_lookup

RETRIEVAL_DIR = f"{ROOT_DIR}/output/full_retrieval"
OUT_DIR = f"{ROOT_DIR}/experiments/query_slice_analysis"


def per_query_recall(rank_lookup, ground_truth, relevant_key, k):
    out = {}
    for qid, gt in ground_truth.items():
        rel_set = set(gt[relevant_key])
        if not rel_set:
            continue
        ranks = rank_lookup.get(qid, {})
        hits = sum(1 for pid in rel_set if ranks.get(pid, 10 ** 9) <= k)
        out[qid] = hits / len(rel_set)
    return out


def per_query_reciprocal_rank(rank_lookup, ground_truth, relevant_key, k):
    out = {}
    for qid, gt in ground_truth.items():
        rel_set = set(gt[relevant_key])
        if not rel_set:
            continue
        ranks = rank_lookup.get(qid, {})
        relevant_ranks = [ranks[pid] for pid in rel_set if pid in ranks and ranks[pid] <= k]
        out[qid] = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0
    return out


def main():
    with open(f"{RETRIEVAL_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)

    bm25_df = pd.read_parquet(f"{RETRIEVAL_DIR}/bm25_retrieval.parquet")
    tt_df = pd.read_parquet(f"{RETRIEVAL_DIR}/two_tower_retrieval.parquet")
    for df in (bm25_df, tt_df):
        df['query_id'] = df['query_id'].astype(str)
        df['product_id'] = df['product_id'].astype(str)

    bm25_lookup = build_rank_lookup(bm25_df)
    tt_lookup = build_rank_lookup(tt_df)

    bm25_r100 = per_query_recall(bm25_lookup, ground_truth, "relevant_broad", 100)
    tt_r100 = per_query_recall(tt_lookup, ground_truth, "relevant_broad", 100)
    bm25_r10 = per_query_recall(bm25_lookup, ground_truth, "relevant_broad", 10)
    tt_r10 = per_query_recall(tt_lookup, ground_truth, "relevant_broad", 10)
    bm25_rr10 = per_query_reciprocal_rank(bm25_lookup, ground_truth, "relevant_broad", 10)
    tt_rr10 = per_query_reciprocal_rank(tt_lookup, ground_truth, "relevant_broad", 10)

    qids = sorted(set(bm25_r100) & set(tt_r100))
    rows = []
    for qid in qids:
        rows.append({
            "query_id": qid,
            "bm25_recall100": bm25_r100[qid], "tt_recall100": tt_r100[qid],
            "bm25_recall10": bm25_r10.get(qid), "tt_recall10": tt_r10.get(qid),
            "bm25_rr10": bm25_rr10.get(qid), "tt_rr10": tt_rr10.get(qid),
        })
    df = pd.DataFrame(rows)
    df['delta_tt_minus_bm25'] = df['tt_recall100'] - df['bm25_recall100']
    df.to_csv(f"{OUT_DIR}/_per_query_recall.csv", index=False)

    print(f"n_queries with non-empty relevant_broad: {len(df)} / 5000")
    print(f"Overall BM25 Recall@100 (macro mean)      = {df['bm25_recall100'].mean():.4f}")
    print(f"Overall Two-Tower Recall@100 (macro mean)  = {df['tt_recall100'].mean():.4f}")
    print(f"(reference from retrieval_metrics.csv: BM25=0.4542, Two-Tower=0.4598)")


if __name__ == "__main__":
    main()
