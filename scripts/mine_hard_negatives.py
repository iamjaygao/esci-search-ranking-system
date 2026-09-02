"""
Phase 3.1 -- BM25 hard-negative mining stats (mining only; does not train a
model). Reuses the existing US-locale BM25 index built by
scripts/build_full_catalog_indices.py (output/full_retrieval/bm25s_index_us)
via retrieval/bm25.py's load_bm25_index()/search_bm25_global() -- read-only,
no reimplementation of BM25 scoring.

For each training query: run BM25 top-100, and select up to 4 products
labeled 'I' (NOT 'C' -- explicitly excluded per task instructions) as hard
negatives. 'C' is neither treated as positive nor negative here.

Because ranking each of the ~18,800 train queries against the full 1.21M
US-locale catalog is nontrivial, this mines a fixed random SAMPLE of queries
(documented, not the full train set) to produce representative statistics
within a reasonable runtime. The actual hard-negative-augmented TRAINING RUN
(Phase 3.2/3.3) is a separate, much more expensive step and is NOT executed
here -- see phase3_hard_negative/REPORT.md for what was and wasn't run.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import EXAMPLES_PATH
from retrieval.bm25 import load_bm25_index, search_bm25_global

OUT_DIR = f"{ROOT_DIR}/experiments/two_tower_v2/phase3_hard_negative"
SEED = 42
SAMPLE_N_QUERIES = 2000
BM25_TOPK_FOR_MINING = 100
N_HARD_NEG_PER_QUERY = 4
LOCALE = "us"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{ROOT_DIR}/experiments/two_tower_v2/splits/train_queries.txt") as f:
        train_queries = [l.strip() for l in f if l.strip()]

    rng = np.random.RandomState(SEED)
    sample_queries = sorted(rng.choice(train_queries, size=min(SAMPLE_N_QUERIES, len(train_queries)), replace=False).tolist())

    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_ex["query_id"] = df_ex["query_id"].astype(str)
    df_ex["product_id"] = df_ex["product_id"].astype(str)
    df_sub = df_ex[df_ex["query_id"].isin(set(sample_queries)) & (df_ex["product_locale"] == LOCALE)]
    query_text = df_sub.groupby("query_id")["query"].first().to_dict()
    labels_by_query = {qid: dict(zip(g["product_id"], g["esci_label"])) for qid, g in df_sub.groupby("query_id")}

    print(f"Loading existing US-locale BM25 index (output/full_retrieval/bm25s_index_us)...")
    bm25_index, item_ids = load_bm25_index(
        index_dir=f"{ROOT_DIR}/output/full_retrieval/bm25s_index_us",
        ids_path=f"{ROOT_DIR}/output/full_retrieval/bm25_ids_us.json",
    )

    n_hard_negs_per_query = []
    rank_positions_used = []
    n_fallback = 0  # queries where fewer than N_HARD_NEG_PER_QUERY 'I' items were found in top-100
    t0 = time.time()
    for i, qid in enumerate(sample_queries):
        qtext = query_text.get(qid)
        if not qtext:
            continue
        res = search_bm25_global(bm25_index, item_ids, qtext, k=BM25_TOPK_FOR_MINING)
        labels = labels_by_query.get(qid, {})
        hard_negs = []
        for rank, row in enumerate(res.itertuples(index=False), start=1):
            pid = row.product_id
            if labels.get(pid) == "I":
                hard_negs.append(rank)
                if len(hard_negs) >= N_HARD_NEG_PER_QUERY:
                    break
        n_hard_negs_per_query.append(len(hard_negs))
        rank_positions_used.extend(hard_negs)
        if len(hard_negs) < N_HARD_NEG_PER_QUERY:
            n_fallback += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(sample_queries)} queries mined ({time.time()-t0:.0f}s elapsed)")

    n_hard_negs_per_query = np.array(n_hard_negs_per_query)
    rank_positions_used = np.array(rank_positions_used)

    stats = {
        "sample_note": f"Mined a fixed random sample of {len(sample_queries)} / {len(train_queries)} train queries "
                        f"(seed={SEED}), not the full train set, for tractable runtime.",
        "num_queries": int(len(n_hard_negs_per_query)),
        "target_hard_negatives_per_query": N_HARD_NEG_PER_QUERY,
        "bm25_topk_searched": BM25_TOPK_FOR_MINING,
        "avg_hard_negatives": float(n_hard_negs_per_query.mean()),
        "median_hard_negatives": float(np.median(n_hard_negs_per_query)),
        "queries_with_zero_hard_negatives": int((n_hard_negs_per_query == 0).sum()),
        "fallback_rate": float(n_fallback / len(n_hard_negs_per_query)),
        "fallback_definition": "fraction of queries for which fewer than 4 'I'-labeled products were found within the BM25 top-100",
        "rank_distribution_of_mined_hard_negatives": {
            "mean_rank": float(rank_positions_used.mean()) if len(rank_positions_used) else None,
            "median_rank": float(np.median(rank_positions_used)) if len(rank_positions_used) else None,
            "p90_rank": float(np.percentile(rank_positions_used, 90)) if len(rank_positions_used) else None,
        },
        "positive_definition_reminder": "positive = E/S (unchanged from V0/V1). hard_negative = I only. 'C' is neither positive nor negative in this mining step.",
    }
    with open(f"{OUT_DIR}/hard_negative_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
