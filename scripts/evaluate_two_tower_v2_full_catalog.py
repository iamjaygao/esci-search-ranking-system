"""
Evaluates the Phase 1 V1 best checkpoint on the REAL, unmodified full-catalog
benchmark (output/full_retrieval/eval_query_ids.json + ground_truth.json,
same 5000-query sample, seed=42 -- unchanged). This is the apples-to-apples
V0-vs-V1 comparison flagged as not-yet-done in FINAL_REPORT.md.

Builds a NEW FAISS index over the same 1,215,854-product US-locale catalog
using the V1 checkpoint (reuses retrieval.two_tower.build_global_tt_index
verbatim, just pointed at a different model path) -- does NOT touch or
overwrite output/full_retrieval/tt_index_us.faiss (the V0 index). Query
encoding + search is batched (Phase 7.1 finding: ~55x faster than the
per-query loop), not a re-run of scripts/run_full_tt_retrieval.py.

Usage:
    python scripts/evaluate_two_tower_v2_full_catalog.py
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import faiss

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import PRODUCTS_PATH
from retrieval.two_tower import build_global_tt_index, _get_best_device
from scripts.evaluate_full_retrieval import build_rank_lookup, recall_at_k, mrr_at_k, rrf_fuse_query, RRF_K

RETRIEVAL_DIR = f"{ROOT_DIR}/output/full_retrieval"
V1_MODEL_PATH = f"{ROOT_DIR}/experiments/two_tower_v2/phase1_correct_training/full_run/checkpoints/checkpoint-9286"
OUT_DIR = f"{ROOT_DIR}/experiments/two_tower_v2/phase1_correct_training/full_run/full_catalog_eval"
LOCALE = "us"
K_VALUES = [10, 50, 100, 200]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert os.path.isdir(V1_MODEL_PATH), f"V1 best checkpoint not found at {V1_MODEL_PATH}"

    print("Loading full US-locale catalog...")
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_pr['product_id'] = df_pr['product_id'].astype(str)
    df_us = df_pr[df_pr['product_locale'] == LOCALE].copy().reset_index(drop=True)
    print(f"US-locale catalog: {len(df_us)} products")

    print(f"\nBuilding V1 FAISS index (checkpoint-9286, best by dev_cosine_recall@100)...")
    t0 = time.time()
    model, tt_index, tt_ids = build_global_tt_index(df_us, model_name=V1_MODEL_PATH)
    build_time = time.time() - t0
    print(f"Index built in {build_time:.1f}s over {len(tt_ids)} products, dim={tt_index.d}")

    faiss.write_index(tt_index, f"{OUT_DIR}/tt_index_us_v1.faiss")
    with open(f"{OUT_DIR}/tt_ids_us_v1.json", "w") as f:
        json.dump(tt_ids, f)

    with open(f"{RETRIEVAL_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)
    with open(f"{RETRIEVAL_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)
    query_ids = eval_meta["query_ids"]
    query_texts = [ground_truth[qid]["query"] for qid in query_ids]

    print(f"\nBatch-encoding {len(query_texts)} eval queries with V1 model...")
    q_emb = model.encode(query_texts, batch_size=256, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    q_emb = np.nan_to_num(q_emb)

    print("Batched FAISS search (Top-200)...")
    t0 = time.time()
    scores, idxs = tt_index.search(q_emb, 200)
    search_time = time.time() - t0
    print(f"Search done in {search_time:.1f}s")

    rows = []
    for qi, qid in enumerate(query_ids):
        for rank in range(200):
            pid_idx = idxs[qi][rank]
            if pid_idx == -1:
                continue
            rows.append({"query_id": qid, "product_id": tt_ids[pid_idx], "rank": rank + 1,
                         "semantic_score": float(scores[qi][rank])})
    tt_df = pd.DataFrame(rows)
    tt_df.to_parquet(f"{OUT_DIR}/two_tower_retrieval_v1.parquet", index=False)
    print(f"Saved {len(tt_df)} rows to two_tower_retrieval_v1.parquet")

    tt_lookup = build_rank_lookup(tt_df)

    rows_metrics = []
    row = {"retriever": "Two-Tower-V1"}
    for k in K_VALUES:
        r, n = recall_at_k(tt_lookup, ground_truth, "relevant_broad", k)
        row[f"recall@{k}"] = r
        er, _ = recall_at_k(tt_lookup, ground_truth, "relevant_exact", k)
        row[f"exact_recall@{k}"] = er
    row["mrr@10"] = mrr_at_k(tt_lookup, ground_truth, "relevant_broad", k=10)
    rows_metrics.append(row)

    bm25_df = pd.read_parquet(f"{RETRIEVAL_DIR}/bm25_retrieval.parquet")
    bm25_df['query_id'] = bm25_df['query_id'].astype(str)
    bm25_df['product_id'] = bm25_df['product_id'].astype(str)
    bm25_lookup = build_rank_lookup(bm25_df)
    rrf_recall_broad, rrf_lookup_100 = None, None
    from scripts.evaluate_full_retrieval import rrf_recall
    rrf_recall_broad, rrf_lookup_100 = rrf_recall(bm25_lookup, tt_lookup, ground_truth, "relevant_broad", top_n=100)
    rrf_recall_exact, _ = rrf_recall(bm25_lookup, tt_lookup, ground_truth, "relevant_exact", top_n=100)
    rrf_mrr = mrr_at_k(rrf_lookup_100, ground_truth, "relevant_broad", k=10)
    rows_metrics.append({
        "retriever": "Hybrid RRF@100 (BM25 + Two-Tower-V1)",
        "recall@100": rrf_recall_broad, "exact_recall@100": rrf_recall_exact, "mrr@10": rrf_mrr,
    })

    metrics_df = pd.DataFrame(rows_metrics)
    metrics_df.to_csv(f"{OUT_DIR}/retrieval_metrics_v1.csv", index=False)

    with open(f"{RETRIEVAL_DIR}/retrieval_metrics.csv") as f:
        v0_metrics = pd.read_csv(f)

    comparison = {
        "v0_two_tower_recall100": float(v0_metrics.loc[v0_metrics['retriever'] == 'Two-Tower', 'recall@100'].iloc[0]),
        "v1_two_tower_recall100": float(row.get("recall@100")),
        "v0_rrf_recall100": float(v0_metrics.loc[v0_metrics['retriever'] == 'Hybrid RRF@100', 'recall@100'].iloc[0]),
        "v1_rrf_recall100": float(rrf_recall_broad),
        "v1_checkpoint": V1_MODEL_PATH,
        "build_time_sec": build_time, "search_time_sec": search_time,
        "eval_query_sample": {"size": len(query_ids), "seed": eval_meta["seed"], "locale": eval_meta["locale"]},
    }
    with open(f"{OUT_DIR}/v0_vs_v1_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + metrics_df.to_string(index=False))
    print("\n" + json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
