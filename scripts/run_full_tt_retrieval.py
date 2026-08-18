"""
Runs real Two-Tower + FAISS retrieval against the US-only full-catalog index
(built by scripts/build_full_catalog_indices.py) for the deterministic
5000-query evaluation set. Records rank, score, and per-query latency.

Usage:
    python scripts/run_full_tt_retrieval.py
"""
import os
import sys
import json
import time

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT_DIR
from retrieval.two_tower import search_tt_global, MODEL_NAME, _get_best_device

OUT_DIR = f"{ROOT_DIR}/output/full_retrieval"
K = 200


def main():
    with open(f"{OUT_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)
    with open(f"{OUT_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)
    query_ids = eval_meta['query_ids']

    print("Loading US-only FAISS index + Two-Tower encoder...")
    device = _get_best_device()
    print("device:", device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    faiss_index = faiss.read_index(f"{OUT_DIR}/tt_index_us.faiss")
    with open(f"{OUT_DIR}/tt_ids_us.json") as f:
        tt_ids = json.load(f)
    print(f"Loaded index with {len(tt_ids)} products, dim={faiss_index.d}")

    rows = []
    latencies = []
    t_start = time.time()
    for i, qid in enumerate(query_ids):
        qtext = ground_truth[qid]['query']
        t0 = time.perf_counter()
        df_res = search_tt_global(model, faiss_index, tt_ids, qtext, k=K)
        latencies.append(time.perf_counter() - t0)
        for rank, row in enumerate(df_res.itertuples(index=False), start=1):
            rows.append({"query_id": qid, "product_id": row.product_id, "rank": rank, "semantic_score": row.semantic_score})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(query_ids)} queries done ({time.time()-t_start:.0f}s elapsed)")

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(f"{OUT_DIR}/two_tower_retrieval.parquet", index=False)
    print(f"Saved {OUT_DIR}/two_tower_retrieval.parquet ({len(df_out)} rows, {df_out['query_id'].nunique()} queries)")

    lat = np.array(latencies) * 1000
    latency_stats = {
        "retriever": "two_tower", "device": device,
        "mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)),
        "n_queries": len(lat),
    }
    print("TT latency:", latency_stats)
    with open(f"{OUT_DIR}/tt_latency.json", "w") as f:
        json.dump(latency_stats, f, indent=2)


if __name__ == "__main__":
    main()
