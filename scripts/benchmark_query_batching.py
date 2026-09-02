"""
Phase 7.1 -- Offline batch query-encoding throughput vs the existing
single-query online loop (scripts/run_full_tt_retrieval.py / search_tt_global).
Does NOT modify run_full_tt_retrieval.py (its output is the existing "online,
one-query-at-a-time" latency benchmark, output/full_retrieval/tt_latency.json,
kept as-is and cited here for comparison). This script only measures the
OFFLINE batched alternative: encode all 5000 eval queries in one batched
model.encode() call, then a single batched faiss_index.search() call.
"""
import os
import sys
import json
import time

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from retrieval.two_tower import MODEL_NAME, _get_best_device

RETRIEVAL_DIR = f"{ROOT_DIR}/output/full_retrieval"
OUT_DIR = f"{ROOT_DIR}/experiments/two_tower_v2/phase7_serving"
K = 200


def main():
    with open(f"{RETRIEVAL_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)
    with open(f"{RETRIEVAL_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)
    query_ids = eval_meta["query_ids"]
    query_texts = [ground_truth[qid]["query"] for qid in query_ids]

    device = _get_best_device()
    print(f"device={device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    faiss_index = faiss.read_index(f"{RETRIEVAL_DIR}/tt_index_us.faiss")

    t0 = time.time()
    q_emb = model.encode(query_texts, batch_size=256, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    q_emb = np.nan_to_num(q_emb)
    encode_time = time.time() - t0

    t0 = time.time()
    scores, idxs = faiss_index.search(q_emb, K)
    search_time = time.time() - t0

    total_time = encode_time + search_time
    n = len(query_ids)

    with open(f"{RETRIEVAL_DIR}/tt_latency.json") as f:
        online_latency = json.load(f)

    result = {
        "n_queries": n,
        "offline_batched": {
            "encode_time_sec": encode_time, "search_time_sec": search_time,
            "total_time_sec": total_time,
            "throughput_qps": n / total_time,
            "effective_per_query_ms": 1000 * total_time / n,
            "method": "one model.encode(list_of_5000, batch_size=256) call + one faiss_index.search(all, k) call",
        },
        "online_single_query_loop_REFERENCE": {
            **online_latency,
            "throughput_qps_equiv": 1000 / online_latency["mean_ms"],
            "method": "existing scripts/run_full_tt_retrieval.py: one model.encode([single_query]) + one faiss_index.search(1,k) call PER QUERY, sequential loop (unmodified)",
            "note": "These two benchmarks measure different things and are not a strict apples-to-apples ratio: the online loop's per-call timing includes per-call encode+search for ONE query; the offline number amortizes encode+search cost across a batch of 5000. Reported side by side, not divided into a single 'speedup' number, to avoid overstating precision.",
        },
    }
    with open(f"{OUT_DIR}/batching_benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
