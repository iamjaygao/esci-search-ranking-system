"""
Phase 7.3 -- ANN index benchmark (IndexFlatIP exact oracle vs HNSW vs IVFFlat)
over the EXISTING US-locale full-catalog Two-Tower embeddings. Does not
re-encode the 1.2M-product catalog (would take hours); reconstructs the
already-computed vectors from output/full_retrieval/tt_index_us.faiss
(read-only) and builds alternative index types over the SAME vectors, so any
ANN-recall difference is attributable to the index structure only, not to
re-encoding noise.

Reports two DIFFERENT metrics, not conflated:
  - ANN_recall_vs_flat: agreement between each ANN index's top-100 and the
    FlatIP oracle's top-100, per query, averaged (a pure index-quality metric).
  - task_recall100: real Recall@100 against ESCI ground truth
    (output/full_retrieval/ground_truth.json, relevant_broad = E/S/C), the
    actual task metric.

Usage:
    python scripts/benchmark_ann.py
"""
import os
import sys
import json
import time
import tempfile

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import ROOT_DIR as CFG_ROOT
from retrieval.two_tower import MODEL_NAME, _get_best_device

RETRIEVAL_DIR = f"{ROOT_DIR}/output/full_retrieval"
OUT_DIR = f"{ROOT_DIR}/experiments/two_tower_v2/phase7_serving"
K = 100


def recall_at_k_macro(rank_lookup, ground_truth, k):
    recalls = []
    for qid, gt in ground_truth.items():
        rel_set = set(gt["relevant_broad"])
        if not rel_set:
            continue
        ranks = rank_lookup.get(qid, {})
        hits = sum(1 for pid in rel_set if ranks.get(pid, 10 ** 9) <= k)
        recalls.append(hits / len(rel_set))
    return float(np.mean(recalls))


def index_size_mb(index):
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        path = tmp.name
    faiss.write_index(index, path)
    size = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size


def latency_stats(index, query_embeddings, k=K):
    lats = []
    for i in range(query_embeddings.shape[0]):
        q = query_embeddings[i:i + 1]
        t0 = time.perf_counter()
        index.search(q, k)
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    return {
        "query_p50_ms": float(np.percentile(lats, 50)),
        "query_p95_ms": float(np.percentile(lats, 95)),
        "query_p99_ms": float(np.percentile(lats, 99)),
        "qps": float(1000.0 / lats.mean()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading existing FlatIP index + ids (no re-encoding of the catalog)...")
    flat_index = faiss.read_index(f"{RETRIEVAL_DIR}/tt_index_us.faiss")
    with open(f"{RETRIEVAL_DIR}/tt_ids_us.json") as f:
        item_ids = json.load(f)
    n, dim = flat_index.ntotal, flat_index.d
    print(f"Catalog: {n} products, dim={dim}")

    t0 = time.time()
    embeddings = flat_index.reconstruct_n(0, n)
    print(f"Reconstructed {n} vectors in {time.time()-t0:.1f}s")

    with open(f"{RETRIEVAL_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)
    with open(f"{RETRIEVAL_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)
    query_ids = eval_meta["query_ids"]
    query_texts = [ground_truth[qid]["query"] for qid in query_ids]

    device = _get_best_device()
    print(f"Encoding {len(query_texts)} eval queries with the existing V0 model (device={device})...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    query_embeddings = model.encode(query_texts, batch_size=256, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    query_embeddings = np.nan_to_num(query_embeddings)

    id_to_pos = {iid: i for i, iid in enumerate(item_ids)}

    results = []

    def eval_index(name, index, params, build_time):
        size_mb = index_size_mb(index)
        lat = latency_stats(index, query_embeddings, K)
        scores, idxs = index.search(query_embeddings, K)
        rank_lookup = {}
        for qi, qid in enumerate(query_ids):
            ranks = {item_ids[idxs[qi][r]]: r + 1 for r in range(K) if idxs[qi][r] != -1}
            rank_lookup[qid] = ranks
        task_recall = recall_at_k_macro(rank_lookup, ground_truth, K)
        return {
            "index_type": name, "index_parameters": params,
            "build_time_sec": build_time, "index_size_mb": size_mb,
            **lat, "task_recall100": task_recall,
        }, rank_lookup

    print("\n=== FlatIP (exact oracle) ===")
    t0 = time.time()
    flat_index2 = faiss.IndexFlatIP(dim)  # rebuild fresh to time "build" fairly (reconstructed vectors, not the loaded index)
    flat_index2.add(embeddings)
    build_time_flat = time.time() - t0
    flat_row, flat_lookup = eval_index("IndexFlatIP", flat_index2, {}, build_time_flat)
    flat_row["ANN_recall_vs_flat"] = 1.0
    results.append(flat_row)
    print(flat_row)

    print("\n=== HNSW ===")
    M = 32
    t0 = time.time()
    hnsw = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(embeddings)
    hnsw.hnsw.efSearch = 64
    build_time_hnsw = time.time() - t0
    hnsw_row, hnsw_lookup = eval_index("HNSW", hnsw, {"M": M, "efConstruction": 40, "efSearch": 64}, build_time_hnsw)
    results.append(hnsw_row)
    print(hnsw_row)

    print("\n=== IVFFlat ===")
    nlist = 4096
    t0 = time.time()
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(embeddings)
    ivf.add(embeddings)
    ivf.nprobe = 32
    build_time_ivf = time.time() - t0
    ivf_row, ivf_lookup = eval_index("IVFFlat", ivf, {"nlist": nlist, "nprobe": 32}, build_time_ivf)
    results.append(ivf_row)
    print(ivf_row)

    # ---- ANN_recall_vs_flat for HNSW/IVF (agreement with FlatIP top-100) ----
    for row, lookup in [(hnsw_row, hnsw_lookup), (ivf_row, ivf_lookup)]:
        agree = []
        for qid in query_ids:
            flat_set = set(flat_lookup.get(qid, {}))
            ann_set = set(lookup.get(qid, {}))
            if flat_set:
                agree.append(len(flat_set & ann_set) / len(flat_set))
        row["ANN_recall_vs_flat"] = float(np.mean(agree))

    df = pd.DataFrame(results)
    df.to_csv(f"{OUT_DIR}/ann_benchmark.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved {OUT_DIR}/ann_benchmark.csv")


if __name__ == "__main__":
    main()
