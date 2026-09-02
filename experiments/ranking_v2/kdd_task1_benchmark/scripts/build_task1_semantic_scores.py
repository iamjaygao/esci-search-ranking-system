"""
Recompute semantic_score on the Task 1 candidate pool.

WHY THIS IS NECESSARY (resolution item 8: "recompute everything derived from
the candidate pool ... normalization"):

  output/two_tower_scores_{train,test}.csv stores ONLY the per-query min-max
  NORMALISED value, computed over the LARGE-version candidate pool
  (retrieval/two_tower.py:138-144). The raw cosine and the per-query min/max
  needed to invert it were never persisted. Min-max normalisation is
  pool-dependent by definition, and the Task 1 pool has a different candidate
  set per query, so those frozen values cannot be joined.

  This is a RECOMPUTE, not a retrieval re-run: the candidate set is fixed by
  the ESCI judgments and only the encoder forward pass is repeated. The
  Two-Tower model is loaded frozen from models/two_tower_finetuned and is NOT
  retrained.

TWO COLUMNS ARE EMITTED:
  semantic_cosine_raw -- pool-INDEPENDENT cosine similarity, persisted for the
                         first time (Phase 0 flagged its absence as a blocker)
  semantic_score      -- per-query min-max over the Task 1 pool, the analogue
                         of the frozen feature

DELIBERATE CORRECTION vs retrieval/two_tower.py: that function de-duplicates
items on item_id alone (line 89), which collapses products that exist in more
than one locale with different titles. In the Task 1 pool 4,359 product_ids
span 2-3 locales, so items are keyed here on (product_id, product_locale).
Recorded in feature_pool_dependency_audit.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
OUT = os.path.join(ROOT, "experiments", "ranking_v2", "kdd_task1_benchmark")
CACHE = os.path.join(OUT, "_cache")

MODEL_PATH = os.path.join(ROOT, "models", "two_tower_finetuned")
EXAMPLES = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")
PRODUCTS = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet")


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    os.makedirs(CACHE, exist_ok=True)
    t_start = time.time()

    print("Loading Task 1 pool ...", flush=True)
    ex = pd.read_parquet(EXAMPLES, columns=[
        "example_id", "query_id", "query", "product_id", "product_locale",
        "esci_label", "small_version", "split"])
    t1 = ex[ex["small_version"] == 1].copy()
    n_anchor = len(t1)
    print(f"  Task1 rows: {n_anchor}, queries: {t1['query_id'].nunique()}", flush=True)

    pr = pd.read_parquet(PRODUCTS, columns=[
        "product_id", "product_locale", "product_title",
        "product_description", "product_bullet_point"])
    assert not pr.duplicated(["product_id", "product_locale"]).any(), "products PK not unique"

    t1 = t1.merge(pr, on=["product_id", "product_locale"], how="left")
    assert len(t1) == n_anchor, f"locale-aware product join changed rows: {len(t1)} != {n_anchor}"

    t1["item_text"] = (t1["product_title"].fillna("") + " " +
                       t1["product_description"].fillna("") + " " +
                       t1["product_bullet_point"].fillna(""))

    # ---- unique items keyed on (product_id, product_locale) ----
    items = t1[["product_id", "product_locale", "item_text"]].drop_duplicates(
        subset=["product_id", "product_locale"], keep="first").reset_index(drop=True)
    item_texts = [t if str(t).strip() else "unknown product" for t in items["item_text"].tolist()]
    print(f"  unique (product_id, locale) items to encode: {len(items)}", flush=True)

    queries = t1[["query_id", "query"]].drop_duplicates(subset="query_id").reset_index(drop=True)
    print(f"  unique queries to encode: {len(queries)}", flush=True)

    device = best_device()
    print(f"\nLoading frozen encoder {MODEL_PATH} on {device} ...", flush=True)
    model = SentenceTransformer(MODEL_PATH, device=device)

    print("Encoding products ...", flush=True)
    t0 = time.time()
    prod_emb = model.encode(item_texts, batch_size=256, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)
    prod_emb = np.nan_to_num(prod_emb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  products encoded in {time.time()-t0:.0f}s -> {prod_emb.shape}", flush=True)

    print("Encoding queries ...", flush=True)
    q_emb = model.encode(queries["query"].astype(str).tolist(), batch_size=256,
                         show_progress_bar=True, convert_to_numpy=True,
                         normalize_embeddings=True)
    q_emb = np.nan_to_num(q_emb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # ---- row-aligned raw cosine ----
    print("\nComputing raw cosine per candidate row ...", flush=True)
    items["_iidx"] = np.arange(len(items))
    queries["_qidx"] = np.arange(len(queries))
    t1 = t1.merge(items[["product_id", "product_locale", "_iidx"]],
                  on=["product_id", "product_locale"], how="left")
    t1 = t1.merge(queries[["query_id", "_qidx"]], on="query_id", how="left")
    assert len(t1) == n_anchor and t1["_iidx"].notna().all() and t1["_qidx"].notna().all()

    iidx = t1["_iidx"].to_numpy(dtype=np.int64)
    qidx = t1["_qidx"].to_numpy(dtype=np.int64)
    cos = np.empty(n_anchor, dtype=np.float32)
    CH = 2_000_000
    for s in range(0, n_anchor, CH):
        e = min(s + CH, n_anchor)
        cos[s:e] = np.einsum("ij,ij->i", q_emb[qidx[s:e]], prod_emb[iidx[s:e]])
    cos = np.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
    t1["semantic_cosine_raw"] = cos

    # ---- per-query min-max ON THE TASK 1 POOL (pool-dependent) ----
    print("Applying per-query min-max normalisation on the Task 1 pool ...", flush=True)
    g = t1.groupby("query_id")["semantic_cosine_raw"]
    mn, mx = g.transform("min"), g.transform("max")
    rng = mx - mn
    t1["semantic_score"] = np.where(rng > 1e-8, (t1["semantic_cosine_raw"] - mn) / rng, 0.0)

    out = t1[["example_id", "query_id", "product_id", "product_locale",
              "semantic_cosine_raw", "semantic_score"]].copy()
    assert len(out) == n_anchor
    assert out["semantic_score"].notna().all() and out["semantic_cosine_raw"].notna().all()
    path = os.path.join(CACHE, "task1_semantic_scores.parquet")
    out.to_parquet(path, index=False)

    meta = {
        "rows": int(len(out)),
        "queries": int(out["query_id"].nunique()),
        "unique_items_encoded": int(len(items)),
        "unique_queries_encoded": int(len(queries)),
        "model_path": os.path.relpath(MODEL_PATH, ROOT),
        "model_frozen": True,
        "device": device,
        "embedding_dim": int(prod_emb.shape[1]),
        "item_key": ["product_id", "product_locale"],
        "item_text_formula": "product_title + ' ' + product_description + ' ' + product_bullet_point (fillna '')",
        "normalization": "per-query min-max over the Task 1 candidate pool",
        "raw_cosine_persisted": True,
        "cosine_raw_stats": {
            "mean": float(out["semantic_cosine_raw"].mean()),
            "std": float(out["semantic_cosine_raw"].std()),
            "min": float(out["semantic_cosine_raw"].min()),
            "max": float(out["semantic_cosine_raw"].max()),
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(CACHE, "task1_semantic_scores_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n" + json.dumps(meta, indent=2), flush=True)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
