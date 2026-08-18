"""
Full-catalog retrieval evaluation: builds a US-locale-only BM25 index and a
US-locale-only Two-Tower FAISS index over the real product catalog (not the
ESCI per-query candidate pool).

Scoped to product_locale == 'us' because:
  1. The global product catalog mixes us/jp/es listings, and the existing
     build_global_bm25_index()/build_global_tt_index() build ONE mixed-locale
     index with no locale field stored per row -- searching it for a US query
     could retrieve jp/es products, which would deflate Recall@K for reasons
     that have nothing to do with retrieval quality.
  2. 11,567 product_ids appear in more than one locale (same ASIN sold in
     multiple markets) -- pre-filtering to a single locale before building
     the index removes this ambiguity entirely rather than patching it after
     the fact.
  3. This is a locale-consistency fix applied identically to both retrievers,
     not a retrieval-quality tuning decision.

Reuses retrieval/bm25.py's build_global_bm25_index() and
retrieval/two_tower.py's build_global_tt_index() verbatim -- no tokenizer or
model changes, only the input dataframe is pre-filtered to one locale.

Usage:
    python scripts/build_full_catalog_indices.py
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PRODUCTS_PATH, ROOT_DIR
from retrieval.bm25 import build_global_bm25_index
from retrieval.two_tower import build_global_tt_index

LOCALE = "us"
OUT_DIR = f"{ROOT_DIR}/output/full_retrieval"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading product catalog...")
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_pr['product_id'] = df_pr['product_id'].astype(str)

    print(f"Full catalog: {len(df_pr)} rows, {df_pr['product_id'].nunique()} unique product_id, "
          f"locales: {dict(df_pr['product_locale'].value_counts())}")

    df_us = df_pr[df_pr['product_locale'] == LOCALE].copy().reset_index(drop=True)
    print(f"US-locale subset: {len(df_us)} rows, {df_us['product_id'].nunique()} unique product_id")
    assert len(df_us) == df_us['product_id'].nunique(), "Unexpected duplicate product_id within us locale"

    print("\n--- Building BM25 index (US-only catalog) ---")
    t0 = time.time()
    bm25_index, bm25_ids = build_global_bm25_index(df_us)
    print(f"BM25 index built in {time.time()-t0:.1f}s over {len(bm25_ids)} products")
    bm25_index.save(f"{OUT_DIR}/bm25s_index_us")
    import json
    with open(f"{OUT_DIR}/bm25_ids_us.json", "w") as f:
        json.dump(bm25_ids, f)
    print(f"Saved {OUT_DIR}/bm25s_index_us and bm25_ids_us.json")

    print("\n--- Building Two-Tower FAISS index (US-only catalog) ---")
    t0 = time.time()
    _, tt_index, tt_ids = build_global_tt_index(df_us)
    print(f"FAISS index built in {time.time()-t0:.1f}s over {len(tt_ids)} products, dim={tt_index.d}")
    import faiss
    faiss.write_index(tt_index, f"{OUT_DIR}/tt_index_us.faiss")
    with open(f"{OUT_DIR}/tt_ids_us.json", "w") as f:
        json.dump(tt_ids, f)
    print(f"Saved {OUT_DIR}/tt_index_us.faiss and tt_ids_us.json")

    print("\nDone. Index product counts:")
    print(f"  BM25 indexed products:  {len(bm25_ids)}")
    print(f"  FAISS indexed products: {len(tt_ids)}")


if __name__ == "__main__":
    main()
