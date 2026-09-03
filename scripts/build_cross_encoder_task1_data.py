"""
Attach the product text fields the Cross-Encoder needs to the FROZEN Task 1
splits.

Why this script is necessary
----------------------------
The frozen Task 1 parquets carry `product_title` and `product_brand` but NOT
`product_color`, `product_bullet_point` or `product_description` -- they were
built for the 17-feature LambdaMART pipeline, which never needed them. Verified
column list:

    example_id, query_id, query, product_id, product_locale, doc_id,
    esci_label, gain, lgb_label, split, product_title, product_brand,
    category, bm25_raw, semantic_cosine_raw, <17 features>

The three missing fields DO exist in the raw products table. This script
LEFT-joins them on (product_id, product_locale) -- locale-aware, asserted not
to fan out -- and writes a cache alongside the experiment.

The frozen split parquets are NEVER modified, and no row is added or removed.

Usage:
    python scripts/build_cross_encoder_task1_data.py --splits train dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reranking.cross_encoder import TEXT_FIELDS  # noqa: E402

BENCH = os.path.join(ROOT, "experiments", "ranking_v2", "kdd_task1_benchmark")
OUT_DIR = os.path.join(ROOT, "experiments", "ranking_v2", "cross_encoder_task1", "_cache")
PRODUCTS = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet")

#: Expected shape of each frozen split, from the Task 1 REPORT. A mismatch is a
#: STOP condition -- this script must never silently rebuild or repair data.
EXPECTED = {"train": (663407, 28733), "dev": (118231, 5071), "test": (336373, 14496)}

KEEP = ["example_id", "query_id", "query", "product_id", "product_locale",
        "doc_id", "esci_label", "gain", "split", "product_title", "product_brand"]
FROM_PRODUCTS = ["product_color", "product_bullet_point", "product_description"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev"],
                    choices=["train", "dev", "test"],
                    help="test is excluded by default: TEST LOCK is in force")
    ap.add_argument("--max_chars_per_field", type=int, default=4000,
                    help="storage-only cap on each raw text field. 4000 chars is "
                         ">10x what 256 tokens can hold, so it cannot affect any "
                         "model with max_length<=256. Set 0 to disable.")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    cap = args.max_chars_per_field or None
    if cap is not None and cap < 2560:
        raise SystemExit(f"--max_chars_per_field={cap} is too small to be storage-only; "
                         "it could truncate content a 256-token model would have seen")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading raw products table (locale-aware key) ...", flush=True)
    pr = pd.read_parquet(PRODUCTS, columns=["product_id", "product_locale"] + FROM_PRODUCTS)
    if pr.duplicated(["product_id", "product_locale"]).any():
        raise SystemExit("STOP: products table has duplicate (product_id, product_locale) keys")
    for c in FROM_PRODUCTS:
        if cap is not None:
            pr[c] = pr[c].astype("string").str.slice(0, cap)

    manifest = {"source_products": os.path.relpath(PRODUCTS, ROOT),
                "join_keys": ["product_id", "product_locale"],
                "fields_from_frozen_split": ["product_title", "product_brand"],
                "fields_joined_from_products": FROM_PRODUCTS,
                "text_field_order": [f"{lbl} <- {col}" for lbl, col in TEXT_FIELDS],
                "max_chars_per_field": cap,
                "frozen_splits_modified": False,
                "splits": {}}

    for split in args.splits:
        src = os.path.join(BENCH, f"{split}_task1.parquet")
        if not os.path.exists(src):
            raise SystemExit(f"STOP: frozen split not found: {src}")
        df = pd.read_parquet(src, columns=KEEP)

        exp_rows, exp_q = EXPECTED[split]
        got_rows, got_q = len(df), df["query_id"].nunique()
        if (got_rows, got_q) != (exp_rows, exp_q):
            raise SystemExit(
                f"STOP: {split} artifact disagrees with the Task 1 REPORT.\n"
                f"  expected {exp_rows} rows / {exp_q} queries\n"
                f"  actual   {got_rows} rows / {got_q} queries\n"
                "Not rebuilding or repairing. Human decision required.")

        n0 = len(df)
        df = df.merge(pr, on=["product_id", "product_locale"], how="left")
        if len(df) != n0:
            raise SystemExit(f"STOP: products join fanned out on {split}: {len(df)} != {n0}")

        nulls = {c: int(df[c].isna().sum()) for c in FROM_PRODUCTS}
        title_null = int(df["product_title"].isna().sum())
        out = os.path.join(args.out_dir, f"{split}_text.parquet")
        df.to_parquet(out, index=False)

        manifest["splits"][split] = {
            "source": os.path.relpath(src, ROOT),
            "output": os.path.relpath(out, ROOT),
            "rows": int(len(df)), "queries": int(df["query_id"].nunique()),
            "rows_match_report": True,
            "null_counts_joined_fields": nulls,
            "null_count_product_title": title_null,
            "size_mb": round(os.path.getsize(out) / 1e6, 1),
        }
        print(f"  {split}: {len(df)} rows / {df['query_id'].nunique()} queries "
              f"-> {out} ({manifest['splits'][split]['size_mb']} MB); "
              f"nulls {nulls}", flush=True)

    with open(os.path.join(args.out_dir, "build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print("wrote", os.path.join(args.out_dir, "build_manifest.json"))


if __name__ == "__main__":
    main()
