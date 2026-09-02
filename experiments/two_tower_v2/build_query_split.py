"""
Phase 1 -- Query-level train/dev split for Two-Tower fine-tuning.
Splits by unique query_id (never by pair) so no query leaks between train
and dev. Fixed seed=42. Does not touch scripts/train_two_tower.py or any
existing model/output file.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from config import EXAMPLES_PATH, PRODUCTS_PATH

OUT_DIR = f"{ROOT_DIR}/experiments/two_tower_v2/splits"
SEED = 42
DEV_RATIO = 0.10
LOCALE = "us"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_examples = pd.read_parquet(EXAMPLES_PATH)
    df_products = pd.read_parquet(PRODUCTS_PATH)
    df = pd.merge(df_examples, df_products, on=["product_id", "product_locale"], how="left")
    df = df[df["small_version"] == 1]
    df = df[df["split"] == "train"]
    df = df[df["product_locale"] == LOCALE]
    df = df[df["esci_label"].isin(["E", "S"])]

    unique_queries = sorted(df["query_id"].astype(str).unique().tolist())
    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(unique_queries)
    n_dev = int(len(shuffled) * DEV_RATIO)
    dev_queries = sorted(shuffled[:n_dev].tolist())
    train_queries = sorted(shuffled[n_dev:].tolist())

    assert set(train_queries).isdisjoint(set(dev_queries)), "train/dev query overlap detected"

    with open(f"{OUT_DIR}/train_queries.txt", "w") as f:
        f.write("\n".join(train_queries))
    with open(f"{OUT_DIR}/dev_queries.txt", "w") as f:
        f.write("\n".join(dev_queries))

    df["query_id"] = df["query_id"].astype(str)
    n_train_pairs = int(df["query_id"].isin(set(train_queries)).sum())
    n_dev_pairs = int(df["query_id"].isin(set(dev_queries)).sum())

    summary = {
        "seed": SEED, "dev_ratio": DEV_RATIO, "locale": LOCALE,
        "n_unique_queries_total": len(unique_queries),
        "n_train_queries": len(train_queries), "n_dev_queries": len(dev_queries),
        "n_train_pairs": n_train_pairs, "n_dev_pairs": n_dev_pairs,
        "disjoint_verified": True,
        "source_filter": {"small_version": 1, "split": "train", "product_locale": LOCALE, "esci_label": ["E", "S"]},
    }
    with open(f"{OUT_DIR}/split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
