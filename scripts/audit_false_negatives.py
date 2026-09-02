"""
Phase 2 -- False-negative audit for the current MultipleNegativesRankingLoss
in-batch-negative training setup. Pure data analysis, no training, no model
loading. Reuses the same positive-pair filter as scripts/train_two_tower.py
(read-only reuse). Outputs experiments/two_tower_v2/phase2_false_negative/false_negative_audit.json.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import EXAMPLES_PATH, PRODUCTS_PATH

OUT_DIR = f"{ROOT_DIR}/experiments/two_tower_v2/phase2_false_negative"
SEED = 42
BATCH_SIZE = 64
LOCALE = "us"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_examples = pd.read_parquet(EXAMPLES_PATH)
    df_products = pd.read_parquet(PRODUCTS_PATH)
    df = pd.merge(df_examples, df_products, on=["product_id", "product_locale"], how="left")
    df = df[df["small_version"] == 1]
    df = df[df["split"] == "train"]
    df = df[df["product_locale"] == LOCALE]
    df_pos = df[df["esci_label"].isin(["E", "S"])].copy()
    df_pos["query_id"] = df_pos["query_id"].astype(str)
    df_pos["product_id"] = df_pos["product_id"].astype(str)

    # ---- 1. Same-query false negatives (multiple positives per query) ----
    pos_per_query = df_pos.groupby("query_id").size()
    n_queries_multi = int((pos_per_query > 1).sum())

    stats = {
        "num_queries": int(df_pos["query_id"].nunique()),
        "num_pairs": int(len(df_pos)),
        "mean_positives_per_query": float(pos_per_query.mean()),
        "median_positives_per_query": float(pos_per_query.median()),
        "p95_positives_per_query": float(pos_per_query.quantile(0.95)),
        "max_positives_per_query": int(pos_per_query.max()),
        "queries_with_multiple_positives": n_queries_multi,
        "queries_with_multiple_positives_pct": 100.0 * n_queries_multi / df_pos["query_id"].nunique(),
    }

    # ---- Empirical same-query batch collision rate under CURRENT (V0) random
    # batching: DataLoader(shuffle=True, batch_size=64, drop_last=True), no seed. ----
    rng = np.random.RandomState(SEED)  # a fixed seed here only for THIS audit's reproducibility,
                                        # not a claim that V0 itself is seeded (it is not).
    n_trials = 5
    collision_rates = []
    for trial in range(n_trials):
        order = rng.permutation(len(df_pos))
        qids = df_pos["query_id"].values[order]
        n_batches = len(qids) // BATCH_SIZE
        collisions = 0
        total_pairs_in_full_batches = n_batches * BATCH_SIZE
        for b in range(n_batches):
            batch_qids = qids[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            vals, counts = np.unique(batch_qids, return_counts=True)
            collisions += int((counts[counts > 1] - 1).sum())  # extra pairs beyond the first, per colliding query
        collision_rates.append(collisions / total_pairs_in_full_batches)

    stats["observed_same_query_batch_collision_rate"] = {
        "batch_size": BATCH_SIZE, "n_shuffles_tested": n_trials,
        "mean_rate": float(np.mean(collision_rates)),
        "per_trial_rates": [float(x) for x in collision_rates],
        "definition": "fraction of (query,product) pairs in full batches that share a query_id with an earlier pair in the same batch, averaged over independent random shuffles",
    }

    # ---- 2. Semantic false negatives: does a training-positive product also
    # appear as a judged-relevant (E/S/C) item for a DIFFERENT query? ----
    df_all_labels = df[df["esci_label"].isin(["E", "S", "C"])][["query_id", "product_id", "esci_label"]].copy()
    df_all_labels["query_id"] = df_all_labels["query_id"].astype(str)
    df_all_labels["product_id"] = df_all_labels["product_id"].astype(str)
    product_to_queries = df_all_labels.groupby("product_id")["query_id"].apply(set).to_dict()

    def n_other_relevant_queries(row):
        qs = product_to_queries.get(row["product_id"], set())
        return len(qs - {row["query_id"]})

    df_pos["n_other_queries_this_product_is_relevant_for"] = df_pos.apply(n_other_relevant_queries, axis=1)
    n_with_cross_query_relevance = int((df_pos["n_other_queries_this_product_is_relevant_for"] > 0).sum())

    stats["semantic_false_negative_risk"] = {
        "definition": "for each training-positive (query,product) pair, whether that SAME product is also ESCI-labeled E/S/C for at least one DIFFERENT query. If that other query's own pair lands in the same random batch, this product -- correct for the other query too -- is still only usable as an in-batch NEGATIVE for it under plain MNRL.",
        "n_pairs_with_cross_query_relevant_product": n_with_cross_query_relevance,
        "pct_of_training_pairs": 100.0 * n_with_cross_query_relevance / len(df_pos),
        "mean_other_relevant_queries_per_product": float(df_pos["n_other_queries_this_product_is_relevant_for"].mean()),
    }

    with open(f"{OUT_DIR}/false_negative_audit.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
