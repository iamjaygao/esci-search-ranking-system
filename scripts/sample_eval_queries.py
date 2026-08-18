"""
Deterministic evaluation query sample for full-catalog retrieval evaluation.

Samples N test-split, US-locale query_ids with a fixed seed so the evaluation
set is reproducible across runs. Also builds retrieval ground truth from
ESCI labels: broad-relevance (E+S+C) and exact-relevance (E only) product_id
sets per query. Unlabeled catalog products are never treated as negatives --
only known-labeled products participate in the ground truth.

Usage:
    python scripts/sample_eval_queries.py
"""
import os
import sys
import json

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, ROOT_DIR

SEED = 42
N_QUERIES = 5000
LOCALE = "us"
OUT_DIR = f"{ROOT_DIR}/output/full_retrieval"
BROAD_LABELS = {"E", "S", "C"}
EXACT_LABELS = {"E"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)

    df_test = df_ex[(df_ex['split'] == 'test') & (df_ex['product_locale'] == LOCALE)].copy()
    all_query_ids = sorted(df_test['query_id'].unique().tolist())
    print(f"US-locale test queries available: {len(all_query_ids)}")

    rng = np.random.RandomState(SEED)
    n = min(N_QUERIES, len(all_query_ids))
    sampled = sorted(rng.choice(all_query_ids, size=n, replace=False).tolist())
    print(f"Sampled {len(sampled)} queries, seed={SEED}")

    with open(f"{OUT_DIR}/eval_query_ids.json", "w") as f:
        json.dump({"seed": SEED, "locale": LOCALE, "n_requested": N_QUERIES,
                   "n_available": len(all_query_ids), "query_ids": sampled}, f, indent=2)
    print(f"Saved {OUT_DIR}/eval_query_ids.json")

    df_eval = df_test[df_test['query_id'].isin(set(sampled))].copy()

    query_text = df_eval.groupby('query_id')['query'].first().to_dict()

    ground_truth = {}
    for qid, group in df_eval.groupby('query_id'):
        broad = set(group[group['esci_label'].isin(BROAD_LABELS)]['product_id'])
        exact = set(group[group['esci_label'].isin(EXACT_LABELS)]['product_id'])
        ground_truth[qid] = {
            "query": query_text[qid],
            "relevant_broad": sorted(broad),
            "relevant_exact": sorted(exact),
            "num_broad": len(broad),
            "num_exact": len(exact),
        }

    with open(f"{OUT_DIR}/ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved {OUT_DIR}/ground_truth.json")

    n_broad = [v['num_broad'] for v in ground_truth.values()]
    n_exact = [v['num_exact'] for v in ground_truth.values()]
    print(f"\nBroad-relevant (E+S+C) per query: mean={np.mean(n_broad):.2f}, median={np.median(n_broad):.0f}, "
          f"min={min(n_broad)}, max={max(n_broad)}, queries_with_zero={sum(1 for x in n_broad if x==0)}")
    print(f"Exact-relevant (E only) per query: mean={np.mean(n_exact):.2f}, median={np.median(n_exact):.0f}, "
          f"min={min(n_exact)}, max={max(n_exact)}, queries_with_zero={sum(1 for x in n_exact if x==0)}")


if __name__ == "__main__":
    main()
