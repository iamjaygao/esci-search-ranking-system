"""
Task 3 -- Per-query NDCG@10 (BM25 baseline vs 17-feature LambdaMART) on the
exact same test query / candidate pool used by evaluation/evaluate_lambdamart.py
(via evaluation.evaluate_advanced.extract_test_advanced_features), plus a
query-level paired bootstrap over the per-query delta. No retraining;
loads the existing output/lambdamart_model.txt and reuses the existing
feature-extraction / metric functions verbatim.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from evaluation.evaluate_advanced import extract_test_advanced_features
from evaluation.metrics import apply_business_ndcg_labels
from scripts.build_query_slices import per_query_ndcg

OUT_DIR = f"{ROOT_DIR}/experiments/audit"
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


def main():
    with open(f"{ROOT_DIR}/output/lambdamart_features.json") as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    idf_map = meta.get("idf_map", {})

    print("=== Extracting test candidate pool (same function as evaluate_lambdamart.py) ===")
    df_test = extract_test_advanced_features(idf_map)

    df_truth = apply_business_ndcg_labels(df_test.copy(), budget_col='user_budget', price_col='price', stars_col='stars_clean')
    df_truth_standard = df_truth.copy()

    df_test = df_test.dropna(subset=feature_cols)
    features = df_test[feature_cols].values

    print("=== Running LambdaMART inference (existing checkpoint, no retraining) ===")
    model = lgb.Booster(model_file=f"{ROOT_DIR}/output/lambdamart_model.txt")
    df_test['predicted_score'] = model.predict(features)
    df_test['relevance'] = df_truth_standard['relevance']

    # Sanity check: reproduce official aggregate NDCG@10 before trusting per-query numbers.
    bm25_ndcg_series = per_query_ndcg(df_test, 'bm25_score', k=10)
    model_ndcg_series = per_query_ndcg(df_test, 'predicted_score', k=10)
    print(f"Reproduced BM25 NDCG@10 (aggregate, this candidate pool)      = {bm25_ndcg_series.mean():.4f}")
    print(f"Reproduced LambdaMART NDCG@10 (aggregate, this candidate pool) = {model_ndcg_series.mean():.4f}  (reference: ~0.846)")

    # Per-query metadata: candidate_count, label counts, query_length, query_mean_idf, query text
    label_counts = df_test.groupby(['query_id', 'esci_label']).size().unstack(fill_value=0)
    for col in ['E', 'S', 'C', 'I']:
        if col not in label_counts.columns:
            label_counts[col] = 0

    grp = df_test.groupby('query_id')
    meta_df = pd.DataFrame({
        "query_text": grp['query'].first(),
        "candidate_count": grp.size(),
        "query_length": grp['query_length'].first(),
        "query_mean_idf": grp['query_mean_idf'].first(),
    })
    meta_df = meta_df.join(label_counts[['E', 'S', 'C', 'I']]).rename(
        columns={'E': 'E_count', 'S': 'S_count', 'C': 'C_count', 'I': 'I_count'}
    )

    per_query = pd.DataFrame({
        "bm25_ndcg10": bm25_ndcg_series,
        "model_ndcg10": model_ndcg_series,
    })
    per_query = per_query.join(meta_df, how='inner')
    per_query['delta'] = per_query['model_ndcg10'] - per_query['bm25_ndcg10']
    per_query = per_query.reset_index().rename(columns={'index': 'query_id'})
    per_query = per_query[['query_id', 'query_text', 'bm25_ndcg10', 'model_ndcg10', 'delta',
                            'candidate_count', 'E_count', 'S_count', 'C_count', 'I_count',
                            'query_length', 'query_mean_idf']]

    per_query.to_csv(f"{OUT_DIR}/per_query_ndcg.csv", index=False)
    print(f"\nSaved per_query_ndcg.csv ({len(per_query)} queries)")

    # ============ Paired bootstrap over per-query delta ============
    deltas = per_query['delta'].values
    n = len(deltas)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    boot_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = deltas[idx].mean()

    observed_mean_delta = float(deltas.mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    frac_le_0 = float(np.mean(boot_means <= 0))
    frac_ge_0 = float(np.mean(boot_means >= 0))
    p_value = float(min(1.0, 2 * min(frac_le_0, frac_ge_0)))

    bootstrap_results = {
        "n_queries": int(n),
        "bm25_mean_ndcg10": float(per_query['bm25_ndcg10'].mean()),
        "lambdamart_mean_ndcg10": float(per_query['model_ndcg10'].mean()),
        "mean_delta": observed_mean_delta,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "p_value_two_sided": p_value,
    }
    with open(f"{OUT_DIR}/bootstrap_results.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    print(json.dumps(bootstrap_results, indent=2))


if __name__ == "__main__":
    main()
