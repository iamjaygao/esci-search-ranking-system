"""
Task 2 (evaluation half) -- Evaluate the newly trained 12-feature LambdaMART
model on the same test candidate pool / metric used for the official
17-feature LambdaMART NDCG@10 (evaluation/evaluate_lambdamart.py), and report
the delta against the 17-feature model.
"""
import os
import sys
import json

import lightgbm as lgb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from evaluation.evaluate_advanced import extract_test_advanced_features
from evaluation.metrics import ndcg_at_k, apply_business_ndcg_labels

OUT_DIR = f"{ROOT_DIR}/experiments/audit"

# From Task 3 (experiments/audit/bootstrap_results.json), computed via the identical
# extract_test_advanced_features + dropna(feature_cols) + ndcg_at_k pipeline.
FULL_17_FEATURE_NDCG10 = None


def main():
    with open(f"{OUT_DIR}/bootstrap_results.json") as f:
        boot = json.load(f)
    full_ndcg10 = boot["lambdamart_mean_ndcg10"]

    with open(f"{OUT_DIR}/model_12feature_features.json") as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    idf_map = meta.get("idf_map", {})
    assert len(feature_cols) == 12

    with open(f"{OUT_DIR}/task2_train_meta.json") as f:
        train_meta = json.load(f)

    df_test = extract_test_advanced_features(idf_map)
    df_truth = apply_business_ndcg_labels(df_test.copy(), budget_col='user_budget', price_col='price', stars_col='stars_clean')
    df_truth_standard = df_truth.copy()

    df_test = df_test.dropna(subset=feature_cols)
    features = df_test[feature_cols].values

    model = lgb.Booster(model_file=f"{OUT_DIR}/model_12feature.txt")
    df_test['predicted_score'] = model.predict(features)
    df_test['relevance'] = df_truth_standard['relevance']

    ndcg10_12feature = ndcg_at_k(df_test.copy(), score_col='predicted_score', k=10)

    result = {
        "ndcg10_17feature": full_ndcg10,
        "ndcg10_12feature": ndcg10_12feature,
        "absolute_delta": ndcg10_12feature - full_ndcg10,
        "best_iteration_17feature": 250,
        "best_iteration_12feature": train_meta["best_iteration"],
        "training_runtime_sec_12feature": train_meta["training_runtime_sec"],
        "excluded_features": train_meta["excluded_features"],
    }
    with open(f"{OUT_DIR}/result_12feature.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
