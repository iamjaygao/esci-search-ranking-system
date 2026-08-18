import os
import sys
import json
import lightgbm as lgb

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT_DIR
from evaluation.metrics import ndcg_at_k, recall_at_k, apply_business_ndcg_labels
from evaluation.evaluate_advanced import extract_test_advanced_features


def main():
    print("\n--- Phase 1: Loading Feature Metadata ---")
    try:
        with open(f'{ROOT_DIR}/output/lambdamart_features.json', "r") as f:
            meta = json.load(f)
        feature_cols = meta["features"]
        idf_map = meta.get("idf_map", {})
    except FileNotFoundError:
        print("ERROR: lambdamart_features.json not found. Run scripts/train_lambdamart.py first.")
        return

    print("\n--- Phase 2: Feature Extraction ---")
    df_test = extract_test_advanced_features(idf_map)

    print("\n--- Phase 3: Generating Ground Truths ---")
    # Same relevance scale used for BM25/Two-Tower/both MLP rerankers, so NDCG@10
    # numbers are directly comparable across the whole leaderboard.
    df_truth = apply_business_ndcg_labels(df_test.copy(), budget_col='user_budget', price_col='price', stars_col='stars_clean')
    df_truth_standard = df_truth.copy()
    df_truth_business = df_truth.copy()
    df_truth_business['relevance'] = df_truth_business['business_relevance']

    df_test = df_test.dropna(subset=feature_cols)
    features = df_test[feature_cols].values

    print("\n--- Phase 4: Running Inference ---")
    model = lgb.Booster(model_file=f'{ROOT_DIR}/output/lambdamart_model.txt')
    df_test['predicted_score'] = model.predict(features)

    print("\n--- Phase 5: Calculating Metrics ---")
    df_test['relevance'] = df_truth_standard['relevance']
    standard_ndcg = ndcg_at_k(df_test.copy(), score_col='predicted_score', k=10)
    standard_recall = recall_at_k(df_test.copy(), df_truth_standard, score_col='predicted_score', k=10)

    df_test_business = df_test.copy()
    df_test_business['relevance'] = df_truth_business['business_relevance']
    business_ndcg = ndcg_at_k(df_test_business, score_col='predicted_score', k=10)

    print("\n" + "=" * 60)
    print(" LAMBDAMART RERANKER EVALUATION")
    print("=" * 60)
    print("1. Standard Textual Relevance (Compared to ESCI Baseline)")
    print(f"   Standard NDCG@10:  {standard_ndcg:.4f}")
    print(f"   Recall@10:         {standard_recall:.4f}")
    print("-" * 60)
    print("2. Business Relevance (Intent, Budget, Quality)")
    print(f"   Business NDCG@10:  {business_ndcg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
