"""
Task 2 -- 12-feature LambdaMART (drops the 5 static/query-independent item
features: log_price, is_price_missing, stars_clean, log_review_count,
is_rating_missing). Mirrors scripts/train_lambdamart.py exactly (split, seed,
hyperparameters, early stopping, label_gain) except for the feature set.
Does not modify or overwrite scripts/train_lambdamart.py or its outputs.
"""
import os
import sys
import json
import time

import lightgbm as lgb
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from config import EXAMPLES_PATH, PRODUCTS_PATH
from reranking.advanced_features import extract_advanced_features

OUT_DIR = f"{ROOT_DIR}/experiments/audit"

LABEL_MAP_ORDINAL = {'E': 3, 'S': 2, 'C': 1, 'I': 0}
LABEL_GAIN = [0, 1, 3, 7]

EXCLUDED = ['log_price', 'is_price_missing', 'stars_clean', 'log_review_count', 'is_rating_missing']


def main():
    bm25_csv_path = f'{ROOT_DIR}/output/bm25_scores_train.csv'
    semantic_csv_path = f'{ROOT_DIR}/output/two_tower_scores_train.csv'
    esci_s_path = f'{ROOT_DIR}/esci-data/esci-s_dataset/esci_s_products.parquet'

    df_all, feature_columns, idf_map = extract_advanced_features(
        EXAMPLES_PATH, PRODUCTS_PATH, bm25_csv_path, semantic_csv_path, esci_s_path,
        excluded_features=EXCLUDED,
    )
    assert len(feature_columns) == 12, f"expected 12 features, got {len(feature_columns)}: {feature_columns}"
    df_all['lgb_label'] = df_all['esci_label'].map(LABEL_MAP_ORDINAL).fillna(0).astype(int)

    print("\nSplitting queries into Train (85%) and Validation (15%)...")
    unique_queries = df_all['query_id'].unique().tolist()
    train_queries, val_queries = train_test_split(unique_queries, test_size=0.15, random_state=42)

    df_train = df_all[df_all['query_id'].isin(train_queries)].sort_values('query_id').reset_index(drop=True)
    df_val = df_all[df_all['query_id'].isin(val_queries)].sort_values('query_id').reset_index(drop=True)

    group_train = df_train.groupby('query_id', sort=False).size().to_numpy()
    group_val = df_val.groupby('query_id', sort=False).size().to_numpy()
    assert group_train.sum() == len(df_train), "train group/row count mismatch"
    assert group_val.sum() == len(df_val), "val group/row count mismatch"

    X_train, y_train = df_train[feature_columns].values, df_train['lgb_label'].values
    X_val, y_val = df_val[feature_columns].values, df_val['lgb_label'].values

    print(f"Train: {len(df_train)} rows / {len(group_train)} queries")
    print(f"Val:   {len(df_val)} rows / {len(group_val)} queries")

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=LABEL_GAIN,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    print("\nStarting LambdaMART Training (12-feature)...")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        eval_at=[10],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)],
    )
    runtime_sec = time.time() - t0

    os.makedirs(OUT_DIR, exist_ok=True)
    model.booster_.save_model(f'{OUT_DIR}/model_12feature.txt')
    with open(f'{OUT_DIR}/model_12feature_features.json', "w") as f:
        json.dump({"features": feature_columns, "idf_map": idf_map}, f)

    best_iteration = int(model.best_iteration_) if model.best_iteration_ else None
    num_trees = model.booster_.num_trees()

    print(f"\nTraining runtime: {runtime_sec:.1f}s")
    print(f"best_iteration_ (sklearn, in-memory): {best_iteration}")
    print(f"num_trees in booster: {num_trees}")

    print("\nFeature importance (gain):")
    importances = model.booster_.feature_importance(importance_type='gain')
    for name, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        print(f"  {name:<22s} {imp:.1f}")

    with open(f"{OUT_DIR}/task2_train_meta.json", "w") as f:
        json.dump({
            "feature_columns": feature_columns,
            "excluded_features": EXCLUDED,
            "train_rows": len(df_train), "train_queries": len(group_train),
            "val_rows": len(df_val), "val_queries": len(group_val),
            "training_runtime_sec": runtime_sec,
            "best_iteration": best_iteration,
            "num_trees_in_saved_model": int(num_trees),
        }, f, indent=2)


if __name__ == "__main__":
    main()
