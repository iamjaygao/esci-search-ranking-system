import os
import sys
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, PRODUCTS_PATH, ROOT_DIR
from reranking.advanced_features import extract_advanced_features

# lambdarank's label is an index into LightGBM's internal gain table, not a
# fractional relevance weight -- so this is a separate encoding from the
# target_score used by the pairwise MLP rerankers (E=1.0/S=0.1/C=0.01/I=0.0).
# Only the ordering (I < C < S < E) needs to match between the two.
LABEL_MAP_ORDINAL = {'E': 3, 'S': 2, 'C': 1, 'I': 0}
LABEL_GAIN = [0, 1, 3, 7]


def train_lambdamart():
    bm25_csv_path = f'{ROOT_DIR}/output/bm25_scores_train.csv'
    semantic_csv_path = f'{ROOT_DIR}/output/two_tower_scores_train.csv'
    esci_s_path = f'{ROOT_DIR}/esci-data/esci-s_dataset/esci_s_products.parquet'

    df_all, feature_columns, idf_map = extract_advanced_features(
        EXAMPLES_PATH, PRODUCTS_PATH, bm25_csv_path, semantic_csv_path, esci_s_path
    )
    df_all['lgb_label'] = df_all['esci_label'].map(LABEL_MAP_ORDINAL).fillna(0).astype(int)

    print("\nSplitting queries into Train (85%) and Validation (15%)...")
    unique_queries = df_all['query_id'].unique().tolist()
    train_queries, val_queries = train_test_split(unique_queries, test_size=0.15, random_state=42)

    # LightGBM needs rows grouped contiguously by query, paired with a `group`
    # array of per-query row counts in that same order.
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

    # Conservative first-pass hyperparameters -- goal is an apples-to-apples
    # comparison against the 17-feature MLP reranker, not a tuned model.
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

    print("\nStarting LambdaMART Training...")
    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        eval_at=[10],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)],
    )

    os.makedirs(f'{ROOT_DIR}/output', exist_ok=True)
    model.booster_.save_model(f'{ROOT_DIR}/output/lambdamart_model.txt')
    with open(f'{ROOT_DIR}/output/lambdamart_features.json', "w") as f:
        json.dump({"features": feature_columns, "idf_map": idf_map}, f)
    print("Saved LambdaMART model and feature metadata.")

    print("\nFeature importance (gain):")
    importances = model.booster_.feature_importance(importance_type='gain')
    for name, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        print(f"  {name:<22s} {imp:.1f}")

    return model


if __name__ == "__main__":
    train_lambdamart()
