"""
Builds read-only caches used by the later audit phases, by IMPORTING (never
modifying) the repo's own feature extractors and scoring the saved models.

Outputs (all under experiments/ranking_v2/audit/_cache/):
  test_scored.parquet   -- official test candidate pool + 17 features
                           + bm25/tt/mlp/lambdamart scores + relevance
  train_features.parquet-- official train candidate pool + 17 features + labels
  cache_meta.json

Nothing here retrains anything: LambdaMART is loaded from
output/lambdamart_model.txt and the MLP from output/best_advanced_reranker.pth.

Usage: python experiments/ranking_v2/audit/scripts/00_build_caches.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from config import EXAMPLES_PATH, PRODUCTS_PATH  # noqa: E402
from reranking.advanced_features import extract_advanced_features, ALL_FEATURES  # noqa: E402
from reranking.advanced_model import AdvancedDeepReranker  # noqa: E402
from evaluation.evaluate_advanced import extract_test_advanced_features  # noqa: E402
from evaluation.metrics import apply_business_ndcg_labels  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "ranking_v2", "audit")
CACHE = os.path.join(OUT, "_cache")

KEEP_COLS = (
    ["query_id", "product_id", "query", "product_locale", "product_title",
     "product_brand", "esci_label", "target_score", "category", "price",
     "stars", "ratings"] + ALL_FEATURES
)


def main():
    os.makedirs(CACHE, exist_ok=True)
    meta = {}

    # ---------------- LambdaMART feature metadata (official) ----------------
    with open(os.path.join(ROOT, "output/lambdamart_features.json")) as f:
        lgb_meta = json.load(f)
    lgb_features = lgb_meta["features"]
    idf_map = lgb_meta.get("idf_map", {})
    meta["lambdamart_features"] = lgb_features
    meta["idf_map_size"] = len(idf_map)

    with open(os.path.join(ROOT, "output/advanced_normalization_stats.json")) as f:
        norm = json.load(f)
    mlp_features = norm["features"]
    meta["mlp_features"] = mlp_features
    meta["feature_sets_identical"] = (lgb_features == mlp_features)
    meta["mlp_idf_map_size"] = len(norm.get("idf_map", {}))
    meta["idf_maps_identical"] = (norm.get("idf_map", {}) == idf_map)

    # ---------------- TEST ----------------
    print("=== Building TEST feature frame (official eval path) ===")
    df_test = extract_test_advanced_features(idf_map)
    meta["test_rows_raw"] = int(len(df_test))

    # exactly what evaluation/evaluate_lambdamart.py does
    df_truth = apply_business_ndcg_labels(
        df_test.copy(), budget_col="user_budget", price_col="price", stars_col="stars_clean"
    )
    df_test["relevance_standard"] = df_truth["relevance"]
    df_test["relevance_business"] = df_truth["business_relevance"]

    df_test = df_test.dropna(subset=lgb_features)
    meta["test_rows_after_dropna"] = int(len(df_test))
    meta["test_queries_after_dropna"] = int(df_test["query_id"].nunique())

    # ---- LambdaMART inference (saved booster, no retraining)
    import lightgbm as lgb
    booster = lgb.Booster(model_file=os.path.join(ROOT, "output/lambdamart_model.txt"))
    meta["lambdamart_num_trees"] = int(booster.num_trees())
    df_test["lambdamart_score"] = booster.predict(df_test[lgb_features].values)

    # ---- MLP inference (saved weights, no retraining)
    mean = np.array(norm["mean"])
    std = np.array(norm["std"])
    Xn = (df_test[mlp_features].values - mean) / std
    device = torch.device("cpu")
    mlp = AdvancedDeepReranker(input_dim=len(mlp_features)).to(device)
    mlp.load_state_dict(torch.load(
        os.path.join(ROOT, "output/best_advanced_reranker.pth"),
        map_location=device, weights_only=True))
    mlp.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xn), 100000):
            xb = torch.tensor(Xn[i:i + 100000], dtype=torch.float32)
            preds.append(mlp(xb).squeeze(-1).numpy())
    df_test["mlp_score"] = np.concatenate(preds)

    # bm25_score / semantic_score columns already present as the raw retriever scores
    df_test["bm25_raw"] = df_test["bm25_score"]
    df_test["tt_raw"] = df_test["semantic_score"]

    keep = [c for c in KEEP_COLS if c in df_test.columns] + [
        "relevance_standard", "relevance_business",
        "lambdamart_score", "mlp_score", "bm25_raw", "tt_raw",
    ]
    df_test[keep].to_parquet(os.path.join(CACHE, "test_scored.parquet"), index=False)
    print("wrote test_scored.parquet", df_test.shape)

    # ---------------- TRAIN ----------------
    print("\n=== Building TRAIN feature frame (official training path) ===")
    df_train, feat_cols, train_idf = extract_advanced_features(
        EXAMPLES_PATH, PRODUCTS_PATH,
        os.path.join(ROOT, "output/bm25_scores_train.csv"),
        os.path.join(ROOT, "output/two_tower_scores_train.csv"),
        os.path.join(ROOT, "esci-data/esci-s_dataset/esci_s_products.parquet"),
    )
    meta["train_rows"] = int(len(df_train))
    meta["train_queries"] = int(df_train["query_id"].nunique())
    meta["train_feature_cols"] = feat_cols
    meta["train_idf_map_size"] = len(train_idf)
    meta["train_idf_map_matches_saved"] = (train_idf == idf_map)

    keep_tr = [c for c in KEEP_COLS if c in df_train.columns]
    df_train[keep_tr].to_parquet(os.path.join(CACHE, "train_features.parquet"), index=False)
    print("wrote train_features.parquet", df_train.shape)

    with open(os.path.join(CACHE, "cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in meta.items() if k != "lambdamart_features"}, indent=2, default=str))


if __name__ == "__main__":
    main()
