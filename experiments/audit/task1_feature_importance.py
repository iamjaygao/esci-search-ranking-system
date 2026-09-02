"""
Task 1 -- LambdaMART feature gain/split importance + best_iteration.
No retraining: loads the existing output/lambdamart_model.txt Booster directly.
"""
import os
import sys
import json

import lightgbm as lgb
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

OUT_DIR = f"{ROOT_DIR}/experiments/audit"

FEATURE_GROUP = {
    'semantic_score': 'semantic',
    'bm25_score': 'lexical',
    'word_overlap': 'lexical',
    'log_price': 'item',
    'is_price_missing': 'item',
    'stars_clean': 'item',
    'log_review_count': 'item',
    'is_rating_missing': 'item',
    'query_length': 'query',
    'query_mean_idf': 'query',
    'query_max_idf': 'query',
    'user_budget': 'query',
    'cheap_intent': 'query',
    'brand_match': 'interaction',
    'color_match': 'interaction',
    'is_over_budget': 'interaction',
    'is_dominant_category': 'interaction',
}


def main():
    with open(f"{ROOT_DIR}/output/lambdamart_features.json") as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    assert len(feature_cols) == 17, f"expected 17 features, got {len(feature_cols)}"

    booster = lgb.Booster(model_file=f"{ROOT_DIR}/output/lambdamart_model.txt")
    num_trees = booster.num_trees()
    best_iteration_attr = booster.best_iteration  # not preserved across save/load in LightGBM text format

    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    assert len(gain) == len(feature_cols) == len(split)

    total_gain = float(gain.sum())
    df = pd.DataFrame({
        "feature": feature_cols,
        "group": [FEATURE_GROUP[f] for f in feature_cols],
        "gain": gain.astype(float),
        "split": split.astype(int),
    })
    df["gain_pct"] = 100.0 * df["gain"] / total_gain
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    df["gain_rank"] = df.index + 1
    df = df[["feature", "group", "gain", "gain_pct", "split", "gain_rank"]]

    df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

    result = {
        "num_features": len(feature_cols),
        "num_trees_in_saved_model": int(num_trees),
        "n_estimators_cap": 1000,
        "early_stopping_rounds": 50,
        "early_stopping_triggered": bool(num_trees < 1000),
        "best_iteration_from_reloaded_booster": int(best_iteration_attr),
        "best_iteration_note": (
            "LightGBM's text model format (Booster.save_model / lgb.Booster(model_file=...)) does not "
            "persist the sklearn-wrapper best_iteration_ value directly -- the reloaded Booster reports "
            "best_iteration=-1 (unset) via that attribute. num_trees()=250 < n_estimators cap of 1000 "
            "confirms early stopping DID trigger during training (otherwise the model would contain 1000 "
            "trees). Separately, in Task 2 (experiments/audit/task2_train_12feature.py), the identical "
            "training procedure was run in-process and log_evaluation printed evaluation scores through "
            "iteration 280, but model.best_iteration_ == model.booster_.num_trees() == 230 after fit() "
            "returned -- confirming empirically that LightGBM's sklearn API rolls the booster back to the "
            "best iteration when early stopping triggers (the trees added after the best iteration are "
            "discarded from the returned/saved booster, not merely ignored at prediction time). Applying "
            "that same, now-confirmed mechanism to the reloaded 17-feature booster: since save_model() "
            "persists whatever booster state existed immediately after fit(), and that state is already "
            "pruned to the best iteration, best_iteration = num_trees_in_saved_model = 250 exactly (not a "
            "guess or an approximation -- it follows from the confirmed truncation behavior). No training "
            "log for the original scripts/train_lambdamart.py run exists in the repo to cross-check this "
            "independently."
        ),
        "best_iteration_inferred": int(num_trees),
    }
    with open(f"{OUT_DIR}/task1_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(df.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
