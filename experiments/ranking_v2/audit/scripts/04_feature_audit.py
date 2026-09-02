"""
Phase E -- Feature audit: per-feature stats, redundancy, importance concentration.

Reads only the caches built by 00_build_caches.py plus the saved LightGBM
booster. No retraining, no feature changes.

Writes: feature_audit.csv, feature_correlation.csv, feature_importance.csv,
        feature_audit.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "experiments", "ranking_v2", "audit")
CACHE = os.path.join(OUT, "_cache")

GROUPS = {
    "query_length": "query", "query_mean_idf": "query", "query_max_idf": "query",
    "user_budget": "query", "cheap_intent": "query",
    "log_price": "popularity/business", "is_price_missing": "product",
    "stars_clean": "popularity/business", "log_review_count": "popularity/business",
    "is_rating_missing": "product",
    "bm25_score": "lexical", "word_overlap": "lexical",
    "semantic_score": "semantic",
    "is_dominant_category": "interaction", "brand_match": "interaction",
    "color_match": "interaction", "is_over_budget": "interaction",
}
ORD = {"E": 3, "S": 2, "C": 1, "I": 0}


def main():
    import lightgbm as lgb

    with open(os.path.join(ROOT, "output/lambdamart_features.json")) as f:
        feats = json.load(f)["features"]

    tr = pd.read_parquet(os.path.join(CACHE, "train_features.parquet"),
                         columns=feats + ["esci_label", "query_id"])
    te = pd.read_parquet(os.path.join(CACHE, "test_scored.parquet"),
                         columns=feats + ["esci_label", "query_id"])

    booster = lgb.Booster(model_file=os.path.join(ROOT, "output/lambdamart_model.txt"))
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    bnames = booster.feature_name()
    # booster stores Column_0..Column_16 because it was fit on a numpy array
    gain_map = dict(zip(feats, gain)) if len(bnames) == len(feats) else {}
    split_map = dict(zip(feats, split))
    total_gain = float(gain.sum())

    # ---------------- importance ----------------
    imp = pd.DataFrame({
        "feature": feats,
        "feature_group": [GROUPS[f] for f in feats],
        "gain": [float(gain_map[f]) for f in feats],
        "split_count": [int(split_map[f]) for f in feats],
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp["gain_pct"] = 100.0 * imp["gain"] / total_gain
    imp["cumulative_gain_pct"] = imp["gain_pct"].cumsum()
    imp["gain_rank"] = np.arange(1, len(imp) + 1)
    imp["split_pct"] = 100.0 * imp["split_count"] / imp["split_count"].sum()
    imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

    concentration = {
        "total_gain": total_gain,
        "top1_features": imp["feature"].head(1).tolist(),
        "top1_gain_pct": float(imp["gain_pct"].head(1).sum()),
        "top3_features": imp["feature"].head(3).tolist(),
        "top3_gain_pct": float(imp["gain_pct"].head(3).sum()),
        "top5_features": imp["feature"].head(5).tolist(),
        "top5_gain_pct": float(imp["gain_pct"].head(5).sum()),
        "semantic_plus_bm25_plus_word_overlap_gain_pct": float(
            imp.loc[imp["feature"].isin(["semantic_score", "bm25_score", "word_overlap"]), "gain_pct"].sum()),
        "features_with_zero_gain": imp.loc[imp["gain"] == 0, "feature"].tolist(),
        "features_with_lt_0p1pct_gain": imp.loc[imp["gain_pct"] < 0.1, "feature"].tolist(),
        "n_trees": int(booster.num_trees()),
    }

    # ---------------- per-feature stats ----------------
    rows = []
    for f in feats:
        for split_name, df in [("train", tr), ("test", te)]:
            s = df[f].astype(float)
            lab = df["esci_label"].map(ORD).astype(float)
            nun = int(s.nunique())
            vc = s.value_counts(normalize=True)
            rows.append({
                "feature": f,
                "feature_group": GROUPS[f],
                "split": split_name,
                "dtype": str(df[f].dtype),
                "n": int(len(s)),
                "missing_rate": float(s.isna().mean()),
                "zero_rate": float((s == 0).mean()),
                "unique_count": nun,
                "modal_value": float(vc.index[0]),
                "modal_share": float(vc.iloc[0]),
                "mean": float(s.mean()), "std": float(s.std()),
                "min": float(s.min()),
                "p5": float(s.quantile(0.05)), "median": float(s.median()),
                "p95": float(s.quantile(0.95)), "max": float(s.max()),
                "pearson_with_ordinal_label": float(s.corr(lab)),
                "spearman_with_ordinal_label": float(s.corr(lab, method="spearman")),
                "gain": float(gain_map[f]),
                "gain_pct": float(100.0 * gain_map[f] / total_gain),
                "split_count": int(split_map[f]),
            })
    fa = pd.DataFrame(rows)

    # notes / flags per feature (computed on train, checked against test)
    notes = {}
    for f in feats:
        t = fa[(fa.feature == f) & (fa.split == "train")].iloc[0]
        v = fa[(fa.feature == f) & (fa.split == "test")].iloc[0]
        n = []
        if t["unique_count"] == 1:
            n.append("CONSTANT")
        if t["modal_share"] >= 0.99:
            n.append(f"NEAR-CONSTANT (modal value {t['modal_value']:g} on {t['modal_share']:.2%} of rows)")
        elif t["modal_share"] >= 0.95:
            n.append(f"highly imbalanced (modal value {t['modal_value']:g} on {t['modal_share']:.2%} of rows)")
        if t["zero_rate"] >= 0.90:
            n.append(f"zero on {t['zero_rate']:.2%} of train rows")
        if t["missing_rate"] > 0:
            n.append(f"missing_rate={t['missing_rate']:.4f}")
        if t["gain"] == 0:
            n.append("ZERO GAIN in the official LambdaMART model (never used for a split)")
        # train/test shift
        denom = abs(t["std"]) if t["std"] > 1e-9 else 1.0
        shift = abs(t["mean"] - v["mean"]) / denom
        if shift > 0.10:
            n.append(f"train/test mean shift = {shift:.3f} SD")
        if t["min"] < 0 and f in ("semantic_score", "bm25_score"):
            n.append("negative sentinel present (semantic_score missing -> -1.0 fill)")
        notes[f] = "; ".join(n)
    fa["notes"] = fa["feature"].map(notes)
    fa.to_csv(os.path.join(OUT, "feature_audit.csv"), index=False)

    # ---------------- correlation ----------------
    corr_tr = tr[feats].astype(float).corr()
    corr_te = te[feats].astype(float).corr()
    crows = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            crows.append({
                "feature_a": a, "feature_b": b,
                "group_a": GROUPS[a], "group_b": GROUPS[b],
                "pearson_train": float(corr_tr.loc[a, b]),
                "pearson_test": float(corr_te.loc[a, b]),
                "abs_pearson_train": float(abs(corr_tr.loc[a, b])),
                "flag_ge_0.90": bool(abs(corr_tr.loc[a, b]) >= 0.90),
                "flag_ge_0.70": bool(abs(corr_tr.loc[a, b]) >= 0.70),
            })
    cdf = pd.DataFrame(crows).sort_values("abs_pearson_train", ascending=False)
    cdf.to_csv(os.path.join(OUT, "feature_correlation.csv"), index=False)

    rep = {
        "E3_importance_concentration": concentration,
        "E1_flags": {f: notes[f] for f in feats if notes[f]},
        "E2_redundancy": {
            "pairs_abs_corr_ge_0.90": cdf[cdf["flag_ge_0.90"]][
                ["feature_a", "feature_b", "pearson_train"]].to_dict("records"),
            "pairs_abs_corr_ge_0.70": cdf[cdf["flag_ge_0.70"]][
                ["feature_a", "feature_b", "pearson_train"]].to_dict("records"),
            "top_10_by_abs_corr": cdf.head(10)[
                ["feature_a", "feature_b", "pearson_train", "pearson_test"]].to_dict("records"),
        },
        "E_label_correlation_ranked": fa[fa.split == "test"][
            ["feature", "feature_group", "pearson_with_ordinal_label",
             "spearman_with_ordinal_label", "gain_pct"]
        ].sort_values("spearman_with_ordinal_label", key=abs, ascending=False).to_dict("records"),
    }
    with open(os.path.join(OUT, "feature_audit.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)

    pd.set_option("display.width", 200)
    print(imp[["gain_rank", "feature", "feature_group", "gain", "gain_pct",
               "cumulative_gain_pct", "split_count"]].to_string(index=False))
    print("\nConcentration:", json.dumps(concentration, indent=1))
    print("\nTop correlations:\n", cdf.head(12).to_string(index=False))
    print("\nFlags:", json.dumps(rep["E1_flags"], indent=1))


if __name__ == "__main__":
    main()
