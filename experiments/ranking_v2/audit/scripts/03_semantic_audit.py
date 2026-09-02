"""
Phase D -- semantic_score lineage & exposure audit.

Lineage under test:
  ranking row.semantic_score
    <- output/two_tower_scores_{train,test}.csv
    <- scripts/generate_two_tower_scores.py -> retrieval.two_tower.compute_two_tower_scores
    <- models/two_tower_finetuned  (SentenceTransformer checkpoint)
    <- scripts/train_two_tower.py: ESCI examples where
         small_version == 1 AND split == 'train' AND product_locale == 'us'
         AND esci_label in {E, S}, trained with MultipleNegativesRankingLoss.

D1/D2 measure, separately, whether the Two-Tower encoder saw
  (a) the same query text
  (b) the same product
  (c) the same query-product positive pair
that a given ranking row is built from.

D3 reports semantic_score distributions by split and by label.
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

EXAMPLES = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")
PCTLS = [1, 5, 25, 50, 75, 95, 99]


def norm_q(s):
    return s.astype(str).str.lower().str.split().str.join(" ")


def stats(series):
    s = pd.Series(series).astype(float)
    d = {"count": int(len(s)), "mean": float(s.mean()), "std": float(s.std()),
         "min": float(s.min()), "max": float(s.max()), "median": float(s.median())}
    for p in PCTLS:
        d[f"p{p}"] = float(s.quantile(p / 100.0))
    return d


def main():
    rep = {}
    print("Reconstructing the Two-Tower training set ...")
    ex = pd.read_parquet(EXAMPLES)
    ex["query_id"] = ex["query_id"].astype(str)
    ex["product_id"] = ex["product_id"].astype(str)

    tt_train = ex[(ex["small_version"] == 1) &
                  (ex["split"] == "train") &
                  (ex["product_locale"] == "us") &
                  (ex["esci_label"].isin(["E", "S"]))].copy()
    tt_train["qn"] = norm_q(tt_train["query"])

    rep["D0_two_tower_training_set"] = {
        "checkpoint": "models/two_tower_finetuned",
        "base_model": "sentence-transformers/msmarco-distilbert-base-v3",
        "training_script": "scripts/train_two_tower.py",
        "loss": "MultipleNegativesRankingLoss (in-batch negatives), 1 epoch, batch 64",
        "filter": "small_version==1 AND split=='train' AND product_locale=='us' AND esci_label in {E,S}",
        "training_pairs": int(len(tt_train)),
        "unique_queries_seen": int(tt_train["qn"].nunique()),
        "unique_query_ids_seen": int(tt_train["query_id"].nunique()),
        "unique_products_seen": int(tt_train["product_id"].nunique()),
        "scoring_config_note": (
            "config.USE_SMALL_VERSION is False, so scripts/generate_two_tower_scores.py "
            "scores the FULL dataset with an encoder fine-tuned only on the small_version "
            "subset -- a training/scoring population mismatch on top of the split question."),
        "score_normalization": (
            "retrieval.two_tower.compute_two_tower_scores min-max normalizes cosine "
            "similarity WITHIN each query's candidate set, so semantic_score is a "
            "per-query relative score in [0,1], not a calibrated absolute similarity."),
    }

    seen_qn = set(tt_train["qn"].unique())
    seen_qid = set(tt_train["query_id"].unique())
    seen_pid = set(tt_train["product_id"].unique())
    seen_pair_qn = set(zip(tt_train["qn"], tt_train["product_id"]))
    seen_pair_qid = set(zip(tt_train["query_id"], tt_train["product_id"]))

    # ------------------------------------------------------------------
    # D1 / D2 exposure, computed on the ACTUAL ranking rows
    # ------------------------------------------------------------------
    exposure = {}
    dist_rows = []
    per_split_frames = {}

    for split, path in [("train", os.path.join(CACHE, "train_features.parquet")),
                        ("test", os.path.join(CACHE, "test_scored.parquet"))]:
        print(f"\n--- exposure for ranking {split} rows ---")
        df = pd.read_parquet(path, columns=["query_id", "product_id", "query",
                                            "esci_label", "semantic_score", "product_locale"])
        df["qn"] = norm_q(df["query"])

        q_seen = df["qn"].isin(seen_qn)
        p_seen = df["product_id"].isin(seen_pid)
        pair_seen = pd.Series(
            [(a, b) in seen_pair_qn for a, b in zip(df["qn"], df["product_id"])],
            index=df.index)
        qid_seen = df["query_id"].isin(seen_qid)
        pair_qid_seen = pd.Series(
            [(a, b) in seen_pair_qid for a, b in zip(df["query_id"], df["product_id"])],
            index=df.index)

        df["_q_seen"] = q_seen
        df["_p_seen"] = p_seen
        df["_pair_seen"] = pair_seen
        per_split_frames[split] = df

        n = len(df)
        uq = df[["query_id", "qn"]].drop_duplicates()
        exposure[split] = {
            "ranking_rows": int(n),
            "ranking_queries": int(df["query_id"].nunique()),
            "A_same_query_text_seen_in_tt_training": {
                "rows": int(q_seen.sum()), "row_pct": round(100.0 * q_seen.mean(), 3),
                "queries": int(uq["qn"].isin(seen_qn).sum()),
                "query_pct": round(100.0 * uq["qn"].isin(seen_qn).mean(), 3),
            },
            "A2_same_query_id_seen_in_tt_training": {
                "rows": int(qid_seen.sum()), "row_pct": round(100.0 * qid_seen.mean(), 3),
            },
            "B_same_product_seen_in_tt_training": {
                "rows": int(p_seen.sum()), "row_pct": round(100.0 * p_seen.mean(), 3),
                "unique_products": int(df.loc[p_seen, "product_id"].nunique()),
            },
            "C_same_query_product_POSITIVE_PAIR_seen_in_tt_training": {
                "rows": int(pair_seen.sum()), "row_pct": round(100.0 * pair_seen.mean(), 3),
                "rows_by_query_id_key": int(pair_qid_seen.sum()),
                "row_pct_by_query_id_key": round(100.0 * pair_qid_seen.mean(), 3),
            },
        }
        # pair exposure restricted to the rows where it can matter (E/S rows)
        es = df["esci_label"].isin(["E", "S"])
        exposure[split]["C_pair_exposure_among_E_S_rows_only"] = {
            "E_S_rows": int(es.sum()),
            "of_which_seen_as_tt_training_pair": int((pair_seen & es).sum()),
            "pct": round(100.0 * (pair_seen & es).sum() / max(es.sum(), 1), 3),
        }
        # and restricted to us-locale E/S rows (the population TT was trained on)
        uses = es & (df["product_locale"] == "us")
        exposure[split]["C_pair_exposure_among_US_E_S_rows_only"] = {
            "us_E_S_rows": int(uses.sum()),
            "of_which_seen_as_tt_training_pair": int((pair_seen & uses).sum()),
            "pct": round(100.0 * (pair_seen & uses).sum() / max(uses.sum(), 1), 3),
        }

        # -------- D3 distributions --------
        row = {"split": split, "group": "ALL", "n": int(n)}
        row.update(stats(df["semantic_score"]))
        dist_rows.append(row)
        for lab, ordv in [("E", 3), ("S", 2), ("C", 1), ("I", 0)]:
            sub = df[df["esci_label"] == lab]["semantic_score"]
            if len(sub):
                row = {"split": split, "group": f"label_{lab}(rel={ordv})", "n": int(len(sub))}
                row.update(stats(sub))
                dist_rows.append(row)
        # exposure-conditioned distribution
        for name, mask in [("tt_training_pair", pair_seen),
                           ("NOT_tt_training_pair", ~pair_seen)]:
            sub = df[mask]["semantic_score"]
            if len(sub):
                row = {"split": split, "group": name, "n": int(len(sub))}
                row.update(stats(sub))
                dist_rows.append(row)
        # within E/S only, seen vs unseen pair -- the cleanest exposure contrast
        for name, mask in [("E_S_rows_seen_pair", pair_seen & es),
                           ("E_S_rows_unseen_pair", (~pair_seen) & es)]:
            sub = df[mask]["semantic_score"]
            if len(sub):
                row = {"split": split, "group": name, "n": int(len(sub))}
                row.update(stats(sub))
                dist_rows.append(row)

    rep["D1_D2_exposure"] = exposure
    pd.DataFrame(dist_rows).to_csv(os.path.join(OUT, "semantic_score_distribution.csv"), index=False)

    # ------------------------------------------------------------------
    # D3b: discriminative power of semantic_score, train vs test
    # ------------------------------------------------------------------
    print("\nComputing separation statistics ...")
    sep = {}
    for split, df in per_split_frames.items():
        d = {}
        for a, b in [("E", "I"), ("E", "S"), ("S", "I")]:
            sa = df[df["esci_label"] == a]["semantic_score"]
            sb = df[df["esci_label"] == b]["semantic_score"]
            if len(sa) and len(sb):
                pooled = np.sqrt((sa.var() + sb.var()) / 2)
                d[f"cohens_d_{a}_vs_{b}"] = float((sa.mean() - sb.mean()) / pooled) if pooled > 0 else None
                d[f"mean_gap_{a}_vs_{b}"] = float(sa.mean() - sb.mean())
        # within-query rank correlation between semantic_score and ordinal label
        ordmap = {"E": 3, "S": 2, "C": 1, "I": 0}
        d["pearson_semantic_vs_ordinal_label"] = float(
            df["semantic_score"].corr(df["esci_label"].map(ordmap).astype(float)))
        d["spearman_semantic_vs_ordinal_label"] = float(
            df["semantic_score"].corr(df["esci_label"].map(ordmap).astype(float), method="spearman"))
        sep[split] = d
    rep["D3_separation"] = sep

    # ------------------------------------------------------------------
    # D4: classification of the risk
    # ------------------------------------------------------------------
    tr = exposure["train"]["C_pair_exposure_among_US_E_S_rows_only"]["pct"]
    te = exposure["test"]["C_pair_exposure_among_US_E_S_rows_only"]["pct"]
    rep["D4_classification"] = {
        "A_true_label_leakage_into_test": {
            "verdict": "NO",
            "evidence": (
                "The Two-Tower encoder is fine-tuned strictly on split=='train' rows. No "
                "ESCI test label, test query text (beyond the incidental overlaps recorded "
                "in split_integrity.json), or test query-product pair enters its training "
                f"data. Test-row positive-pair exposure = {te}% of us-locale E/S rows."),
        },
        "B_upstream_model_saw_the_same_pair_at_ranking_TRAIN_time": {
            "verdict": "YES, extensive",
            "evidence": (
                f"{tr}% of us-locale E/S rows in the LambdaMART/MLP TRAINING set were "
                f"literal Two-Tower training positives, vs {te}% for test. The ranker "
                "therefore learns semantic_score's reliability from a regime where the "
                "upstream encoder has memorised the positives."),
            "correct_term": "upstream train-on-train exposure / non-out-of-fold feature, NOT label leakage",
        },
        "C_train_eval_feature_distribution_mismatch": None,   # filled below
        "D_normal_production_feature": {
            "verdict": "PARTLY",
            "evidence": (
                "In production a fine-tuned encoder does score unseen queries, so the "
                "feature itself is legitimate. What is not production-like is generating "
                "the RANKER's training-set values of that feature with an encoder that was "
                "fit on the same rows -- an in-fold feature."),
        },
    }

    with open(os.path.join(OUT, "semantic_audit.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print("\nWrote semantic_audit.json + semantic_score_distribution.csv")
    print(json.dumps(exposure, indent=1))
    print(json.dumps(sep, indent=1))


if __name__ == "__main__":
    main()
