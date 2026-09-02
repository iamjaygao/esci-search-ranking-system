"""
Phase B -- Benchmark integrity audit (split / duplicates / labels / candidates).

READ-ONLY with respect to every pre-existing file. Writes only into
experiments/ranking_v2/audit/.

Audits two universes, because the repo has two different notions of
"the ranking dataset":

  U1 "judgment universe"  = raw ESCI examples parquet (all locales)
  U2 "ranking candidate pool" = what actually reaches the ranker, i.e.
     output/bm25_scores_{split}.csv OUTER-JOIN output/two_tower_scores_{split}.csv,
     inner-joined to that split's queries and to the products table
     (this is exactly what reranking/advanced_features.extract_advanced_features
     and evaluation/evaluate_advanced.extract_test_advanced_features build).

Usage: python experiments/ranking_v2/audit/scripts/01_split_label_candidates.py
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

EXAMPLES = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")
PRODUCTS = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet")

LABEL_ORDINAL = {"E": 3, "S": 2, "C": 1, "I": 0}
LABEL_FRACTIONAL = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

PCTLS = [1, 5, 25, 50, 75, 95, 99]


def describe(series, prefix=""):
    s = pd.Series(series).astype(float)
    d = {
        f"{prefix}count": int(s.shape[0]),
        f"{prefix}mean": float(s.mean()),
        f"{prefix}std": float(s.std()),
        f"{prefix}min": float(s.min()),
        f"{prefix}max": float(s.max()),
    }
    for p in PCTLS:
        d[f"{prefix}p{p}"] = float(s.quantile(p / 100.0))
    d[f"{prefix}median"] = float(s.median())
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    print("Loading examples parquet ...")
    ex = pd.read_parquet(EXAMPLES)
    ex["query_id"] = ex["query_id"].astype(str)
    ex["product_id"] = ex["product_id"].astype(str)
    report["examples_columns"] = list(ex.columns)
    report["examples_rows"] = int(len(ex))
    report["split_values"] = sorted(ex["split"].unique().tolist())
    report["locale_values"] = sorted(ex["product_locale"].unique().tolist())
    report["small_version_counts"] = {
        str(k): int(v) for k, v in ex["small_version"].value_counts().items()
    }
    report["note_no_dev_split"] = (
        "ESCI ships only 'train' and 'test'. The ranking pipeline has no held-out dev "
        "split; scripts/train_lambdamart.py and scripts/train_adv_reranker.py carve a "
        "random 15% of TRAIN QUERIES (sklearn train_test_split, random_state=42) as an "
        "internal early-stopping validation set. That internal split is regenerated at "
        "train time and is not persisted anywhere."
    )

    # ------------------------------------------------------------------
    # B1. query-level split integrity (U1: raw judgment universe)
    # ------------------------------------------------------------------
    print("B1: query-level split overlap ...")
    q = ex[["query_id", "query", "split"]].drop_duplicates()
    # a query_id maps to exactly one query string?
    qid_to_texts = q.groupby("query_id")["query"].nunique()
    report["query_id_with_multiple_texts"] = int((qid_to_texts > 1).sum())
    qid_to_split = ex.groupby("query_id")["split"].nunique()
    report["query_id_in_multiple_splits"] = int((qid_to_split > 1).sum())

    qsets = {}
    for sp in ["train", "test"]:
        sub = ex[ex["split"] == sp]
        qsets[sp] = {
            "query_id": set(sub["query_id"].unique()),
            "raw": set(sub["query"].astype(str).unique()),
            "lower_strip": set(sub["query"].astype(str).str.lower().str.strip().unique()),
            # pipeline normalization actually used for feature extraction is
            # `str(q).lower().split()` (advanced_features.get_idf_stats /
            # fast_overlap) -> whitespace-collapsed lowercase
            "pipeline_norm": set(
                sub["query"].astype(str).str.lower().str.split().str.join(" ").unique()
            ),
        }

    b1 = {}
    for key in ["query_id", "raw", "lower_strip", "pipeline_norm"]:
        tr, te = qsets["train"][key], qsets["test"][key]
        inter = tr & te
        b1[key] = {
            "train_unique": len(tr),
            "test_unique": len(te),
            "train_test_overlap": len(inter),
            "train_test_overlap_pct_of_test": round(100.0 * len(inter) / max(len(te), 1), 4),
            "overlap_examples": sorted(list(inter))[:20] if key != "query_id" else sorted(list(inter))[:20],
        }
    b1["dev_split"] = (
        "NOT AVAILABLE -- no persisted dev split exists for the ranking pipeline. "
        "train/dev overlap and dev/test overlap are therefore undefined for the "
        "official ranking benchmark. (An unrelated Two-Tower-v2 experiment does keep "
        "experiments/two_tower_v2/splits/{train,dev}_queries.txt; that split is NOT "
        "used by the ranking models audited here.)"
    )
    report["B1_query_split_overlap"] = b1

    # ------------------------------------------------------------------
    # B2. query-product duplicates / conflicting labels (U1)
    # ------------------------------------------------------------------
    print("B2: duplicates & label conflicts ...")
    b2 = {}
    for sp in ["train", "test"]:
        sub = ex[ex["split"] == sp]
        n = len(sub)
        dup_pairs = int(n - len(sub[["query_id", "product_id"]].drop_duplicates()))
        # conflicting label for the same (query_id, product_id)
        g = sub.groupby(["query_id", "product_id"])["esci_label"].nunique()
        b2[sp] = {
            "rows": n,
            "unique_query_product_pairs": int(len(sub[["query_id", "product_id"]].drop_duplicates())),
            "duplicate_query_product_rows": dup_pairs,
            "query_product_pairs_with_conflicting_label": int((g > 1).sum()),
        }
    # cross-split identical (normalized query text, product_id)
    def keyed(sp):
        sub = ex[ex["split"] == sp][["query", "product_id", "esci_label"]].copy()
        sub["qn"] = sub["query"].astype(str).str.lower().str.split().str.join(" ")
        return sub[["qn", "product_id", "esci_label"]].drop_duplicates()

    ktr, kte = keyed("train"), keyed("test")
    merged = ktr.merge(kte, on=["qn", "product_id"], suffixes=("_train", "_test"))
    b2["cross_split_same_normalized_query_and_product"] = {
        "pairs": int(len(merged)),
        "pairs_with_same_label": int((merged["esci_label_train"] == merged["esci_label_test"]).sum()),
        "pairs_with_conflicting_label": int((merged["esci_label_train"] != merged["esci_label_test"]).sum()),
    }
    # cross-split shared products (not a leak by itself, recorded for D)
    ptr = set(ex[ex["split"] == "train"]["product_id"].unique())
    pte = set(ex[ex["split"] == "test"]["product_id"].unique())
    b2["product_id_overlap_train_test"] = {
        "train_unique_products": len(ptr),
        "test_unique_products": len(pte),
        "overlap": len(ptr & pte),
        "overlap_pct_of_test_products": round(100.0 * len(ptr & pte) / max(len(pte), 1), 4),
        "interpretation": "shared catalog items are expected and are NOT label leakage by themselves",
    }
    report["B2_duplicates"] = b2

    # ------------------------------------------------------------------
    # B3 + B4. label + candidate distributions, U1 and U2
    # ------------------------------------------------------------------
    print("B3/B4: building U2 ranking candidate pools ...")
    pr = pd.read_parquet(PRODUCTS, columns=["product_id", "product_locale", "product_title", "product_brand"])
    pr["product_id"] = pr["product_id"].astype(str)
    report["products_rows"] = int(len(pr))
    report["products_unique_product_id"] = int(pr["product_id"].nunique())
    report["products_product_id_appearing_in_multiple_locales"] = int(
        (pr.groupby("product_id")["product_locale"].nunique() > 1).sum()
    )

    pr_ids_only = pr[["product_id"]].drop_duplicates()

    label_rows = []
    cand_rows = []
    u2_summary = {}

    for sp in ["train", "test"]:
        sub = ex[ex["split"] == sp]

        # ---- U1 label distribution
        vc = sub["esci_label"].value_counts()
        for lab in ["E", "S", "C", "I"]:
            label_rows.append({
                "universe": "U1_raw_esci_judgments",
                "split": sp,
                "esci_label": lab,
                "ordinal_label": LABEL_ORDINAL[lab],
                "fractional_label": LABEL_FRACTIONAL[lab],
                "count": int(vc.get(lab, 0)),
                "pct": round(100.0 * vc.get(lab, 0) / len(sub), 4),
            })

        # ---- U1 candidate distribution
        cpq = sub.groupby("query_id").size()
        rel_esc = sub[sub["esci_label"].isin(["E", "S", "C"])].groupby("query_id").size().reindex(cpq.index, fill_value=0)
        rel_e = sub[sub["esci_label"] == "E"].groupby("query_id").size().reindex(cpq.index, fill_value=0)
        for name, ser in [("candidates_per_query", cpq), ("relevant_ESC_per_query", rel_esc), ("relevant_E_per_query", rel_e)]:
            row = {"universe": "U1_raw_esci_judgments", "split": sp, "quantity": name}
            row.update(describe(ser))
            cand_rows.append(row)

        # ---- U2: the actual ranking candidate pool
        bm25 = pd.read_csv(os.path.join(ROOT, f"output/bm25_scores_{sp}.csv"))
        bm25.columns = ["query_id", "product_id", "bm25_score"]
        sem = pd.read_csv(os.path.join(ROOT, f"output/two_tower_scores_{sp}.csv"))
        sem.columns = ["query_id", "product_id", "semantic_score"]
        for d in (bm25, sem):
            d["query_id"] = d["query_id"].astype(str)
            d["product_id"] = d["product_id"].astype(str)

        cand = bm25.merge(sem, on=["query_id", "product_id"], how="outer")
        qtexts = sub[["query_id", "query"]].drop_duplicates()
        cand = cand.merge(qtexts, on="query_id", how="inner")

        n_before_prod_join = len(cand)
        # replicate the pipeline's product join (product_id only, both locales)
        cand_pipeline = cand.merge(pr[["product_id", "product_locale"]], on="product_id", how="inner")
        n_after_prod_join = len(cand_pipeline)
        # what a locale-correct join would give
        cand_ids = cand.merge(pr_ids_only, on="product_id", how="inner")

        labels = sub[["query_id", "product_id", "esci_label"]].drop_duplicates()
        cand_pipeline = cand_pipeline.merge(labels, on=["query_id", "product_id"], how="left")

        u2_summary[sp] = {
            "bm25_score_rows": int(len(bm25)),
            "tt_score_rows": int(len(sem)),
            "outer_join_rows": int(len(bm25.merge(sem, on=["query_id", "product_id"], how="outer"))),
            "rows_after_query_join": n_before_prod_join,
            "rows_after_pipeline_product_join_on_product_id_only": n_after_prod_join,
            "rows_if_joined_on_unique_product_id": int(len(cand_ids)),
            "row_inflation_from_locale_unaware_product_join": n_after_prod_join - int(len(cand_ids)),
            "row_inflation_pct": round(100.0 * (n_after_prod_join - len(cand_ids)) / max(len(cand_ids), 1), 4),
            "unique_queries_in_pool": int(cand_pipeline["query_id"].nunique()),
            "unlabeled_rows_in_pool": int(cand_pipeline["esci_label"].isna().sum()),
            "unlabeled_pct": round(100.0 * cand_pipeline["esci_label"].isna().sum() / len(cand_pipeline), 4),
            "rows_scored_by_bm25_only": int(cand_pipeline["semantic_score"].isna().sum()),
            "rows_scored_by_tt_only": int(cand_pipeline["bm25_score"].isna().sum()),
        }

        vc2 = cand_pipeline["esci_label"].value_counts(dropna=False)
        for lab in ["E", "S", "C", "I"]:
            label_rows.append({
                "universe": "U2_ranking_candidate_pool",
                "split": sp,
                "esci_label": lab,
                "ordinal_label": LABEL_ORDINAL[lab],
                "fractional_label": LABEL_FRACTIONAL[lab],
                "count": int(vc2.get(lab, 0)),
                "pct": round(100.0 * vc2.get(lab, 0) / len(cand_pipeline), 4),
            })
        label_rows.append({
            "universe": "U2_ranking_candidate_pool",
            "split": sp,
            "esci_label": "UNLABELED",
            "ordinal_label": 0,
            "fractional_label": 0.0,
            "count": int(cand_pipeline["esci_label"].isna().sum()),
            "pct": round(100.0 * cand_pipeline["esci_label"].isna().sum() / len(cand_pipeline), 4),
        })

        cpq2 = cand_pipeline.groupby("query_id").size()
        rel2 = cand_pipeline[cand_pipeline["esci_label"].isin(["E", "S", "C"])].groupby("query_id").size().reindex(cpq2.index, fill_value=0)
        rel2e = cand_pipeline[cand_pipeline["esci_label"] == "E"].groupby("query_id").size().reindex(cpq2.index, fill_value=0)
        for name, ser in [("candidates_per_query", cpq2), ("relevant_ESC_per_query", rel2), ("relevant_E_per_query", rel2e)]:
            row = {"universe": "U2_ranking_candidate_pool", "split": sp, "quantity": name}
            row.update(describe(ser))
            cand_rows.append(row)

        u2_summary[sp]["queries_with_fewer_than_10_candidates"] = int((cpq2 < 10).sum())
        u2_summary[sp]["queries_with_zero_ESC_relevant"] = int((rel2 == 0).sum())
        u2_summary[sp]["queries_with_all_same_label"] = int(
            (cand_pipeline.groupby("query_id")["esci_label"].nunique(dropna=False) == 1).sum()
        )
        u2_summary[sp]["candidate_cap_TOP_K"] = 150
        u2_summary[sp]["queries_hitting_TOP_K_cap"] = int((cpq2 >= 150).sum())

    report["B3_B4_U2_pool_summary"] = u2_summary

    pd.DataFrame(label_rows).to_csv(os.path.join(OUT, "label_distribution.csv"), index=False)
    pd.DataFrame(cand_rows).to_csv(os.path.join(OUT, "candidate_distribution.csv"), index=False)

    # ------------------------------------------------------------------
    # label_gain consistency across the pipeline
    # ------------------------------------------------------------------
    report["label_gain_consistency"] = {
        "lambdamart_training_label_map": "E=3,S=2,C=1,I=0 (scripts/train_lambdamart.py LABEL_MAP_ORDINAL)",
        "lambdamart_training_label_gain": [0, 1, 3, 7],
        "mlp_training_target_score_map": "E=1.0,S=0.1,C=0.01,I=0.0 (reranking/advanced_features.py)",
        "evaluation_relevance_map": "E=1.0,S=0.1,C=0.01,I=0.0 (evaluation/metrics.apply_business_ndcg_labels)",
        "evaluation_gain_formula": "2**relevance - 1 (evaluation/metrics.dcg)",
        "effective_evaluation_gain_vector_E_S_C_I": [
            float(2 ** 1.0 - 1), float(2 ** 0.1 - 1), float(2 ** 0.01 - 1), 0.0
        ],
        "VERDICT": (
            "INCONSISTENT. Training gain is [I=0,C=1,S=3,E=7]; the reported NDCG@10 gain "
            "is [I=0, C=0.00696, S=0.07177, E=1.0]. Relative S:E gain is 3/7=0.43 at train "
            "time vs 0.072 at eval time -- a ~6x difference in how much a Substitute is "
            "worth. The eval convention also differs from the standard ESCI convention "
            "(2**{3,2,1,0}-1 = [7,3,1,0]), so these NDCG@10 numbers are NOT comparable to "
            "published ESCI leaderboard numbers."
        ),
    }

    with open(os.path.join(OUT, "split_integrity.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("B1")}, indent=2)[:2000])
    print("\nWrote split_integrity.json / label_distribution.csv / candidate_distribution.csv")


if __name__ == "__main__":
    main()
