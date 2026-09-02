"""Consolidates every audit artifact into experiments/ranking_v2/audit/audit_summary.json."""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
OUT = os.path.join(ROOT, "experiments", "ranking_v2", "audit")


def j(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def main():
    split = j("split_integrity.json")
    ev = j("evaluation_audit.json")
    sem = j("semantic_audit.json")
    feat = j("feature_audit.json")
    cap = j("model_capacity_audit.json")
    boot = j("bootstrap_results.json")
    fail = j("failure_mode_decomposition.json")
    loc = j("locale_feature_health.json")
    parse = j("esci_s_parsing_audit.json")

    mc = pd.read_csv(os.path.join(OUT, "model_comparison.csv"))
    gains = pd.read_csv(os.path.join(OUT, "eval_gain_convention_sensitivity.csv"))
    sl = pd.read_csv(os.path.join(OUT, "query_slice_results.csv"))

    def headline(pool):
        return {r["model"]: round(r["ndcg_at_10"], 4)
                for _, r in mc[mc["pool"] == pool].iterrows()}

    summary = {
        "audit_date": "2026-09-02",
        "git_commit_at_audit_start": "9b173e95307b6af4335d46518f703265a9608d57",
        "scope": "Phase 0 read-only audit of the ESCI ranking pipeline. No retraining, "
                 "no tuning, no changes to any pre-existing file.",

        "verified_baselines": {
            "pool_official_inflated_682233_rows": headline("official"),
            "pool_clean_locale_matched_638016_rows": headline("clean"),
            "reproduction_status": {
                "bm25_0.8188_clean_all_locale": "VERIFIED (exact)",
                "two_tower_0.8267_clean_all_locale": "VERIFIED (exact)",
                "mlp_0.8458": "VERIFIED (exact, on the inflated pool)",
                "lambdamart_0.8464": "VERIFIED (exact, on the inflated pool)",
                "lambdamart_without_semantic_0.832": "VERIFIED as an MLP ablation, NOT a LambdaMART "
                                                     "one -- output/ablations/* were produced by "
                                                     "scripts/run_feature_ablation.py which retrains "
                                                     "the MLP (full=0.84578). No LambdaMART "
                                                     "no-semantic_score artifact exists.",
                "bm25_0.804_variant": "NOT VERIFIED -- no artifact in the repo produces 0.804. The "
                                      "closest recomputation is binary E-vs-rest BM25 = 0.8041.",
            },
            "gain_convention_sensitivity": gains.to_dict("records"),
            "non_degenerate_subset": ev["C_nondegenerate_subset"],
        },

        "P0_findings": {
            "P0-1_train_eval_gain_mismatch": {
                "detail": split["label_gain_consistency"],
                "impact": "LambdaMART optimises S:E = 3:7 (0.43); the reported metric scores "
                          "S:E = 0.0718:1.0 (0.072). Objective and metric disagree by ~6x on "
                          "how much a Substitute is worth.",
                "corroborating_evidence": "88.3% of the total NDCG@10 shortfall comes from "
                                          "queries containing at least one S; 88.6% of the 7,282 "
                                          "queries where LambdaMART loses to BM25 contain an S "
                                          "vs 66.4% base rate.",
            },
            "P0-2_candidate_pool_duplication": ev["C0_candidate_pool_integrity"],
            "P0-3_mixed_pool_baseline_comparison": ev["C_bm25_baseline_reconciliation"],
        },

        "P1_findings": {
            "P1-1_log_review_count_constant_zero": {
                "ratings_parsing": parse["ratings"],
                "impact": "feature is identically 0 on 100% of train and test rows; it is not a "
                          "weak feature, it is a dead one.",
            },
            "P1-2_price_parsing_corruption": parse["price"],
            "P1-3_stars_parsing_truncation": parse["stars"],
            "P1-4_non_english_feature_collapse": loc,
            "P1-5_degenerate_queries": ev["C_degenerate_queries"],
            "P1-6_split_text_overlap": {
                "query_ids_in_both_splits": 1,
                "query_id": "79706 ('piano'), 3 train rows + 31 test rows, disjoint products",
                "normalized_query_text_overlap": split["B1_query_split_overlap"]["pipeline_norm"][
                    "train_test_overlap"],
                "pct_of_test_queries": split["B1_query_split_overlap"]["pipeline_norm"][
                    "train_test_overlap_pct_of_test"],
                "cross_split_same_query_text_and_product": split["B2_duplicates"][
                    "cross_split_same_normalized_query_and_product"],
            },
            "P1-7_transductive_imputation": {
                "detail": "price/stars/ratings medians and the is_dominant_category top-20-BM25 "
                          "mode are computed within whichever split's frame is being processed "
                          "(reranking/advanced_features.py:118-134, "
                          "evaluation/evaluate_advanced.py:105-121). Test-time imputation "
                          "therefore uses test-set statistics.",
                "severity_note": "Effect is small in practice because the imputed features carry "
                                 "<2.6% of total gain, but it is a genuine protocol defect.",
            },
        },

        "semantic_score": {
            "lineage": sem["D0_two_tower_training_set"],
            "exposure": sem["D1_D2_exposure"],
            "separation": sem["D3_separation"],
            "classification": sem["D4_classification"],
            "verdict": "Upstream train-on-train exposure (category B) is real and large "
                       "(25.97% of us-locale E/S ranking-TRAIN rows were literal Two-Tower "
                       "training positives vs 0.004% for test), but the resulting feature "
                       "distribution mismatch (category C) is negligible: train/test "
                       "semantic_score means differ by 0.005, and Cohen's d for E-vs-I is "
                       "0.729 (train) vs 0.681 (test). There is NO label leakage (category A).",
        },

        "features": {
            "official_17_features": [r["feature"] for r in
                                     sorted(feat["E3_importance_concentration"]["top5_features"] and
                                            [], key=str)] or None,
            "importance_concentration": feat["E3_importance_concentration"],
            "flags": feat["E1_flags"],
            "redundancy_ge_0.90": feat["E2_redundancy"]["pairs_abs_corr_ge_0.90"],
            "redundancy_ge_0.70": feat["E2_redundancy"]["pairs_abs_corr_ge_0.70"],
            "marginal_value_vs_gain_share": {
                "note": "output/ablations/* are MLP leave-one-out retrains (full = 0.84578).",
                "no_semantic_score": -0.013847,
                "no_bm25_score": -0.005556,
                "no_word_overlap": -0.003150,
                "no_brand_match": -0.001329,
                "no_query_mean_idf": -0.000251,
                "interpretation": "semantic_score holds 55.0% of LambdaMART split gain but "
                                  "removing it costs only 0.0138 NDCG -- gain share measures "
                                  "how often the tree splits on it, not how much unique "
                                  "information it carries.",
            },
        },

        "model_capacity": {
            "top10_agreement": cap["F1_top10_agreement"],
            "lambdamart_minus_mlp_clean_pool": next(
                r for r in boot["results"]["clean"] if r["comparison"] == "lambdamart - mlp"),
            "verdict": "On the clean pool LambdaMART and the MLP are statistically "
                       "indistinguishable (mean delta +0.00043, 95% CI [-0.00012, +0.00098], "
                       "p=0.126) despite very different inductive biases, and agree on 88.9% "
                       "of top-10 slots. Evidence suggests the shared 17-feature "
                       "representation, not ranker capacity, is the binding constraint.",
        },

        "failure_modes": fail,
        "slices": sl.to_dict("records"),
        "bootstrap": boot["results"],

        "deliverable_status": {
            "REPORT.md": "generated",
            "repository_inventory.md": "generated",
            "audit_summary.json": "generated",
            "split_integrity.json": "generated",
            "label_distribution.csv": "generated",
            "candidate_distribution.csv": "generated",
            "semantic_score_distribution.csv": "generated",
            "feature_audit.csv": "generated",
            "feature_correlation.csv": "generated",
            "feature_importance.csv": "generated",
            "model_comparison.csv": "generated",
            "query_slice_results.csv": "generated (us-locale only -- the pre-existing slice "
                                       "rules in scripts/build_query_slices.py are English-token "
                                       "rules and are only defined for product_locale=='us')",
            "worst_queries.csv": "generated",
            "largest_losses_vs_bm25.csv": "generated",
            "largest_gains_vs_bm25.csv": "generated",
            "bootstrap_results.json": "generated",
        },
    }
    summary["features"]["official_17_features"] = pd.read_csv(
        os.path.join(OUT, "feature_importance.csv"))["feature"].tolist()

    with open(os.path.join(OUT, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("wrote audit_summary.json")


if __name__ == "__main__":
    main()
