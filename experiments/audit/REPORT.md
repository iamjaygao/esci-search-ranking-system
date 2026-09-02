# ESCI Ranking Experimental Supplement

## Experimental Configuration

| Item | Value |
|---|---|
| Random seed | 42 (query-level train/val split via `train_test_split`; `LGBMRanker(random_state=42)`) |
| Train split | ESCI official `train` split, all locales, 99,684 unique query_ids |
| Internal train sub-split | 85% of train-split queries → 84,731 queries / 1,799,751 rows |
| Internal validation sub-split | 15% of train-split queries → 14,953 queries / 319,934 rows (used only for early stopping, not for final test evaluation) |
| Test split | ESCI official `test` split, all locales, 30,969 unique query_ids, no overlap with train |
| learning_rate | 0.05 |
| num_boost_round (n_estimators cap) | 1000 |
| Early stopping | `lgb.early_stopping(stopping_rounds=50)`, monitored metric `ndcg@10` on the validation split |
| Label mapping (ordinal, LightGBM label index) | E=3, S=2, C=1, I=0 |
| label_gain | [0, 1, 3, 7] |
| 17-feature best_iteration | 250 |
| 12-feature best_iteration | 230 |

Note: the task brief assumed `learning_rate=0.01`. The actual `scripts/train_lambdamart.py` uses `learning_rate=0.05`. All experiments in this supplement reuse the value found in the real training script (0.05), per the instruction to reuse the full model's actual training configuration rather than the brief's assumed numbers.

## 1. LambdaMART Feature Importance

Extracted directly from the existing `output/lambdamart_model.txt` Booster (no retraining). `num_trees()=250` (the saved model), below the `n_estimators` cap of 1000, confirming early stopping triggered.

`best_iteration` is not persisted by LightGBM's text model format when reloaded via `lgb.Booster(model_file=...)` (reloaded value: -1/unset). It was derived as follows: in Task 2, an identical training run (12-feature model, same script structure) was executed in-process, where `log_evaluation` printed scores through iteration 280 but `model.best_iteration_ == model.booster_.num_trees() == 230` after `fit()` returned — i.e., LightGBM's sklearn API rolls the returned/saved booster back to the best iteration when early stopping triggers, discarding the trees added after the best iteration. Applying this confirmed mechanism to the 17-feature booster: `best_iteration = num_trees_in_saved_model = 250`. No training log for the original `scripts/train_lambdamart.py` run exists in the repo to cross-check this independently.

| feature | group | gain | gain_pct | split | gain_rank |
|---|---|---:|---:|---:|---:|
| semantic_score | semantic | 392371.01 | 54.99% | 1130 | 1 |
| bm25_score | lexical | 162206.82 | 22.73% | 995 | 2 |
| word_overlap | lexical | 80479.79 | 11.28% | 647 | 3 |
| is_dominant_category | interaction | 25400.23 | 3.56% | 442 | 4 |
| query_mean_idf | query | 13341.97 | 1.87% | 840 | 5 |
| brand_match | interaction | 12243.77 | 1.72% | 341 | 6 |
| log_price | item | 11017.78 | 1.54% | 1154 | 7 |
| stars_clean | item | 6294.31 | 0.88% | 720 | 8 |
| query_length | query | 4543.02 | 0.64% | 499 | 9 |
| query_max_idf | query | 2349.63 | 0.33% | 332 | 10 |
| is_rating_missing | item | 2067.06 | 0.29% | 213 | 11 |
| is_price_missing | item | 795.19 | 0.11% | 114 | 12 |
| color_match | interaction | 322.18 | 0.05% | 62 | 13 |
| user_budget | query | 32.52 | 0.005% | 9 | 14 |
| cheap_intent | query | 9.37 | 0.001% | 2 | 15 |
| log_review_count | item | 0.00 | 0.00% | 0 | 16 |
| is_over_budget | interaction | 0.00 | 0.00% | 0 | 17 |

Full data: `experiments/audit/feature_importance.csv`.

For reference (not the same quantity as gain importance, per task instructions), the pre-existing leave-one-out ablations recorded elsewhere in the repo:

| Removed feature | NDCG@10 drop |
|---|---:|
| semantic_score | ≈0.0138 |
| bm25_score | ≈0.0056 |
| word_overlap | ≈0.003 |

## 2. 12-Feature Model

Dropped features: `log_price`, `is_price_missing`, `stars_clean`, `log_review_count`, `is_rating_missing` (query-independent item features). Retrained with identical train/val split, seed, hyperparameters, early stopping, label mapping, and label_gain as the 17-feature model.

| Metric | 17-feature | 12-feature | Absolute delta |
|---|---:|---:|---:|
| NDCG@10 | 0.8464 | 0.8447 | -0.0017 |
| best_iteration | 250 | 230 | -20 |

Training runtime (12-feature): 5.51 seconds (feature extraction from raw ESCI/BM25/Two-Tower/ESCI-S data + LightGBM fit, single run, this machine).

Model and result files: `experiments/audit/model_12feature.txt`, `experiments/audit/model_12feature_features.json`, `experiments/audit/result_12feature.json`.

## 3. Per-query NDCG and Bootstrap

Computed on the exact same test candidate pool used by `evaluation/evaluate_lambdamart.py` (via `evaluation.evaluate_advanced.extract_test_advanced_features`), n=30,969 test queries (all have at least one E/S/C-labeled candidate, so NDCG@10 is defined for all of them).

| Metric | Value |
|---|---:|
| BM25 mean NDCG@10 (this candidate pool) | 0.8155 |
| LambdaMART mean NDCG@10 | 0.8464 |
| Mean delta (LambdaMART - BM25) | 0.0309 |
| 95% bootstrap CI | [0.0293, 0.0326] |
| p-value (two-sided, paired bootstrap) | < 0.0001 (0 of 10,000 resamples had mean delta ≤ 0) |
| n_bootstrap | 10,000 |
| bootstrap seed | 42 |

**Discrepancy note**: BM25 NDCG@10 computed on the LambdaMART-matched candidate pool = 0.8155, differing from the previously-recorded all-locale BM25 baseline of 0.8188 (`scripts/build_query_slices.py`, which independently re-verifies against 0.8188 within a 0.001 tolerance). The two numbers come from different candidate-pool construction paths: `build_query_slices.py` merges `bm25_scores_test.csv` directly against ESCI labels; `extract_test_advanced_features` additionally joins the products table on `product_id` only (not `product_id`+locale — a pre-existing issue already documented in `scripts/analyze_topk_overlap.py`) and applies `dropna(subset=feature_cols)`. Per Task 3 instructions ("use the existing implementation corresponding to the LambdaMART comparison"), 0.8155 is the number used here, since it is computed on the identical query/candidate set as the LambdaMART predictions in this section. Existing code was not modified.

Full per-query data: `experiments/audit/per_query_ndcg.csv`. Full bootstrap output: `experiments/audit/bootstrap_results.json`.

## 4. Win / Loss / Tie

| group | query_count | percentage | mean_delta |
|---|---:|---:|---:|
| win | 12,249 | 39.55% | 0.1419 |
| loss | 7,248 | 23.40% | -0.1077 |
| tie | 11,472 | 37.04% | 0.0000 |

### Loss-query slice vs. all test queries

| variable | all queries mean | all queries median | loss queries mean | loss queries median | mean difference | median difference |
|---|---:|---:|---:|---:|---:|---:|
| query_length | 3.2704 | 3.0 | 3.3463 | 3.0 | 0.0759 | 0.0 |
| query_mean_idf | 8.7425 | 8.7122 | 8.8415 | 8.8326 | 0.0990 | 0.1203 |
| candidate_count | 22.0295 | 16.0 | 23.6184 | 16.0 | 1.5888 | 0.0 |
| E_count | 13.5203 | 13.0 | 11.3241 | 10.0 | -2.1962 | -3.0 |
| S_count | 5.3255 | 3.0 | 8.0043 | 6.0 | 2.6788 | 3.0 |

Full data: `experiments/audit/loss_slice_stats.csv`.

### Worst 20 queries (lowest delta = model_ndcg10 - bm25_ndcg10)

| query_id | query_text | bm25_ndcg10 | model_ndcg10 | delta | candidate_count | E | S | C | I |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9806 | apple tv+ app for fire stick | 1.0000 | 0.0763 | -0.9237 | 38 | 9 | 29 | 0 | 0 |
| 71766 | muñeco blade | 0.9437 | 0.0477 | -0.8961 | 27 | 1 | 3 | 5 | 18 |
| 130588 | ｈショーツ | 1.0000 | 0.1772 | -0.8228 | 16 | 2 | 14 | 0 | 0 |
| 108560 | vida charging station | 1.0000 | 0.1772 | -0.8228 | 16 | 2 | 13 | 0 | 1 |
| 5533 | 941m mercury | 1.0000 | 0.1772 | -0.8228 | 16 | 2 | 0 | 0 | 14 |
| 49822 | hearing amplifiers for seniors for tv | 0.9537 | 0.1605 | -0.7932 | 19 | 7 | 12 | 0 | 0 |
| 3170 | 28 pulgadas tv sin smart | 0.9418 | 0.1561 | -0.7857 | 40 | 1 | 35 | 0 | 4 |
| 62477 | lightweight makeup organizer | 1.0000 | 0.2198 | -0.7802 | 16 | 4 | 12 | 0 | 0 |
| 27240 | cocaine purity test | 0.9742 | 0.2274 | -0.7468 | 16 | 1 | 14 | 1 | 0 |
| 49541 | hcg drops for weight loss | 1.0000 | 0.2600 | -0.7400 | 16 | 1 | 15 | 0 | 0 |
| 43372 | funda xs max iphone pluto | 1.0000 | 0.2600 | -0.7400 | 16 | 1 | 15 | 0 | 0 |
| 5185 | 75 ft telephone cords for landline phones | 1.0000 | 0.2600 | -0.7400 | 16 | 1 | 15 | 0 | 0 |
| 5574 | 990k | 1.0000 | 0.2600 | -0.7400 | 35 | 1 | 34 | 0 | 0 |
| 109015 | viver kombucha | 1.0000 | 0.2600 | -0.7400 | 40 | 1 | 28 | 0 | 11 |
| 79943 | pilas para calculadora casio | 0.8629 | 0.1232 | -0.7397 | 19 | 3 | 12 | 0 | 4 |
| 97343 | st sixtones 特典なし | 0.7521 | 0.0141 | -0.7381 | 40 | 6 | 12 | 15 | 7 |
| 102921 | throw pillows for couch | 0.8941 | 0.1588 | -0.7353 | 19 | 9 | 0 | 10 | 0 |
| 53660 | intex プール フレーム | 0.9409 | 0.2279 | -0.7131 | 60 | 37 | 19 | 4 | 0 |
| 126831 | 床保護マット 180 | 0.8435 | 0.1312 | -0.7123 | 16 | 5 | 0 | 0 | 11 |
| 37789 | entertainment stand for tv up to 90 | 1.0000 | 0.2891 | -0.7109 | 16 | 1 | 0 | 0 | 15 |

Full data: `experiments/audit/worst_20_queries.csv`.

## 5. Recall@100 by Relevance Definition

Query universe: existing `output/full_retrieval/eval_query_ids.json` (seed=42, 5,000 US-locale test queries, unchanged). Retrieval outputs: existing `bm25_retrieval.parquet`, `two_tower_retrieval.parquet`. RRF: k=60, reusing `rrf_fuse_query`/`recall_at_k`/`build_rank_lookup` imported from `scripts/evaluate_full_retrieval.py` (unchanged).

| Relevant denominator | BM25 Recall@100 | Two-Tower Recall@100 | RRF Recall@100 | mean relevant items / query |
|---|---:|---:|---:|---:|
| E only | 0.5265 | 0.5295 | 0.5979 | 12.24 |
| E + S | 0.4551 | 0.4645 | 0.5222 | 16.42 |
| E + S + C | 0.4542 | 0.4598 | 0.5194 | 16.91 |

The E+S+C row reproduces the previously-recorded values (BM25≈0.45, Two-Tower≈0.46, RRF≈0.52) exactly (0.4542 / 0.4598 / 0.5194), matching `output/full_retrieval/retrieval_metrics.csv`'s `exact_recall@100` columns computed by the pre-existing pipeline.

Full data: `experiments/audit/recall_denominators.csv`.

## 6. Retrieval Coverage

Same query universe and relevance definition as Task 5's "E + S + C" row (k=100), recomputed from raw ESCI labels + retrieval parquet files (not copied from `output/full_retrieval/retriever_overlap_summary.json`).

### Micro (item-instance-counted) coverage decomposition

| Category | Count | % of relevant item instances |
|---|---:|---:|
| Both (BM25 ∩ Two-Tower) | 27,716 | 32.77% |
| BM25-only | 10,168 | 12.02% |
| Two-Tower-only | 11,057 | 13.07% |
| Both-miss | 35,631 | 42.13% |
| Union | 48,941 | 57.87% |
| Total relevant item instances | 84,572 | 100.00% |

These reproduce the previously-recorded values (BM25-only≈12%, TT-only≈13%, both≈33%, both-miss≈42%, union≈58%) to within rounding.

### Macro (per-query-averaged) Recall@100, same query set/definition

| Retriever | Macro Recall@100 |
|---|---:|
| BM25 | 0.4542 |
| Two-Tower | 0.4598 |
| RRF (k=60) | 0.5194 |

### Consistency check: does macro Recall@100 equal the micro-implied recall?

| Quantity | Value |
|---|---:|
| macro BM25 recall@100 | 0.45420 |
| micro (both+bm25_only)/total | 0.44795 |
| difference | 0.00625 |
| macro Two-Tower recall@100 | 0.45980 |
| micro (both+tt_only)/total | 0.45846 |
| difference | 0.00134 |
| macro RRF recall@100 | 0.51941 |
| micro union recall (both+bm25_only+tt_only)/total | 0.57869 |
| bm25_only% + tt_only% + both% + both_miss% | 100.00% |

The BM25-only/TT-only/both/both-miss percentages sum to exactly 100%, and BM25 recall = BM25-only + both / macro Two-Tower recall ≈ micro-implied Two-Tower recall (difference 0.0013). Macro BM25 recall and the micro-implied BM25 recall differ by 0.0063; macro RRF recall (0.5194) and micro union recall (0.5787) differ by 0.0593 and are not the same quantity. Macro recall is the mean of per-query recall (each query weighted equally); micro coverage is the count of individual (query, relevant-item) instances pooled across all queries and normalized by the total instance count (queries with more relevant items contribute proportionally more). These are two different averaging schemes over the same underlying hit/miss data and are not algebraically required to be equal; both are reported here rather than reconciled or adjusted to match.

Full data: `experiments/audit/retrieval_coverage.json`.
