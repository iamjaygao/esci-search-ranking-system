# Audit — Retrieval Query-Slicing / Exact-Match Analysis

Scope: this audits the **full-catalog retrieval Recall@100 pipeline**
(`output/full_retrieval/*`, built by `scripts/sample_eval_queries.py` +
`scripts/run_full_bm25_retrieval.py` + `scripts/run_full_tt_retrieval.py` +
`scripts/evaluate_full_retrieval.py`), which is the pipeline that produced the
BM25/Two-Tower/RRF Recall@100 numbers this analysis is meant to slice.

## 1. Test query set

- File: `output/full_retrieval/eval_query_ids.json`
- Fields: `seed` (=42), `locale` (="us"), `n_requested` (=5000), `n_available`,
  `query_ids` (list).
- **5,000 queries**, sampled deterministically (seed=42) from all US-locale
  ESCI `test`-split queries. This matches the ~5,000 figure assumed in the
  task brief.
- This is a **different, smaller query universe** than the 17-feature
  LambdaMART reranking evaluation used in the earlier audit
  (`experiments/audit/`), which used all 30,969 test-split queries (all
  locales) on a fixed ~16-160-candidate pool per query. The two pipelines are
  not interchangeable; this analysis stays entirely inside the 5,000-query
  full-catalog retrieval pipeline.

## 2. BM25 top-K retrieval results

- File: `output/full_retrieval/bm25_retrieval.parquet`
- Columns: `query_id`, `product_id`, `rank` (1..200), `bm25_score`
- 1,000,000 rows = 5,000 queries × 200 ranks each (full Top-200 per query
  stored, not just Top-100).

## 3. Two-Tower top-K retrieval results

- File: `output/full_retrieval/two_tower_retrieval.parquet`
- Columns: `query_id`, `product_id`, `rank` (1..200), `semantic_score`
- Same shape as BM25: 1,000,000 rows, Top-200 per query.

## 4. Ground truth / ESCI labels

- File: `output/full_retrieval/ground_truth.json`
- Per query_id: `{"query": <raw text>, "relevant_broad": [product_ids],
  "relevant_exact": [product_ids], "num_broad": int, "num_exact": int}`
- Built in `scripts/sample_eval_queries.py` directly from the raw ESCI
  examples table (`config.EXAMPLES_PATH`), restricted to `split == 'test'`
  and `product_locale == 'us'`, for exactly the 5,000 sampled query_ids.
- `relevant_broad` = ESCI labels `{E, S, C}`. `relevant_exact` = `{E}` only.
  There is no pre-existing "E+S" (excl. C) ground-truth set for this
  pipeline — if needed it would have to be constructed fresh from the same
  raw ESCI examples table the same way (this analysis does not need it).

## 5. Current Recall@100 definition

- Code: `recall_at_k()` in `scripts/evaluate_full_retrieval.py`.
- Per query: `hits = |{relevant items ranked <= k by this retriever}|`,
  `recall = hits / |relevant_set|`. Queries with an empty relevant set for
  the given key are **skipped** (not counted as 0 or excluded from the
  denominator in a way that inflates the average — they simply don't
  contribute a term to the mean).
- Aggregate metric = **macro-average** (unweighted mean) of per-query recall
  across all queries with a non-empty relevant set. This is the same
  averaging scheme documented and cross-checked in `experiments/audit/`
  (Task 6) previously.
- Existing aggregate results: `output/full_retrieval/retrieval_metrics.csv`
  — BM25 recall@100(broad)=0.4542, recall@100(exact)=0.5265; Two-Tower
  recall@100(broad)=0.4598, recall@100(exact)=0.5295; Hybrid RRF@100
  recall@100(broad)=0.5194, recall@100(exact)=0.5979 (RRF uses
  `rrf_fuse_query()`, k=60, fusing the Top-200 lists of both retrievers into
  a fused Top-100).
- **Relevant labels**: broad = `{E, S, C}`, exact = `{E}` — as defined in
  `output/full_retrieval/evaluation_config.json` and matching what was
  audited previously. No mismatch found vs. the task brief's assumptions.

## 6. Per-query Recall@100

**Does not already exist.** `retrieval_metrics.csv` only stores the
aggregate (macro-mean) recall per retriever; no per-query CSV/parquet for
this retrieval pipeline is saved anywhere in the repo. This analysis computes
it fresh (Step 3 below), reusing `recall_at_k`'s exact hit-counting logic but
capturing the per-query values instead of only the mean.

(Note: a per-query recall/NDCG file *does* exist for the *other* pipeline —
`output/query_slices/query_level_metrics.csv` — but that is per-query NDCG@10
on the fixed-candidate reranking task, not Recall@100 on this full-catalog
retrieval task. Not reused here to avoid conflating the two pipelines.)

## 7. Query raw text field

`query` key inside each entry of `ground_truth.json` (identical to the
`query` column in the raw ESCI examples parquet for that query_id).

## 8. Existing query-slicing / error-analysis code

`scripts/build_query_slices.py` + `scripts/analyze_query_slices.py` +
`output/query_slices/` already implement SKU/model regex, brand-phrase
matching (with a `NON_BRAND_PLACEHOLDERS` denylist), a color vocabulary, and
attribute (size/capacity/quantity/dimension) regexes — but **for the
fixed-candidate reranking NDCG task**, not for this Recall@100 full-catalog
retrieval task. Query universe, candidate pool, and metric all differ.

This analysis reuses the same **conservative regex philosophy** (SKU token
patterns, brand-placeholder denylist) as a matter of engineering consistency,
reimplemented fresh in `experiments/query_slice_analysis/` for the 5,000
query, Recall@100 pipeline. The brand vocabulary itself is rebuilt from
scratch from the catalog's `product_brand` field (see Step 2 in REPORT.md),
per the task instructions, rather than reusing the other pipeline's
per-candidate-pool brand matching.

## 9. Can existing predictions be reused directly?

**Yes.** No retraining and no new retrieval calls are needed:
`bm25_retrieval.parquet`, `two_tower_retrieval.parquet`, and
`ground_truth.json` fully cover BM25 Top-200, Two-Tower Top-200, and
ESCI-label ground truth for exactly the 5,000-query eval set. Brand/title
metadata additionally comes from `esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet`
(read-only), filtered to `product_locale == 'us'` (the catalog has ~11,567
product_ids that repeat across locales with different metadata — a
pre-existing, already-documented issue in `scripts/analyze_topk_overlap.py`
— so the locale filter is applied here to avoid picking up a non-US record
for a shared ASIN).

## 10. Minor observation (not a blocker)

Recomputing Hybrid RRF@100 recall (broad) independently in this audit gave
0.51941 vs. 0.51938 stored in `retrieval_metrics.csv` — a ~3e-5 difference.
`rrf_fuse_query()` builds `all_pids = set(bm25_ranks) | set(tt_ranks)` before
sorting by score; Python set iteration order is not guaranteed stable across
runs, so exact-tie RRF scores can be ordered differently between runs,
producing tiny non-deterministic differences in which items land in the
fused Top-100 boundary. This does not affect the analysis below (slices use
Recall@100 computed fresh, consistently, within this same script run).
