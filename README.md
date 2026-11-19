# Search Ranking System (Amazon ESCI)

> **Contribution note:** This started as a three-person course project.
> The following are my own work, added after the original submission:
>
> - **Full-catalog retrieval layer (Setting 2)** — the US-catalog index
>   build, full-catalog BM25 and Two-Tower/FAISS retrieval runs, RRF fusion,
>   the deterministic 5,000-query evaluation set, the Recall@K evaluation
>   harness, and the retrieval-overlap analysis. The underlying BM25 and
>   Two-Tower modules (`retrieval/bm25.py`, `retrieval/two_tower.py`) were
>   written mainly by teammates.
> - **LambdaMART reranker** — the LightGBM trainer and evaluation, including
>   the `label_gain` handling needed to compare a tree model against the
>   pairwise MLPs. The 17-feature extraction pipeline
>   (`reranking/advanced_features.py`) was written by a teammate.
> - **Root-caused the `BatchNorm`/pairwise-training interaction** in the
>   inherited baseline reranker (see the debugging note below).
> - **Separated the two experimental settings** — the original write-up
>   reported Recall@K and NDCG@K as one end-to-end pipeline; they are
>   module-level evaluations on different candidate sources and are now
>   documented as such.
> - **Found that the whole evaluation was running on the wrong candidate
>   pool** — the Tasks 2/3 (classification) release rather than the Task 1
>   (ranking) release — and rebuilt the benchmark on Task 1. See
>   [Setting 1b](#setting-1b).
> - **Reproduced Amazon's official Task 1 baseline** on that same pool, with
>   our evaluation queries removed from its training data, to make the first
>   like-for-like comparison possible.
> - **Cross-encoder reranker** (XLM-R base), which is the only model here that
>   beats the official baseline.

A multi-stage e-commerce search project built on Amazon ESCI. The project implements lexical retrieval, dense retrieval, hybrid fusion, and learning-to-rank components, and evaluates retrieval coverage and ranking quality under two controlled experimental settings rather than treating all reported metrics as one end-to-end result.

---

## Key Results

Every number below is quoted with its candidate pool; numbers from different pools are never compared with each other.

- **Full-catalog retrieval** (5,000 US queries, 1,215,854-product catalog, E/S/C relevant): fusing BM25 and Two-Tower with RRF raised Recall@100 from 0.4649 (Two-Tower V1 alone) to 0.5301; BM25 alone reached 0.4542.
- **Retriever complementarity** (V0 encoder, same benchmark): 12.0% of relevant items in the top 100 were found only by BM25, 13.1% only by Two-Tower, and 32.8% by both.
- **Corrected ranking benchmark:** found that the ranking evaluation had been running on the Tasks 2/3 pool, where 21.06% of test queries score NDCG 1.0 under *any* ordering (vs. 1 of 14,496 queries in the Task 1 pool), and rebuilt it on **KDD Cup Task 1**.
- **Against Amazon's official baseline** (reproduced locally, 5,071 Task 1 DEV queries, NDCG@20): an XLM-R cross-encoder scored 0.8456 vs. 0.8071 (Δ +0.0385, 95% CI [0.0348, 0.0421]); the 17-feature LambdaMART scored 0.7950, **below** the baseline (Δ −0.0122, 95% CI [−0.0160, −0.0082]).
- **Tree vs. MLP on identical features** (deduplicated Tasks 2/3 test pool, 30,969 queries, official linear gain): LambdaMART scored NDCG@10 0.8521 vs. 0.8222 for BM25 (Δ +0.0298, 95% CI [0.0284, 0.0313]), but beat the 17-feature MLP by only +0.0009 (95% CI [0.0004, 0.0014]), with 55.1% of queries scored identically.
- **Serving benchmarks** (V0 encoder, 1.2M vectors): replacing exact FAISS `IndexFlatIP` with HNSW cut search-only p50 latency from 85.0 ms to 0.26 ms while Recall@100 fell from 0.4598 to 0.4263; batching all 5,000 queries gave 1,014.9 QPS (encode + search) vs. 18.4 QPS for a one-query-at-a-time loop.
- Added business-aware ranking evaluation that penalizes over-budget results and uses rating quality as a tie-break signal.

## System Components

| Component | Role |
|---|---|
| BM25 | Lexical matching and lexical retrieval signal |
| Two-Tower | Semantic matching with a fine-tuned SentenceTransformer encoder |
| FAISS IndexFlatIP | Full-catalog dense retrieval index (HNSW / IVFFlat benchmarked offline, not adopted) |
| RRF (k=60) | Rank-based fusion of BM25 and Two-Tower retrieval lists |
| MLP rerankers | Pairwise neural reranking on fixed ESCI candidates |
| LambdaMART | Learning-to-rank reranker on the same 17-feature set as the advanced MLP |

The components are evaluated in two different settings below. The reported full-catalog RRF output was not fed into the MLP/LambdaMART experiments used for the NDCG results.

---

## Experimental Setting 1 — ESCI Reference-Candidate Scoring & Ranking

### What this setting measures

This setting uses the query-product candidate pairs already provided by Amazon ESCI. BM25 and Two-Tower generate lexical and semantic scores for those existing candidates; they do not perform catalog-wide candidate generation in this experiment.

Holding the candidate pool fixed isolates the ranking problem: if a relevant product is already available to the ranker, can the system place it near the top? Ranking quality is evaluated primarily with NDCG@K.

### Candidate data

- Candidate source: ESCI-provided query-product reference pairs.
- BM25 and Two-Tower score the same candidate set.
- The configured `TOP_K=150` belongs to this fixed-candidate scoring / demo path.
- The downstream MLP and LambdaMART rerankers are trained and evaluated on this fixed-candidate setting.
- The advanced 17-feature models additionally use ESCI-S enrichment fields such as price, star rating, review count, and category.

### Retrieval signals on the fixed candidate pool

**BM25 scoring.** BM25 provides the lexical baseline and the normalized `bm25_score` feature for downstream rerankers.

**Two-Tower scoring.** The dense encoder is based on `msmarco-distilbert-base-v3` (SentenceTransformer), fine-tuned on the **US-locale** subset of the ESCI training split's `small_version == 1` data (329,447 E/S-labeled query-product pairs) using **MultipleNegativesRankingLoss** for one epoch (batch size 64). This is the **V0** encoder; it scores the existing ESCI candidates rather than searching the whole catalog.

> **V0 vs. V1 encoder.** V0 ran 5,147 optimizer steps against the library-default `warmup_steps=10000`, so its learning rate never left warmup (final LR ≈ 52% of target, no decay). **V1** (`checkpoint-9286`) fixes this: 3 epochs, `warmup_ratio=0.10`, seed 42, checkpoint selected on a query-disjoint dev split, trained on 297,112 pairs (10% of queries held out for dev). V1 is used only for the full-catalog Recall@K result; all other Two-Tower numbers (fixed-candidate scoring, overlap analysis, serving benchmarks) use V0. Five things changed together between V0 and V1 and no ablation separates them, so the V1 gain cannot be attributed to any single fix.

Product text is `title + description + bullet_points`, matching the representation used at inference/scoring time.

> **Training-label note:** Two-Tower training uses E/S positives. This is separate from the broad full-catalog retrieval evaluation definition, where E/S/C are counted as relevant.

### Reranking models

Three reranker variants were trained on the fixed candidate setting: two pairwise MLPs using **MarginRankingLoss** over ESCI-labeled query/positive/negative triplets, and one LightGBM LambdaMART model.

#### Baseline reranker — 7 features

`DeepESCIReranker`: MLP with two hidden layers (64 → 32), LayerNorm + ReLU + Dropout.

| Feature | Description |
|---|---|
| `bm25_score` | Normalized BM25 retrieval score |
| `semantic_score` | Normalized Two-Tower cosine similarity |
| `word_overlap` | Stemmed query↔title token overlap fraction |
| `query_length` | Number of query tokens |
| `title_length` | Number of title tokens |
| `has_brand` | Whether the product has a brand field |
| `bullet_count` | Number of product bullet points |

Two additional features (`log_product_freq`, `log_brand_freq` — log frequency of the product/brand within the current candidate pool) were dropped: they're recomputed independently on train vs. test, so the same product/brand gets a different value in each split, and the model was picking up on this as a near-identity fingerprint instead of a generalizable signal.

Architecture: `7 → Linear(64) → LN → ReLU → Dropout(0.3) → Linear(32) → LN → ReLU → Dropout(0.2) → Linear(1)`
Training: AdamW (lr=1e-4, wd=1e-2), ReduceLROnPlateau, up to 50 epochs, early stopping (patience=6).

> **Debugging note:** the original version (`BatchNorm`, no dropout, `Sigmoid` output, plain `Adam` with no weight decay) trained to a train loss near 0 while validation loss diverged — a classic overfitting signature, but the actual mechanism was subtler than "needs more regularization." `Sigmoid` bounds the output to (0,1), which makes `MarginRankingLoss(margin=1.0)` almost unsatisfiable and pushes the network to saturate; removing `Sigmoid` alone made it *worse*, because an unbounded score plus a margin loss with no weight decay lets the network trivially inflate the pos/neg gap by scaling all weights up. The real root cause was `BatchNorm`: positives and negatives are forwarded through the model in two separate batches during training, so `BatchNorm` normalizes each side against its own batch statistics — letting the network separate them for free using the normalization itself, with none of that trick available once `eval()` switches to running population statistics. Swapping in `LayerNorm` (which normalizes per-sample, with no train/eval discrepancy) fixed it outright.

#### Advanced reranker — 17 features

`AdvancedDeepReranker`: MLP with two hidden layers (64 → 32), LayerNorm + LeakyReLU + Dropout. Extends the baseline with ESCI-S enrichment data (price, ratings, reviews, category).

**Query intent features**

| Feature | Description |
|---|---|
| `user_budget` | Budget parsed from query (e.g. "under $50") via regex |
| `cheap_intent` | Query contains "cheap", "affordable", or "budget" |
| `query_mean_idf` | Mean IDF of query tokens (proxy for specificity) |
| `query_max_idf` | Max IDF of query tokens |

**Item authority features** (from ESCI-S enrichment)

| Feature | Description |
|---|---|
| `log_price` | Log-transformed price (category/global median imputation) |
| `is_price_missing` | Price missingness indicator |
| `stars_clean` | Parsed star rating |
| `log_review_count` | Log-transformed review count |
| `is_rating_missing` | Rating missingness indicator |

**Interaction features**

| Feature | Description |
|---|---|
| `is_over_budget` | Product price exceeds stated user budget |
| `brand_match` | Query mentions the product brand |
| `color_match` | Query and title share a color token |
| `is_dominant_category` | Product matches the plurality category in BM25 top-20 |

Architecture: `17 → Linear(64) → LN → LeakyReLU(0.1) → Dropout(0.3) → Linear(32) → LN → LeakyReLU(0.1) → Dropout(0.2) → Linear(1)`
Training: AdamW (lr=1e-4, wd=1e-2), CosineAnnealingLR.

**Business NDCG:** A custom evaluation metric that hard-penalizes over-budget results and uses star rating as a quality tie-breaker within the same ESCI label.

#### LambdaMART reranker — 17 features (LightGBM)

A third reranker using `LightGBM`'s `lambdarank` objective on the **same 17 features** as the advanced MLP, for an apples-to-apples comparison of model paradigm (gradient-boosted trees vs. MLP) on identical inputs.

- Training label is a separate ordinal encoding (`I=0, C=1, S=2, E=3` with explicit `label_gain=[0, 1, 3, 7]`) — `lambdarank` indexes into an internal gain table by label, so it needs small integer grades rather than the fractional `target_score` used by the pairwise MLPs. Every model is *evaluated* on the same relevance scale, so the NDCG@10 column is comparable across rows.
  - **Known defect in this configuration:** `label_gain=[0, 1, 3, 7]` is not the gain vector the evaluator uses, so the training objective and the reported metric optimise different things. The net effect of aligning it is small (**+0.0006** NDCG@10, `experiments/ranking_v2/benchmark_repair/paired_comparisons.json`). It is an attributability defect rather than a performance one: until it was aligned, no experiment increment could be cleanly attributed.
- No feature normalization needed — tree splits are invariant to monotonic transforms, which also sidesteps the whole class of train/test normalization-mismatch bugs the MLP rerankers are prone to.
- Conservative, untuned hyperparameters (`num_leaves=31`, `learning_rate=0.05`, `n_estimators=1000` with early stopping on validation NDCG@10) — the goal was a fair comparison, not a tuned model.
- **Feature importance (gain)** is dominated by the two retrieval signals: `semantic_score` contributes ~2.4x the gain of `bm25_score`, with `word_overlap` a distant third. `log_review_count` and `is_over_budget` contribute zero gain — candidates for pruning. On a later candidate pool, `log_review_count` was found to be identically zero, i.e. the effective feature count is 16 rather than 17.

### Ranking results

Standard NDCG@10 on the ESCI test split. These results belong to the fixed-candidate/reranking setting and should be interpreted separately from the full-catalog Recall@K experiment in Setting 2.

> **Provenance — read before quoting these numbers.**
> `candidate_pool`: large-version (Tasks 2/3) ESCI test, deduplicated
> (`benchmark_repair/data/test_clean.parquet`, **638,016 rows / 30,969 queries**).
> `metric_version`: official KDD linear gain — E 1.0 / S 0.1 / C 0.01 / I 0.0 used directly as
> the DCG numerator, scored by `official_kdd_ndcg.py`, cutoff @10.
> This is **not** the KDD Cup Task 1 benchmark, and these values are **not comparable** to
> any published ESCI ranking number or to the Task 1 numbers in
> [Setting 1b](#setting-1b). Source: `experiments/ranking_v2/official_metric_final/`.

| Model | NDCG@10 |
|---|---:|
| BM25 (lexical retrieval) | 0.8222 |
| Two-Tower (dense retrieval, V0) | 0.8313 |
| Advanced reranker (17 features, MLP) | 0.8512 |
| **LambdaMART (17 features, LightGBM)** | **0.8521** |

Paired bootstrap (10,000 resamples, seed 42): LambdaMART − BM25 = **+0.0298**, 95% CI [0.0284, 0.0313]; LambdaMART − MLP = **+0.0009**, 95% CI [0.0004, 0.0014], p = 0.0008, with 55.1% of queries scored identically.

> **Retired figures.** Earlier versions of this README reported BM25 0.8188 → LambdaMART 0.8464 (MLP 0.8458). Those mixed two pools — BM25 from a 638,016-row join, the models from a 682,233-row pool with 6.93% duplicated rows — under a `2**rel − 1` gain, and are superseded by the table above. The 7-feature baseline MLP was not re-scored on the repaired pool, and its earlier before/after-fix figures have no saved result file, so no number is reported for it.

> **Locale scope note:** The Two-Tower encoder is English-only (`msmarco-distilbert-base-v3`) and was trained on US-locale pairs only. Its all-locale NDCG therefore reflects zero-shot transfer to ES and JP, where the tokenizer produces heavy `[UNK]` on Japanese text. Per-locale results are not separated in this experiment.

LambdaMART's lead over the advanced MLP is statistically significant but very small: on identical features, the model family barely matters, and ranking quality at this feature budget is bottlenecked by feature coverage rather than architecture. Note that this comparison also differs in training data and objective, not only in architecture.

### What this experiment answers

This controlled setting measures ranking quality conditional on candidate availability. Because the candidate pool is fixed, NDCG differences are easier to attribute to scoring features and reranking models rather than to changes in retrieval coverage.

---

<a id="setting-1b"></a>

## Experimental Setting 1b — Competition-Aligned Benchmark (KDD Cup Task 1)

### Why there is a second benchmark

ESCI ships **two** candidate pools: a reduced one for Task 1 (ranking) and a larger one for Tasks 2/3 (classification). Setting 1 above ran on the larger one. The two share a dataset name and a metric name, but they are different tasks — and the Tasks 2/3 pool is substantially easier to score well on, because **21.06%** of its test queries (6,523 / 30,969) are *degenerate*: every candidate carries the same relevance label, so any ordering whatsoever scores a perfect 1.0. On the Task 1 pool that share is **0.007%** (1 query in 14,496).

The benchmark was therefore rebuilt on the Task 1 pool. Scores on the two pools are much further apart than any model difference, but they are different populations, so the gap between them should not be read as a single "drop" number.

### Task 1 test results

`candidate_pool`: Task 1 test, **336,373 rows / 14,496 queries**.
`metric_version`: official competition gain — E 1.0 / S 0.1 / C 0.01 / I 0.0 used **directly** as the DCG numerator, *not* `2**rel − 1`.

| Model | NDCG (full) | @20 | @10 |
|---|---:|---:|---:|
| Random floor | 0.7503 | 0.6861 | 0.5670 |
| BM25 | 0.8198 | 0.7683 | 0.6819 |
| Two-Tower | 0.8245 | 0.7730 | 0.6886 |
| **LambdaMART (17 features)** | **0.8429** | **0.7951** | **0.7199** |
| Oracle ceiling | 1.0000 | 1.0000 | 1.0000 |

Full-list is the headline cutoff here: Task 1 averages 23.2 candidates per query and caps at 40, so full, @20 and @10 genuinely differ.

> **Open inconsistency:** `experiments/competition_alignment/current_vs_official_split_audit.json` records the Task 1 test artifact as "never evaluated", which conflicts with `kdd_task1_benchmark/task1_baseline_comparison.csv`, the source of this table. Until that is settled, treat these test numbers as provisional, and do not compare them with the DEV numbers below (different population).

### Against Amazon's official baseline

The 0.83 published in the upstream esci-data README is **not** a usable comparison target: it is produced through a helper that swaps the S and C gain weights, and it is reported on the competition's *public test* split — which cannot be recovered from the public release, because the released `test` split is public ∪ private concatenated with the boundary erased (30.01% of queries, matching 15% + 15%).

So the baseline was retrained locally from the official recipe, with our evaluation queries (**118,231 rows**) removed from its training pool first, leaving both models trained on the same **28,733** Task 1 training queries and scored on the same **5,071** DEV queries by the same scorer.

The cross-encoder is `FacebookAI/xlm-roberta-base` fine-tuned with a 4-class E/S/C/I head on **663,407** query–product pairs (**28,733** queries) from Task 1 train, with the DEV queries held out. Its ranking score is P(E)·1.0 + P(S)·0.1 + P(C)·0.01, and the checkpoint was selected on dev NDCG@20, not on loss (`experiments/ranking_v2/cross_encoder_task1/FROZEN_V2.json`).

`candidate_pool`: Task 1 DEV, 5,071 queries / 118,231 rows. Same `metric_version` as above.

| Model | NDCG (full) | @20 | @10 | us@20 | es@20 | jp@20 |
|---|---:|---:|---:|---:|---:|---:|
| Random floor | 0.7489 | 0.6843 | 0.5648 | 0.7101 | 0.6317 | 0.6512 |
| BM25 | 0.8187 | 0.7655 | 0.6792 | 0.7895 | 0.7466 | 0.7113 |
| Two-Tower | 0.8253 | 0.7732 | 0.6913 | 0.8062 | 0.7458 | 0.6996 |
| *Amazon official baseline (reproduced)* | *0.8521* | *0.8071* | *0.7358* | *0.8345* | *0.7823* | *0.7477* |
| LambdaMART (17 features) | 0.8430 | 0.7950 | 0.7210 | 0.8189 | 0.7767 | 0.7405 |
| **Cross-encoder (XLM-R base)** | **0.8834** | **0.8456** | **0.7924** | **0.8558** | **0.8486** | **0.8140** |
| Oracle ceiling | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Query-level paired bootstrap on @20, n=10,000, seed=42:

| Step | Δ@20 | 95% CI | p | W/L/T |
|---|---:|---|---:|---|
| BM25 − Random | +0.0811 | [+0.0761, +0.0862] | <0.0001 | 3359/1700/12 |
| Two-Tower − BM25 | +0.0077 | [+0.0034, +0.0120] | 0.0008 | 2628/2378/65 |
| Official baseline − Two-Tower | +0.0340 | [+0.0300, +0.0380] | <0.0001 | 2915/2094/62 |
| **LambdaMART − official baseline** | **−0.0122** | **[−0.0160, −0.0082]** | **<0.0001** | 2353/2636/82 |
| **Cross-encoder − LambdaMART** | **+0.0506** | **[+0.0465, +0.0547]** | **<0.0001** | 3248/1703/120 |
| **Cross-encoder − official baseline** | **+0.0385** | **[+0.0348, +0.0421]** | **<0.0001** | 3229/1696/146 |

On the 3,133 US-locale DEV queries alone, the cross-encoder scores NDCG@20 0.8558 vs. 0.8345 for the official baseline (Δ +0.0213, 95% CI [0.0173, 0.0253], W/L/T 1852/1160/121).

**The feature-based LambdaMART loses to the official baseline** by 0.0122 at @20, and its point estimates are also lower at full-list and @10 (table above; only @20 was bootstrapped). This was invisible for most of the project because the two had never been placed on the same split. The cross-encoder clears the baseline by **+0.0385** at @20. The official-baseline numbers follow Amazon's recipe: a cross-encoder for US and a bi-encoder for ES/JP.

### Caveats that travel with these numbers

- **No leaderboard comparison is possible.** The KDD Cup 2022 private test split is unrecoverable from the public release, so no number here can be compared to the 0.9043 winning score. This is a permanent limitation, not pending work.
- **The official scorer was never executed.** Parity was established against an independent pure-Python reimplementation (13/13 cases, including 300 random queries × {@10, @20, full}; max |Δ| = 1.1e-16). That is self-consistency, not external validation — Terrier was never run.
- **LambdaMART carries an upstream exposure.** Its `semantic_score` feature comes from a Two-Tower trained on the Task 1 training split, which includes 100% of the US DEV queries (61.8% of DEV). This is train-on-train exposure, not label leakage; its magnitude was not measured and is perfectly confounded with locale.
- Full artifacts: `experiments/ranking_v2/kdd_task1_benchmark/` and `experiments/competition_alignment/`.

---

## Experimental Setting 2 — Full-Catalog Retrieval

### What this setting measures

This is a separate experiment for true candidate generation. Instead of starting from ESCI-provided query-product pairs, the system searches the product catalog directly and asks: can relevant products be recovered from the catalog at all? Retrieval quality is evaluated with Recall@K.

### BM25 full-catalog retrieval

A global BM25 index is built over the product catalog and queried directly. The experiment retrieves up to Top-200 and evaluates K ∈ {10, 50, 100, 200}.

### Two-Tower full-catalog retrieval

All product representations are encoded once with the fine-tuned Two-Tower and stored in FAISS IndexFlatIP. At query time, the query encoder retrieves products directly from this catalog-wide dense index.

### Hybrid retrieval — Reciprocal Rank Fusion

BM25 and Two-Tower independently produce ranked product lists. Their union is fused with RRF, k=60:

```text
RRF(d) = 1 / (60 + rank_BM25(d)) + 1 / (60 + rank_TT(d))
```

A product missing from one retriever contributes zero from that side. The candidate union is sorted by RRF score and truncated to the requested Top-K. RRF avoids directly combining raw BM25 and dense-similarity scores, which live on different numerical scales.

### Full-catalog retrieval results

| Retriever | Broad Recall@100 (V1 encoder) | Broad Recall@100 (V0 encoder, superseded) |
|---|---:|---:|
| BM25 | 0.4542 | 0.4542 |
| Two-Tower | 0.4649 | 0.4598 |
| **Hybrid (RRF)** | **0.5301** | 0.5194 |

Evaluated on a deterministic 5,000-query sample (seed=42) of the 22,458 US-locale test queries, not the full set. Both encoder runs use the identical benchmark; V1 is the current result (sources: `experiments/two_tower_v2/phase1_correct_training/full_run/full_catalog_eval/retrieval_metrics_v1.csv`, V0: `output/full_retrieval/retrieval_metrics.csv`). The 5,000-query benchmark was used only for final reporting, not for checkpoint selection. V1 is a single run at one seed, so the +0.0051 Two-Tower gain over V0 has no confidence interval.

The Two-Tower is only slightly stronger than BM25 in standalone Recall@100. Its main value is complementarity: it retrieves relevant products that BM25 misses.

### Retrieval complementarity

At the micro / relevant-item-instance level, Top-100 retrieval ownership (V0 encoder, 84,572 relevant instances) is:

| Retrieval ownership | Share of relevant item instances |
|---|---:|
| BM25 only | 12.023% |
| Two-Tower only | 13.074% |
| Both | 32.772% |
| Neither | 42.131% |

Although BM25 and Two-Tower have very similar standalone Recall@100, they fail on different items. The Two-Tower uniquely recovers 13.1% of relevant item instances that BM25 misses, while BM25 uniquely recovers 12.0% that the Two-Tower misses. Their standalone recall looks similar because these unique gains and losses nearly cancel, not because the retrievers return the same products.

#### Macro Recall vs. micro overlap

The overlap percentages above are micro-averaged over relevant query-item instances, while the reported Recall@100 metric is macro-averaged over queries:

```text
Macro Recall@100 = mean_q(hits_q / relevant_q)
Micro coverage    = sum_q(hits_q) / sum_q(relevant_q)
```

Therefore, BM25-only + Both should not be expected to equal the reported BM25 Recall@100 exactly. BM25's micro-level coverage is 12.023% + 32.772% = 44.795%, while its macro Recall@100 is 45.42%. The difference comes from aggregation weights: queries with more relevant items receive more weight under micro averaging, and BM25's per-query recall has a small negative correlation with the number of relevant items for the query.

The micro-level oracle union covers 12.023% + 13.074% + 32.772% = 57.869% of relevant item instances. This is a micro-level union coverage ceiling only and should not be directly subtracted from macro Recall@100 values.

### Serving benchmarks

All on the **V0** encoder and the 5,000-query benchmark (`experiments/two_tower_v2/phase7_serving/`):

| Measurement | Result |
|---|---|
| Per-query p50 retrieval latency | BM25 (`bm25s`, CPU) 2.69 ms; Two-Tower (Apple MPS) 53.9 ms |
| Exact vs. ANN index, search-only p50 (pre-encoded queries) | `IndexFlatIP` 85.0 ms → HNSW (M=32, efC=40, efSearch=64) 0.26 ms |
| Recall@100 cost of HNSW | 0.4598 → 0.4263 (−7.3% relative) |
| Throughput, encode + search | batched (batch_size 256) 1,014.9 QPS; one-query-at-a-time loop 18.4 QPS |

The two throughput figures measure different things and are reported side by side, not as a speedup ratio. The ANN indices were built offline from the existing flat index and benchmarked only; the retrieval results above use exact `IndexFlatIP`.

### What this experiment answers

This setting measures candidate-generation coverage: whether relevant products can be found from the catalog before ranking. It complements, but is intentionally separate from, the fixed-candidate ranking experiment above.

---

## How the Two Settings Relate

The two settings isolate different search failure modes:

- **Full-catalog retrieval:** was the relevant product found at all?
- **Fixed-candidate ranking:** once a candidate is available, was it ordered correctly?

This separation makes error diagnosis cleaner. A low Recall@K points to a candidate-generation problem; a low NDCG@K with a fixed candidate pool points to a ranking problem.

The project therefore implements the major components of a multi-stage search system, but the reported experiments are controlled module-level evaluations. The reported full-catalog RRF candidates were not passed into the MLP/LambdaMART rerankers, so Recall@K and NDCG@K should not be interpreted as consecutive metrics from one executed end-to-end pipeline.

---

## Evaluation Architecture

**Setting 1 — Fixed reference candidates**

```text
ESCI query-product candidates
            │
            ├── BM25 lexical score
            └── Two-Tower semantic score
                        │
                        ▼
              Feature extraction
                        │
                        ▼
              MLP / LambdaMART
                        │
                        ▼
                     NDCG@K
```

**Setting 2 — Full-catalog retrieval**

```text
                    User Query
                   /          \
                  ▼            ▼
          BM25 global index   Two-Tower + FAISS
                  \            /
                   \          /
                    ▼        ▼
                    RRF (k=60)
                        │
                        ▼
                 Retrieval Top-K
                        │
                        ▼
                     Recall@K
```

---

## Original architecture diagrams

### Preliminary design
<img width="8192" height="6999" alt="Preliminary Architecture" src="https://github.com/user-attachments/assets/2c2ea3c6-ed8a-48fa-a84b-04c393847488" />

### Final architecture (iterated)
<img width="716" height="713" alt="Architecture v2 - retrieval" src="https://github.com/user-attachments/assets/6f28f390-4609-4aa5-88b8-16e25ebe1104" />
<img width="716" height="663" alt="Architecture v2 - reranking" src="https://github.com/user-attachments/assets/ed6933d5-990f-408f-8ac6-2348cdcb01c4" />
<img width="716" height="608" alt="Architecture v2 - features" src="https://github.com/user-attachments/assets/16b3a438-c6fa-4f34-afa0-cd9aee0364aa" />
<img width="716" height="661" alt="Architecture v2 - training" src="https://github.com/user-attachments/assets/de4f82c0-cd76-401a-8112-8e82c8abeb15" />

---

## Demo

The interactive search app (`interactive_search.py`) runs a full end-to-end query using the advanced reranker. Results are displayed in a sortable GUI table with columns: Rank, Score, Brand, Price, Stars, Reviews, Category, Title. MMR (Maximal Marginal Relevance, λ=0.6) is applied post-reranking to penalize repeated brands and improve diversity. This applies only to the interactive demo. No reported metric in Setting 1 or Setting 2 involves MMR.

Requires Steps 1–5 from Setup to be completed first.

```bash
python interactive_search.py
```

---

## Dataset

**Amazon ESCI** — ~2.6M (2,621,288) human-annotated query-product pairs across us/jp/es locales. The full-catalog retrieval experiment (Setting 2) is scoped to the 1,215,854 US-locale products.

> ESCI is released as **two different candidate pools**, and they are not interchangeable:
>
> | | queries | judgements | used by |
> |---|---:|---:|---|
> | `small_version == 1` — **Task 1**, ranking | 48,300 | 1,118,011 | Setting 1b |
> | `large_version == 1` — **Tasks 2/3**, classification | 130,652 | 2,621,288 | Settings 1 and 2 |
>
> Both are "ESCI" and both are scored with "NDCG", which is exactly why the mix-up described
> in [Setting 1b](#setting-1b) was
> easy to make. Any NDCG figure from this project should be quoted together with its pool.

| Label | Meaning | Relevance weight |
|---|---|---|
| E (Exact) | The product directly answers the query | 1.0 |
| S (Substitute) | A related but not ideal product | 0.1 |
| C (Complement) | A product often bought alongside | 0.01 |
| I (Irrelevant) | Not related | 0.0 |

**ESCI-S** — an enrichment dataset adding structured fields (price, star ratings, review counts, product category) used by the advanced reranker. Source: [shuttie/esci-s](https://github.com/shuttie/esci-s) ([esci.json.zst](https://esci-s.s3.amazonaws.com/esci.json.zst), ~3.4GB compressed, ~1.66M products). Place it at `esci-data/esci-s_dataset/esci.json.zst` and run `python utility/convert_esci_to_parquet.py` to produce `esci_s_products.parquet` (only `asin`/`price`/`stars`/`ratings`/`category` are kept — the raw dataset also has a per-row-variable `attr` struct field for book metadata that would otherwise break schema-consistent chunked parquet writing).

---

## Setup

> The ESCI dataset and trained model weights are not included in this repo. Follow these steps in order to train from scratch.

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

> **macOS + LightGBM:** the LightGBM wheel links against Homebrew's OpenMP runtime, which isn't installed by default: `brew install libomp`.

**2. Fine-tune the Two-Tower encoder** (once)
```bash
python scripts/train_two_tower.py
```
Saves weights to `models/two_tower_finetuned/`.

**3. Generate retrieval scores** (once, or when data changes)
```bash
python scripts/generate_bm25_scores.py
python scripts/generate_two_tower_scores.py
```
Produces `output/bm25_scores_{train,test}.csv` and `output/two_tower_scores_{train,test}.csv`.

**4. Build search indices** (once)
```bash
python scripts/build_indices.py
```
Pre-computes the global BM25 and FAISS indices used at query time.

**5. Train the reranker**

Baseline (7-feature):
```bash
python scripts/train_reranker.py
```

Advanced (17-feature, requires ESCI-S):
```bash
python scripts/train_adv_reranker.py
```

LambdaMART (17-feature, requires ESCI-S):
```bash
python scripts/train_lambdamart.py
```

**6. Evaluate**
```bash
python evaluation/evaluate_retrieval.py   # BM25 and Two-Tower baselines
python evaluation/evaluate_reranker.py    # Baseline reranker
python evaluation/evaluate_advanced.py    # Advanced reranker (standard + business NDCG)
python evaluation/evaluate_lambdamart.py  # LambdaMART reranker (standard + business NDCG)
```

---

## Project Structure

```
.
├── interactive_search.py          # Tkinter GUI demo (advanced reranker + MMR)
├── config.py                      # Global paths and settings
├── requirements.txt
│
├── retrieval/
│   ├── bm25.py                    # BM25 scoring (fixed-candidate) + global BM25 index (full-catalog/demo)
│   └── two_tower.py               # Two-Tower scoring (fixed-candidate) + global FAISS index (full-catalog/demo)
│
├── reranking/
│   ├── model.py                   # DeepESCIReranker (7-feature baseline)
│   ├── features.py                # Feature extraction + PairwiseESCIDataset
│   ├── advanced_model.py          # AdvancedDeepReranker (17-feature)
│   └── advanced_features.py       # Advanced feature extraction + dataset
│
├── evaluation/
│   ├── metrics.py                 # NDCG@K, Recall@K, business NDCG
│   ├── evaluate_retrieval.py      # BM25 / Two-Tower NDCG@10 (fixed-candidate)
│   ├── evaluate_reranker.py       # 7-feature MLP NDCG@10 (fixed-candidate)
│   ├── evaluate_advanced.py       # 17-feature MLP standard + business NDCG@10 (fixed-candidate)
│   └── evaluate_lambdamart.py     # LambdaMART standard + business NDCG@10 (fixed-candidate)
│
├── scripts/
│   ├── train_two_tower.py
│   ├── train_reranker.py
│   ├── train_adv_reranker.py
│   ├── train_lambdamart.py            # LightGBM LambdaMART reranker (17-feature)
│   ├── generate_bm25_scores.py
│   ├── generate_two_tower_scores.py
│   ├── build_indices.py               # global BM25/FAISS indices for the live demo
│   ├── run_pipeline.py
│   │
│   ├── build_full_catalog_indices.py  # US-locale BM25 + FAISS index over the full ~1.2M product catalog
│   ├── sample_eval_queries.py         # deterministic 5,000-query eval set + broad/exact ground truth
│   ├── run_full_bm25_retrieval.py     # full-catalog BM25 retrieval (Top-200) + latency
│   ├── run_full_tt_retrieval.py       # full-catalog Two-Tower/FAISS retrieval (Top-200) + latency
│   ├── evaluate_full_retrieval.py     # Recall@K/ExactRecall@K, RRF hybrid fusion, overlap analysis
│   ├── analyze_label_coverage.py      # ESCI label coverage of full-catalog Top-100 results
│   │
│   ├── run_feature_ablation.py        # single-feature ablation for the 17-feature MLP
│   ├── run_group_ablation.py          # feature-group ablation for the 17-feature MLP
│   ├── build_query_slices.py          # BM25 vs Two-Tower NDCG by query slice (fixed-candidate)
│   ├── analyze_query_slices.py        # slice-level NDCG/win-rate aggregation
│   ├── analyze_topk_overlap.py        # BM25 vs MLP vs LambdaMART Top-10 overlap (fixed-candidate)
│   └── generate_interview_figures.py  # generates chart assets under screenshots/
│
├── utility/
│   └── convert_esci_to_parquet.py     # Chunked ESCI-S json.zst -> parquet conversion
│
├── tests/
│   ├── test_two_tower.py              # Unit tests (no real data or GPU required)
│   ├── test_baseline_search.py        # CLI interactive search (baseline reranker)
│   └── test_idf_scorer.py             # Query specificity/IDF scoring sanity check
│
├── models/                            # Saved model weights (gitignored)
└── output/                            # Scores, indices, normalization stats, full-catalog results (gitignored)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning | PyTorch |
| Transformer encoder | `sentence-transformers` / `msmarco-distilbert-base-v3` |
| Gradient-boosted ranking | LightGBM (`LGBMRanker`, `lambdarank` objective) |
| Semantic vector index | FAISS (IndexFlatIP — exact inner-product search for all reported retrieval; HNSW / IVFFlat benchmarked offline only) |
| Lexical retrieval | `bm25s` |
| Data | pandas, pyarrow, numpy |
| GUI | Tkinter |
