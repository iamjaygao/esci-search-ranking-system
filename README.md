# Search Ranking System (Amazon ESCI)

A multi-stage e-commerce search project built on Amazon ESCI. The project implements lexical retrieval, dense retrieval, hybrid fusion, and learning-to-rank components, and evaluates retrieval coverage and ranking quality under two controlled experimental settings rather than treating all reported metrics as one end-to-end result.

---

## Key Results

- On the ESCI reference-candidate ranking setting, improved NDCG@10 from 0.8188 (BM25 baseline) to 0.8464 with a 17-feature LightGBM LambdaMART reranker; the 17-feature MLP reached 0.8458, suggesting that performance at this feature budget was driven more by feature coverage than by model family.
- On a separate full-catalog retrieval setting, BM25 reached 0.4542 Broad Recall@100, Two-Tower reached 0.4598, and RRF hybrid retrieval improved Recall@100 to 0.5194.
- Item-level overlap analysis showed distinct retrieval error modes: 12.023% of relevant item instances were BM25-only, 13.074% were Two-Tower-only, and 32.772% were found by both.
- Added business-aware ranking evaluation that penalizes over-budget results and uses rating quality as a tie-break signal.

## System Components

| Component | Role |
|---|---|
| BM25 | Lexical matching and lexical retrieval signal |
| Two-Tower | Semantic matching with a fine-tuned SentenceTransformer encoder |
| FAISS IndexFlatIP | Full-catalog dense retrieval index |
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

**Two-Tower scoring.** The dense encoder is based on `msmarco-distilbert-base-v3` (SentenceTransformer), fine-tuned on the ESCI training split's `small_version == 1` subset using **MultipleNegativesRankingLoss** for one epoch (batch size 64) on E/S-labeled query-product pairs. In this setting, it scores the existing ESCI candidates rather than searching the whole catalog.

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

- Training label is a separate ordinal encoding (`I=0, C=1, S=2, E=3` with explicit `label_gain=[0, 1, 3, 7]`) — `lambdarank` indexes into an internal gain table by label, so it needs small integer grades rather than the fractional `target_score` used by the pairwise MLPs. Evaluation still uses the same relevance scale as every other model, so NDCG@10 numbers are directly comparable.
- No feature normalization needed — tree splits are invariant to monotonic transforms, which also sidesteps the whole class of train/test normalization-mismatch bugs the MLP rerankers are prone to.
- Conservative, untuned hyperparameters (`num_leaves=31`, `learning_rate=0.05`, `n_estimators=1000` with early stopping on validation NDCG@10) — the goal was a fair comparison, not a tuned model.
- **Feature importance (gain)** is dominated by the two retrieval signals: `semantic_score` contributes ~2.4x the gain of `bm25_score`, with `word_overlap` a distant third. `log_review_count` and `is_over_budget` contribute zero gain — candidates for pruning.

### Ranking results

Standard NDCG@10 on the ESCI test split. These results belong to the fixed-candidate/reranking setting and should be interpreted separately from the full-catalog Recall@K experiment in Setting 2:

| Model | NDCG@10 | Business NDCG@10 |
|---|---|---|
| BM25 (lexical retrieval) | 0.8188 | — |
| Two-Tower (dense retrieval) | 0.8267 | — |
| Baseline reranker (7 features, MLP) | 0.8444 | — |
| Advanced reranker (17 features, MLP) | 0.8458 | 0.8529 |
| **LambdaMART (17 features, LightGBM)** | **0.8464** | **0.8536** |

The baseline reranker originally scored **0.7822** — worse than doing no reranking at all — due to a `BatchNorm`/pairwise-training interaction (see the debugging note above). After the fix, it closes almost all of the gap to the 17-feature models despite using less than half the features, and LambdaMART essentially ties the advanced MLP: at this feature budget, ranking quality is bottlenecked by feature coverage, not by model architecture.

### What this experiment answers

This controlled setting measures ranking quality conditional on candidate availability. Because the candidate pool is fixed, NDCG differences are easier to attribute to scoring features and reranking models rather than to changes in retrieval coverage.

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

| Retriever | Broad Recall@100 |
|---|---:|
| BM25 | 0.4542 |
| Two-Tower | 0.4598 |
| **Hybrid (RRF)** | **0.5194** |

Evaluated on a deterministic 5,000-query sample (seed=42) of the 22,458 US-locale test queries, not the full set.

The Two-Tower is only slightly stronger than BM25 in standalone Recall@100. Its main value is complementarity: it retrieves relevant products that BM25 misses.

### Retrieval complementarity

At the micro / relevant-item-instance level, Top-100 retrieval ownership is:

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

**Amazon ESCI** — ~2.6M (2,621,288) human-annotated query-product pairs across us/jp/es locales. The full-catalog retrieval experiment (Setting 2) is scoped to the 1,215,854 US-locale products:

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
| Semantic vector index | FAISS (IndexFlatIP — exact/brute-force inner-product search, not ANN) |
| Lexical retrieval | `bm25s` |
| Data | pandas, pyarrow, numpy |
| GUI | Tkinter |
