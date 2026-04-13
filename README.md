# Hybrid Search & Reranking

Two-stage neural product search over 2.6M Amazon products. A query like *"running shoes for seniors"* retrieves and re-scores candidates in real time using a combination of keyword search, semantic embeddings, and a learned neural reranker.

**Team:** Gyula Planky, Jian Gao, Hyuk Jin Chung — Deep Learning course project

---

## Pipeline

```
User Query
    │
    ├─── BM25 (lexical)  ──────────┐
    │                              ├──▶  Merge top-150 candidates
    └─── Two-Tower (semantic) ─────┘
                                            │
                                            ▼
                                   Neural Reranker
                                   (17-feature MLP)
                                            │
                                            ▼
                                   Top results + MMR diversity
```

**Why both retrievers?**

| BM25 wins when… | Two-Tower wins when… |
|---|---|
| "iPhone 15 Pro Max" (exact match) | "gift for outdoorsy dad" (no exact words) |
| "SKU-12345" (code lookup) | "comfortable WFH chair" (semantic intent) |

---

## Architecture

### Preliminary design
<img width="8192" height="6999" alt="Preliminary Architecture" src="https://github.com/user-attachments/assets/2c2ea3c6-ed8a-48fa-a84b-04c393847488" />

### Final architecture (iterated)
<img width="716" height="713" alt="Architecture v2 - retrieval" src="https://github.com/user-attachments/assets/6f28f390-4609-4aa5-88b8-16e25ebe1104" />
<img width="716" height="663" alt="Architecture v2 - reranking" src="https://github.com/user-attachments/assets/ed6933d5-990f-408f-8ac6-2348cdcb01c4" />
<img width="716" height="608" alt="Architecture v2 - features" src="https://github.com/user-attachments/assets/16b3a438-c6fa-4f34-afa0-cd9aee0364aa" />
<img width="716" height="661" alt="Architecture v2 - training" src="https://github.com/user-attachments/assets/de4f82c0-cd76-401a-8112-8e82c8abeb15" />

---

## Stage 1 — Retrieval

### BM25
Classic lexical retrieval using `bm25s` (C-native). For live queries, a global index is pre-built over the full product catalog. Scores are min-max normalized; top-150 kept per query.

### Two-Tower
Base model: `msmarco-distilbert-base-v3` (SentenceTransformer), fine-tuned on ESCI using **MultipleNegativesRankingLoss** for one epoch (batch size 64) on E/S-labeled query-product pairs. At inference, all products are encoded once and stored in a **FAISS IndexFlatIP** for fast ANN search.

---

## Stage 2 — Reranking

Two generations of reranker were trained, both using **pairwise MarginRankingLoss** over ESCI-labeled (query, positive, negative) triplets.

### Baseline reranker — 9 features

`DeepESCIReranker`: 3-layer MLP with BatchNorm + ReLU.

| Feature | Description |
|---|---|
| `bm25_score` | Normalized BM25 retrieval score |
| `semantic_score` | Normalized Two-Tower cosine similarity |
| `word_overlap` | Stemmed query↔title token overlap fraction |
| `query_length` | Number of query tokens |
| `title_length` | Number of title tokens |
| `has_brand` | Whether the product has a brand field |
| `bullet_count` | Number of product bullet points |
| `log_product_freq` | Log frequency of the product in training data |
| `log_brand_freq` | Log frequency of the brand in training data |

Architecture: `9 → Linear(64) → BN → ReLU → Linear(32) → BN → ReLU → Linear(1) → Sigmoid`
Training: Adam (lr=1e-3), ReduceLROnPlateau, up to 50 epochs, early stopping (patience=6).

### Advanced reranker — 17 features

`AdvancedDeepReranker`: 3-layer MLP with LayerNorm + LeakyReLU + Dropout. Extends the baseline with ESCI-S enrichment data (price, ratings, reviews, category).

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

---

## Demo

The interactive search app (`interactive_search.py`) runs a full end-to-end query using the advanced reranker. Results are displayed in a sortable GUI table with columns: Rank, Score, Brand, Price, Stars, Reviews, Category, Title. MMR (Maximal Marginal Relevance, λ=0.6) is applied post-reranking to penalize repeated brands and improve diversity.

Requires Steps 1–5 from Setup to be completed first.

```bash
python interactive_search.py
```

---

## Dataset

**Amazon ESCI** — ~2.6M US product listings with human-annotated query-product relevance labels:

| Label | Meaning | Relevance weight |
|---|---|---|
| E (Exact) | The product directly answers the query | 1.0 |
| S (Substitute) | A related but not ideal product | 0.1 |
| C (Complement) | A product often bought alongside | 0.01 |
| I (Irrelevant) | Not related | 0.0 |

**ESCI-S** — an enrichment dataset adding structured fields (price, star ratings, review counts, product category) used by the advanced reranker.

---

## Setup

> The ESCI dataset and trained model weights are not included in this repo. Follow these steps in order to train from scratch.

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

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

Baseline (9-feature):
```bash
python scripts/train_reranker.py
```

Advanced (17-feature, requires ESCI-S):
```bash
python scripts/train_adv_reranker.py
```

**6. Evaluate**
```bash
python evaluation/evaluate_retrieval.py   # BM25 and Two-Tower baselines
python evaluation/evaluate_reranker.py    # Baseline reranker
python evaluation/evaluate_advanced.py    # Advanced reranker (standard + business NDCG)
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
│   ├── bm25.py                    # BM25 scoring and global index
│   └── two_tower.py               # Two-Tower scoring and FAISS index
│
├── reranking/
│   ├── model.py                   # DeepESCIReranker (9-feature baseline)
│   ├── features.py                # Feature extraction + PairwiseESCIDataset
│   ├── advanced_model.py          # AdvancedDeepReranker (17-feature)
│   └── advanced_features.py       # Advanced feature extraction + dataset
│
├── evaluation/
│   ├── metrics.py                 # NDCG@K, Recall@K, business NDCG
│   ├── evaluate_retrieval.py
│   ├── evaluate_reranker.py
│   └── evaluate_advanced.py
│
├── scripts/
│   ├── train_two_tower.py
│   ├── train_reranker.py
│   ├── train_adv_reranker.py
│   ├── generate_bm25_scores.py
│   ├── generate_two_tower_scores.py
│   ├── build_indices.py
│   └── run_pipeline.py
│
├── tests/
│   ├── test_two_tower.py          # Unit tests (no real data or GPU required)
│   └── test_baseline_search.py    # CLI interactive search (baseline reranker)
│
├── models/                        # Saved model weights (gitignored)
└── output/                        # Scores, indices, normalization stats (gitignored)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning | PyTorch |
| Transformer encoder | `sentence-transformers` / `msmarco-distilbert-base-v3` |
| Semantic ANN index | FAISS (IndexFlatIP) |
| Lexical retrieval | `bm25s` |
| Data | pandas, pyarrow, numpy |
| GUI | Tkinter |
