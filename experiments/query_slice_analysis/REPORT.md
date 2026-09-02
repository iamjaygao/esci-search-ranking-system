# Retrieval Query-Slicing / Exact-Match Analysis

Pipeline: `output/full_retrieval/*` (5,000 US-locale ESCI test queries, seed=42,
Top-200 BM25 / Two-Tower retrieval, Recall@100 against `{E,S,C}` ESCI labels).
See `AUDIT.md` for full provenance. No retraining, no modification of existing
files; all new artifacts live under `experiments/query_slice_analysis/`.

Slice definitions (`build_query_assignments.py`) are constructed **without**
using any BM25/Two-Tower retrieval output, to avoid circularity:
- `is_sku_model` / `is_storage_numeric`: regex over raw query text only.
- `is_brand_heavy`: catalog-wide brand vocabulary (2,674 brands with
  US-catalog frequency ≥ 50, from `product_brand`), matched against raw
  query text only — not per-candidate brands.
- `is_strong_lexical` / `is_low_lexical_overlap`: stemmed token-overlap
  between the query and its **ground-truth relevant product titles**
  (top quartile / bottom quartile of the max overlap score across each
  query's `relevant_broad` items). Never uses a retrieval score to decide
  slice membership.

Bootstrap: query-level resampling with replacement, 1,000 resamples per
slice, 95% CI on `Δ = TT_Recall@100 − BM25_Recall@100` (descriptive
uncertainty only — no formal significance test is claimed).

## Overall baseline

| | BM25 Recall@100 | Two-Tower Recall@100 | Δ (TT−BM25) | 95% CI |
|---|---:|---:|---:|---:|
| Overall (n=5000) | 0.4542 | 0.4598 | +0.0056 | [-0.0010, +0.0128] |

Overall, Two-Tower nominally edges out BM25, but the CI crosses zero — not a
robust difference at the whole-test-set level. The slices below show this
reverses/strengthens directionally in specific subpopulations.

## Slice metrics

| slice | n | size | BM25 R@100 | TT R@100 | Δ(TT−BM25) | BM25 wins | TT wins | ties | 95% CI |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| SKU / model-like | 199 | stable | 0.5773 | 0.5344 | -0.0428 | 87 | 64 | 48 | [-0.0806, -0.0053] |
| Storage / numeric-spec | 136 | stable | 0.4922 | 0.4826 | -0.0096 | 60 | 57 | 19 | [-0.0505, +0.0303] |
| Brand-heavy | 744 | stable | 0.5604 | 0.5460 | -0.0144 | 310 | 295 | 139 | [-0.0340, +0.0049] |
| Strong lexical | 2872 | stable | 0.5314 | 0.5180 | -0.0134 | 1127 | 1210 | 535 | [-0.0230, -0.0045] |
| Low lexical-overlap | 1367 | stable | 0.2805 | 0.3149 | +0.0344 | 396 | 569 | 402 | [+0.0204, +0.0478] |

All five slices exceed n=100 ("stable" per Step 4's size threshold); none
needed to be marked TOO SMALL or exploratory. Full data with Recall@10/MRR@10
columns: `slice_metrics.csv`.

Full data: `query_assignments.csv`, `slice_metrics.csv`, `examples.csv`.

## Answers

### Q1. Is BM25 actually better than Two-Tower on SKU/model queries?

Yes, and the effect is robust. n=199, BM25 Recall@100=0.5773 vs. TT=0.5344
(Δ=-0.0428), 95% bootstrap CI **[-0.0806, -0.0053]** — entirely below zero.
Representative examples (BM25 recall=1.0, TT recall≈0): `insta360 one`,
`ps3 console`, `panasonic zs100 battery`, `x1 carbon 7th gen`.

**Verdict: SUPPORTED.**

### Q2. Is BM25 actually better than Two-Tower on storage/numeric-spec queries?

n=136, BM25 Recall@100=0.4922 vs. TT=0.4826 (Δ=-0.0096), 95% bootstrap CI
**[-0.0505, +0.0303]** — crosses zero. The point estimate leans toward BM25,
but the direction is not distinguishable from noise at this sample size.

**Verdict: NOT SUPPORTED** (directionally consistent, not statistically robust).

### Q3. Brand-heavy queries — who is actually stronger?

n=744, BM25 Recall@100=0.5604 vs. TT=0.5460 (Δ=-0.0144), 95% bootstrap CI
**[-0.0340, +0.0049]** — crosses zero (upper bound is just above 0). BM25
nominally leads on brand-heavy queries but the difference is not robust by
this bootstrap.

**Verdict: NOT SUPPORTED** (no predisposed answer was assumed; the data show
a BM25-leaning point estimate that does not clear the bootstrap CI threshold).

### Q4. Strong-lexical queries — who is stronger?

n=2872, BM25 Recall@100=0.5314 vs. TT=0.5180 (Δ=-0.0134), 95% bootstrap CI
**[-0.0230, -0.0045]** — entirely below zero. Note: because of a mass point
at overlap_score=1.0 (median=1.0, P75=1.0), this "top quartile" slice
actually covers 57% of all queries, not a narrow tail — reported as-is
rather than adjusted to produce a smaller slice.

**Verdict: SUPPORTED** (BM25 stronger).

### Q5. Low lexical-overlap queries — who is stronger?

n=1367, BM25 Recall@100=0.2805 vs. TT=0.3149 (Δ=+0.0344), 95% bootstrap CI
**[+0.0204, +0.0478]** — entirely above zero. Representative examples (BM25
recall=0, TT recall=1.0): `post-partum depression workbook`, `rated book`,
`harrypotter slythern sweatshirt`, `tempered chocolate kit`. Both retrievers'
absolute recall is much lower here (0.28 / 0.31) than overall (0.45 / 0.46) —
this slice is intrinsically harder for both retrievers, not just a place
where TT wins.

**Verdict: SUPPORTED** (Two-Tower stronger).

### Q6. Do these results support "BM25 and Two-Tower are complementary"?

Along the **lexical-overlap axis** specifically, yes: BM25 has a robust edge
on SKU/model queries (Q1) and on the strong-lexical slice (Q4), while
Two-Tower has a robust edge on the low-lexical-overlap slice (Q5), and all
three of those CIs exclude zero. That is a genuine complementary pattern on
this data.

However, complementarity is **not** uniform across every "exact-match-like"
category tested: storage/numeric-spec (Q2) and brand-heavy (Q3) queries show
the same directional lean (BM25 nominally ahead) but neither CI excludes
zero, so those two categories do not independently support a complementarity
claim at the level of rigor used here.

**Verdict: PARTIALLY SUPPORTED** — true for the lexical-overlap-defined
slices (SKU/model, strong-lexical, low-lexical-overlap), not established for
storage/numeric-spec or brand-heavy queries specifically.

### Q7. Does this support the resume claim "dense retrieval underperformed BM25 on exact-match queries such as brand names and SKUs"?

Splitting the compound claim as instructed:
- **SKUs**: supported (Q1, CI excludes zero).
- **Brand names**: not supported at this rigor level (Q3, CI crosses zero).

Because the two halves of the claim disagree, the claim as a whole cannot be
marked simply SUPPORTED.

**Verdict: PARTIALLY SUPPORTED.**

## Caveats

- Bootstrap CIs here are descriptive uncertainty only (percentile bootstrap,
  1,000 resamples, seed=42); no formal hypothesis test was run, and no
  "statistically significant" language is used beyond "CI excludes zero."
- The "strong lexical" slice is large (57% of all queries) because of a mass
  point at overlap_score=1.0 in the underlying distribution, not because the
  P75 threshold was chosen to enlarge it — reported as found.
- `is_brand_heavy` and `is_sku_model` are not mutually exclusive from the
  other slices (a query can be both SKU-like and brand-heavy); slice metrics
  are computed independently per slice, not as a partition.
- Brand vocabulary uses a frequency threshold (≥50 catalog occurrences) and
  excludes known non-brand placeholder values (`generic`, `unbranded`,
  `vinyl`, etc., reusing the denylist already vetted in
  `scripts/build_query_slices.py`); a small amount of residual noise in
  catalog `product_brand` values is possible but not manually reviewed here.
