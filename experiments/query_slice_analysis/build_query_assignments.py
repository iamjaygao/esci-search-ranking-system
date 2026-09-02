"""
Step 2 -- Build query slice assignments for the 5,000-query full-catalog
retrieval eval set (output/full_retrieval/eval_query_ids.json /
ground_truth.json). No retraining, no modification of existing files.

Slice definitions are constructed WITHOUT using any BM25/Two-Tower retrieval
output, to avoid circularity:
  - is_sku_model, is_storage_numeric: regex over the raw query text only.
  - is_brand_heavy: catalog-wide brand vocabulary (from product_brand),
    matched against the raw query text only.
  - is_strong_lexical / is_low_lexical_overlap: stemmed token-overlap between
    the query and its GROUND-TRUTH relevant product titles (not retrieved
    candidates).

Reuses the conservative SKU/model regex, brand-placeholder denylist, and
model-line-prefix vocabulary already vetted in scripts/build_query_slices.py
(a different pipeline -- the fixed-candidate reranking task -- but the same
regex philosophy), imported read-only. Does not modify that file.
"""
import os
import sys
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from config import PRODUCTS_PATH
from scripts.build_query_slices import (
    normalize_text, tokenize, model_sku_flag, brand_explicit_flag,
    NON_BRAND_PLACEHOLDERS,
)

RETRIEVAL_DIR = f"{ROOT_DIR}/output/full_retrieval"
OUT_DIR = f"{ROOT_DIR}/experiments/query_slice_analysis"
LOCALE = "us"

MIN_BRAND_FREQ = 50  # catalog-wide minimum occurrence count to enter the brand vocabulary

# ============================================================
# Storage / numeric-spec regex (Step 2B) -- written fresh, since the
# fixed-candidate pipeline's DIMENSION_PATTERN doesn't cover hyphenated
# "55-inch" or resolution tokens like "4k"/"1080p".
# ============================================================
CAPACITY_UNIT = re.compile(r'\b\d+(\.\d+)?\s?-?\s?(gb|tb|mb|kg|lb|lbs|oz|ml)\b|\b\d+(\.\d+)?(l)\b')
DIMENSION_UNIT = re.compile(r'\b\d+(\.\d+)?\s?-?\s?(inch|in|mm|cm|ft|foot|feet)\b|\b\d+(\.\d+)?"')
RESOLUTION_TOKEN = re.compile(r'\b(4k|8k|1080p|720p|2160p|hd|uhd)\b')
RAM_STORAGE_TOKEN = re.compile(r'\b\d+\s?-?\s?(gb|tb)\s?(ram|storage|memory)?\b')

STORAGE_NUMERIC_PATTERN = re.compile(
    '|'.join(p.pattern for p in [CAPACITY_UNIT, DIMENSION_UNIT, RESOLUTION_TOKEN, RAM_STORAGE_TOKEN])
)


def main():
    with open(f"{RETRIEVAL_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)
    query_ids = [str(q) for q in eval_meta["query_ids"]]

    with open(f"{RETRIEVAL_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)

    print(f"Loaded {len(query_ids)} eval queries (seed={eval_meta['seed']}, locale={eval_meta['locale']})")

    # ---- Catalog (us-locale only) for brand vocabulary + relevant-product titles ----
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_pr['product_id'] = df_pr['product_id'].astype(str)
    df_pr_us = df_pr[df_pr['product_locale'] == LOCALE].drop_duplicates('product_id')
    print(f"US-locale catalog: {len(df_pr_us)} products")

    # ---- Brand vocabulary: catalog-wide, frequency-thresholded, denylist-filtered ----
    brand_counts = Counter()
    for b in df_pr_us['product_brand'].dropna():
        b_norm = normalize_text(b)
        if b_norm and len(b_norm) >= 2 and b_norm not in NON_BRAND_PLACEHOLDERS:
            brand_counts[b_norm] += 1
    brand_vocab = sorted([b for b, c in brand_counts.items() if c >= MIN_BRAND_FREQ])
    print(f"Brand vocabulary: {len(brand_vocab)} brands with catalog frequency >= {MIN_BRAND_FREQ} "
          f"(out of {len(brand_counts)} distinct normalized brand strings)")
    with open(f"{OUT_DIR}/brand_vocab.json", "w") as f:
        json.dump({"min_freq": MIN_BRAND_FREQ, "n_brands": len(brand_vocab), "brands": brand_vocab}, f, indent=2)

    # Precompile brand regexes once (word-boundary phrase match).
    brand_patterns = [(b, re.compile(r'\b' + re.escape(b) + r'\b')) for b in brand_vocab]

    def is_brand_heavy(query_norm):
        return any(p.search(query_norm) for _, p in brand_patterns)

    # ---- Ground-truth relevant-product titles (for circularity-free lexical slices) ----
    title_lookup = df_pr_us.set_index('product_id')['product_title'].to_dict()
    stemmer = PorterStemmer()

    def stem_set(text):
        return set(stemmer.stem(w) for w in str(text).lower().split() if w)

    rows = []
    lexical_scores = {}
    for qid in query_ids:
        gt = ground_truth[qid]
        query_text = gt["query"]
        query_norm = normalize_text(query_text)

        # A. SKU / model-like
        sku_reasons = model_sku_flag(query_norm)
        is_sku_model = len(sku_reasons) > 0

        # B. Storage / numeric-spec
        is_storage_numeric = bool(STORAGE_NUMERIC_PATTERN.search(query_norm))

        # C. Brand-heavy (catalog-wide vocabulary, not per-candidate)
        brand_heavy = is_brand_heavy(query_norm)

        # D/E. Lexical overlap vs. ground-truth relevant product titles (max over relevant_broad)
        q_stems = stem_set(query_text)
        overlap_score = None
        if q_stems and gt["relevant_broad"]:
            best = 0.0
            for pid in gt["relevant_broad"]:
                title = title_lookup.get(pid)
                if not title:
                    continue
                t_stems = stem_set(title)
                if not t_stems:
                    continue
                frac = len(q_stems & t_stems) / len(q_stems)
                if frac > best:
                    best = frac
            overlap_score = best
            lexical_scores[qid] = overlap_score

        rows.append({
            "query_id": qid,
            "query": query_text,
            "is_sku_model": is_sku_model,
            "is_storage_numeric": is_storage_numeric,
            "is_brand_heavy": brand_heavy,
            "lexical_overlap_score": overlap_score,
            "num_broad": gt["num_broad"],
        })

    df = pd.DataFrame(rows)

    # Strong-lexical / low-lexical-overlap thresholds, computed only over queries
    # with a defined overlap score (i.e. >=1 relevant_broad item with a title).
    scored = df.dropna(subset=['lexical_overlap_score'])
    p75 = scored['lexical_overlap_score'].quantile(0.75)
    p25 = scored['lexical_overlap_score'].quantile(0.25)
    print(f"lexical_overlap_score: n_scored={len(scored)}/{len(df)}, "
          f"P25={p25:.4f}, median={scored['lexical_overlap_score'].median():.4f}, P75={p75:.4f}")

    df['is_strong_lexical'] = (df['lexical_overlap_score'] >= p75).fillna(False)
    df['is_low_lexical_overlap'] = (df['lexical_overlap_score'] <= p25).fillna(False)
    # Queries with no relevant_broad item (overlap undefined) cannot be assigned to
    # either lexical slice -- explicit, not silently dropped.
    df.loc[df['lexical_overlap_score'].isna(), ['is_strong_lexical', 'is_low_lexical_overlap']] = False

    out_cols = ['query_id', 'query', 'is_sku_model', 'is_storage_numeric', 'is_brand_heavy',
                'is_strong_lexical', 'is_low_lexical_overlap']
    df[out_cols].to_csv(f"{OUT_DIR}/query_assignments.csv", index=False)

    # Keep overlap score + num_broad around for downstream scripts (not one of the
    # required deliverables, but needed to reproduce thresholds without recompute).
    df[['query_id', 'lexical_overlap_score', 'num_broad']].to_csv(f"{OUT_DIR}/_lexical_overlap_scores.csv", index=False)

    print("\n=== Slice sizes ===")
    for col in ['is_sku_model', 'is_storage_numeric', 'is_brand_heavy', 'is_strong_lexical', 'is_low_lexical_overlap']:
        print(f"  {col}: {int(df[col].sum())} / {len(df)}")
    print(f"\nSaved {OUT_DIR}/query_assignments.csv")


if __name__ == "__main__":
    main()
