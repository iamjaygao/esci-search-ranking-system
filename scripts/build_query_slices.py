"""
Day 3: Exact Lexical/Attribute vs Semantic query-slice evaluation.

No retraining. No new retrieval. This reuses the exact same fixed-candidate
test-set files (output/bm25_scores_test.csv, output/two_tower_scores_test.csv)
and the exact same NDCG implementation (evaluation/metrics.dcg) that produced
the global BM25 (0.8188) and Two-Tower (0.8267) numbers, and slices them by
query type to test whether BM25 wins on exact lexical/attribute intent while
Two-Tower wins on semantic intent.

Scope: restricted to product_locale == 'us' test queries. All slice rules
(brand matching, model/SKU regex, attribute regex, color vocabulary, the IDF
percentile threshold) are fixed in this file BEFORE any NDCG is computed --
the only thing inspected before computing metrics is whether the rules fire
on sensible examples (sanity-check samples), never whether BM25 or Two-Tower
"wins" under them.

Usage:
    python scripts/build_query_slices.py
"""
import os
import sys
import re
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, PRODUCTS_PATH, ROOT_DIR
from evaluation.metrics import dcg

OUT_DIR = f"{ROOT_DIR}/output/query_slices"
BM25_TEST_CSV = f"{ROOT_DIR}/output/bm25_scores_test.csv"
TT_TEST_CSV = f"{ROOT_DIR}/output/two_tower_scores_test.csv"
IDF_STATS_PATH = f"{ROOT_DIR}/output/advanced_normalization_stats.json"

STANDARD_LABEL_MAP = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
LOCALE = "us"

# ============================================================
# 0. Pre-registered slice rules (fixed BEFORE any NDCG is computed)
# ============================================================

# Reused verbatim from reranking/advanced_features.py's color_match feature.
COLOR_WORDS = {'red', 'black', 'blue', 'white', 'green', 'yellow', 'silver',
               'gold', 'grey', 'gray', 'pink', 'purple', 'brown'}

# The product_brand field contains placeholder/non-brand values (data-quality
# artifact of the underlying catalog, not something we can fix upstream).
# Excluding these before brand phrase-matching -- discovered during the
# sanity-check pass on brand_explicit samples (e.g. "claypool lennon delirium
# vinyl" was matching brand="vinyl", a format descriptor, not a brand). This
# is a correctness fix applied before any slice NDCG is computed, identical
# regardless of which retriever it would favor.
NON_BRAND_PLACEHOLDERS = {
    "generic", "unbranded", "unknown", "other", "various", "none",
    "na", "n a", "brand", "no brand", "noname", "no name",
    "vinyl", "metal", "glass", "silicone",
}

# Known consumer-electronics / product-line prefixes used only in the
# bigram rule (prefix immediately followed by a digit-bearing token), e.g.
# "iphone 15", "galaxy s24", "rtx 4090". Fixed before running any analysis.
MODEL_LINE_PREFIXES = {
    "iphone", "ipad", "imac", "macbook", "airpods", "watch", "chromebook",
    "galaxy", "pixel", "surface", "playstation", "ps", "xbox", "switch",
    "rtx", "gtx", "ryzen", "radeon", "core", "thinkpad", "ideapad",
    "vivobook", "zenbook", "inspiron", "xps", "legion", "gopro", "kindle",
    "echo", "fire",
}

FUSED_ALNUM_TOKEN = re.compile(r'^[a-z]{1,6}[0-9]{1,4}[a-z0-9]*$')
# Requires a letter AND a digit somewhere in the token -- a real SKU/model
# code is always alphanumeric (wh-1000xm5, rtx-4090); a bare "4-8" is an age
# range, not a product code. Tightened during the manual sanity-check pass
# (before computing any slice NDCG) after finding "crafts for kids ages 4-8"
# incorrectly flagged as model/SKU-like.
HYPHENATED_ALNUM_TOKEN = re.compile(r'^(?=.*[a-z])(?=.*[0-9])[a-z0-9]+-[a-z0-9]+$')
DIGIT_BEARING_TOKEN = re.compile(r'^[a-z0-9]{1,6}$')
HAS_DIGIT = re.compile(r'[0-9]')

SIZE_NUMERIC = re.compile(r'\bsize\s*\d+\b')
SIZE_LETTER = re.compile(r'\b(xxs|xs|xxl|xxxl)\b')
VAGUE_SIZE_WORD = re.compile(r'\b(small|medium|large|s|m|l)\b')  # tracked, NOT used in has_exact_attribute

CAPACITY_LONGUNIT = re.compile(r'\b\d+(\.\d+)?\s?(gb|tb|mb|kg|lb|lbs|oz)\b')
CAPACITY_SHORTUNIT = re.compile(r'\b\d+(\.\d+)?(ml|l)\b')  # no space allowed: short/ambiguous unit letters

QUANTITY_PATTERN = re.compile(r'\b(\d+)\s*-?\s*pack\b|\bpack of \d+\b|\bset of \d+\b|\b\d+\s?(pcs|piece|pieces|count|ct)\b')

DIMENSION_PATTERN = re.compile(r'\b\d+(\.\d+)?\s?(inch|in|mm|cm|ft|foot|feet)\b|\b\d+(\.\d+)?"')

IDF_PERCENTILE_FOR_LEXICAL_SPECIFIC = 75  # top quartile, fixed before running


def normalize_text(s):
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def tokenize(s):
    return s.split() if s else []


def brand_explicit_flag(query_norm, candidate_brands_norm):
    """Phrase-aware, word-boundary brand match against this query's own
    candidate pool brands. Rejects brands shorter than 2 chars and known
    non-brand placeholder values."""
    matched = []
    for b in candidate_brands_norm:
        if not b or len(b) < 2 or b in NON_BRAND_PLACEHOLDERS:
            continue
        pattern = r'\b' + re.escape(b) + r'\b'
        if re.search(pattern, query_norm):
            matched.append(b)
    return matched


def model_sku_flag(query_norm):
    tokens = tokenize(query_norm)
    reasons = []
    for tok in tokens:
        if '-' in tok and HYPHENATED_ALNUM_TOKEN.match(tok):
            reasons.append(f"hyphenated:{tok}")
        elif FUSED_ALNUM_TOKEN.match(tok):
            reasons.append(f"fused:{tok}")
    for i in range(len(tokens) - 1):
        if tokens[i] in MODEL_LINE_PREFIXES:
            nxt = tokens[i + 1]
            if DIGIT_BEARING_TOKEN.match(nxt) and HAS_DIGIT.search(nxt):
                reasons.append(f"bigram:{tokens[i]}_{nxt}")
    return reasons


def exact_attribute_flags(query_norm):
    types = []
    tokens = tokenize(query_norm)
    has_model_prefix = any(t in MODEL_LINE_PREFIXES for t in tokens)
    size_hit = bool(SIZE_NUMERIC.search(query_norm))
    letter_match = SIZE_LETTER.search(query_norm)
    if letter_match:
        # "xs" collides with the iPhone XS model name (found during the
        # manual sanity-check pass: 26/35 standalone "xs" queries were about
        # iPhone XS, not clothing size). Suppress only that specific token,
        # only when a model-line prefix is present elsewhere in the query --
        # xl/xxl/xxs/xxxl have no such collision and are left untouched.
        if letter_match.group(1) == 'xs' and has_model_prefix:
            pass
        else:
            size_hit = True
    if size_hit:
        types.append("size")
    if CAPACITY_LONGUNIT.search(query_norm) or CAPACITY_SHORTUNIT.search(query_norm):
        types.append("capacity")
    if QUANTITY_PATTERN.search(query_norm):
        types.append("quantity")
    if DIMENSION_PATTERN.search(query_norm):
        types.append("dimensions")
    tokens = set(tokenize(query_norm))
    if tokens & COLOR_WORDS:
        types.append("color")
    return types


# ============================================================
# 1. Per-query NDCG using the exact same dcg() as evaluation/metrics.py
# ============================================================

def per_query_ndcg(df, score_col, k=10):
    out = {}
    for qid, group in df.groupby('query_id'):
        sorted_group = group.sort_values(by=score_col, ascending=False)
        rel = sorted_group['relevance'].values
        ideal_rel = sorted(rel, reverse=True)
        dcg_val = dcg(rel, k)
        idcg_val = dcg(ideal_rel, k)
        if idcg_val > 0:
            out[qid] = dcg_val / idcg_val
    return pd.Series(out, name=score_col + "_ndcg")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Phase 1: Load test truth + retrieval score files (no retraining) ===")
    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)
    df_pr['product_id'] = df_pr['product_id'].astype(str)

    df_test_all = df_ex[df_ex['split'] == 'test'].copy()
    df_test_all['relevance'] = df_test_all['esci_label'].map(STANDARD_LABEL_MAP).fillna(0.0)
    df_truth_all = df_test_all[['query_id', 'product_id', 'esci_label', 'relevance']].rename(
        columns={'product_id': 'item_id'}
    ).drop_duplicates()

    df_bm25_all = pd.read_csv(BM25_TEST_CSV, dtype={'query_id': str, 'item_id': str})
    df_tt_all = pd.read_csv(TT_TEST_CSV, dtype={'query_id': str, 'item_id': str})

    print("\n=== Phase 2a: Reproduce ALL-LOCALE global baseline (must match 0.8188 / 0.8267) ===")
    df_bm25_merged_all = pd.merge(df_bm25_all, df_truth_all, on=['query_id', 'item_id'], how='left')
    df_bm25_merged_all['relevance'] = df_bm25_merged_all['relevance'].fillna(0.0)
    df_tt_merged_all = pd.merge(df_tt_all, df_truth_all, on=['query_id', 'item_id'], how='left')
    df_tt_merged_all['relevance'] = df_tt_merged_all['relevance'].fillna(0.0)

    global_bm25_all = per_query_ndcg(df_bm25_merged_all, 'bm25_score', k=10).mean()
    global_tt_all = per_query_ndcg(df_tt_merged_all, 'two_tower_score', k=10).mean()
    print(f"All-locale BM25 NDCG@10       = {global_bm25_all:.4f}  (reference: 0.8188)")
    print(f"All-locale Two-Tower NDCG@10  = {global_tt_all:.4f}  (reference: 0.8267)")

    bm25_ok = abs(global_bm25_all - 0.8188) < 0.001
    tt_ok = abs(global_tt_all - 0.8267) < 0.001
    if not (bm25_ok and tt_ok):
        raise AssertionError(
            f"Baseline reproduction FAILED (bm25_ok={bm25_ok}, tt_ok={tt_ok}). "
            "Stopping before any slice analysis, per instructions -- go debug evaluation mismatch first."
        )
    print("Baseline reproduction OK on the full (all-locale) test set. Proceeding to slice analysis.")

    print(f"\n=== Phase 2b: Restrict to product_locale == '{LOCALE}' for the slice analysis ===")
    print("Reason (fixed before any slicing/metrics below): brand phrase-matching, the model/SKU regex, "
          "the attribute regex (size/color/capacity/quantity/dimensions), and the color vocabulary are all "
          "English-token rules. Applying them to jp/es query text would silently fail to fire (undercounting "
          "every slice), diluting slice purity rather than testing the hypothesis. This is a scoping decision, "
          "not a result -- it is applied identically regardless of which retriever it would favor.")
    df_test = df_test_all[df_test_all['product_locale'] == LOCALE].copy()
    n_us = df_test['query_id'].nunique()
    n_all = df_test_all['query_id'].nunique()
    print(f"us-locale test queries: {n_us} / {n_all} total ({n_us/n_all:.1%})")

    df_truth = df_truth_all[df_truth_all['query_id'].isin(set(df_test['query_id']))]
    us_query_ids = set(df_test['query_id'].unique())
    df_bm25_merged = df_bm25_merged_all[df_bm25_merged_all['query_id'].isin(us_query_ids)].copy()
    df_tt_merged = df_tt_merged_all[df_tt_merged_all['query_id'].isin(us_query_ids)].copy()
    unjudged_bm25 = int(pd.merge(df_bm25_all[df_bm25_all['query_id'].isin(us_query_ids)], df_truth, on=['query_id', 'item_id'], how='left')['relevance'].isna().sum())
    unjudged_tt = int(pd.merge(df_tt_all[df_tt_all['query_id'].isin(us_query_ids)], df_truth, on=['query_id', 'item_id'], how='left')['relevance'].isna().sum())

    print(f"Unjudged (retrieved-but-unlabeled) rows within us-locale subset: bm25={unjudged_bm25}, tt={unjudged_tt} "
          f"(expected ~0, since this candidate pool IS the ESCI-labeled set)")

    bm25_ndcg_per_q = per_query_ndcg(df_bm25_merged, 'bm25_score', k=10)
    tt_ndcg_per_q = per_query_ndcg(df_tt_merged, 'two_tower_score', k=10)
    global_bm25 = bm25_ndcg_per_q.mean()
    global_tt = tt_ndcg_per_q.mean()
    print(f"\nus-locale-only BM25 NDCG@10      = {global_bm25:.4f}  (this is the in-scope baseline for all slice deltas below)")
    print(f"us-locale-only Two-Tower NDCG@10  = {global_tt:.4f}")

    print("\n=== Phase 3: Build query metadata ===")
    grp = df_test.groupby('query_id')
    query_text = grp['query'].first()
    label_counts = df_test.groupby(['query_id', 'esci_label']).size().unstack(fill_value=0)
    for col in ['E', 'S', 'C', 'I']:
        if col not in label_counts.columns:
            label_counts[col] = 0
    num_candidates = grp.size()

    meta = pd.DataFrame({
        'query': query_text,
        'product_locale': LOCALE,
        'num_candidates': num_candidates,
        'num_E': label_counts['E'],
        'num_S': label_counts['S'],
        'num_C': label_counts['C'],
        'num_I': label_counts['I'],
    })
    meta.index.name = 'query_id'

    print("=== Phase 4: Compute slice flags (rules fixed before this point) ===")
    # Candidate brands per query, for phrase-aware brand matching.
    df_test_brand = df_test.merge(
        df_pr[['product_id', 'product_locale', 'product_brand']],
        on=['product_id', 'product_locale'], how='left'
    )
    brands_per_query = df_test_brand.groupby('query_id')['product_brand'].apply(
        lambda s: sorted(set(normalize_text(b) for b in s.dropna().unique()))
    )

    with open(IDF_STATS_PATH) as f:
        idf_map = json.load(f).get("idf_map", {})

    def query_mean_idf(q):
        words = str(q).lower().split()
        if not words:
            return 0.0
        return float(np.mean([idf_map.get(w, 10.0) for w in words]))

    rows = []
    for qid, qtext in query_text.items():
        qnorm = normalize_text(qtext)
        cand_brands = brands_per_query.get(qid, [])
        brand_matches = brand_explicit_flag(qnorm, cand_brands)
        model_reasons = model_sku_flag(qnorm)
        attr_types = exact_attribute_flags(qnorm)
        mean_idf = query_mean_idf(qtext)
        rows.append({
            'query_id': qid,
            'query_norm': qnorm,
            'query_length': len(qnorm.split()),
            'brand_explicit': len(brand_matches) > 0,
            'brand_matched': ";".join(brand_matches),
            'model_like': len(model_reasons) > 0,
            'model_reasons': ";".join(model_reasons),
            'attribute_types': ";".join(attr_types),
            'has_exact_attribute': len(attr_types) > 0,
            'has_size': 'size' in attr_types,
            'has_color': 'color' in attr_types,
            'has_capacity': 'capacity' in attr_types,
            'has_quantity': 'quantity' in attr_types,
            'has_dimensions': 'dimensions' in attr_types,
            'query_mean_idf': mean_idf,
        })
    flags = pd.DataFrame(rows).set_index('query_id')

    idf_threshold = np.percentile(flags['query_mean_idf'].values, IDF_PERCENTILE_FOR_LEXICAL_SPECIFIC)
    flags['lexical_specific'] = flags['query_mean_idf'] >= idf_threshold
    flags['semantic_intent'] = (
        (flags['query_length'] >= 4)
        & (~flags['brand_explicit'])
        & (~flags['model_like'])
        & (~flags['has_exact_attribute'])
        & (~flags['lexical_specific'])
    )
    flags['other'] = (
        (~flags['brand_explicit'])
        & (~flags['model_like'])
        & (~flags['has_exact_attribute'])
        & (~flags['lexical_specific'])
        & (~flags['semantic_intent'])
    )

    print(f"IDF top-{100-IDF_PERCENTILE_FOR_LEXICAL_SPECIFIC}% threshold (query_mean_idf >= {idf_threshold:.4f}), "
          f"fixed from the observed distribution, not tuned on outcomes.")

    query_level = meta.join(flags)
    query_level = query_level.join(bm25_ndcg_per_q).join(tt_ndcg_per_q)
    query_level = query_level.rename(columns={
        'bm25_score_ndcg': 'bm25_ndcg_at_10',
        'two_tower_score_ndcg': 'two_tower_ndcg_at_10',
    })
    query_level = query_level.dropna(subset=['bm25_ndcg_at_10', 'two_tower_ndcg_at_10'])
    query_level['delta_tt_minus_bm25'] = query_level['two_tower_ndcg_at_10'] - query_level['bm25_ndcg_at_10']

    query_level.reset_index().to_csv(f"{OUT_DIR}/query_level_metrics.csv", index=False)
    print(f"Saved {OUT_DIR}/query_level_metrics.csv ({len(query_level)} queries)")

    definitions = {
        "locale_scope": LOCALE,
        "locale_scope_reason": (
            "Brand phrase-matching, the model/SKU regex, the attribute regex, and the color "
            "vocabulary are all English-token rules; restricting to us-locale avoids diluting "
            "slice purity on jp/es text. Fixed before any slice NDCG was computed."
        ),
        "color_words": sorted(COLOR_WORDS),
        "non_brand_placeholders": sorted(NON_BRAND_PLACEHOLDERS),
        "model_line_prefixes": sorted(MODEL_LINE_PREFIXES),
        "regex": {
            "fused_alnum_token": FUSED_ALNUM_TOKEN.pattern,
            "hyphenated_alnum_token": HYPHENATED_ALNUM_TOKEN.pattern,
            "size_numeric": SIZE_NUMERIC.pattern,
            "size_letter": SIZE_LETTER.pattern,
            "vague_size_word_tracked_not_used": VAGUE_SIZE_WORD.pattern,
            "capacity_longunit": CAPACITY_LONGUNIT.pattern,
            "capacity_shortunit": CAPACITY_SHORTUNIT.pattern,
            "quantity": QUANTITY_PATTERN.pattern,
            "dimensions": DIMENSION_PATTERN.pattern,
        },
        "idf_percentile_for_lexical_specific": IDF_PERCENTILE_FOR_LEXICAL_SPECIFIC,
        "idf_threshold_value": float(idf_threshold),
        "semantic_intent_definition": "query_length>=4 AND NOT brand_explicit AND NOT model_like AND NOT has_exact_attribute AND NOT lexical_specific",
        "other_definition": "NOT brand_explicit AND NOT model_like AND NOT has_exact_attribute AND NOT lexical_specific AND NOT semantic_intent",
        "known_limitations": [
            "fused_alnum_token also fires on non-model alpha+digit tokens with the same shape as real "
            "model codes (e.g. 'ww2', 'd3' for vitamin D3) -- an accepted precision/recall tradeoff, "
            "since narrowing it would also exclude the S24/M2/PS5-style codes it's meant to catch.",
            "dimension/size regexes require an explicit unit or 'size' keyword, so bare specs like "
            "'10.2 ipad' (meaning 10.2-inch) are missed (false negative, not false positive).",
            "brand matching is restricted to brands appearing among that query's own candidate pool; "
            "a query naming a brand with zero matching candidates in the pool cannot be detected this way.",
        ],
        "all_locale_baseline_reproduction": {"bm25_ndcg_at_10": float(global_bm25_all), "tt_ndcg_at_10": float(global_tt_all)},
        "us_locale_only_baseline": {"bm25_ndcg_at_10": float(global_bm25), "tt_ndcg_at_10": float(global_tt)},
    }
    with open(f"{OUT_DIR}/slice_definitions.json", "w") as f:
        json.dump(definitions, f, indent=2)
    print(f"Saved {OUT_DIR}/slice_definitions.json")

    return query_level, global_bm25, global_tt, idf_threshold, unjudged_bm25, unjudged_tt


if __name__ == "__main__":
    main()
