"""
Day 3 (part 2): aggregates output/query_slices/query_level_metrics.csv (produced
by scripts/build_query_slices.py) into slice-level NDCG/win-rate tables,
representative examples, and sanity-check sample files. No new scoring, no
retraining -- pure aggregation of already-computed per-query NDCG@10 values.

Usage:
    python scripts/analyze_query_slices.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, PRODUCTS_PATH, ROOT_DIR

OUT_DIR = f"{ROOT_DIR}/output/query_slices"
BM25_TEST_CSV = f"{ROOT_DIR}/output/bm25_scores_test.csv"
TT_TEST_CSV = f"{ROOT_DIR}/output/two_tower_scores_test.csv"
LOW_SAMPLE_THRESHOLD = 100


def slice_row(df, name, mask, total_n):
    sub = df[mask]
    n = len(sub)
    bm25 = sub['bm25_ndcg_at_10'].mean()
    tt = sub['two_tower_ndcg_at_10'].mean()
    delta = tt - bm25
    winner = "Two-Tower" if delta > 1e-6 else ("BM25" if delta < -1e-6 else "Tie")
    bm25_win_pct = (sub['delta_tt_minus_bm25'] < -1e-9).mean() * 100 if n else np.nan
    tt_win_pct = (sub['delta_tt_minus_bm25'] > 1e-9).mean() * 100 if n else np.nan
    tie_pct = 100 - bm25_win_pct - tt_win_pct if n else np.nan

    # Paired Wilcoxon signed-rank test on per-query deltas: is the median
    # delta significantly different from 0? Non-parametric since per-query
    # NDCG deltas are not remotely normal (lots of exact ties at 0).
    nonzero = sub['delta_tt_minus_bm25'][sub['delta_tt_minus_bm25'].abs() > 1e-9]
    if len(nonzero) >= 10:
        try:
            stat, p_value = wilcoxon(nonzero)
        except ValueError:
            p_value = np.nan
    else:
        p_value = np.nan

    return {
        "slice": name,
        "num_queries": n,
        "pct_test": 100 * n / total_n,
        "bm25_ndcg_at_10": bm25,
        "tt_ndcg_at_10": tt,
        "delta_tt_minus_bm25": delta,
        "winner": winner,
        "bm25_win_pct": bm25_win_pct,
        "tt_win_pct": tt_win_pct,
        "tie_pct": tie_pct,
        "wilcoxon_p": p_value,
        "low_sample_size": n < LOW_SAMPLE_THRESHOLD,
    }


def main():
    df = pd.read_csv(f"{OUT_DIR}/query_level_metrics.csv")
    total_n = len(df)

    slices = {
        "Global": df['query'].notna(),
        "Brand": df['brand_explicit'],
        "Model/SKU": df['model_like'],
        "Exact Attribute": df['has_exact_attribute'],
        "Size": df['has_size'],
        "Color": df['has_color'],
        "Capacity/Storage": df['has_capacity'],
        "Quantity": df['has_quantity'],
        "Dimensions": df['has_dimensions'],
        "Lexical Specific (top-25% IDF)": df['lexical_specific'],
        "Semantic Intent": df['semantic_intent'],
        "Other": df['other'],
    }

    rows = [slice_row(df, name, mask, total_n) for name, mask in slices.items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/slice_summary.csv", index=False)

    md_lines = ["| Slice | #Queries | % Test | BM25 NDCG@10 | TT NDCG@10 | Δ TT-BM25 | Winner | Wilcoxon p |",
                "|---|---:|---:|---:|---:|---:|---|---:|"]
    for _, r in summary.iterrows():
        flag = " *(low n)*" if r['low_sample_size'] else ""
        p_str = f"{r['wilcoxon_p']:.2e}" if pd.notna(r['wilcoxon_p']) else "n/a"
        sig = "*" if pd.notna(r['wilcoxon_p']) and r['wilcoxon_p'] < 0.05 else ""
        md_lines.append(
            f"| {r['slice']}{flag} | {r['num_queries']} | {r['pct_test']:.1f}% | "
            f"{r['bm25_ndcg_at_10']:.4f} | {r['tt_ndcg_at_10']:.4f} | "
            f"{r['delta_tt_minus_bm25']:+.4f} | {r['winner']} | {p_str}{sig} |"
        )
    with open(f"{OUT_DIR}/slice_summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))

    win_rates = summary[["slice", "bm25_win_pct", "tt_win_pct", "tie_pct"]].copy()
    win_rates.to_csv(f"{OUT_DIR}/slice_win_rates.csv", index=False)
    print("\nWin rates:")
    print(win_rates.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # ---- Representative examples: top-ranked item per method, per query ----
    print("\nBuilding representative examples (with top-ranked product title/label)...")
    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)
    df_pr['product_id'] = df_pr['product_id'].astype(str)
    us_qids = set(df['query_id'].astype(str))
    df_ex_us = df_ex[(df_ex['split'] == 'test') & (df_ex['query_id'].isin(us_qids))]
    title_label = df_ex_us.merge(
        df_pr[['product_id', 'product_locale', 'product_title']],
        on=['product_id', 'product_locale'], how='left'
    )[['query_id', 'product_id', 'product_title', 'esci_label']].rename(columns={'product_id': 'item_id'})
    title_label['item_id'] = title_label['item_id'].astype(str)

    bm25_scores = pd.read_csv(BM25_TEST_CSV, dtype={'query_id': str, 'item_id': str})
    tt_scores = pd.read_csv(TT_TEST_CSV, dtype={'query_id': str, 'item_id': str})
    bm25_top1 = bm25_scores.sort_values('bm25_score', ascending=False).groupby('query_id').first()['item_id']
    tt_top1 = tt_scores.sort_values('two_tower_score', ascending=False).groupby('query_id').first()['item_id']

    def lookup(qid, item_id):
        row = title_label[(title_label['query_id'] == qid) & (title_label['item_id'] == item_id)]
        if row.empty:
            return "", ""
        return row.iloc[0]['product_title'], row.iloc[0]['esci_label']

    def build_examples(sub, n=10):
        recs = []
        for _, r in sub.iterrows():
            qid = str(r['query_id'])
            bm25_title, bm25_label = lookup(qid, bm25_top1.get(qid, ""))
            tt_title, tt_label = lookup(qid, tt_top1.get(qid, ""))
            recs.append({
                "query": r['query'],
                "brand_explicit": r['brand_explicit'],
                "model_like": r['model_like'],
                "has_exact_attribute": r['has_exact_attribute'],
                "attribute_types": r['attribute_types'],
                "lexical_specific": r['lexical_specific'],
                "semantic_intent": r['semantic_intent'],
                "bm25_ndcg_at_10": r['bm25_ndcg_at_10'],
                "tt_ndcg_at_10": r['two_tower_ndcg_at_10'],
                "delta_tt_minus_bm25": r['delta_tt_minus_bm25'],
                "bm25_top1_title": bm25_title,
                "bm25_top1_label": bm25_label,
                "tt_top1_title": tt_title,
                "tt_top1_label": tt_label,
            })
        return pd.DataFrame(recs).head(n)

    def top_examples_for_mask(mask, label):
        sub = df[mask].copy()
        sub['abs_delta'] = sub['delta_tt_minus_bm25'].abs()
        bm25_win = sub[sub['delta_tt_minus_bm25'] < 0].sort_values('abs_delta', ascending=False)
        tt_win = sub[sub['delta_tt_minus_bm25'] > 0].sort_values('abs_delta', ascending=False)
        return build_examples(bm25_win, 10), build_examples(tt_win, 10)

    bm25_win_all, tt_win_all = [], []
    for slice_name, mask in [("Global", df['query'].notna()), ("Brand", df['brand_explicit']),
                              ("Model/SKU", df['model_like']), ("Exact Attribute", df['has_exact_attribute']),
                              ("Semantic Intent", df['semantic_intent'])]:
        b, t = top_examples_for_mask(mask, slice_name)
        b.insert(0, "slice", slice_name)
        t.insert(0, "slice", slice_name)
        bm25_win_all.append(b)
        tt_win_all.append(t)

    pd.concat(bm25_win_all, ignore_index=True).to_csv(f"{OUT_DIR}/bm25_win_examples.csv", index=False)
    pd.concat(tt_win_all, ignore_index=True).to_csv(f"{OUT_DIR}/two_tower_win_examples.csv", index=False)
    print(f"Saved {OUT_DIR}/bm25_win_examples.csv and two_tower_win_examples.csv")

    # ---- Sample files for sanity-check review ----
    def save_samples(mask, cols, path, n_pos=20, n_neg=20, seed=7):
        pos = df[mask]
        neg = df[~mask]
        pos_s = pos.sample(min(n_pos, len(pos)), random_state=seed)[cols]
        neg_s = neg.sample(min(n_neg, len(neg)), random_state=seed)[cols]
        pos_s = pos_s.copy(); pos_s['flag'] = True
        neg_s = neg_s.copy(); neg_s['flag'] = False
        pd.concat([pos_s, neg_s], ignore_index=True).to_csv(path, index=False)

    save_samples(df['brand_explicit'], ['query', 'brand_matched'], f"{OUT_DIR}/brand_samples.csv")
    save_samples(df['model_like'], ['query', 'model_reasons'], f"{OUT_DIR}/model_like_samples.csv")
    save_samples(df['has_exact_attribute'], ['query', 'attribute_types'], f"{OUT_DIR}/exact_attribute_samples.csv", n_pos=30, n_neg=20)
    save_samples(df['semantic_intent'], ['query'], f"{OUT_DIR}/semantic_intent_samples.csv")
    print(f"Saved brand/model/exact_attribute/semantic_intent sample CSVs to {OUT_DIR}/")

    return summary, win_rates


if __name__ == "__main__":
    main()
