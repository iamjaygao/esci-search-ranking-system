"""
Full-Catalog Retrieval Label Coverage Analysis.

Does NOT rerun retrieval, rebuild indices, or retrain anything. Reuses the
existing output/full_retrieval/{bm25,two_tower}_retrieval.parquet and
eval_query_ids.json exactly as produced by scripts/run_full_bm25_retrieval.py
/ scripts/run_full_tt_retrieval.py. Hybrid RRF@100 is reconstructed using the
same rrf_fuse_query() function (k=60) imported from
scripts/evaluate_full_retrieval.py -- not reimplemented, not retuned.

Quantifies how many of the Top-100 retrieved (query, product) pairs actually
have an ESCI relevance judgment (E/S/C/I) vs. none at all (unlabeled/unknown),
to justify why full-catalog retrieval is evaluated with Recall@K rather than
a fabricated full-catalog NDCG.

Usage:
    python scripts/analyze_label_coverage.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, PRODUCTS_PATH, ROOT_DIR
from scripts.evaluate_full_retrieval import rrf_fuse_query, build_rank_lookup, RRF_K

OUT_DIR = f"{ROOT_DIR}/output/full_retrieval"
LOCALE = "us"
TOPK = 100
LABELS = ["E", "S", "C", "I"]


def main():
    print("=== Loading existing retrieval outputs (no rerun) ===")
    with open(f"{OUT_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)
    query_ids = [str(q) for q in eval_meta["query_ids"]]
    print(f"Eval query set: {len(query_ids)} queries, seed={eval_meta['seed']}, locale={eval_meta['locale']}")

    bm25_df = pd.read_parquet(f"{OUT_DIR}/bm25_retrieval.parquet")
    tt_df = pd.read_parquet(f"{OUT_DIR}/two_tower_retrieval.parquet")
    bm25_df['query_id'] = bm25_df['query_id'].astype(str)
    tt_df['query_id'] = tt_df['query_id'].astype(str)
    bm25_df['product_id'] = bm25_df['product_id'].astype(str)
    tt_df['product_id'] = tt_df['product_id'].astype(str)

    bm25_top100 = bm25_df[bm25_df['rank'] <= TOPK].copy()
    tt_top100 = tt_df[tt_df['rank'] <= TOPK].copy()

    # Reconstruct Hybrid RRF@100 using the EXACT existing function (k=60, unchanged).
    print(f"Reconstructing Hybrid RRF@{TOPK} using existing rrf_fuse_query() (k={RRF_K}, unchanged)...")
    bm25_lookup = build_rank_lookup(bm25_df)  # full top-200 lists, as RRF expects
    tt_lookup = build_rank_lookup(tt_df)
    hybrid_rows = []
    for qid in query_ids:
        fused = rrf_fuse_query(bm25_lookup.get(qid, {}), tt_lookup.get(qid, {}), rrf_k=RRF_K, top_n=TOPK)
        for rank, pid in enumerate(fused, start=1):
            hybrid_rows.append({"query_id": qid, "product_id": pid, "rank": rank})
    hybrid_top100 = pd.DataFrame(hybrid_rows)

    retrievers = {"BM25@100": bm25_top100, "Two-Tower@100": tt_top100, "Hybrid RRF@100": hybrid_top100}

    # ============ Sanity check 1, 2, 5 ============
    print("\n=== Sanity checks (pre-join) ===")
    for name, df in retrievers.items():
        n_expected = len(query_ids) * TOPK
        ok_count = len(df) == n_expected
        ok_dupes = df.duplicated(subset=['query_id', 'product_id']).sum() == 0
        print(f"  {name}: rows={len(df)} (expect {n_expected}) ok={ok_count}; no dup (qid,pid) pairs: {ok_dupes}")
        assert ok_count, f"{name} row count mismatch"
        assert ok_dupes, f"{name} has duplicate (query_id, product_id) pairs"
    qid_sets = [set(df['query_id']) for df in retrievers.values()]
    same_queries = all(s == set(query_ids) for s in qid_sets)
    print(f"  Same 5,000 query_ids across BM25/TT/Hybrid: {same_queries}")
    assert same_queries

    # ============ Ground-truth judged-pair table (test split, us-locale, E/S/C/I all count as judged) ============
    print("\n=== Building ESCI judged-pair table (test split, us-locale) ===")
    df_ex = pd.read_parquet(EXAMPLES_PATH)
    df_ex['query_id'] = df_ex['query_id'].astype(str)
    df_ex['product_id'] = df_ex['product_id'].astype(str)
    judged = df_ex[(df_ex['split'] == 'test') & (df_ex['product_locale'] == LOCALE) &
                   (df_ex['query_id'].isin(set(query_ids)))][['query_id', 'product_id', 'esci_label']].drop_duplicates()
    print(f"Judged pairs available for these {len(query_ids)} queries: {len(judged)}")
    assert judged['esci_label'].isin(LABELS).all(), "Unexpected label value outside E/S/C/I"

    # ============ Join + aggregate stats ============
    print("\n=== Joining retrieval outputs to judged-pair table ===")
    summary_rows = []
    query_level_frames = []
    for name, df in retrievers.items():
        merged = df.merge(judged, on=['query_id', 'product_id'], how='left')
        merged['judged'] = merged['esci_label'].notna()
        total = len(merged)
        judged_n = int(merged['judged'].sum())
        unlabeled_n = total - judged_n
        label_counts = merged['esci_label'].value_counts()
        e = int(label_counts.get('E', 0)); s = int(label_counts.get('S', 0))
        c = int(label_counts.get('C', 0)); i = int(label_counts.get('I', 0))
        assert e + s + c + i == judged_n, "E+S+C+I != judged count"
        assert judged_n + unlabeled_n == total

        summary_rows.append({
            "retriever": name, "total_pairs": total, "judged": judged_n,
            "judged_pct": 100 * judged_n / total, "unlabeled": unlabeled_n,
            "unlabeled_pct": 100 * unlabeled_n / total,
            "E": e, "E_pct": 100 * e / total, "S": s, "S_pct": 100 * s / total,
            "C": c, "C_pct": 100 * c / total, "I": i, "I_pct": 100 * i / total,
        })

        # Query-level coverage
        ql = merged.groupby('query_id')['judged'].agg(['sum', 'count']).rename(
            columns={'sum': 'judged_count_top100', 'count': 'total_count_top100'})
        ql['unlabeled_count_top100'] = ql['total_count_top100'] - ql['judged_count_top100']
        ql['judged_fraction_top100'] = ql['judged_count_top100'] / ql['total_count_top100']
        ql['retriever'] = name
        ql = ql.reset_index()
        query_level_frames.append(ql)

        merged['retriever'] = name
        retrievers[name] = merged  # keep enriched version for examples later

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUT_DIR}/label_coverage_summary.csv", index=False)

    md_lines = ["| Retriever | Total Pairs | Judged | Judged % | Unlabeled | Unlabeled % | E | S | C | I |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary_df.iterrows():
        md_lines.append(f"| {r['retriever']} | {r['total_pairs']} | {r['judged']} | {r['judged_pct']:.2f}% | "
                         f"{r['unlabeled']} | {r['unlabeled_pct']:.2f}% | {r['E']} | {r['S']} | {r['C']} | {r['I']} |")
    with open(f"{OUT_DIR}/label_coverage_summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))

    # ============ Sanity checks (post-join) ============
    print("\n=== Sanity checks (post-join) ===")
    for name, df in retrievers.items():
        no_i_in_unlabeled = df.loc[~df['judged'], 'esci_label'].isna().all()
        print(f"  {name}: no unlabeled item assigned a label (incl. 'I'): {no_i_in_unlabeled}")
        assert no_i_in_unlabeled

    # ============ Query-level coverage stats ============
    print("\n=== Query-level coverage ===")
    query_level_df = pd.concat(query_level_frames, ignore_index=True)
    query_level_df.to_parquet(f"{OUT_DIR}/query_level_label_coverage.parquet", index=False)

    qstats_rows = []
    for name in retrievers:
        sub = query_level_df[query_level_df['retriever'] == name]
        jc = sub['judged_count_top100']
        jf = sub['judged_fraction_top100']
        qstats_rows.append({
            "retriever": name,
            "mean_judged_count": jc.mean(), "median_judged_count": jc.median(),
            "p25_judged_count": jc.quantile(0.25), "p75_judged_count": jc.quantile(0.75),
            "min_judged_count": jc.min(), "max_judged_count": jc.max(),
            "mean_unlabeled_fraction": 1 - jf.mean(), "median_unlabeled_fraction": 1 - jf.median(),
        })
    qstats_df = pd.DataFrame(qstats_rows)
    print(qstats_df.to_string(index=False))
    qstats_df.to_csv(f"{OUT_DIR}/query_level_coverage_stats.csv", index=False)

    # ============ Unlabeled examples ============
    print("\n=== Sampling unlabeled examples ===")
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_pr['product_id'] = df_pr['product_id'].astype(str)
    df_pr_us = df_pr[df_pr['product_locale'] == LOCALE][['product_id', 'product_title']].drop_duplicates('product_id')
    query_text = judged.copy()  # not used directly; get query text from eval ground_truth or df_ex
    df_ex_qtext = df_ex[df_ex['query_id'].isin(set(query_ids))][['query_id', 'query']].drop_duplicates('query_id')

    all_unlabeled = pd.concat([df[~df['judged']][['query_id', 'product_id', 'retriever', 'rank']]
                                for df in retrievers.values()], ignore_index=True)
    sample = all_unlabeled.sample(min(50, len(all_unlabeled)), random_state=42)
    sample = sample.merge(df_ex_qtext, on='query_id', how='left').merge(df_pr_us, on='product_id', how='left')
    sample = sample[['query', 'product_id', 'product_title', 'retriever', 'rank']]
    sample.to_csv(f"{OUT_DIR}/unlabeled_examples.csv", index=False)
    print(f"Saved {OUT_DIR}/unlabeled_examples.csv ({len(sample)} rows, from {len(all_unlabeled)} total unlabeled pairs)")

    # ============ Final answers ============
    print("\n=== Final answers ===")
    for _, r in summary_df.iterrows():
        print(f"  {r['retriever']}: {r['unlabeled']} / {r['total_pairs']} unlabeled ({r['unlabeled_pct']:.2f}%)")
    best = summary_df.loc[summary_df['judged_pct'].idxmax(), 'retriever']
    print(f"  Highest judged coverage: {best}")
    majority_unlabeled = (summary_df['unlabeled_pct'] > 50).all()
    print(f"  Unlabeled is majority for all retrievers: {majority_unlabeled}")

    return summary_df, qstats_df


if __name__ == "__main__":
    main()
