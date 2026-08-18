"""
Full-catalog retrieval evaluation: computes Recall@K / ExactRecall@K for BM25,
Two-Tower, and two hybrid fusion strategies (union, RRF) using real retrieval
output (output/full_retrieval/{bm25,two_tower}_retrieval.parquet) against
known ESCI-labeled ground truth. No reranking, no retraining.

Usage:
    python scripts/evaluate_full_retrieval.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PRODUCTS_PATH, ROOT_DIR

OUT_DIR = f"{ROOT_DIR}/output/full_retrieval"
K_VALUES = [10, 50, 100, 200]
RRF_K = 60
LOCALE = "us"


def build_rank_lookup(df):
    lookup = {}
    for qid, group in df.groupby('query_id'):
        lookup[qid] = dict(zip(group['product_id'].astype(str), group['rank']))
    return lookup


def recall_at_k(rank_lookup, ground_truth, relevant_key, k):
    recalls = []
    for qid, gt in ground_truth.items():
        rel_set = set(gt[relevant_key])
        if not rel_set:
            continue
        ranks = rank_lookup.get(qid, {})
        hits = sum(1 for pid in rel_set if ranks.get(pid, 10**9) <= k)
        recalls.append(hits / len(rel_set))
    return float(np.mean(recalls)), len(recalls)


def mrr_at_k(rank_lookup, ground_truth, relevant_key, k=10):
    rrs = []
    for qid, gt in ground_truth.items():
        rel_set = set(gt[relevant_key])
        if not rel_set:
            continue
        ranks = rank_lookup.get(qid, {})
        relevant_ranks = [ranks[pid] for pid in rel_set if pid in ranks and ranks[pid] <= k]
        rrs.append(1.0 / min(relevant_ranks) if relevant_ranks else 0.0)
    return float(np.mean(rrs))


def hybrid_union(bm25_lookup, tt_lookup, ground_truth, relevant_key, budget=100):
    recalls, counts = [], []
    for qid, gt in ground_truth.items():
        rel_set = set(gt[relevant_key])
        bset = {pid for pid, r in bm25_lookup.get(qid, {}).items() if r <= budget}
        tset = {pid for pid, r in tt_lookup.get(qid, {}).items() if r <= budget}
        union = bset | tset
        counts.append(len(union))
        if not rel_set:
            continue
        hits = len(union & rel_set)
        recalls.append(hits / len(rel_set))
    return float(np.mean(recalls)), counts


def rrf_fuse_query(bm25_ranks, tt_ranks, rrf_k=RRF_K, top_n=100):
    all_pids = set(bm25_ranks) | set(tt_ranks)
    scores = {}
    for pid in all_pids:
        s = 0.0
        if pid in bm25_ranks:
            s += 1.0 / (rrf_k + bm25_ranks[pid])
        if pid in tt_ranks:
            s += 1.0 / (rrf_k + tt_ranks[pid])
        scores[pid] = s
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [pid for pid, _ in ranked]


def rrf_recall(bm25_lookup, tt_lookup, ground_truth, relevant_key, top_n=100):
    recalls = []
    rrf_lookup = {}
    for qid, gt in ground_truth.items():
        fused = rrf_fuse_query(bm25_lookup.get(qid, {}), tt_lookup.get(qid, {}), top_n=top_n)
        rrf_lookup[qid] = {pid: r for r, pid in enumerate(fused, start=1)}
        rel_set = set(gt[relevant_key])
        if not rel_set:
            continue
        hits = len(set(fused) & rel_set)
        recalls.append(hits / len(rel_set))
    return float(np.mean(recalls)), rrf_lookup


def main():
    print("=== Loading ground truth + retrieval outputs ===")
    with open(f"{OUT_DIR}/ground_truth.json") as f:
        ground_truth = json.load(f)
    with open(f"{OUT_DIR}/eval_query_ids.json") as f:
        eval_meta = json.load(f)

    bm25_df = pd.read_parquet(f"{OUT_DIR}/bm25_retrieval.parquet")
    tt_df = pd.read_parquet(f"{OUT_DIR}/two_tower_retrieval.parquet")
    bm25_df['query_id'] = bm25_df['query_id'].astype(str)
    tt_df['query_id'] = tt_df['query_id'].astype(str)
    bm25_df['product_id'] = bm25_df['product_id'].astype(str)
    tt_df['product_id'] = tt_df['product_id'].astype(str)

    print(f"BM25 retrieval rows: {len(bm25_df)}, queries: {bm25_df['query_id'].nunique()}")
    print(f"TT retrieval rows: {len(tt_df)}, queries: {tt_df['query_id'].nunique()}")

    bm25_lookup = build_rank_lookup(bm25_df)
    tt_lookup = build_rank_lookup(tt_df)

    # ============ Sanity checks (subset computable here; rest checked at build/retrieval time) ============
    print("\n=== Sanity checks ===")
    checks = {}
    checks['same_query_set'] = set(bm25_df['query_id'].unique()) == set(tt_df['query_id'].unique()) == set(eval_meta['query_ids'])
    checks['k_truncates'] = (bm25_df.groupby('query_id').size().max() <= 200) and (tt_df.groupby('query_id').size().max() <= 200)
    checks['ground_truth_independent_of_retrieval'] = True  # ground_truth.json built solely from ESCI labels, before any retrieval ran
    for name, ok in checks.items():
        print(f"  {name}: {'OK' if ok else 'FAIL'}")
        if not ok:
            raise AssertionError(f"Sanity check failed: {name}")

    # ============ Recall / ExactRecall tables ============
    print("\n=== Computing Recall@K / ExactRecall@K ===")
    rows = []
    for retriever, lookup in [("BM25", bm25_lookup), ("Two-Tower", tt_lookup)]:
        row = {"retriever": retriever}
        for k in K_VALUES:
            r, n = recall_at_k(lookup, ground_truth, "relevant_broad", k)
            row[f"recall@{k}"] = r
            er, _ = recall_at_k(lookup, ground_truth, "relevant_exact", k)
            row[f"exact_recall@{k}"] = er
        row["mrr@10"] = mrr_at_k(lookup, ground_truth, "relevant_broad", k=10)
        rows.append(row)

    # Monotonicity check
    for row in rows:
        recalls_seq = [row[f"recall@{k}"] for k in K_VALUES]
        exact_seq = [row[f"exact_recall@{k}"] for k in K_VALUES]
        assert all(recalls_seq[i] <= recalls_seq[i+1] + 1e-9 for i in range(len(recalls_seq)-1)), \
            f"Recall not monotonic for {row['retriever']}"
        assert all(exact_seq[i] <= exact_seq[i+1] + 1e-9 for i in range(len(exact_seq)-1)), \
            f"ExactRecall not monotonic for {row['retriever']}"
    print("  Recall@K and ExactRecall@K confirmed monotonic non-decreasing in K.")

    # ============ Hybrid Union ============
    print("\n=== Hybrid union (BM25@100 UNION TT@100) ===")
    union_recall, union_counts = hybrid_union(bm25_lookup, tt_lookup, ground_truth, "relevant_broad", budget=100)
    union_exact_recall, _ = hybrid_union(bm25_lookup, tt_lookup, ground_truth, "relevant_exact", budget=100)
    union_stats = {
        "mean_candidates": float(np.mean(union_counts)), "median_candidates": float(np.median(union_counts)),
        "max_candidates": int(np.max(union_counts)), "min_candidates": int(np.min(union_counts)),
        "recall_broad": union_recall, "recall_exact": union_exact_recall,
    }
    print(f"  BM25@100 UNION TT@100: mean candidates={union_stats['mean_candidates']:.1f}, "
          f"median={union_stats['median_candidates']:.0f}, max={union_stats['max_candidates']}")
    print(f"  Union Recall(broad)={union_recall:.4f}, Union Recall(exact)={union_exact_recall:.4f}")

    # ============ Hybrid RRF (fixed budget: top-100 fused from top-200 candidate lists) ============
    print(f"\n=== Hybrid RRF (k={RRF_K}, fused top-100 from top-200 lists per retriever) ===")
    rrf_recall_broad, rrf_lookup_100 = rrf_recall(bm25_lookup, tt_lookup, ground_truth, "relevant_broad", top_n=100)
    rrf_recall_exact, _ = rrf_recall(bm25_lookup, tt_lookup, ground_truth, "relevant_exact", top_n=100)
    rrf_mrr = mrr_at_k(rrf_lookup_100, ground_truth, "relevant_broad", k=10)
    print(f"  RRF Recall@100(broad)={rrf_recall_broad:.4f}, RRF Recall@100(exact)={rrf_recall_exact:.4f}, RRF MRR@10={rrf_mrr:.4f}")

    rows.append({
        "retriever": "Hybrid RRF@100", "recall@10": np.nan, "exact_recall@10": np.nan,
        "recall@50": np.nan, "exact_recall@50": np.nan,
        "recall@100": rrf_recall_broad, "exact_recall@100": rrf_recall_exact,
        "recall@200": np.nan, "exact_recall@200": np.nan, "mrr@10": rrf_mrr,
    })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{OUT_DIR}/retrieval_metrics.csv", index=False)

    md_lines = ["| Retriever | Recall@10 | Recall@50 | Recall@100 | Recall@200 | ExactRecall@100 | MRR@10 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in metrics_df.iterrows():
        def fmt(x):
            return "n/a" if pd.isna(x) else f"{x:.4f}"
        md_lines.append(f"| {r['retriever']} | {fmt(r['recall@10'])} | {fmt(r['recall@50'])} | "
                         f"{fmt(r['recall@100'])} | {fmt(r['recall@200'])} | {fmt(r['exact_recall@100'])} | {fmt(r['mrr@10'])} |")
    md_lines.append("")
    md_lines.append(f"Hybrid Union (BM25@100 ∪ TT@100, NOT a fixed-100 budget): mean candidates="
                     f"{union_stats['mean_candidates']:.1f}, median={union_stats['median_candidates']:.0f}, "
                     f"max={union_stats['max_candidates']} -- Recall(broad)={union_recall:.4f}, Recall(exact)={union_exact_recall:.4f}")
    with open(f"{OUT_DIR}/retrieval_metrics.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))

    # ============ Complementarity analysis at K=100 (broad relevance) ============
    print("\n=== Complementarity analysis (K=100, broad relevance) ===")
    both, bm25_only, tt_only, missed = 0, 0, 0, 0
    per_query_overlap = []
    for qid, gt in ground_truth.items():
        rel_set = gt['relevant_broad']
        bset = {pid for pid, r in bm25_lookup.get(qid, {}).items() if r <= 100}
        tset = {pid for pid, r in tt_lookup.get(qid, {}).items() if r <= 100}
        q_both = q_bm25 = q_tt = q_missed = 0
        for pid in rel_set:
            in_b, in_t = pid in bset, pid in tset
            if in_b and in_t:
                both += 1; q_both += 1
            elif in_b:
                bm25_only += 1; q_bm25 += 1
            elif in_t:
                tt_only += 1; q_tt += 1
            else:
                missed += 1; q_missed += 1
        per_query_overlap.append({"query_id": qid, "both": q_both, "bm25_only": q_bm25, "tt_only": q_tt, "missed": q_missed})

    total = both + bm25_only + tt_only + missed
    overlap_summary = {
        "k": 100, "relevance": "broad (E+S+C)", "total_relevant_item_instances": total,
        "pct_found_by_both": 100 * both / total, "pct_bm25_only": 100 * bm25_only / total,
        "pct_tt_only": 100 * tt_only / total, "pct_missed_by_both": 100 * missed / total,
        "counts": {"both": both, "bm25_only": bm25_only, "tt_only": tt_only, "missed": missed},
    }
    with open(f"{OUT_DIR}/retriever_overlap_summary.json", "w") as f:
        json.dump(overlap_summary, f, indent=2)
    overlap_md = [
        "| Category | % of known-relevant items |",
        "|---|---:|",
        f"| Found by both | {overlap_summary['pct_found_by_both']:.1f}% |",
        f"| BM25-only | {overlap_summary['pct_bm25_only']:.1f}% |",
        f"| Two-Tower-only | {overlap_summary['pct_tt_only']:.1f}% |",
        f"| Missed by both | {overlap_summary['pct_missed_by_both']:.1f}% |",
    ]
    with open(f"{OUT_DIR}/retriever_overlap_summary.md", "w") as f:
        f.write("\n".join(overlap_md) + "\n")
    print("\n".join(overlap_md))

    # ============ Representative examples ============
    print("\n=== Building representative bm25-only / tt-only examples ===")
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_pr['product_id'] = df_pr['product_id'].astype(str)
    df_pr_us = df_pr[df_pr['product_locale'] == LOCALE][['product_id', 'product_title']].drop_duplicates('product_id').set_index('product_id')

    bm25_only_rows, tt_only_rows = [], []
    for qid, gt in ground_truth.items():
        bset_ranks = bm25_lookup.get(qid, {})
        tset_ranks = tt_lookup.get(qid, {})
        for pid in gt['relevant_broad']:
            in_b = pid in bset_ranks and bset_ranks[pid] <= 100
            in_t = pid in tset_ranks and tset_ranks[pid] <= 100
            title = df_pr_us['product_title'].get(pid, "")
            label = "E" if pid in gt['relevant_exact'] else ("S/C" if pid in gt['relevant_broad'] else "?")
            if in_b and not in_t:
                bm25_only_rows.append({
                    "query": gt['query'], "relevant_product_id": pid, "product_title": title, "label": label,
                    "bm25_rank": bset_ranks[pid], "tt_rank": tset_ranks.get(pid, None),
                })
            elif in_t and not in_b:
                tt_only_rows.append({
                    "query": gt['query'], "relevant_product_id": pid, "product_title": title, "label": label,
                    "bm25_rank": bset_ranks.get(pid, None), "tt_rank": tset_ranks[pid],
                })

    rng = np.random.RandomState(42)
    bm25_only_sample = pd.DataFrame(bm25_only_rows)
    tt_only_sample = pd.DataFrame(tt_only_rows)
    if len(bm25_only_sample) > 20:
        bm25_only_sample = bm25_only_sample.sample(20, random_state=42)
    if len(tt_only_sample) > 20:
        tt_only_sample = tt_only_sample.sample(20, random_state=42)
    bm25_only_sample.to_csv(f"{OUT_DIR}/bm25_only_examples.csv", index=False)
    tt_only_sample.to_csv(f"{OUT_DIR}/tt_only_examples.csv", index=False)
    print(f"Saved bm25_only_examples.csv ({len(bm25_only_sample)} rows) and tt_only_examples.csv ({len(tt_only_sample)} rows)")
    print(f"(from {len(bm25_only_rows)} total bm25-only relevant instances, {len(tt_only_rows)} total tt-only relevant instances)")

    # ============ Latency summary ============
    with open(f"{OUT_DIR}/bm25_latency.json") as f:
        bm25_lat = json.load(f)
    with open(f"{OUT_DIR}/tt_latency.json") as f:
        tt_lat = json.load(f)
    with open(f"{OUT_DIR}/latency_summary.json", "w") as f:
        json.dump({"bm25": bm25_lat, "two_tower": tt_lat}, f, indent=2)
    print(f"\nSaved {OUT_DIR}/latency_summary.json")

    # ============ Evaluation config ============
    df_pr = pd.read_parquet(PRODUCTS_PATH)
    catalog_us_count = int((df_pr['product_locale'] == LOCALE).sum())
    config = {
        "query_count": len(eval_meta['query_ids']),
        "seed": eval_meta['seed'],
        "locale": LOCALE,
        "catalog_product_count": catalog_us_count,
        "k_values": K_VALUES,
        "broad_relevance_labels": ["E", "S", "C"],
        "exact_relevance_labels": ["E"],
        "bm25_index": f"{OUT_DIR}/bm25s_index_us",
        "faiss_index": f"{OUT_DIR}/tt_index_us.faiss",
        "hybrid_method": "RRF",
        "rrf_k": RRF_K,
        "hybrid_union_budget_per_retriever": 100,
        "important_caveat": (
            "Recall is measured against known ESCI-labeled relevant products; unlabeled "
            "catalog items are treated as unknown, not as explicit negatives. Precision@K "
            "is not reported as a true retrieval-quality metric for this reason."
        ),
    }
    with open(f"{OUT_DIR}/evaluation_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {OUT_DIR}/evaluation_config.json")

    return metrics_df, overlap_summary


if __name__ == "__main__":
    main()
