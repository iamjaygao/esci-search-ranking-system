"""
Phase D -- the single authoritative NDCG@10 scorer for ESCI ranking V2.

One implementation, no model-specific logic. Every V2 model must be scored
through `evaluate()` or `per_query_ndcg()` in this module.

Convention (see official_gain_mapping.json):

    relevance : E=1.00, S=0.10, C=0.01, I=0.00
    gain      : 2**relevance - 1        (identical to evaluation/metrics.dcg)
    discount  : 1 / log2(rank + 1), rank 1-based
    NDCG@k    : DCG@k / IDCG@k, per query
    aggregate : unweighted arithmetic mean over queries (each query weight 1)

Documented behaviours
---------------------
* candidate count < k  -- the list is simply shorter. DCG and IDCG are both
  computed over the same short list, so NDCG is well defined and is often
  exactly 1.0. No padding, no penalty. (Same as the V1 scorer.)

* no relevant item (IDCG == 0, i.e. every candidate is I) -- the query is
  EXCLUDED from the mean, and `n_excluded_no_relevant` is reported. This
  matches evaluation/metrics.ndcg_at_k, which skips `idcg_val <= 0`.
  On the frozen pool this excludes 0 test queries.

* ties -- DETERMINISTIC, and different from V1. Rows are sorted by
  (-score, product_id) using a stable mergesort, so equal scores are broken by
  ascending product_id rather than by incoming frame order. This makes the
  metric independent of row ordering, which V1's default quicksort was not
  (34.2% of BM25-scored rows are tied with another row in the same query).
  Set `tie_break="pessimistic"` to instead rank tied rows worst-relevance-first,
  which gives a lower bound on the score.
"""
import numpy as np
import pandas as pd

K_DEFAULT = 10
OFFICIAL_RELEVANCE = {"I": 0.0, "C": 0.01, "S": 0.10, "E": 1.00}


def relevance_from_labels(labels):
    """ESCI label series -> official graded relevance."""
    return pd.Series(labels).map(OFFICIAL_RELEVANCE).astype(float)


def dcg(relevances, k=K_DEFAULT):
    """sum over the top-k of (2**rel - 1) / log2(rank + 1), rank 1-based."""
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    return float(np.sum((2.0 ** rel - 1.0) / np.log2(np.arange(2, rel.size + 2))))


def _order(group, score_col, tie_break):
    if tie_break == "deterministic":
        # ascending product_id breaks ties; mergesort keeps it stable
        g = group.sort_values("product_id", kind="mergesort")
        return g.sort_values(score_col, ascending=False, kind="mergesort")
    if tie_break == "pessimistic":
        g = group.sort_values("relevance", ascending=True, kind="mergesort")
        return g.sort_values(score_col, ascending=False, kind="mergesort")
    raise ValueError(f"unknown tie_break: {tie_break}")


def per_query_ndcg(df, score_col, k=K_DEFAULT, relevance_col="relevance",
                   query_col="query_id", tie_break="deterministic"):
    """Returns a pd.Series indexed by query_id. Queries with IDCG==0 are absent."""
    for c in (score_col, relevance_col, query_col, "product_id"):
        if c not in df.columns:
            raise ValueError(f"missing required column: {c}")
    if df[score_col].isna().any():
        raise ValueError(f"{score_col} contains NaN -- refusing to score")

    out = {}
    for qid, g in df.groupby(query_col, sort=False):
        ideal = dcg(np.sort(g[relevance_col].values)[::-1], k)
        if ideal <= 0:
            continue
        ranked = _order(g, score_col, tie_break)
        out[qid] = dcg(ranked[relevance_col].values, k) / ideal
    return pd.Series(out, name=f"{score_col}_ndcg@{k}", dtype=float)


def evaluate(df, score_col, k=K_DEFAULT, relevance_col="relevance",
             query_col="query_id", tie_break="deterministic"):
    """Aggregate report for one score column on one pool."""
    s = per_query_ndcg(df, score_col, k, relevance_col, query_col, tie_break)
    n_total = int(df[query_col].nunique())
    return {
        "score_col": score_col,
        "k": k,
        "tie_break": tie_break,
        "n_queries_total": n_total,
        "n_queries_scored": int(len(s)),
        "n_excluded_no_relevant": n_total - int(len(s)),
        "ndcg_at_k": float(s.mean()) if len(s) else 0.0,
        "ndcg_at_k_median": float(s.median()) if len(s) else 0.0,
        "ndcg_at_k_std": float(s.std()) if len(s) else 0.0,
        "n_rows": int(len(df)),
    }
