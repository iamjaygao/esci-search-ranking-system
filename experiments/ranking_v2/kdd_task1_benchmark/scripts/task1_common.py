"""
Shared helpers for the KDD Task 1 benchmark.

Resolution item 3: the arm-B exclusion MUST be a shared helper, not an inline
filter. Any future training pool that draws from the large-version data has to
call `exclude_task1_test_queries()` / `build_arm_b_pool()` -- never re-implement
the filter locally.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
BASE = os.path.join(ROOT, "experiments", "ranking_v2", "kdd_task1_benchmark")

EXAMPLES = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")
PRODUCTS = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet")

#: The sole query_id that ESCI itself assigns to both train and test.
#: Documented as FAIL_UPSTREAM in gate_a.json. NOT filtered out of any pool.
KNOWN_UPSTREAM_CROSS_SPLIT_QUERY_IDS = frozenset({79706})

LABEL_ORDINAL = {"I": 0, "C": 1, "S": 2, "E": 3}
OFFICIAL_GAIN = {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00}


def load_examples(columns=None):
    return pd.read_parquet(EXAMPLES, columns=columns)


def task1_test_query_ids(examples=None):
    """The frozen Task 1 evaluation query set: small_version==1 AND split=='test'."""
    ex = examples if examples is not None else load_examples(
        columns=["query_id", "small_version", "split"])
    return set(ex.loc[(ex["small_version"] == 1) & (ex["split"] == "test"), "query_id"].unique())


def exclude_task1_test_queries(df, examples=None, query_col="query_id"):
    """THE shared arm-B exclusion.

    Removes every row whose query_id is in the frozen Task 1 test query set.
    This is the only sanctioned way to build a training pool that draws on
    large_version data, and it is what makes assertion A3' hold even though the
    raw-data assertion A3 fails upstream (query_id 79706 carries 3 large-only
    train rows while its 31 small_version==1 rows are in Task 1 test).

    Returns (filtered_df, info_dict).
    """
    test_q = task1_test_query_ids(examples)
    before = len(df)
    mask = df[query_col].isin(test_q)
    out = df.loc[~mask].copy()
    info = {
        "rows_before": int(before),
        "rows_removed": int(mask.sum()),
        "rows_after": int(len(out)),
        "queries_removed": int(df.loc[mask, query_col].nunique()),
        "removed_query_ids_sample": sorted(df.loc[mask, query_col].unique().tolist())[:20],
        "task1_test_query_count": len(test_q),
        "helper": "task1_common.exclude_task1_test_queries",
    }
    return out, info


def build_arm_b_pool(examples=None):
    """Arm B: large_version==1 AND split=='train', with Task 1 test queries removed
    via the shared helper. Returns (pool_df, info_dict)."""
    ex = examples if examples is not None else load_examples()
    raw = ex[(ex["large_version"] == 1) & (ex["split"] == "train")]
    pool, info = exclude_task1_test_queries(raw, examples=ex)
    info["definition"] = ("large_version==1 AND split=='train' AND "
                          "query_id NOT IN task1_test_queries "
                          "(applied via task1_common.exclude_task1_test_queries)")
    return pool, info


def build_arm_a_pool(train_query_ids, examples=None):
    """Arm A: the Task 1 training split only (after the 85/15 query-level carve)."""
    ex = examples if examples is not None else load_examples()
    raw = ex[(ex["small_version"] == 1) & (ex["split"] == "train")]
    pool = raw[raw["query_id"].isin(set(train_query_ids))].copy()
    # defensive: run the shared exclusion anyway; it must be a no-op here
    pool2, info = exclude_task1_test_queries(pool, examples=ex)
    assert len(pool2) == len(pool), "arm A unexpectedly intersected Task 1 test queries"
    info["definition"] = ("small_version==1 AND split=='train' AND "
                          "query_id in train_task1 "
                          "(shared exclusion applied, expected to be a no-op)")
    return pool2, info


def normalize_query_text(s):
    """Lowercase + whitespace-collapse. Matches the normalisation used by the
    upstream feature code (`str(q).lower().split()`)."""
    return s.astype(str).str.lower().str.split().str.join(" ")
