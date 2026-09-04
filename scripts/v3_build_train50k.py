"""
V3 STEP 1 -- freeze the 50K supervised training subset.

Selected BY QUERY (never splitting a query's candidate list), locale-stratified to
the full TRAIN row shares, ~50,000 labeled query-product rows.

Guarantees, all asserted:
  * no DEV query enters TRAIN
  * no TEST data is read at all
  * every candidate of a selected query is included
Once written this subset is frozen; Jina and Qwen must consume identical pairs.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CE = os.path.join(ROOT, "experiments/ranking_v2/cross_encoder_task1/_cache")
BENCH = os.path.join(ROOT, "experiments/ranking_v2/kdd_task1_benchmark")
OUT = os.path.join(ROOT, "experiments/ranking_v3/supervised_bakeoff")
TARGET_ROWS, SEED = 50_000, 42
GAIN = {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00}


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_parquet(os.path.join(CE, "train_text.parquet"),
                         columns=["example_id", "query_id", "product_locale", "esci_label"])
    assert len(tr) == 663407 and tr["query_id"].nunique() == 28733, "not the frozen TRAIN pool"

    dev_q = set(pd.read_parquet(os.path.join(BENCH, "dev_task1.parquet"),
                                columns=["query_id"])["query_id"])
    assert not (set(tr["query_id"]) & dev_q), "TRAIN pool already intersects DEV"

    per_q = tr.groupby("query_id").agg(rows=("example_id", "size"),
                                       locale=("product_locale", "first")).reset_index()
    full_rows = {lc: int(tr.loc[tr.product_locale == lc].shape[0]) for lc in ["us", "es", "jp"]}
    full_share = {lc: full_rows[lc] / len(tr) for lc in full_rows}
    row_target = {lc: int(round(TARGET_ROWS * full_share[lc])) for lc in full_share}

    rng = np.random.RandomState(SEED)
    picked = []
    for lc in ["us", "es", "jp"]:
        pool = per_q[per_q.locale == lc].sort_values("query_id")
        order = rng.permutation(pool["query_id"].values)
        rmap = dict(zip(pool["query_id"], pool["rows"]))
        acc = 0
        for qid in order:
            if acc >= row_target[lc]:
                break
            picked.append(qid)
            acc += rmap[qid]
    picked = np.sort(np.array(picked))
    sel = tr[tr["query_id"].isin(set(picked))].copy()

    assert not (set(sel["query_id"]) & dev_q), "DEV LEAK: selected queries intersect DEV"
    assert sel.groupby("query_id").size().equals(
        per_q.set_index("query_id").loc[picked, "rows"]), "a query's candidate list was split"

    lab = sel["esci_label"].value_counts()
    loc_rows = sel["product_locale"].value_counts()
    loc_q = sel.drop_duplicates("query_id")["product_locale"].value_counts()
    ex_ids = np.sort(sel["example_id"].values)

    man = {
        "name": "train50k", "frozen": True, "seed": SEED,
        "source": os.path.relpath(os.path.join(CE, "train_text.parquet"), ROOT),
        "source_rows": 663407, "source_queries": 28733,
        "selection": "by query (whole candidate list), locale-stratified to full-TRAIN row share",
        "target_rows": TARGET_ROWS,
        "exact_row_count": int(len(sel)),
        "query_count": int(sel["query_id"].nunique()),
        "rows_per_query_mean": round(float(len(sel) / sel["query_id"].nunique()), 3),
        "locale_row_counts": {lc: int(loc_rows.get(lc, 0)) for lc in ["us", "es", "jp"]},
        "locale_row_share": {lc: round(float(loc_rows.get(lc, 0) / len(sel)), 4)
                             for lc in ["us", "es", "jp"]},
        "full_train_locale_row_share": {lc: round(full_share[lc], 4) for lc in full_share},
        "locale_query_counts": {lc: int(loc_q.get(lc, 0)) for lc in ["us", "es", "jp"]},
        "label_counts": {k: int(lab.get(k, 0)) for k in ["E", "S", "C", "I"]},
        "label_share_pct": {k: round(100 * float(lab.get(k, 0) / len(sel)), 3)
                            for k in ["E", "S", "C", "I"]},
        "official_gain_targets": GAIN,
        "dev_leakage_check": {"dev_queries": len(dev_q), "intersection": 0, "PASS": True},
        "test_split_touched": False,
        "sha256_query_ids": hashlib.sha256(",".join(map(str, picked.tolist())).encode()).hexdigest(),
        "sha256_example_ids": hashlib.sha256(",".join(map(str, ex_ids.tolist())).encode()).hexdigest(),
        "query_ids": [int(x) for x in picked],
    }
    json.dump(man, open(os.path.join(OUT, "train50k_manifest.json"), "w"), indent=2)
    np.save(os.path.join(OUT, "train50k_example_ids.npy"), ex_ids)

    print(f"rows={man['exact_row_count']}  queries={man['query_count']}  "
          f"rows/query={man['rows_per_query_mean']}")
    print(f"locale rows  : {man['locale_row_counts']}  share {man['locale_row_share']}")
    print(f"full TRAIN    : {man['full_train_locale_row_share']}")
    print(f"locale queries: {man['locale_query_counts']}")
    print(f"labels        : {man['label_counts']}  -> {man['label_share_pct']} %")
    print(f"DEV leak      : {man['dev_leakage_check']['intersection']} (PASS)")
    print(f"sha256(qids)  : {man['sha256_query_ids'][:16]}")
    print("\nwrote train50k_manifest.json + train50k_example_ids.npy")


if __name__ == "__main__":
    main()
