"""
V3 successive-halving funnel -- build the fixed, locale-stratified query subsets.

  round 1  screening    300 queries    all 6 configs
  round 2  confirmation 1000 queries   top-2 configs only   (SUPERSET of the 300)
  round 3  full DEV     5071 queries   top-1/2 only

Both subsets are written once and reused by every model, so no configuration is
ever screened on a different query set than another. The 1,000 strictly contains
the 300, so round-2 numbers are directly comparable to round-1 numbers on the
shared part.

DEV only. Never reads TEST.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_TEXT = os.path.join(ROOT, "experiments/ranking_v2/cross_encoder_task1/_cache/dev_text.parquet")
V3 = os.path.join(ROOT, "experiments/ranking_v3/zero_shot_rerankers")
SEED = 42
SIZES = {"screen300": 300, "confirm1000": 1000}


def largest_remainder(counts: dict, total: int) -> dict:
    """Proportional allocation that sums exactly to `total`."""
    n = sum(counts.values())
    raw = {k: v / n * total for k, v in counts.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = total - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:rem]:
        base[k] += 1
    return base


def main():
    os.makedirs(V3, exist_ok=True)
    df = pd.read_parquet(DEV_TEXT, columns=["query_id", "product_locale"])
    assert len(df) == 118231 and df["query_id"].nunique() == 5071, "not the frozen DEV pool"

    q = df.drop_duplicates("query_id")[["query_id", "product_locale"]].sort_values("query_id")
    pool = {lc: np.sort(q.loc[q["product_locale"] == lc, "query_id"].values)
            for lc in ["us", "es", "jp"]}
    full_counts = {lc: len(v) for lc, v in pool.items()}

    alloc300 = largest_remainder(full_counts, SIZES["screen300"])
    alloc1000 = largest_remainder(full_counts, SIZES["confirm1000"])

    rng = np.random.RandomState(SEED)
    chosen300, chosen1000 = {}, {}
    for lc in ["us", "es", "jp"]:
        perm = rng.permutation(pool[lc])              # one shuffle per locale
        chosen300[lc] = np.sort(perm[:alloc300[lc]])
        chosen1000[lc] = np.sort(perm[:alloc1000[lc]])   # nested by construction
        assert set(chosen300[lc]).issubset(set(chosen1000[lc]))

    ids300 = np.sort(np.concatenate([chosen300[lc] for lc in pool]))
    ids1000 = np.sort(np.concatenate([chosen1000[lc] for lc in pool]))
    assert len(ids300) == 300 and len(ids1000) == 1000
    assert set(ids300).issubset(set(ids1000)), "1000 must contain the 300"

    # candidate-row coverage of each subset
    rows = df.groupby("query_id").size()
    meta = {
        "seed": SEED,
        "dev_source": os.path.relpath(DEV_TEXT, ROOT),
        "dev_queries": 5071, "dev_rows": 118231,
        "test_split_touched": False,
        "stratification": "proportional by product_locale, largest-remainder to hit the exact size",
        "nesting": "confirm1000 strictly contains screen300 (same per-locale permutation prefix)",
        "full_locale_counts": full_counts,
        "subsets": {},
    }
    for name, ids, alloc in [("screen300", ids300, alloc300),
                             ("confirm1000", ids1000, alloc1000)]:
        sub = q[q["query_id"].isin(set(ids))]
        lc_counts = sub["product_locale"].value_counts().to_dict()
        r = int(rows[rows.index.isin(set(ids))].sum())
        payload = {"n_queries": int(len(ids)), "n_rows": r,
                   "locale_counts": {k: int(lc_counts.get(k, 0)) for k in ["us", "es", "jp"]},
                   "locale_alloc_target": alloc,
                   "locale_share": {k: round(lc_counts.get(k, 0) / len(ids), 4)
                                    for k in ["us", "es", "jp"]},
                   "query_ids": [int(x) for x in ids]}
        payload["sha256_of_ids"] = hashlib.sha256(
            ",".join(map(str, payload["query_ids"])).encode()).hexdigest()
        meta["subsets"][name] = payload
        with open(os.path.join(V3, f"subset_{name}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"{name}: {len(ids)} queries / {r} rows  locales={payload['locale_counts']}  "
              f"share={payload['locale_share']}")

    full_share = {k: round(v / 5071, 4) for k, v in full_counts.items()}
    print(f"full DEV locale share = {full_share}")
    meta["full_locale_share"] = full_share
    with open(os.path.join(V3, "funnel_subsets.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nwrote subset_screen300.json, subset_confirm1000.json, funnel_subsets.json")


if __name__ == "__main__":
    main()
