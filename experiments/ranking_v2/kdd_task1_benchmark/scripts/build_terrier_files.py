"""
Section 9.2 / 9.3 -- generate the qrels and run files for a Terrier trec_eval
parity check, plus the synthetic test set used to determine the IDCG==0 policy.

Runs WITHOUT Terrier: it only writes the input files and records the Python-side
NDCG for each. When a JRE + Terrier 5.5 become available, run_terrier_parity.sh
consumes these files and completes Gate B.

qrels mapping -- CORRECTED, not the reference helper's:
    I -> 1, C -> 2, S -> 3, E -> 4
composed with  -m 'ndcg.1=0,2=0.01,3=0.1,4=1'  this yields
    E=1.0, S=0.1, C=0.01, I=0.0   (the competition definition)
The reference helper uses {E:4, S:2, C:3, I:1}, which yields S=0.01 and C=0.1.
See gain_convention_conflict.json.

doc_id -- locale-prefixed f"{product_locale}_{product_id}". Mandatory here:
4,359 product_ids in the Task 1 pool appear under 2-3 locales, and the reference
helper's product_id-only keying would collapse them.

run file -- the reference script does NOT write model scores. It converts the
RANK POSITION into a strictly decreasing synthetic score
(prepare_trec_eval_files.py line 78). That is reproduced exactly, so Terrier and
the Python scorer evaluate the identical ordering and the tie-break strategy is
removed from the parity test.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from task1_common import BASE  # noqa: E402
import kdd_task1_ndcg as SC  # noqa: E402

TREC_DIR = os.path.join(BASE, "trec_eval_data")
QRELS_MAP_CORRECTED = {"I": 1, "C": 2, "S": 3, "E": 4}
MAX_TREC_SCORE, MIN_TREC_SCORE = 128, 0


def synthetic_scores(n):
    """Verbatim from prepare_trec_eval_files.py line 78."""
    return np.arange(MIN_TREC_SCORE, MAX_TREC_SCORE, MAX_TREC_SCORE / n).round(3)[::-1][:n]


def write_qrels(df, path):
    q = df[["query_id", "doc_id", "esci_label"]].copy()
    q["iteration"] = 0
    q["rel"] = q["esci_label"].map(QRELS_MAP_CORRECTED).astype(int)
    q[["query_id", "iteration", "doc_id", "rel"]].to_csv(
        path, index=False, header=False, sep=" ")
    return len(q)


def write_run(df, score_col, path, run_id="task1"):
    """Materialise the Python scorer's final ordering into TREC synthetic scores."""
    tbl_order = []
    qcodes, quniq = pd.factorize(df["query_id"], sort=True)
    pid = pd.factorize(df["product_id"], sort=True)[0]
    loc = pd.factorize(df["product_locale"], sort=True)[0]
    order = np.lexsort((loc, pid, -df[score_col].to_numpy(dtype=float), qcodes))
    d = df.iloc[order].reset_index(drop=True)
    d["_q"] = qcodes[order]

    rows, n_tie_fail = [], 0
    for _, g in d.groupby("_q", sort=False):
        n = len(g)
        syn = synthetic_scores(n)
        if len(set(syn.tolist())) != n:
            n_tie_fail += 1
        rows.append(pd.DataFrame({
            "query_id": g["query_id"].to_numpy(),
            "iteration": "Q0",
            "doc_id": g["doc_id"].to_numpy(),
            "rank": np.arange(n),
            "score": syn,
            "run_id": run_id,
        }))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False, header=False, sep=" ")
    return len(out), n_tie_fail


def synthetic_dataset():
    """Section 9.5 test 1: one all-I query, one normal query, one with tied gains."""
    rows = [
        ("sq_normal", "us", "p1", "E", 4.0), ("sq_normal", "us", "p2", "S", 3.0),
        ("sq_normal", "us", "p3", "C", 2.0), ("sq_normal", "us", "p4", "I", 1.0),
        ("sq_allI", "us", "p5", "I", 3.0), ("sq_allI", "us", "p6", "I", 2.0),
        ("sq_allI", "us", "p7", "I", 1.0),
        ("sq_tied", "us", "p8", "E", 2.0), ("sq_tied", "us", "p9", "E", 2.0),
        ("sq_tied", "us", "pa", "S", 1.0), ("sq_tied", "us", "pb", "I", 1.0),
    ]
    df = pd.DataFrame(rows, columns=["query_id", "product_locale", "product_id",
                                     "esci_label", "score"])
    df["doc_id"] = df["product_locale"] + "_" + df["product_id"]
    df["gain"] = SC.gain_from_labels(df["esci_label"])
    return df


def main():
    os.makedirs(TREC_DIR, exist_ok=True)
    manifest = {"trec_dir": os.path.relpath(TREC_DIR, BASE),
                "qrels_mapping": QRELS_MAP_CORRECTED,
                "gain_spec": "ndcg.1=0,2=0.01,3=0.1,4=1",
                "doc_id_format": "{product_locale}_{product_id}",
                "run_score_materialization": "prepare_trec_eval_files.py line 78, verbatim",
                "datasets": {}}

    # ---- test 1: synthetic ----
    syn = synthetic_dataset()
    nq = write_qrels(syn, os.path.join(TREC_DIR, "synthetic.qrels"))
    nr, tf = write_run(syn, "score", os.path.join(TREC_DIR, "synthetic.results"), "synthetic")
    tbl = SC.per_query_ndcg_table(syn, "score")
    manifest["datasets"]["synthetic"] = {
        "qrels_rows": nq, "run_rows": nr, "queries": int(syn["query_id"].nunique()),
        "synthetic_score_tie_failures": tf,
        "python_ndcg_full_by_policy": {p: SC.aggregate(tbl, p)["ndcg_full"]
                                       for p in SC.ZERO_IDCG_POLICIES},
        "zero_idcg_query_count": int((tbl["idcg_full"] <= 0).sum()),
        "purpose": "verify the formula AND empirically determine Terrier's IDCG==0 behaviour",
    }

    # ---- tests 2 and 3: dev only, never test ----
    dev_path = os.path.join(BASE, "dev_task1.parquet")
    if os.path.exists(dev_path):
        dev = pd.read_parquet(dev_path)
        rng = np.random.RandomState(20240902)
        dev = dev.copy()
        dev["random_score"] = rng.rand(len(dev))
        for name, col in [("dev_random", "random_score"), ("dev_bm25", "bm25_score")]:
            nq = write_qrels(dev, os.path.join(TREC_DIR, f"{name}.qrels"))
            nr, tf = write_run(dev, col, os.path.join(TREC_DIR, f"{name}.results"), name)
            tbl = SC.per_query_ndcg_table(dev, col)
            manifest["datasets"][name] = {
                "qrels_rows": nq, "run_rows": nr, "queries": int(dev["query_id"].nunique()),
                "synthetic_score_tie_failures": tf,
                "score_col": col,
                "random_seed": 20240902 if col == "random_score" else None,
                "python_ndcg_full_by_policy": {p: SC.aggregate(tbl, p)["ndcg_full"]
                                               for p in SC.ZERO_IDCG_POLICIES},
                "zero_idcg_query_count": int((tbl["idcg_full"] <= 0).sum()),
            }
    else:
        manifest["datasets"]["dev_random"] = {"status": "dev_task1.parquet not built yet"}
        manifest["datasets"]["dev_bm25"] = {"status": "dev_task1.parquet not built yet"}

    with open(os.path.join(TREC_DIR, "trec_files_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
