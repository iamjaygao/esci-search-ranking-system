"""
V3 STEP 0 -- zero-cost ensemble audit on already-saved confirm1000 predictions.

No inference is re-run. Weight selection and reporting are on DISJOINT query sets
so the reported number is not the number the weights were fitted on:

    weight selection : screen300   (300 queries, nested inside confirm1000)
    confirmation     : the remaining 700 queries of confirm1000

Scores from different rerankers are not comparable in scale, so fusion operates
on query-level linear normalised rank (best=1, worst=0), which is scale-free.
Coarse 0.1-step fixed-weight grid only -- no gating, no stacking.

Decision rule (pre-registered by the user):
    holdout700 Δ@20 <  +0.002  -> stop pursuing simple ensembles
    holdout700 Δ@20 >= +0.005  -> keep as a final-route candidate

DEV only. TEST never read.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments/ranking_v2/kdd_task1_benchmark/scripts"))
from reranking.cross_encoder import load_task1_scorer  # noqa: E402

V3 = os.path.join(ROOT, "experiments/ranking_v3/zero_shot_rerankers")
OUT = os.path.join(ROOT, "experiments/ranking_v3/supervised_bakeoff")
V2_PRED = os.path.join(ROOT, "experiments/ranking_v2/cross_encoder_task1/full_run/fulldev_epoch2_predictions.parquet")
KEY = ["query_id", "product_id", "product_locale"]
STEP = 0.1


def norm_rank(df: pd.DataFrame, col: str) -> np.ndarray:
    """Query-level linear normalised rank: best=1.0, worst=0.0. Scale-free.
    Ties broken deterministically by (product_id, product_locale) to match the scorer."""
    d = df[KEY + [col]].copy()
    d["_o"] = np.arange(len(d))
    d = d.sort_values(["query_id", col, "product_id", "product_locale"],
                      ascending=[True, False, True, True], kind="mergesort")
    g = d.groupby("query_id", sort=False)
    d["_r"] = g.cumcount()                       # 0 = best
    d["_n"] = g["_o"].transform("size")
    d["_v"] = np.where(d["_n"] > 1, 1.0 - d["_r"] / (d["_n"] - 1), 1.0)
    return d.sort_values("_o")["_v"].to_numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = load_task1_scorer()
    ids1000 = json.load(open(os.path.join(V3, "subset_confirm1000.json")))["query_ids"]
    ids300 = set(json.load(open(os.path.join(V3, "subset_screen300.json")))["query_ids"])
    ids700 = sorted(set(ids1000) - ids300)
    assert len(ids300) == 300 and len(ids700) == 700 and not (ids300 & set(ids700))

    base = pd.read_parquet(V2_PRED)
    base = base[base["query_id"].isin(set(ids1000))].copy()
    parts = {"V2": base[KEY + ["esci_label", "gain", "score"]].rename(columns={"score": "s_V2"})}
    for name, tag in [("Jina", "jina_plain_confirm1000"), ("Qwen", "qwen_plain_confirm1000")]:
        p = pd.read_parquet(os.path.join(V3, tag, "predictions.parquet"))
        parts[name] = p[KEY + ["score"]].rename(columns={"score": f"s_{name}"})

    df = parts["V2"]
    for n in ["Jina", "Qwen"]:
        df = df.merge(parts[n], on=KEY, how="inner")
    assert len(df) == 23711, f"join lost rows: {len(df)}"
    for n in ["V2", "Jina", "Qwen"]:
        df[f"r_{n}"] = norm_rank(df, f"s_{n}")

    def ndcg(sub: pd.DataFrame, col: str) -> float:
        return sc.evaluate(sub, col, gain_col="gain", query_col="query_id")["ndcg_at_20"]

    d300, d700 = df[df.query_id.isin(ids300)].copy(), df[df.query_id.isin(set(ids700))].copy()
    solo = {}
    for n in ["V2", "Jina", "Qwen"]:
        solo[n] = {"sel300": round(ndcg(d300, f"s_{n}"), 6),
                   "holdout700": round(ndcg(d700, f"s_{n}"), 6)}
    print("single models  (@20)")
    for n, v in solo.items():
        print(f"  {n:<6} sel300={v['sel300']:.4f}  holdout700={v['holdout700']:.4f}")
    B700, B300 = solo["V2"]["holdout700"], solo["V2"]["sel300"]

    combos = {"V2+Jina": ["V2", "Jina"], "V2+Qwen": ["V2", "Qwen"],
              "V2+Jina+Qwen": ["V2", "Jina", "Qwen"]}
    grid = [round(x, 1) for x in np.arange(0, 1.0 + 1e-9, STEP)]
    results = {}
    print("\nweight grid on sel300, reported on holdout700 (never fitted there)")
    for cname, members in combos.items():
        cands = []
        for w in itertools.product(grid, repeat=len(members)):
            if abs(sum(w) - 1.0) > 1e-9 or any(x == 0 for x in w):
                continue
            d300["_f"] = sum(wi * d300[f"r_{m}"] for wi, m in zip(w, members))
            cands.append((ndcg(d300, "_f"), w))
        cands.sort(key=lambda t: -t[0])
        best_sel, best_w = cands[0]
        d700["_f"] = sum(wi * d700[f"r_{m}"] for wi, m in zip(best_w, members))
        got = ndcg(d700, "_f")
        results[cname] = {
            "members": members, "grid_step": STEP, "n_grid_points": len(cands),
            "selected_weights": dict(zip(members, best_w)),
            "sel300_ndcg_at_20": round(best_sel, 6),
            "sel300_delta_vs_V2": round(best_sel - B300, 6),
            "holdout700_ndcg_at_20": round(got, 6),
            "holdout700_V2_ndcg_at_20": B700,
            "holdout700_delta_vs_V2": round(got - B700, 6),
            "top3_on_sel300": [{"w": dict(zip(members, w)), "ndcg_at_20": round(s, 6)}
                               for s, w in cands[:3]],
        }
        r = results[cname]
        print(f"  {cname:<14} w={r['selected_weights']}  sel300={best_sel:.4f} "
              f"({r['sel300_delta_vs_V2']:+.4f})  holdout700={got:.4f} "
              f"({r['holdout700_delta_vs_V2']:+.4f})")

    best = max(results, key=lambda k: results[k]["holdout700_delta_vs_V2"])
    d = results[best]["holdout700_delta_vs_V2"]
    verdict = ("KEEP as a final-route candidate (>= +0.005)" if d >= 0.005 else
               "STOP pursuing simple ensembles (< +0.002)" if d < 0.002 else
               "INCONCLUSIVE: between +0.002 and +0.005 -- neither rule fires")
    payload = {
        "method": "query-level linear normalised rank fusion, fixed weights, 0.1 grid",
        "weight_selection_subset": {"name": "screen300", "n_queries": 300},
        "confirmation_subset": {"name": "confirm1000_minus_screen300", "n_queries": 700,
                                "query_ids_sha256_note": "disjoint from the selection subset"},
        "no_inference_rerun": True, "test_split_touched": False,
        "single_models": solo, "combinations": results,
        "best_combination": best, "best_holdout700_delta": d, "verdict": verdict,
        "decision_rule": {"stop_below": 0.002, "keep_at_or_above": 0.005},
    }
    json.dump(payload, open(os.path.join(OUT, "ensemble_audit.json"), "w"), indent=2)
    print(f"\nbest = {best}, holdout700 Δ@20 = {d:+.4f}\nVERDICT: {verdict}")
    print("wrote experiments/ranking_v3/supervised_bakeoff/ensemble_audit.json")


if __name__ == "__main__":
    main()
