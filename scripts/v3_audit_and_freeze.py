"""
V3 Phase 0 -- freeze V2 and audit the DEV pool before any modern reranker runs.

Produces:
  experiments/ranking_v2/cross_encoder_task1/FROZEN_V2.json   (section 3)
  experiments/ranking_v3/zero_shot_rerankers/dev_pool_audit.json (sections 8 + 12)

Reads only DEV. Never touches TEST.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments/ranking_v2/kdd_task1_benchmark/scripts"))

from reranking.cross_encoder import build_product_texts, load_task1_scorer  # noqa: E402

BENCH = os.path.join(ROOT, "experiments/ranking_v2/kdd_task1_benchmark")
CE_DIR = os.path.join(ROOT, "experiments/ranking_v2/cross_encoder_task1")
V3 = os.path.join(ROOT, "experiments/ranking_v3/zero_shot_rerankers")
DEV_TEXT = os.path.join(CE_DIR, "_cache/dev_text.parquet")

PCTL = [50, 90, 95, 99]


def sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dist(a, name):
    a = np.asarray(a, dtype=float)
    d = {"mean": round(float(a.mean()), 2), "max": int(a.max()), "min": int(a.min())}
    for p in PCTL:
        d[f"p{p}"] = float(np.percentile(a, p))
    return {name: d}


def main():
    os.makedirs(V3, exist_ok=True)

    # ---------------- section 3: freeze V2 ----------------
    m = json.load(open(os.path.join(CE_DIR, "full_run/fulldev_epoch2_metrics.json")))
    m1 = json.load(open(os.path.join(CE_DIR, "full_run/fulldev_epoch1_metrics.json")))
    loc2 = json.load(open(os.path.join(CE_DIR, "full_run/fulldev_epoch2_locale_metrics.json")))
    tcfg = json.load(open(os.path.join(CE_DIR, "full_run/checkpoints/config.json")))
    hist = json.load(open(os.path.join(CE_DIR, "full_run/checkpoints/train_history.json")))
    sc = load_task1_scorer()

    frozen = {
        "status": "frozen",
        "frozen_at": "2026-09-03",
        "selected_checkpoint": "epoch_2",
        "selection_metric": "dev_ndcg_at_20",
        "selection_note": (
            "epoch_2 wins on all three cutoffs. NOTE: dev cross-entropy LOSS got WORSE in "
            f"epoch 2 ({hist['history'][0]['dev_loss']} -> {hist['history'][1]['dev_loss']}) "
            "while dev NDCG improved -- selecting on loss would have picked the wrong "
            "checkpoint. Selection was made on the ranking metric, never on loss."),
        "dev_full": m["ndcg_full"],
        "dev_ndcg_at_20": m["ndcg_at_20"],
        "dev_ndcg_at_10": m["ndcg_at_10"],
        "test_split_touched": False,
        "epoch_1_reference": {"dev_full": m1["ndcg_full"], "dev_ndcg_at_20": m1["ndcg_at_20"],
                              "dev_ndcg_at_10": m1["ndcg_at_10"]},
        "dev_locale_metrics": loc2,
        # verified-from-code provenance
        "candidate_pool": m["candidate_pool"],
        "metric_version": m["metric_version"],
        "scorer": m["scorer"],
        "scorer_sha256": sc.scorer_sha256(),
        "gain_convention": {k: v for k, v in sc.GAIN_COMPETITION.items()},
        "gain_applied_as": "used DIRECTLY as the DCG numerator; 2**gain-1 is NOT applied",
        "zero_idcg_policy": m["zero_idcg_policy"],
        "zero_idcg_query_count": m["zero_idcg_query_count"],
        "tie_break": "score DESC, product_id ASC, product_locale ASC (deterministic)",
        "label_mapping_model_head": {"id2label": tcfg["id2label"], "label2id": tcfg["label2id"]},
        "ranking_score_definition": m["score_definition"],
        "dev_source": os.path.relpath(os.path.join(BENCH, "dev_task1.parquet"), ROOT),
        "dev_text_source": os.path.relpath(DEV_TEXT, ROOT),
        "query_count": m["query_count"],
        "row_count": m["row_count"],
        "model_name": tcfg["model_name"],
        "max_length": tcfg["max_length"],
        "train_rows": tcfg["train_rows"], "train_queries": tcfg["train_queries"],
        "training": hist["history"],
        "checkpoint_path": os.path.relpath(
            os.path.join(CE_DIR, "full_run/checkpoints/epoch_2"), ROOT),
        "V3_BASELINE_TO_BEAT_ndcg_at_20": m["ndcg_at_20"],
    }
    with open(os.path.join(CE_DIR, "FROZEN_V2.json"), "w") as f:
        json.dump(frozen, f, indent=2, default=str)
    print(f"FROZEN_V2.json written: @20={m['ndcg_at_20']} full={m['ndcg_full']} "
          f"@10={m['ndcg_at_10']}  test_touched={frozen['test_split_touched']}")

    # ---------------- sections 8 + 12: DEV pool audit ----------------
    dv = pd.read_parquet(DEV_TEXT)
    assert len(dv) == 118231 and dv["query_id"].nunique() == 5071, \
        f"DEV pool mismatch: {len(dv)}/{dv['query_id'].nunique()}"

    cpq = dv.groupby("query_id").size()
    qloc = dv.drop_duplicates("query_id").set_index("query_id")["product_locale"]
    audit = {
        "dev_source": os.path.relpath(DEV_TEXT, ROOT),
        "dev_sha256": sha256(DEV_TEXT),
        "rows": int(len(dv)), "queries": int(dv["query_id"].nunique()),
        "test_split_touched": False,
        "candidates_per_query": dist(cpq.values, "all")["all"],
        "candidates_per_query_by_locale": {
            lc: dist(cpq[qloc == lc].values, lc)[lc] for lc in ["us", "es", "jp"]},
        "locale_query_counts": {lc: int((qloc == lc).sum()) for lc in ["us", "es", "jp"]},
        "field_null_rate": {},
        "field_char_length": {},
        "token_length": {},
    }
    for lbl, col in [("title", "product_title"), ("brand", "product_brand"),
                     ("color", "product_color"), ("bullet", "product_bullet_point"),
                     ("description", "product_description")]:
        audit["field_null_rate"][lbl] = round(float(dv[col].isna().mean()), 6)
        L = dv[col].fillna("").astype(str).str.len().values
        audit["field_char_length"][lbl] = dist(L, lbl)[lbl]
    audit["field_char_length"]["query"] = dist(
        dv["query"].astype(str).str.len().values, "q")["q"]

    # ---- token lengths under both tokenizer families ----
    from transformers import AutoTokenizer
    plain = build_product_texts(dv)          # V2 "plain" construction
    sample = np.random.RandomState(0).choice(len(dv), size=20000, replace=False)
    qs = dv["query"].astype(str).tolist()

    def structured(i):
        g = lambda c: ("" if pd.isna(dv[c].iloc[i]) else str(dv[c].iloc[i]))  # noqa: E731
        return (f"Query:\n{qs[i]}\n\nTitle:\n{g('product_title')}\n\nBrand:\n{g('product_brand')}"
                f"\n\nColor:\n{g('product_color')}\n\nBullet Points:\n{g('product_bullet_point')}"
                f"\n\nDescription:\n{g('product_description')}")

    for tname, tid in [("xlm-roberta (BGE v2-m3)", "BAAI/bge-reranker-v2-m3"),
                       ("qwen3 (Jina v3.5 / Qwen3-Reranker)", "Qwen/Qwen3-Reranker-0.6B")]:
        try:
            tk = AutoTokenizer.from_pretrained(tid)
        except Exception as e:  # noqa: BLE001
            audit["token_length"][tname] = {"error": str(e)[:120]}
            continue
        blk = {}
        for lbl, col in [("title", "product_title"), ("brand", "product_brand"),
                         ("color", "product_color"), ("bullet", "product_bullet_point"),
                         ("description", "product_description")]:
            txt = [("" if pd.isna(dv[col].iloc[i]) else str(dv[col].iloc[i])) for i in sample]
            n = [len(x) for x in tk(txt, add_special_tokens=False)["input_ids"]]
            blk[lbl] = dist(n, lbl)[lbl]
        qn = [len(x) for x in tk([qs[i] for i in sample], add_special_tokens=False)["input_ids"]]
        blk["query"] = dist(qn, "q")["q"]
        pn = [len(x) for x in tk([plain[i] for i in sample], add_special_tokens=False)["input_ids"]]
        blk["combined_plain"] = dist(pn, "c")["c"]
        sn = [len(x) for x in tk([structured(i) for i in sample],
                                 add_special_tokens=False)["input_ids"]]
        blk["combined_structured"] = dist(sn, "c")["c"]
        blk["_note"] = f"measured on a random {len(sample)}-row sample, seed 0"
        audit["token_length"][tname] = blk

    with open(os.path.join(V3, "dev_pool_audit.json"), "w") as f:
        json.dump(audit, f, indent=2, default=str)

    c = audit["candidates_per_query"]
    print(f"\ncandidates/query: mean {c['mean']} p50 {c['p50']} p90 {c['p90']} "
          f"p95 {c['p95']} max {c['max']}")
    print(f"locale queries: {audit['locale_query_counts']}")
    print("\nnull rate:", audit["field_null_rate"])
    for tname, blk in audit["token_length"].items():
        if "error" in blk:
            print(f"\n{tname}: {blk['error']}"); continue
        print(f"\n{tname} token lengths (mean / p90 / p99 / max):")
        for k in ["query", "title", "brand", "color", "bullet", "description",
                  "combined_plain", "combined_structured"]:
            d = blk[k]
            print(f"   {k:<22}{d['mean']:>9.1f}{d['p90']:>9.0f}{d['p99']:>9.0f}{d['max']:>9d}")
    print("\nwrote dev_pool_audit.json")


if __name__ == "__main__":
    main()
