"""
ESCI Ranking V3 -- zero-shot shootout of modern pretrained rerankers.

ONE harness, shared by every model: same DEV rows, same query grouping, same
product data, same candidate pool, same frozen scorer, same result schema.

TEST LOCK
---------
This script is DEV-only by construction. There is no --split flag and no
override. `_assert_dev_only()` hard-fails on anything that is not the frozen
DEV artifact. Nothing here can read, infer on, or score TEST.

No training of any kind: models are loaded pretrained and run in inference mode.

Models (official inference formulation for each, never forced into a foreign one):
  jina  jinaai/jina-reranker-v3.5    LISTWISE  model.rerank(query, [docs]) -> one pass
  bge   BAAI/bge-reranker-v2-m3      POINTWISE XLMRobertaForSequenceClassification logit
  qwen  Qwen/Qwen3-Reranker-0.6B     POINTWISE sentence_transformers.CrossEncoder (official)

Input constructions (both frozen before any run, section 7):
  plain       exactly V2's product text: "title: .. / brand: .. / color: .. /
              bullet: .. / description: ..", empty fields omitted
  structured  "Title:\\n..\\n\\nBrand:\\n..\\n\\nColor:\\n..\\n\\nBullet Points:\\n..
              \\n\\nDescription:\\n.." ; empty fields render as an empty value,
              never "nan"/"None". The query is supplied through each model's own
              query channel rather than embedded in the document string, because
              all three APIs take the query separately.

Truncation (section 8): never a blind tail cut. Priority order is
query > title > brand > color, which costs 139 tokens even at p99; whatever
remains of the budget goes to bullet, then description.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments/ranking_v2/kdd_task1_benchmark/scripts"))

from reranking.cross_encoder import load_task1_scorer  # noqa: E402

DEV_TEXT = os.path.join(ROOT, "experiments/ranking_v2/cross_encoder_task1/_cache/dev_text.parquet")
V3 = os.path.join(ROOT, "experiments/ranking_v3/zero_shot_rerankers")
EXPECT_ROWS, EXPECT_QUERIES = 118231, 5071

MODELS = {
    "jina": {"id": "jinaai/jina-reranker-v3.5", "kind": "listwise", "params": "0.6B",
             "arch": "JinaForRanking (Qwen3-0.6B backbone, LBNL listwise)"},
    "bge": {"id": "BAAI/bge-reranker-v2-m3", "kind": "pointwise", "params": "568M",
            "arch": "XLMRobertaForSequenceClassification (cross-encoder)"},
    "qwen": {"id": "Qwen/Qwen3-Reranker-0.6B", "kind": "pointwise", "params": "0.6B",
             "arch": "Qwen3ForCausalLM + yes/no logit head"},
}

FIELDS = [("title", "product_title"), ("brand", "product_brand"), ("color", "product_color"),
          ("bullet", "product_bullet_point"), ("description", "product_description")]
PRIORITY = ["title", "brand", "color"]      # always kept whole
BUDGETED = ["bullet", "description"]        # share whatever is left


# ------------------------------------------------------------------ guards
def _assert_dev_only(path: str, df: pd.DataFrame) -> None:
    if "test" in os.path.basename(path).lower():
        raise RuntimeError("TEST LOCK: test evaluation is forbidden before final model freeze.")
    if len(df) != EXPECT_ROWS or df["query_id"].nunique() != EXPECT_QUERIES:
        raise RuntimeError(
            f"TEST LOCK / pool guard: expected the frozen DEV pool "
            f"({EXPECT_ROWS} rows / {EXPECT_QUERIES} queries), got "
            f"{len(df)} / {df['query_id'].nunique()}. Refusing to run.")
    if "split" in df.columns and set(df["split"].unique()) != {"train"}:
        # dev rows carry split=='train' because dev is carved out of the ESCI train split
        raise RuntimeError(f"unexpected split values: {set(df['split'].unique())}")


def clean(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "<na>", "null"} else " ".join(s.split())


# ------------------------------------------------------------------ text
def build_docs(df: pd.DataFrame, style: str, tokenizer, budget: int) -> tuple[list[str], dict]:
    """Priority-aware document construction. Returns (texts, truncation_stats)."""
    vals = {lbl: [clean(x) for x in df[col].tolist()] for lbl, col in FIELDS}
    n = len(df)

    def render(parts: dict) -> str:
        if style == "plain":
            return "\n".join(f"{l}: {parts[l]}" for l, _ in FIELDS if parts[l])
        return ("Title:\n{title}\n\nBrand:\n{brand}\n\nColor:\n{color}\n\n"
                "Bullet Points:\n{bullet}\n\nDescription:\n{description}").format(**parts)

    # full-length pass to find what needs budgeting
    full = [render({l: vals[l][i] for l, _ in FIELDS}) for i in range(n)]
    if tokenizer is None or budget is None:
        return full, {"budget": budget, "truncated_rows": 0, "truncation_fraction": 0.0,
                      "policy": "no tokenizer budget applied"}

    lens = np.array([len(x) for x in tokenizer(full, add_special_tokens=False)["input_ids"]])
    need = lens > budget
    stats = {"budget": budget, "rows": n, "truncated_rows": int(need.sum()),
             "truncation_fraction": round(float(need.mean()), 6),
             "full_len_mean": round(float(lens.mean()), 2),
             "full_len_p95": float(np.percentile(lens, 95)),
             "full_len_max": int(lens.max()),
             "policy": "keep query>title>brand>color whole; remaining budget to bullet then description"}
    if not need.any():
        return full, stats

    idx = np.flatnonzero(need)
    # cost of the priority skeleton per row that needs trimming
    skel = [render({**{l: vals[l][i] for l in PRIORITY}, "bullet": "", "description": ""})
            for i in idx]
    skel_len = np.array([len(x) for x in tokenizer(skel, add_special_tokens=False)["input_ids"]])
    out = list(full)
    for k, i in enumerate(idx):
        remain = max(0, budget - int(skel_len[k]) - 8)   # 8 = separator/special slack
        parts = {l: vals[l][i] for l, _ in FIELDS}
        for fld in BUDGETED:
            if remain <= 0:
                parts[fld] = ""
                continue
            ids = tokenizer(parts[fld], add_special_tokens=False)["input_ids"]
            if len(ids) <= remain:
                remain -= len(ids)
            else:
                parts[fld] = tokenizer.decode(ids[:remain], skip_special_tokens=True)
                remain = 0
        out[i] = render(parts)
    stats["priority_skeleton_len_mean"] = round(float(skel_len.mean()), 2)
    stats["priority_skeleton_overflow_rows"] = int((skel_len >= budget).sum())
    return out, stats


# ------------------------------------------------------------------ runners
def run_bge(df, docs, model_id, device, batch_size, budget, log):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    qs = df["query"].astype(str).tolist()
    scores = np.empty(len(df), dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(df), batch_size):
            e = min(s + batch_size, len(df))
            enc = tok(qs[s:e], docs[s:e], padding=True, truncation="only_second",
                      max_length=budget, return_tensors="pt").to(device)
            scores[s:e] = mdl(**enc).logits.view(-1).float().cpu().numpy()
            if (s // batch_size) % log == 0:
                print(f"    {e}/{len(df)}  {e/(time.time()-t0):.0f} rows/s", flush=True)
    return scores, {"unit": "pairs", "n_units": int(len(df))}


def run_qwen(df, docs, model_id, device, batch_size, budget, log):
    """Official `transformers` formulation, transcribed from the Qwen3-Reranker model card:
    chat-template prefix/suffix + P(yes) vs P(no) on the final-position logits.

    The card's alternative sentence_transformers.CrossEncoder path fails on this
    stack ("Cannot handle batch sizes > 1 if no padding token is defined"), so the
    canonical transformers path is used instead -- same formulation, explicit padding.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    tid_yes, tid_no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")

    task = ("Given a web search query, retrieve relevant passages that answer the query")
    prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
              'the Query and the Instruct provided. Note that the answer can only be "yes" or '
              '"no".<|im_end|>\n<|im_start|>user\n')
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pre_ids = tok.encode(prefix, add_special_tokens=False)
    suf_ids = tok.encode(suffix, add_special_tokens=False)
    inner = budget - len(pre_ids) - len(suf_ids)

    qs = df["query"].astype(str).tolist()
    texts = [f"<Instruct>: {task}\n<Query>: {q}\n<Document>: {d}" for q, d in zip(qs, docs)]
    scores = np.empty(len(df), dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            e = min(s + batch_size, len(texts))
            enc = tok(texts[s:e], padding=False, truncation="longest_first",
                      max_length=inner, return_attention_mask=False)
            enc["input_ids"] = [pre_ids + x + suf_ids for x in enc["input_ids"]]
            enc = tok.pad(enc, padding=True, return_tensors="pt").to(device)
            lg = mdl(**enc).logits[:, -1, :]
            pair = torch.stack([lg[:, tid_no], lg[:, tid_yes]], dim=1).float()
            scores[s:e] = torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().numpy()
            if (s // batch_size) % log == 0:
                print(f"    {e}/{len(texts)}  {e/(time.time()-t0):.1f} rows/s", flush=True)
    return scores, {"unit": "pairs", "n_units": int(len(df)),
                    "formulation": "P(yes) via log_softmax over [no, yes] final-token logits"}


def run_jina(df, docs, model_id, device, batch_size, budget, log):
    """LISTWISE: one model.rerank() call per query over its whole candidate set."""
    import torch
    from transformers import AutoModel
    mdl = AutoModel.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
    mdl = mdl.to(device).eval()
    scores = np.empty(len(df), dtype=np.float64)
    pos = np.arange(len(df))
    groups = df.groupby("query_id", sort=False).indices
    t0, done, cand = time.time(), 0, []
    with torch.no_grad():
        for qid, rows in groups.items():
            rows = np.asarray(rows)
            q = str(df["query"].iloc[rows[0]])
            dl = [docs[i] for i in rows]
            cand.append(len(dl))
            res = mdl.rerank(q, dl)
            for r in res:
                scores[rows[r["index"]]] = float(r["relevance_score"])
            done += 1
            if done % log == 0:
                print(f"    {done}/{len(groups)} queries  {done/(time.time()-t0):.2f} q/s",
                      flush=True)
            if done % 50 == 0 and device == "mps":
                torch.mps.empty_cache()   # listwise contexts are large; keep the allocator tidy
    return scores, {"unit": "queries", "n_units": int(len(groups)),
                    "avg_candidates_per_query": round(float(np.mean(cand)), 2),
                    "max_candidates_in_one_call": int(max(cand)),
                    "chunking_used": False,
                    "chunking_note": "no chunking: the largest candidate set (94 docs) fits "
                                     "comfortably in the 131K context"}


def run_jina_mlx(df, docs, model_id, device, batch_size, budget, log):
    """LISTWISE via the OFFICIAL MLX build (jinaai/jina-reranker-v3.5-mlx).

    Why MLX instead of the transformers path: on this machine the transformers/MPS
    path OOMs above 32 candidates at a 512-token budget (measured: 8->5.4 GB,
    16->15.3, 24->25.5, 32->51.4, 40 OOM), while the DEV pool needs up to 94.
    MLX is Jina's own recommended Apple Silicon implementation, handles all 94 in
    one pass, and is faster (2.8-7.1 docs/s vs 1.8-3.4).

    Cross-validated against the transformers path on identical input: scores agree
    within ~0.01, Spearman 0.90, no NaN; the only order difference was a genuine
    near-tie (0.3083 vs 0.3130). The RuntimeWarnings emitted by the vendor's
    rerank.py cosine lines are the same spurious numpy/BLAS FP-flag artifact seen
    elsewhere in this repo, not real numerical failure -- hence suppressed here.
    """
    import warnings
    from huggingface_hub import snapshot_download
    p = snapshot_download(model_id + "-mlx")
    if p not in sys.path:
        sys.path.insert(0, p)
    from rerank import MLXReranker
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = MLXReranker(model_path=p, projector_path=os.path.join(p, "projector.safetensors"))

    scores = np.empty(len(df), dtype=np.float64)
    groups = df.groupby("query_id", sort=False).indices
    t0, done, cand = time.time(), 0, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for qid, rows in groups.items():
            rows = np.asarray(rows)
            q = str(df["query"].iloc[rows[0]])
            dl = [docs[i] for i in rows]
            cand.append(len(dl))
            for r in mdl.rerank(q, dl):
                scores[rows[r["index"]]] = float(r["relevance_score"])
            done += 1
            if done % log == 0:
                print(f"    {done}/{len(groups)} queries  {done/(time.time()-t0):.2f} q/s",
                      flush=True)
    assert not np.isnan(scores).any(), "MLX produced NaN scores"
    return scores, {"unit": "queries", "n_units": int(len(groups)),
                    "avg_candidates_per_query": round(float(np.mean(cand)), 2),
                    "max_candidates_in_one_call": int(max(cand)),
                    "chunking_used": False,
                    "chunking_note": "no chunking: MLX handles the full candidate set "
                                     "(largest 94 docs) in a single listwise pass",
                    "implementation": "official MLX build (jina-reranker-v3.5-mlx)",
                    "implementation_reason": "transformers/MPS OOMs above 32 candidates"}


RUNNERS = {"bge": run_bge, "qwen": run_qwen,
           "jina": run_jina_mlx, "jina_torch": run_jina}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--input", required=True, choices=["plain", "structured"])
    ap.add_argument("--budget", type=int, default=512, help="token budget per (query,doc)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--subset", default=None,
                    help="funnel subset name (screen300 | confirm1000) or a path to a "
                         "JSON file with a query_ids list. Omit to run the full DEV pool.")
    ap.add_argument("--max_queries", type=int, default=None, help="tiny smoke test only")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")

    spec = MODELS[args.model]
    suffix = (f"_{args.subset}" if args.subset else "") + ("_smoke" if args.max_queries else "")
    tag = args.tag or f"{args.model}_{args.input}{suffix}"
    out = os.path.join(V3, tag)
    os.makedirs(out, exist_ok=True)

    sc = load_task1_scorer()
    df = pd.read_parquet(DEV_TEXT)
    _assert_dev_only(DEV_TEXT, df)                      # <-- TEST LOCK
    print(f"[{tag}] DEV pool OK: {len(df)} rows / {df['query_id'].nunique()} queries", flush=True)

    # ---- funnel subset: identical fixed query set for every model ----
    subset_meta = {"subset": None, "n_queries": int(df["query_id"].nunique()),
                   "n_rows": int(len(df))}
    if args.subset:
        p = args.subset if os.path.sep in args.subset else os.path.join(
            V3, f"subset_{args.subset}.json")
        if not os.path.exists(p):
            raise SystemExit(f"subset file not found: {p} "
                             "(run scripts/v3_build_funnel_subsets.py first)")
        sm = json.load(open(p))
        ids = set(sm["query_ids"])
        df = df[df["query_id"].isin(ids)].reset_index(drop=True)
        if df["query_id"].nunique() != sm["n_queries"] or len(df) != sm["n_rows"]:
            raise SystemExit(f"subset mismatch: got {len(df)} rows / "
                             f"{df['query_id'].nunique()} queries, expected "
                             f"{sm['n_rows']} / {sm['n_queries']}")
        subset_meta = {"subset": args.subset, "subset_file": os.path.relpath(p, ROOT),
                       "sha256_of_ids": sm["sha256_of_ids"],
                       "n_queries": sm["n_queries"], "n_rows": sm["n_rows"],
                       "locale_counts": sm["locale_counts"]}
        print(f"[{tag}] subset '{args.subset}': {len(df)} rows / "
              f"{df['query_id'].nunique()} queries  locales={sm['locale_counts']}  "
              f"ids_sha256={sm['sha256_of_ids'][:12]}", flush=True)

    if args.max_queries:
        keep = np.sort(df["query_id"].unique())[:args.max_queries]
        df = df[df["query_id"].isin(set(keep))].reset_index(drop=True)
        print(f"[{tag}] SMOKE subset: {len(df)} rows / {df['query_id'].nunique()} queries",
              flush=True)

    from transformers import AutoTokenizer
    tok_for_budget = AutoTokenizer.from_pretrained(spec["id"], trust_remote_code=True)
    t_text = time.time()
    docs, trunc = build_docs(df, args.input, tok_for_budget, args.budget)
    text_secs = time.time() - t_text
    print(f"[{tag}] text built in {text_secs:.0f}s; truncated "
          f"{100*trunc['truncation_fraction']:.2f}% (budget {args.budget})", flush=True)

    t0 = time.time()
    scores, unit = RUNNERS[args.model](df, docs, spec["id"], device,
                                       args.batch_size, args.budget, args.log_every)
    infer_secs = time.time() - t0

    peak = None
    try:
        if device == "mps":
            peak = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        elif device == "cuda":
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass

    pred = df[["query_id", "product_id", "product_locale", "esci_label", "gain"]].copy()
    pred["score"] = scores
    assert pred["score"].notna().all(), "NaN scores"

    res = sc.evaluate(pred, "score", gain_col="gain", query_col="query_id")
    tbl = sc.per_query_ndcg_table(pred, "score", gain_col="gain", query_col="query_id")
    qloc = pred.drop_duplicates("query_id").set_index("query_id")["product_locale"]
    cpq = pred.groupby("query_id").size()

    keep = tbl[tbl["idcg_full"] > 0]
    lv = qloc.reindex(keep.index)
    locale = {}
    for lc in ["us", "es", "jp"]:
        s = keep[lv == lc]
        locale[lc] = {"query_count": int(len(s)),
                      "ndcg_full": round(float(s["ndcg_full"].mean()), 6) if len(s) else None,
                      "ndcg_at_20": round(float(s["ndcg_at_20"].mean()), 6) if len(s) else None,
                      "ndcg_at_10": round(float(s["ndcg_at_10"].mean()), 6) if len(s) else None}
    locale["overall"] = {"query_count": res["n_queries_scored"],
                         "ndcg_full": round(res["ndcg_full"], 6),
                         "ndcg_at_20": round(res["ndcg_at_20"], 6),
                         "ndcg_at_10": round(res["ndcg_at_10"], 6)}

    summary = {
        "tag": tag, "model_key": args.model, "model_id": spec["id"],
        "params": spec["params"], "architecture": spec["arch"], "kind": spec["kind"],
        "input": args.input, "budget_tokens": args.budget,
        "zero_shot": True, "trained_in_this_phase": False,
        "split": "dev", "test_split_touched": False, "is_smoke": bool(args.max_queries),
        "funnel": subset_meta,
        "rows": int(len(df)), "queries": int(df["query_id"].nunique()),
        "ndcg_full": round(res["ndcg_full"], 6),
        "ndcg_at_20": round(res["ndcg_at_20"], 6),
        "ndcg_at_10": round(res["ndcg_at_10"], 6),
        "candidate_pool": res["candidate_pool"], "metric_version": res["metric_version"],
        "zero_idcg_policy": res["zero_idcg_policy"],
        "zero_idcg_query_count": res["zero_idcg_query_count"],
        "scorer": os.path.relpath(sc.__file__, ROOT),
        "locale": locale, "truncation": trunc,
        "score_stats": {"min": float(scores.min()), "max": float(scores.max()),
                        "mean": float(scores.mean()), "std": float(scores.std())},
        "candidates_per_query": {"mean": round(float(cpq.mean()), 2),
                                 "p50": float(cpq.quantile(.5)), "p90": float(cpq.quantile(.9)),
                                 "p95": float(cpq.quantile(.95)), "max": int(cpq.max())},
    }
    runtime = {"tag": tag, "hardware": "Apple M4 Max, 68.7 GB unified", "device": device,
               "dtype": "auto (bf16 for qwen3-family, fp32 for xlm-r)",
               "batch_size": args.batch_size, "peak_device_memory_gb": peak,
               "text_build_seconds": round(text_secs, 1),
               "inference_seconds": round(infer_secs, 1),
               "rows": int(len(df)), "queries": int(df["query_id"].nunique()),
               "rows_per_sec": round(len(df) / infer_secs, 2),
               "queries_per_sec": round(df["query_id"].nunique() / infer_secs, 3), **unit}

    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=str)
    json.dump(runtime, open(os.path.join(out, "runtime.json"), "w"), indent=2, default=str)
    pq = tbl.copy()
    pq["locale"] = qloc.reindex(pq.index)
    pq["num_candidates"] = cpq.reindex(pq.index)
    pq.index.name = "query_id"
    pq.reset_index()[["query_id", "locale", "num_candidates", "ndcg_full",
                      "ndcg_at_20", "ndcg_at_10"]].to_csv(
        os.path.join(out, "per_query_ndcg.csv"), index=False)
    pred.to_parquet(os.path.join(out, "predictions.parquet"), index=False)

    print(f"\n[{tag}] full={summary['ndcg_full']:.4f}  @20={summary['ndcg_at_20']:.4f}  "
          f"@10={summary['ndcg_at_10']:.4f}   ({infer_secs:.0f}s, "
          f"{runtime['rows_per_sec']:.1f} rows/s, peak {peak} GB)")
    for lc in ["us", "es", "jp"]:
        m = locale[lc]
        print(f"    {lc}: full={m['ndcg_full']} @20={m['ndcg_at_20']} @10={m['ndcg_at_10']}")
    gc.collect()


if __name__ == "__main__":
    main()
