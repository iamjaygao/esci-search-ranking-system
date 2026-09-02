"""
Sections 5-7 -- build the Task 1 benchmark: features, train/dev/test split,
split-integrity gate, and the feature pool-dependency record.

Feature rule (resolution item 8, replacing the classify-or-STOP rule):
    JOIN     raw per-pair scores only
    RECOMPUTE everything derived from the candidate pool
    When in doubt, recompute. No STOP.
    STOP only if a derived feature needs an intermediate value that is
    unavailable and would require re-running RETRIEVAL.

In practice every one of the 17 features is recomputed on the Task 1 pool,
because the two headline signals are pool-dependent at every level:

  bm25_score      retrieval/bm25.py builds a BM25 index over THAT QUERY'S OWN
                  candidates (_score_single_group), so IDF and avgdl come from
                  the candidate set -- even the "raw" BM25 is pool-dependent --
                  and the stored value is additionally per-query min-max
                  normalised.
  semantic_score  per-query min-max over the pool; the raw cosine was never
                  persisted. Recomputed by build_task1_semantic_scores.py with
                  the FROZEN encoder (inference only, not a retrieval re-run).

Neither needs retrieval to be re-run: the candidate set is fixed by the ESCI
judgments. So there is no STOP.

Section 6 split: test = small_version==1 AND split=='test', frozen.
train/dev = query-level 85/15 of small_version==1 AND split=='train',
seed 42, stratified by product_locale at the query level.
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from task1_common import (  # noqa: E402
    BASE, LABEL_ORDINAL, OFFICIAL_GAIN, PRODUCTS, ROOT, load_examples,
    normalize_query_text,
)

sys.path.insert(0, ROOT)
from reranking.advanced_features import ALL_FEATURES  # noqa: E402

DATA = BASE
CACHE = os.path.join(BASE, "_cache")
ESCI_S = os.path.join(ROOT, "esci-data/esci-s_dataset/esci_s_products.parquet")

COLORS = {'red', 'black', 'blue', 'white', 'green', 'yellow', 'silver', 'gold',
          'grey', 'gray', 'pink', 'purple', 'brown'}

JOINED, RECOMPUTED, STOPS = [], [], []


def rec(name, category, reason):
    (JOINED if category == "joined" else RECOMPUTED).append(
        {"feature": name, "category": category, "reason": reason})


def _score_group_raw_and_minmax(query_id, group):
    """Transcribed from retrieval/bm25.py::_score_single_group, with two changes:
    the raw score is also returned (the original discards it), and the progress
    bars are silenced. The BM25 index is still built over THAT QUERY'S OWN
    candidates, exactly as the original does -- that is the property that makes
    even the raw score pool-dependent."""
    import bm25s
    query_text = group["query_text"].iloc[0]
    item_texts = group["item_text"].tolist()
    item_ids = group["item_id"].tolist()

    corpus_tokens = bm25s.tokenize(item_texts, show_progress=False)
    query_tokens = bm25s.tokenize([query_text], show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    doc_indices, raw_scores = retriever.retrieve(
        query_tokens, k=len(item_texts), show_progress=False)

    top_idx, top_raw = doc_indices[0], raw_scores[0]
    mn, mx = top_raw.min(), top_raw.max()
    norm = (top_raw - mn) / (mx - mn) if (mx - mn) > 1e-8 else np.zeros_like(top_raw)
    return [{"query_id": query_id, "product_id": item_ids[idx],
             "bm25_raw": float(top_raw[i]), "bm25_minmax": float(norm[i])}
            for i, idx in enumerate(top_idx)]


def compute_bm25_task1(df):
    """Recompute BM25 on the Task 1 candidate pool: raw + per-query min-max in
    one indexing pass."""
    from joblib import Parallel, delayed
    work = df[["query_id", "query_text", "item_id", "item_text"]]
    grouped = list(work.groupby("query_id"))
    print(f"  BM25 over {len(grouped)} queries ...", flush=True)
    nested = Parallel(n_jobs=-1, batch_size=200)(
        delayed(_score_group_raw_and_minmax)(qid, g) for qid, g in grouped)
    return pd.DataFrame([r for sub in nested for r in sub])


def main():
    os.makedirs(CACHE, exist_ok=True)

    # ---------------- pool ----------------
    print("Building Task 1 pool (locale-aware) ...", flush=True)
    ex = load_examples()
    t1 = ex[ex["small_version"] == 1].copy()
    n_anchor = len(t1)

    pr = pd.read_parquet(PRODUCTS, columns=[
        "product_id", "product_locale", "product_title", "product_description",
        "product_bullet_point", "product_brand", "product_color"])
    t1 = t1.merge(pr, on=["product_id", "product_locale"], how="left")
    assert len(t1) == n_anchor, f"product join fanned out: {len(t1)} != {n_anchor}"

    esci_s = pd.read_parquet(ESCI_S)
    idc = "asin" if "asin" in esci_s.columns else esci_s.columns[0]
    esci_s = esci_s.rename(columns={idc: "product_id"})
    n_es_raw = len(esci_s)
    esci_s = esci_s.drop_duplicates(subset="product_id", keep="first")
    t1 = t1.merge(esci_s[["product_id", "price", "stars", "ratings", "category"]],
                  on="product_id", how="left")
    assert len(t1) == n_anchor, f"ESCI-S join fanned out: {len(t1)} != {n_anchor}"
    print(f"  ESCI-S de-duplicated on asin: {n_es_raw} -> {len(esci_s)}", flush=True)

    t1["item_text"] = (t1["product_title"].fillna("") + " " +
                       t1["product_description"].fillna("") + " " +
                       t1["product_bullet_point"].fillna(""))
    t1["query_text"] = t1["query"]
    t1["item_id"] = t1["product_id"]

    # ---------------- bm25 (RECOMPUTE) ----------------
    bm_path = os.path.join(CACHE, "task1_bm25.parquet")
    if os.path.exists(bm_path):
        bm = pd.read_parquet(bm_path)
        print("  reusing cached task1_bm25.parquet", flush=True)
    else:
        bm = compute_bm25_task1(t1)
        bm.to_parquet(bm_path, index=False)
    t1 = t1.merge(bm, on=["query_id", "product_id"], how="left")
    assert len(t1) == n_anchor, "bm25 join fanned out"
    assert t1["bm25_minmax"].notna().all(), "missing BM25 rows"
    t1["bm25_score"] = t1["bm25_minmax"]
    rec("bm25_score", "recomputed",
        "retrieval/bm25.py scores against a per-QUERY mini index, so IDF and avgdl are "
        "candidate-set statistics; additionally min-max normalised per query. Pool-dependent "
        "at every level. Recomputed on the Task 1 pool; raw score persisted as bm25_raw.")

    # ---------------- semantic (RECOMPUTE) ----------------
    sem_path = os.path.join(CACHE, "task1_semantic_scores.parquet")
    if not os.path.exists(sem_path):
        STOPS.append("task1_semantic_scores.parquet missing -- run build_task1_semantic_scores.py")
        print("STOP:", STOPS[-1]); sys.exit(2)
    sem = pd.read_parquet(sem_path)
    t1 = t1.merge(sem[["example_id", "semantic_cosine_raw", "semantic_score"]],
                  on="example_id", how="left")
    assert len(t1) == n_anchor, "semantic join fanned out"
    assert t1["semantic_score"].notna().all(), "missing semantic rows"
    rec("semantic_score", "recomputed",
        "per-query min-max over the candidate pool; the raw cosine was never persisted by "
        "retrieval/two_tower.py so the frozen column cannot be inverted or re-normalised. "
        "Re-encoded with the FROZEN checkpoint (inference only, not a retrieval re-run); "
        "raw cosine persisted as semantic_cosine_raw.")

    # ---------------- query features ----------------
    t1["query_length"] = t1["query"].astype(str).apply(lambda x: len(x.split()))
    rec("query_length", "joined", "function of the query string alone; pool-independent")

    def parse_budget(q):
        m = re.search(r'under\s*\$?(\d+)', str(q).lower())
        return float(m.group(1)) if m else -1.0
    t1["user_budget"] = t1["query"].apply(parse_budget)
    rec("user_budget", "joined", "regex on the query string; pool-independent")
    t1["cheap_intent"] = t1["query"].str.lower().str.contains('cheap|affordable|budget').astype(float)
    rec("cheap_intent", "joined", "regex on the query string; pool-independent")

    # idf_map rebuilt from TASK 1 TRAIN queries (the analogue of the original,
    # which built it from the large-version train queries)
    tq = t1[t1["split"] == "train"][["query_id", "query"]].drop_duplicates()
    words = " ".join(tq["query"].str.lower().tolist()).split()
    N = len(tq)
    wc = Counter(words)
    idf_map = {w: np.log((N + 1) / (c + 1)) + 1 for w, c in wc.items()}

    def idf_stats(q):
        ws = str(q).lower().split()
        if not ws:
            return 0.0, 0.0
        v = [idf_map.get(w, 10.0) for w in ws]
        return float(np.mean(v)), float(np.max(v))
    st = t1["query"].apply(idf_stats)
    t1["query_mean_idf"] = [x[0] for x in st]
    t1["query_max_idf"] = [x[1] for x in st]
    for f in ["query_mean_idf", "query_max_idf"]:
        rec(f, "recomputed",
            f"IDF table rebuilt from the {N} Task 1 TRAIN queries (the original was built from "
            "large-version train queries); train-set-dependent, so it must track the benchmark")

    # ---------------- item features ----------------
    t1["category"] = t1["category"].fillna("Unknown").astype(str)
    t1["price"] = pd.to_numeric(
        t1["price"].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors="coerce")
    t1["ratings"] = pd.to_numeric(t1["ratings"], errors="coerce")

    t1["is_price_missing"] = t1["price"].isna().astype(float)
    rec("is_price_missing", "joined", "null indicator on the raw attribute; pool-independent")
    cmp_ = t1.groupby("category")["price"].transform("median")
    gmp = t1["price"].median() if not t1["price"].isna().all() else 25.0
    imputed_price = t1["price"].fillna(cmp_).fillna(gmp).fillna(0.0)
    t1["log_price"] = np.log1p(imputed_price)
    rec("log_price", "recomputed",
        "category-median then global-median imputation are IN-POOL statistics, so the value "
        "changes with the candidate population")

    t1["stars_clean"] = t1["stars"].astype(str).str.extract(r'([\d\.]+)').astype(float)
    t1["is_rating_missing"] = t1["stars_clean"].isna().astype(float)
    rec("is_rating_missing", "joined", "null indicator on the raw attribute; pool-independent")
    cms = t1.groupby("category")["stars_clean"].transform("median")
    gms = t1["stars_clean"].median() if not t1["stars_clean"].isna().all() else 4.0
    t1["stars_clean"] = t1["stars_clean"].fillna(cms).fillna(gms).fillna(0.0)
    rec("stars_clean", "recomputed", "category-median imputation is an in-pool statistic")

    cmr = t1.groupby("category")["ratings"].transform("median")
    gmr = t1["ratings"].median() if not t1["ratings"].isna().all() else 0.0
    imputed_ratings = t1["ratings"].fillna(cmr).fillna(gmr).fillna(0.0)
    t1["log_review_count"] = np.log1p(imputed_ratings)
    rec("log_review_count", "recomputed",
        "category-median imputation is an in-pool statistic. NOTE: ESCI-S `ratings` is a string "
        "like '1,116 ratings', so pd.to_numeric yields NaN for every row and this feature "
        "collapses to log1p(0)=0 -- the Phase 0 P1-1 defect, reproduced faithfully here because "
        "constraint 4 forbids changing feature definitions in this phase.")

    # ---------------- interaction ----------------
    stem = PorterStemmer()
    uq = t1["query"].astype(str).unique()
    ut = t1["product_title"].astype(str).unique()
    qsm = {q: set(stem.stem(w) for w in q.lower().split()) for q in uq}
    tsm = {t: set(stem.stem(w) for w in t.lower().split()) for t in ut}

    def overlap(q, t):
        a = qsm.get(q, set())
        if not a:
            return 0.0
        return len(a & tsm.get(t, set())) / len(a)
    t1["word_overlap"] = [overlap(str(q), str(t)) for q, t in
                          zip(t1["query"], t1["product_title"])]
    rec("word_overlap", "joined", "stemmed overlap of query tokens with the product title; "
                                  "function of the (query, product) pair alone")

    t1["is_over_budget"] = ((t1["user_budget"] > 0) & (imputed_price > t1["user_budget"])).astype(float)
    rec("is_over_budget", "recomputed", "derived from the in-pool imputed price")

    def brand(q, b):
        return 0.0 if pd.isna(b) else (1.0 if str(b).lower() in str(q).lower() else 0.0)
    t1["brand_match"] = [brand(q, b) for q, b in zip(t1["query"], t1["product_brand"])]
    rec("brand_match", "joined", "substring test on the (query, brand) pair; pool-independent")

    def color(q, t):
        qc = set(str(q).lower().split()) & COLORS
        return 0.0 if not qc else (1.0 if qc & set(str(t).lower().split()) else 0.0)
    t1["color_match"] = [color(q, t) for q, t in zip(t1["query"], t1["product_title"])]
    rec("color_match", "joined", "colour term shared by query and title; pool-independent")

    top20 = t1.sort_values(["query_id", "bm25_score"], ascending=[True, False]).groupby("query_id").head(20)
    dom = top20.groupby("query_id")["category"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    t1["is_dominant_category"] = (t1["category"] == t1["query_id"].map(dom)).astype(float)
    rec("is_dominant_category", "recomputed",
        "mode of the category among that query's top-20 BM25 candidates -- an explicit in-pool "
        "statistic that also depends on the recomputed BM25 ordering")

    # ---------------- labels ----------------
    t1["gain"] = t1["esci_label"].map(OFFICIAL_GAIN).astype(float)
    t1["lgb_label"] = t1["esci_label"].map(LABEL_ORDINAL).astype(int)
    t1["doc_id"] = t1["product_locale"].astype(str) + "_" + t1["product_id"].astype(str)

    missing = [f for f in ALL_FEATURES if f not in t1.columns]
    assert not missing, f"missing features: {missing}"
    nan_counts = {f: int(t1[f].isna().sum()) for f in ALL_FEATURES if t1[f].isna().any()}
    if nan_counts:
        STOPS.append(f"NaN in features after computation: {nan_counts}")
        print("STOP:", STOPS[-1]); sys.exit(2)

    # ---------------- section 6 split ----------------
    print("\nSplitting train/dev (query-level, seed 42, stratified by locale) ...", flush=True)
    tr_all = t1[t1["split"] == "train"]
    te = t1[t1["split"] == "test"].copy()
    qdf = tr_all[["query_id", "product_locale"]].drop_duplicates("query_id").sort_values("query_id")
    train_q, dev_q = train_test_split(
        qdf["query_id"].tolist(), test_size=0.15, random_state=42,
        stratify=qdf["product_locale"].tolist())
    train_q, dev_q = set(train_q), set(dev_q)
    tr = tr_all[tr_all["query_id"].isin(train_q)].copy()
    dv = tr_all[tr_all["query_id"].isin(dev_q)].copy()
    assert len(tr) + len(dv) == len(tr_all), "train/dev partition lost rows"

    keep = (["example_id", "query_id", "query", "product_id", "product_locale", "doc_id",
             "esci_label", "gain", "lgb_label", "split",
             "product_title", "product_brand", "category",
             "bm25_raw", "semantic_cosine_raw"] + ALL_FEATURES)
    keep = list(dict.fromkeys(keep))

    frames = {}
    for name, d in [("train", tr), ("dev", dv), ("test", te)]:
        d = d[keep].sort_values(["query_id", "example_id"]).reset_index(drop=True)
        d.to_parquet(os.path.join(DATA, f"{name}_task1.parquet"), index=False)
        frames[name] = d
        print(f"  {name}_task1.parquet: {len(d)} rows / {d['query_id'].nunique()} queries", flush=True)

    # ---------------- split integrity (STOP-gate) ----------------
    print("\nSplit integrity ...", flush=True)
    qs = {k: set(v["query_id"]) for k, v in frames.items()}
    si = {"query_counts": {k: len(v) for k, v in qs.items()},
          "row_counts": {k: int(len(v)) for k, v in frames.items()},
          "overlaps": {}, "duplicates": {}, "locale_distribution": {},
          "depth_distribution": {}, "label_distribution": {}}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        ov = qs[a] & qs[b]
        si["overlaps"][f"{a}__{b}"] = {"size": len(ov), "PASS": len(ov) == 0,
                                       "sample": sorted(ov)[:20]}
        if ov:
            STOPS.append(f"{a}/{b} query overlap = {len(ov)}: {sorted(ov)[:20]}")
    for k, d in frames.items():
        dup = d.duplicated(subset=["query_id", "product_id", "product_locale"], keep=False)
        n = int(dup.sum())
        si["duplicates"][k] = {
            "count": n, "PASS": n == 0,
            "sample": (d.loc[dup, ["query_id", "product_id", "product_locale", "esci_label"]]
                       .head(10).to_dict("records") if n else [])}
        if n:
            STOPS.append(f"{k} has {n} duplicate (query_id, product_id, product_locale) rows")
        qd = d.drop_duplicates("query_id")["product_locale"].value_counts(normalize=True)
        si["locale_distribution"][k] = {loc: round(float(qd.get(loc, 0)), 4)
                                        for loc in ["us", "es", "jp"]}
        dep = d.groupby("query_id").size()
        si["depth_distribution"][k] = {
            "mean": round(float(dep.mean()), 4), "median": float(dep.median()),
            "p5": float(dep.quantile(.05)), "p95": float(dep.quantile(.95)),
            "min": int(dep.min()), "max": int(dep.max())}
        lv = d["esci_label"].value_counts(normalize=True)
        si["label_distribution"][k] = {l: round(100 * float(lv.get(l, 0)), 4)
                                       for l in ["E", "S", "C", "I"]}
    si["train_dev_locale_match"] = si["locale_distribution"]["train"] == si["locale_distribution"]["dev"]
    si["split_method"] = ("query-level train_test_split(test_size=0.15, random_state=42, "
                          "stratify=product_locale); a query belongs to exactly one split; "
                          "no row-level splitting")
    si["STOPS"] = STOPS
    si["PASS"] = len(STOPS) == 0
    with open(os.path.join(DATA, "split_integrity.json"), "w") as f:
        json.dump(si, f, indent=2, default=str)
    for k, v in si["overlaps"].items():
        print(f"  overlap {k}: {v['size']} -> {'PASS' if v['PASS'] else 'FAIL'}")
    for k, v in si["duplicates"].items():
        print(f"  duplicates {k}: {v['count']} -> {'PASS' if v['PASS'] else 'FAIL'}")
    print(f"  locale dist: {si['locale_distribution']}")
    print(f"  depth mean:  {{k: v['mean'] for k, v in si['depth_distribution'].items()}}"
          .replace("{k: v['mean'] for k, v in si['depth_distribution'].items()}",
                   str({k: v["mean"] for k, v in si["depth_distribution"].items()})))

    if STOPS:
        print("\n=== STOP ===")
        for s in STOPS:
            print(" -", s)
        sys.exit(2)

    # ---------------- feature pool-dependency record ----------------
    audit = {
        "rule_applied": ("resolution item 8: join raw per-pair scores only; recompute "
                         "everything derived from the candidate pool; when in doubt, "
                         "recompute; this is a RECORD, not a gate"),
        "downstream_path_from_gate_a": "REUSE (A1 TRUE) -- but see note",
        "note_on_reuse": (
            "Gate A established that small_version==1 rows are a strict subset of "
            "large_version==1, so per-row values COULD be joined by key. In practice nothing "
            "was joined from the large-version artifacts: the only two stored per-row scores "
            "(output/bm25_scores_*.csv, output/two_tower_scores_*.csv) are both per-query "
            "min-max normalised over the LARGE pool, which is exactly the pool-dependent case "
            "the rule says to recompute. The 'joined' features below are computed directly "
            "from (query, product) attributes rather than copied from a large-version file."),
        "joined_pool_independent": JOINED,
        "recomputed_pool_dependent": RECOMPUTED,
        "counts": {"joined": len(JOINED), "recomputed": len(RECOMPUTED),
                   "total": len(JOINED) + len(RECOMPUTED)},
        "feature_list": ALL_FEATURES,
        "raw_scores_persisted_for_the_first_time": {
            "bm25_raw": "un-normalised BM25 from the per-query mini index",
            "semantic_cosine_raw": "un-normalised cosine similarity from the frozen encoder",
        },
        "retrieval_rerun_required": False,
        "stop_triggered": False,
        "esci_s_join": {
            "key": "product_id only (ESCI-S has no locale column)",
            "rows_before_dedup": n_es_raw, "rows_after_dedup": int(len(esci_s)),
            "consequence": "a product_id present in several locales receives the same "
                           "price/stars/category; a limitation of ESCI-S, recorded not fixed",
        },
        "idf_map": {"built_from": "Task 1 TRAIN queries", "n_queries": int(N),
                    "vocab": len(idf_map)},
    }
    with open(os.path.join(DATA, "feature_pool_dependency_audit.json"), "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nfeature audit: {len(JOINED)} joined / {len(RECOMPUTED)} recomputed")
    print("wrote train/dev/test_task1.parquet, split_integrity.json, "
          "feature_pool_dependency_audit.json")


if __name__ == "__main__":
    main()
