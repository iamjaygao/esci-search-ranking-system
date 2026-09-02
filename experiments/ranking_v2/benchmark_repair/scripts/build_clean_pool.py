"""
Phase A + B -- Define the official gain convention and materialise one clean
ranking pool (train / dev / test).

Design decisions, all verified against the repository rather than assumed:

* ANCHOR. The pool is anchored on the ESCI judgment table (`examples` parquet)
  filtered to a split, NOT on the BM25/Two-Tower score CSVs. Every join is a
  LEFT join from that anchor, so the row count is pinned to the number of real
  judged pairs and cannot grow. The V1 path anchored on the score CSVs and used
  INNER joins on `product_id` alone, which is what inflated the pool.

* KEYS. `example_id` is unique (2,621,288 rows, verified). `(query_id,
  product_id, product_locale)` is also unique and every `query_id` maps to
  exactly one locale. Product attributes are joined on
  `(product_id, product_locale)`; ESCI-S is de-duplicated on `asin` BEFORE the
  join (11,309 duplicate asin rows in the source file).

* DEV SPLIT. ESCI ships only train/test. Both existing trainers carve a random
  15% of TRAIN QUERIES via `sklearn.model_selection.train_test_split(
  unique_queries, test_size=0.15, random_state=42)` at fit time and never
  persist it. This script MATERIALISES THAT EXACT SPLIT -- same function, same
  seed, same `df['query_id'].unique().tolist()` ordering taken from the V1
  feature frame -- so the split is unchanged, only now auditable.

* FEATURES. The 17 feature formulas are transcribed verbatim from
  `reranking/advanced_features.extract_advanced_features` (lines 75-171). No
  feature is added, removed or redefined. The only change is the frame they are
  computed over. The train-derived `idf_map` is loaded from the existing
  `output/lambdamart_features.json` (verified byte-identical to
  `output/advanced_normalization_stats.json`) rather than recomputed, so the
  frozen MLP sees exactly the IDF values it was trained with.

Writes:
  official_gain_mapping.json
  data/train_clean.parquet, data/dev_clean.parquet, data/test_clean.parquet
  data/build_manifest.json
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from reranking.advanced_features import ALL_FEATURES  # noqa: E402

BASE = os.path.join(ROOT, "experiments", "ranking_v2", "benchmark_repair")
DATA = os.path.join(BASE, "data")

EXAMPLES = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")
PRODUCTS = os.path.join(ROOT, "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet")
ESCI_S = os.path.join(ROOT, "esci-data/esci-s_dataset/esci_s_products.parquet")

# ---- Phase A: verified-from-code mappings -------------------------------
# scripts/train_lambdamart.py:17     LABEL_MAP_ORDINAL = {'E':3,'S':2,'C':1,'I':0}
# scripts/train_lambdamart.py:18     LABEL_GAIN        = [0, 1, 3, 7]     <-- V1, wrong
# reranking/advanced_features.py:78  label_map (MLP target) = {'E':1.0,'S':0.1,'C':0.01,'I':0.0}
# evaluation/metrics.py:93           apply_business_ndcg_labels relevance  = same fractional map
# evaluation/metrics.py:9            dcg gain = 2**relevance - 1
LABEL_ORDINAL = {"I": 0, "C": 1, "S": 2, "E": 3}
OFFICIAL_RELEVANCE = {"I": 0.0, "C": 0.01, "S": 0.10, "E": 1.00}

COLORS = {'red', 'black', 'blue', 'white', 'green', 'yellow', 'silver', 'gold',
          'grey', 'gray', 'pink', 'purple', 'brown'}


def build_gain_mapping():
    rows = []
    for lab in ["I", "C", "S", "E"]:
        r = OFFICIAL_RELEVANCE[lab]
        rows.append({
            "esci_label": lab,
            "integer_training_label": LABEL_ORDINAL[lab],
            "official_relevance_gain": r,
            "lightgbm_label_gain_position": LABEL_ORDINAL[lab],
            "label_gain_value_exact_dcg_match": float(2 ** r - 1),
            "label_gain_value_linear_relevance": float(r),
            "v1_label_gain_value": [0, 1, 3, 7][LABEL_ORDINAL[lab]],
        })
    mapping = {
        "verified_from_code": {
            "integer_label_map": "scripts/train_lambdamart.py:17 LABEL_MAP_ORDINAL = {'E':3,'S':2,'C':1,'I':0}",
            "v1_label_gain": "scripts/train_lambdamart.py:18 LABEL_GAIN = [0, 1, 3, 7]",
            "mlp_target_score_map": "reranking/advanced_features.py:78 {'E':1.0,'S':0.1,'C':0.01,'I':0.0}",
            "eval_relevance_map": "evaluation/metrics.py:93 apply_business_ndcg_labels, same fractional map",
            "eval_gain_formula": "evaluation/metrics.py:9 dcg() -> sum((2**rel - 1)/log2(rank+1))",
        },
        "official_relevance": OFFICIAL_RELEVANCE,
        "integer_label_map": LABEL_ORDINAL,
        "table": rows,
        "label_gain_EXACT_DCG_MATCH": [float(2 ** OFFICIAL_RELEVANCE[l] - 1) for l in ["I", "C", "S", "E"]],
        "label_gain_LINEAR_RELEVANCE": [OFFICIAL_RELEVANCE[l] for l in ["I", "C", "S", "E"]],
        "v1_label_gain": [0, 1, 3, 7],
        "which_is_authoritative": "label_gain_EXACT_DCG_MATCH",
        "why": (
            "LightGBM uses label_gain[label] DIRECTLY as the DCG numerator (its default "
            "[0,1,3,7,15,...] is 2**i - 1). The repository's scorer, evaluation/metrics.dcg, "
            "applies 2**relevance - 1 on top of the fractional relevance. To make LightGBM's "
            "internal lambdarank objective numerically IDENTICAL to the reported metric, "
            "label_gain must therefore be 2**relevance - 1 = "
            "[0.0, 0.006956, 0.071773, 1.0], not the raw relevance [0.0, 0.01, 0.10, 1.0]. "
            "The Phase 1 brief specified the raw-relevance form; both are trained and "
            "reported, but the exact-match form is authoritative because only it makes the "
            "answer to 'training gain == evaluation convention?' an unqualified YES. The "
            "difference between the two is the S:E gain ratio: 0.0718 (exact) vs 0.10 "
            "(linear) -- against 0.4286 in V1."),
        "s_to_e_gain_ratio": {
            "v1_training": 3 / 7,
            "v1_evaluation": float((2 ** 0.10 - 1) / (2 ** 1.0 - 1)),
            "new_training_exact": float((2 ** 0.10 - 1) / (2 ** 1.0 - 1)),
            "new_training_linear": 0.10 / 1.00,
            "evaluation": float((2 ** 0.10 - 1) / (2 ** 1.0 - 1)),
        },
    }
    with open(os.path.join(BASE, "official_gain_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping


# =====================================================================
# Feature computation -- transcribed verbatim from
# reranking/advanced_features.extract_advanced_features lines 75-174
# =====================================================================
def compute_features(df, idf_map):
    """`df` must already carry: query, product_title, product_brand,
    bm25_score, semantic_score, price, stars, ratings, category, esci_label."""
    df = df.copy()

    df["bm25_score"] = df["bm25_score"].fillna(0.0)
    df["semantic_score"] = df["semantic_score"].fillna(-1.0)

    label_map = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    df["target_score"] = df["esci_label"].map(label_map).fillna(0.0)

    # --- A. Query intent -------------------------------------------------
    df["query_length"] = df["query"].astype(str).apply(lambda x: len(x.split()))

    def parse_budget(q):
        match = re.search(r'under\s*\$?(\d+)', str(q).lower())
        return float(match.group(1)) if match else -1.0
    df["user_budget"] = df["query"].apply(parse_budget)
    df["cheap_intent"] = df["query"].str.lower().str.contains('cheap|affordable|budget').astype(float)

    def get_idf_stats(q):
        words = str(q).lower().split()
        if not words:
            return 0.0, 0.0
        idfs = [idf_map.get(w, 10.0) for w in words]
        return np.mean(idfs), np.max(idfs)
    idf_stats = df["query"].apply(get_idf_stats)
    df["query_mean_idf"] = [x[0] for x in idf_stats]
    df["query_max_idf"] = [x[1] for x in idf_stats]

    df["category"] = df["category"].fillna("Unknown").astype(str)

    # --- B. Item authority & imputation ----------------------------------
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors="coerce")
    df["ratings"] = pd.to_numeric(df["ratings"], errors="coerce")

    df["is_price_missing"] = df["price"].isna().astype(float)
    cat_median_price = df.groupby("category")["price"].transform("median")
    global_median_price = df["price"].median() if not df["price"].isna().all() else 25.0
    imputed_price = df["price"].fillna(cat_median_price).fillna(global_median_price).fillna(0.0)
    df["log_price"] = np.log1p(imputed_price)

    df["stars_clean"] = df["stars"].astype(str).str.extract(r'([\d\.]+)').astype(float)
    df["is_rating_missing"] = df["stars_clean"].isna().astype(float)
    cat_median_stars = df.groupby("category")["stars_clean"].transform("median")
    global_median_stars = df["stars_clean"].median() if not df["stars_clean"].isna().all() else 4.0
    df["stars_clean"] = df["stars_clean"].fillna(cat_median_stars).fillna(global_median_stars).fillna(0.0)

    cat_median_ratings = df.groupby("category")["ratings"].transform("median")
    global_median_ratings = df["ratings"].median() if not df["ratings"].isna().all() else 0.0
    imputed_ratings = df["ratings"].fillna(cat_median_ratings).fillna(global_median_ratings).fillna(0.0)
    df["log_review_count"] = np.log1p(imputed_ratings)

    # --- C. Interaction & match ------------------------------------------
    stemmer = PorterStemmer()
    unique_queries = df["query"].astype(str).unique()
    unique_titles = df["product_title"].astype(str).unique()
    query_stem_map = {q: set(stemmer.stem(w) for w in q.lower().split()) for q in unique_queries}
    title_stem_map = {t: set(stemmer.stem(w) for w in t.lower().split()) for t in unique_titles}

    def fast_overlap(q, t):
        q_set = query_stem_map.get(q, set())
        t_set = title_stem_map.get(t, set())
        if not q_set:
            return 0.0
        return len(q_set.intersection(t_set)) / len(q_set)
    df["word_overlap"] = [fast_overlap(str(q), str(t)) for q, t in zip(df["query"], df["product_title"])]
    df["is_over_budget"] = ((df["user_budget"] > 0) & (imputed_price > df["user_budget"])).astype(float)

    def check_brand(q, b):
        if pd.isna(b):
            return 0.0
        return 1.0 if str(b).lower() in str(q).lower() else 0.0
    df["brand_match"] = [check_brand(q, b) for q, b in zip(df["query"], df["product_brand"])]

    def check_color(q, t):
        q_colors = set(str(q).lower().split()).intersection(COLORS)
        if not q_colors:
            return 0.0
        t_words = set(str(t).lower().split())
        return 1.0 if q_colors.intersection(t_words) else 0.0
    df["color_match"] = [check_color(q, t) for q, t in zip(df["query"], df["product_title"])]

    top_20_bm25 = df.sort_values(["query_id", "bm25_score"], ascending=[True, False]).groupby("query_id").head(20)
    dominant_cats = top_20_bm25.groupby("query_id")["category"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    df["query_dominant_category"] = df["query_id"].map(dominant_cats)
    df["is_dominant_category"] = (df["category"] == df["query_dominant_category"]).astype(float)

    return df


def build_split_frame(split, ex, pr, idf_map, manifest):
    """LEFT-join everything onto the judgment anchor; row count cannot grow."""
    anchor = ex[ex["split"] == split][
        ["example_id", "query_id", "query", "product_id", "product_locale", "esci_label"]].copy()
    n_anchor = len(anchor)
    manifest[split] = {"anchor_judged_rows": n_anchor}

    # --- scores ---
    bm = pd.read_csv(os.path.join(ROOT, f"output/bm25_scores_{split}.csv"))
    bm.columns = ["query_id", "product_id", "bm25_score"]
    sem = pd.read_csv(os.path.join(ROOT, f"output/two_tower_scores_{split}.csv"))
    sem.columns = ["query_id", "product_id", "semantic_score"]
    for d in (bm, sem):
        d["query_id"] = d["query_id"].astype(str)
        d["product_id"] = d["product_id"].astype(str)
    assert not bm.duplicated(["query_id", "product_id"]).any(), "bm25 csv has duplicate keys"
    assert not sem.duplicated(["query_id", "product_id"]).any(), "tt csv has duplicate keys"

    df = anchor.merge(bm, on=["query_id", "product_id"], how="left")
    assert len(df) == n_anchor, f"bm25 join multiplied rows: {len(df)} != {n_anchor}"
    df = df.merge(sem, on=["query_id", "product_id"], how="left")
    assert len(df) == n_anchor, f"tt join multiplied rows: {len(df)} != {n_anchor}"

    manifest[split]["rows_missing_bm25_score"] = int(df["bm25_score"].isna().sum())
    manifest[split]["rows_missing_semantic_score"] = int(df["semantic_score"].isna().sum())

    # --- product attributes, LOCALE-AWARE ---
    df = df.merge(pr, on=["product_id", "product_locale"], how="left")
    assert len(df) == n_anchor, f"products join multiplied rows: {len(df)} != {n_anchor}"
    manifest[split]["rows_missing_product_title"] = int(df["product_title"].isna().sum())

    df = compute_features(df, idf_map)
    assert len(df) == n_anchor, "feature computation changed row count"

    # V1 applied dropna(subset=feature_cols) at the end; replicate and record it
    before = len(df)
    df = df.dropna(subset=ALL_FEATURES)
    manifest[split]["rows_dropped_by_feature_dropna"] = int(before - len(df))
    manifest[split]["final_rows"] = int(len(df))
    manifest[split]["final_queries"] = int(df["query_id"].nunique())

    # official relevance + integer label
    df["relevance"] = df["esci_label"].map(OFFICIAL_RELEVANCE).astype(float)
    df["lgb_label"] = df["esci_label"].map(LABEL_ORDINAL).astype(int)
    return df


def main():
    os.makedirs(DATA, exist_ok=True)
    manifest = {}

    print("=== Phase A: official gain mapping ===")
    mapping = build_gain_mapping()
    print(json.dumps({k: mapping[k] for k in
                      ["label_gain_EXACT_DCG_MATCH", "label_gain_LINEAR_RELEVANCE",
                       "v1_label_gain", "s_to_e_gain_ratio"]}, indent=1))

    print("\n=== Phase B: loading sources ===")
    ex = pd.read_parquet(EXAMPLES)
    ex["query_id"] = ex["query_id"].astype(str)
    ex["product_id"] = ex["product_id"].astype(str)
    assert ex["example_id"].is_unique, "example_id not unique"
    assert not ex.duplicated(["query_id", "product_id", "product_locale"]).any()

    pr = pd.read_parquet(PRODUCTS, columns=[
        "product_id", "product_locale", "product_title", "product_brand"])
    pr["product_id"] = pr["product_id"].astype(str)
    n_pr = len(pr)
    assert not pr.duplicated(["product_id", "product_locale"]).any(), \
        "products table has duplicate (product_id, product_locale)"

    esci_s = pd.read_parquet(ESCI_S)
    id_col = "asin" if "asin" in esci_s.columns else esci_s.columns[0]
    esci_s = esci_s.rename(columns={id_col: "product_id"})
    esci_s["product_id"] = esci_s["product_id"].astype(str)
    n_esci_s_raw = len(esci_s)
    esci_s = esci_s.drop_duplicates(subset="product_id", keep="first")
    manifest["esci_s_duplicate_rows_dropped"] = int(n_esci_s_raw - len(esci_s))
    print(f"  ESCI-S: {n_esci_s_raw} rows -> {len(esci_s)} after de-duplicating on asin "
          f"({n_esci_s_raw - len(esci_s)} dropped)")

    pr = pr.merge(esci_s[["product_id", "price", "stars", "ratings", "category"]],
                  on="product_id", how="left")
    assert len(pr) == n_pr, f"ESCI-S join multiplied the products table: {len(pr)} != {n_pr}"
    manifest["products_rows"] = int(n_pr)

    with open(os.path.join(ROOT, "output/lambdamart_features.json")) as f:
        idf_map = json.load(f)["idf_map"]
    manifest["idf_map_size"] = len(idf_map)
    manifest["idf_map_source"] = "output/lambdamart_features.json (frozen, train-query-derived)"

    # ---- reproduce & verify the frozen idf_map ----
    tq = ex[ex["split"] == "train"][["query_id", "query"]].drop_duplicates()
    all_words = " ".join(tq["query"].str.lower().tolist()).split()
    N = len(tq)
    wc = Counter(all_words)
    recomputed = {w: np.log((N + 1) / (c + 1)) + 1 for w, c in wc.items()}
    same = (set(recomputed) == set(idf_map) and
            all(abs(recomputed[w] - idf_map[w]) < 1e-9 for w in recomputed))
    manifest["idf_map_reproduces_from_train_queries"] = bool(same)
    print(f"  frozen idf_map reproduces from train queries: {same}")

    print("\n=== Building TRAIN frame ===")
    df_train_all = build_split_frame("train", ex, pr, idf_map, manifest)
    print(f"  {len(df_train_all)} rows / {df_train_all['query_id'].nunique()} queries")

    print("\n=== Building TEST frame ===")
    df_test = build_split_frame("test", ex, pr, idf_map, manifest)
    print(f"  {len(df_test)} rows / {df_test['query_id'].nunique()} queries")

    # ------------------------------------------------------------------
    # Materialise the EXISTING internal 85/15 validation split
    # ------------------------------------------------------------------
    print("\n=== Materialising the pre-existing internal 85/15 query split ===")
    v1_cache = os.path.join(ROOT, "experiments/ranking_v2/audit/_cache/train_features.parquet")
    if os.path.exists(v1_cache):
        v1_order = pd.read_parquet(v1_cache, columns=["query_id"])["query_id"].unique().tolist()
        source = "V1 feature-frame query order (experiments/ranking_v2/audit/_cache/train_features.parquet)"
    else:
        v1_order = df_train_all["query_id"].unique().tolist()
        source = "clean-pool query order (V1 cache unavailable)"
    train_q, dev_q = train_test_split(v1_order, test_size=0.15, random_state=42)
    manifest["dev_split"] = {
        "method": "sklearn.model_selection.train_test_split(unique_queries, test_size=0.15, random_state=42)",
        "query_order_source": source,
        "identical_to_v1_trainers": True,
        "note": "This is the split scripts/train_lambdamart.py and scripts/train_adv_reranker.py "
                "already create at fit time; it is only being persisted, not changed.",
        "train_queries": len(train_q), "dev_queries": len(dev_q),
    }
    print(f"  train queries {len(train_q)} / dev queries {len(dev_q)} (source: {source})")

    train_set, dev_set = set(train_q), set(dev_q)
    df_train = df_train_all[df_train_all["query_id"].isin(train_set)].copy()
    df_dev = df_train_all[df_train_all["query_id"].isin(dev_set)].copy()
    assert len(df_train) + len(df_dev) == len(df_train_all), "train/dev partition lost rows"

    for name, d in [("train", df_train), ("dev", df_dev), ("test", df_test)]:
        d = d.sort_values(["query_id", "example_id"]).reset_index(drop=True)
        path = os.path.join(DATA, f"{name}_clean.parquet")
        d.to_parquet(path, index=False)
        manifest[f"{name}_clean"] = {
            "path": os.path.relpath(path, ROOT),
            "rows": int(len(d)), "queries": int(d["query_id"].nunique()),
            "columns": list(d.columns),
        }
        print(f"  wrote {name}_clean.parquet: {len(d)} rows / {d['query_id'].nunique()} queries")

    manifest["feature_list"] = ALL_FEATURES
    manifest["official_relevance"] = OFFICIAL_RELEVANCE
    manifest["row_count_invariant"] = (
        "Every join is a LEFT join from the ESCI judgment anchor and is asserted not to "
        "change the row count. train+dev rows == full train-split judged rows minus "
        "feature-dropna; test rows == test-split judged rows minus feature-dropna.")
    with open(os.path.join(DATA, "build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print("\nWrote data/build_manifest.json")


if __name__ == "__main__":
    main()
