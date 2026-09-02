"""
Sections 8.3, 11-14, 18 -- rebuild every baseline on the Task 1 benchmark and
emit the final tables.

Nothing here is tuned. LambdaMART is retrained because section 12.2 requires it
(the gain convention changed and the pool changed); every hyperparameter is
copied verbatim from scripts/train_lambdamart.py and only `label_gain` and the
training rows differ.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch  # noqa: F401  (import before lightgbm; libomp clash on macOS)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from task1_common import (  # noqa: E402
    BASE, ROOT, build_arm_a_pool, build_arm_b_pool, load_examples,
)
import kdd_task1_ndcg as SC  # noqa: E402

sys.path.insert(0, ROOT)
from reranking.advanced_features import ALL_FEATURES  # noqa: E402

MODELS = os.path.join(BASE, "models", "lambdamart_task1")
POOL = "task1_small_v1"
ZERO_POLICY = SC.DEFAULT_ZERO_IDCG_POLICY          # "exclude"
ZERO_POLICY_SOURCE = SC.DEFAULT_ZERO_IDCG_POLICY_SOURCE

FIXED_PARAMS = dict(objective="lambdarank", metric="ndcg", n_estimators=1000,
                    learning_rate=0.05, num_leaves=31, min_child_samples=20,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=42, n_jobs=-1)
LABEL_GAIN_OFFICIAL = [0.0, 0.01, 0.10, 1.0]       # I, C, S, E

N_FLOOR_SEEDS = 100
N_TIE_SEEDS = 20


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def groups(df):
    g = df.groupby("query_id", sort=False).size().to_numpy()
    assert g.sum() == len(df)
    return g


def main():
    import lightgbm as lgb
    os.makedirs(MODELS, exist_ok=True)
    t_start = time.time()

    tr = pd.read_parquet(os.path.join(BASE, "train_task1.parquet"))
    dv = pd.read_parquet(os.path.join(BASE, "dev_task1.parquet"))
    te = pd.read_parquet(os.path.join(BASE, "test_task1.parquet"))
    print(f"train {len(tr)}/{tr.query_id.nunique()}  dev {len(dv)}/{dv.query_id.nunique()}  "
          f"test {len(te)}/{te.query_id.nunique()}", flush=True)

    qloc = te.drop_duplicates("query_id").set_index("query_id")["product_locale"]

    # =================== 11.3 degenerate analysis ===========================
    print("\n[11.3] degenerate analysis", flush=True)
    deg = {}
    for name, d in [("train", tr), ("dev", dv), ("test", te)]:
        gq = d.groupby("query_id")["gain"]
        n_levels = gq.nunique()
        maxg = gq.max()
        degen = n_levels <= 1
        all_i = maxg <= 0
        loc = d.drop_duplicates("query_id").set_index("query_id")["product_locale"]
        blk = {"queries": int(len(n_levels)),
               "degenerate_count": int(degen.sum()),
               "degenerate_fraction": round(float(degen.mean()), 6),
               "all_I_count": int(all_i.sum()),
               "all_I_fraction": round(float(all_i.mean()), 6),
               "note": "all_I is a subset of degenerate; all_I has IDCG == 0",
               "by_locale": {}}
        for lc in ["us", "es", "jp"]:
            m = loc.reindex(degen.index) == lc
            blk["by_locale"][lc] = {
                "queries": int(m.sum()),
                "degenerate_count": int((degen & m).sum()),
                "degenerate_fraction": round(float(degen[m].mean()), 6) if m.sum() else None,
                "all_I_count": int((all_i & m).sum()),
                "all_I_fraction": round(float(all_i[m].mean()), 6) if m.sum() else None}
        deg[name] = blk
        print(f"  {name}: degenerate {blk['degenerate_count']}/{blk['queries']} "
              f"({100*blk['degenerate_fraction']:.2f}%), all_I {blk['all_I_count']}", flush=True)
    deg["large_version_comparison"] = {
        "large_test_degenerate_count": 6523, "large_test_queries": 30969,
        "large_test_degenerate_fraction": round(6523 / 30969, 6),
        "task1_test_degenerate_fraction": deg["test"]["degenerate_fraction"],
        "expectation": ("the reduced version filters out 'easy' queries, so the Task 1 "
                        "degenerate share was expected to be LOWER than large-version's 21.06%"),
        "expectation_met": deg["test"]["degenerate_fraction"] < 6523 / 30969,
    }
    with open(os.path.join(BASE, "degenerate_query_analysis.json"), "w") as f:
        json.dump(deg, f, indent=2, default=str)

    nondeg_test = set(te.groupby("query_id")["gain"].nunique().pipe(lambda s: s[s > 1]).index)

    # =================== 11.1 random floor ==================================
    print(f"\n[11.1] random floor, {N_FLOOR_SEEDS} seeds", flush=True)
    floor = {"n_seeds": N_FLOOR_SEEDS, "zero_idcg_policy": ZERO_POLICY, "splits": {}}
    for name, d in [("test", te), ("dev", dv)]:
        loc_d = d.drop_duplicates("query_id").set_index("query_id")["product_locale"]
        nd = set(d.groupby("query_id")["gain"].nunique().pipe(lambda s: s[s > 1]).index)
        acc = {k: {c: [] for c in ["ndcg_full", "ndcg_at_10", "ndcg_at_20"]}
               for k in ["overall", "us", "es", "jp", "non_degenerate"]}
        rng = np.random.RandomState(12345)
        d = d.copy()
        for s in range(N_FLOOR_SEEDS):
            d["_r"] = np.random.RandomState(rng.randint(0, 2**31 - 1)).rand(len(d))
            tbl = SC.per_query_ndcg_table(d, "_r")
            keep = tbl["idcg_full"] > 0
            for c in ["ndcg_full", "ndcg_at_10", "ndcg_at_20"]:
                v = tbl.loc[keep, c]
                acc["overall"][c].append(float(v.mean()))
                lv = loc_d.reindex(v.index)
                for lc in ["us", "es", "jp"]:
                    acc[lc][c].append(float(v[lv == lc].mean()))
                acc["non_degenerate"][c].append(float(v[v.index.isin(nd)].mean()))
        floor["splits"][name] = {
            grp: {c: {"mean": round(float(np.mean(a)), 6), "std": round(float(np.std(a)), 6),
                      "p5": round(float(np.percentile(a, 5)), 6),
                      "p95": round(float(np.percentile(a, 95)), 6)}
                  for c, a in cols.items()}
            for grp, cols in acc.items()}
        print(f"  {name} overall full: {floor['splits'][name]['overall']['ndcg_full']}", flush=True)
    with open(os.path.join(BASE, "random_ranking_floor.json"), "w") as f:
        json.dump(floor, f, indent=2, default=str)

    # =================== 11.2 oracle ceiling ================================
    print("\n[11.2] oracle ceiling", flush=True)
    te_o = te.copy()
    te_o["_oracle"] = te_o["gain"]
    otbl = SC.per_query_ndcg_table(te_o, "_oracle")
    oa = SC.aggregate(otbl, ZERO_POLICY)
    assert abs(oa["ndcg_full"] - 1.0) < 1e-12, f"oracle ndcg_full != 1.0: {oa['ndcg_full']}"
    with open(os.path.join(BASE, "oracle_ceiling.json"), "w") as f:
        json.dump({"ndcg_full": oa["ndcg_full"], "ndcg_at_10": oa["ndcg_at_10"],
                   "ndcg_at_20": oa["ndcg_at_20"],
                   "zero_idcg_policy": ZERO_POLICY,
                   "zero_idcg_query_count": oa["zero_idcg_query_count"],
                   "assertion": "oracle ndcg_full == 1.0 after applying the IDCG==0 policy",
                   "PASS": True}, f, indent=2)
    print(f"  oracle full={oa['ndcg_full']:.10f} @10={oa['ndcg_at_10']:.10f}", flush=True)

    # =================== 12 LambdaMART (retrain) ============================
    print("\n[12] retraining LambdaMART on Task 1 (label_gain = official linear)", flush=True)
    t0 = time.time()
    model = lgb.LGBMRanker(label_gain=LABEL_GAIN_OFFICIAL, **FIXED_PARAMS)
    model.fit(tr[ALL_FEATURES].values, tr["lgb_label"].values, group=groups(tr),
              eval_set=[(dv[ALL_FEATURES].values, dv["lgb_label"].values)],
              eval_group=[groups(dv)], eval_at=[10],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    elapsed = time.time() - t0
    lgb_path = os.path.join(MODELS, "lambdamart_task1.txt")
    model.booster_.save_model(lgb_path)
    lm_meta = {
        "label_gain": LABEL_GAIN_OFFICIAL,
        "integer_label_map": {"I": 0, "C": 1, "S": 2, "E": 3},
        "hyperparameters": FIXED_PARAMS,
        "hyperparameters_source": "scripts/train_lambdamart.py, copied verbatim; nothing tuned",
        "best_iteration": int(model.best_iteration_),
        "num_trees": int(model.booster_.num_trees()),
        "early_stopping_triggered": bool(model.booster_.num_trees() < FIXED_PARAMS["n_estimators"]),
        "train_seconds": round(elapsed, 2),
        "train_rows": int(len(tr)), "train_queries": int(tr.query_id.nunique()),
        "dev_rows": int(len(dv)), "dev_queries": int(dv.query_id.nunique()),
        "lightgbm_internal_dev_ndcg_at_10": float(model.best_score_["valid_0"]["ndcg@10"]),
        "eval_at": [10],
        "note_eval_at": ("early stopping monitors ndcg@10 because that is the frozen config's "
                         "eval_at; the HEADLINE metric is full-list NDCG. Both are reported."),
        "feature_importance_gain_pct": None,
    }
    imp = model.booster_.feature_importance("gain")
    tot = float(imp.sum())
    lm_meta["feature_importance_gain_pct"] = {
        f: round(100.0 * float(g) / tot, 4)
        for f, g in sorted(zip(ALL_FEATURES, imp), key=lambda x: -x[1])}
    te["lambdamart_score"] = model.booster_.predict(te[ALL_FEATURES].values)
    dv["lambdamart_score"] = model.booster_.predict(dv[ALL_FEATURES].values)
    dev_lm = SC.aggregate(SC.per_query_ndcg_table(dv, "lambdamart_score"), ZERO_POLICY)
    lm_meta["dev_official_scorer"] = {k: dev_lm[k] for k in
                                      ["ndcg_full", "ndcg_at_10", "ndcg_at_20"]}
    lm_meta["objective_metric_alignment_dev_at_10"] = {
        "lightgbm_internal": lm_meta["lightgbm_internal_dev_ndcg_at_10"],
        "official_scorer_at_10": dev_lm["ndcg_at_10"],
        "abs_diff": abs(lm_meta["lightgbm_internal_dev_ndcg_at_10"] - dev_lm["ndcg_at_10"]),
        "note": "LightGBM uses label_gain directly as the DCG numerator, so with the official "
                "linear gain its internal ndcg@10 is the same function as the scorer's @10; "
                "the residual is tie-breaking.",
    }
    with open(os.path.join(MODELS, "training_meta.json"), "w") as f:
        json.dump(lm_meta, f, indent=2, default=str)
    print(f"  trees={lm_meta['num_trees']} best_iter={lm_meta['best_iteration']} "
          f"({elapsed:.1f}s); dev full={dev_lm['ndcg_full']:.6f}", flush=True)

    # =================== baselines ==========================================
    ex_cols = set(load_examples(columns=None).columns)
    has_rank_col = bool({"rank", "position", "ranking_position"} & ex_cols)
    te["_oracle"] = te["gain"]
    te["_random"] = np.random.RandomState(20240902).rand(len(te))

    baselines = {
        "Random": "_random",
        "BM25": "bm25_score",
        "Two-Tower": "semantic_score",
        "LambdaMART": "lambdamart_score",
        "Oracle": "_oracle",
    }

    print("\n[12.3] baseline table", flush=True)
    rows, per_query = [], {}
    swapped_gain = SC.gain_from_labels(te["esci_label"], SC.GAIN_SWAPPED)
    te_sw = te.copy()
    te_sw["gain"] = swapped_gain

    for name, col in baselines.items():
        tbl = SC.per_query_ndcg_table(te, col)
        agg = SC.aggregate(tbl, ZERO_POLICY)
        per_query[name] = tbl
        nd = tbl[tbl.index.isin(nondeg_test) & (tbl["idcg_full"] > 0)]
        # random tie-break sensitivity
        tie_vals = [SC.aggregate(SC.per_query_ndcg_table(te, col, tie_break="random", seed=s),
                                 ZERO_POLICY)["ndcg_full"] for s in range(N_TIE_SEEDS)]
        sw = SC.aggregate(SC.per_query_ndcg_table(te_sw, col), ZERO_POLICY)
        rows.append({
            "model": name,
            "ndcg_full": round(agg["ndcg_full"], 6),
            "ndcg_at_10": round(agg["ndcg_at_10"], 6),
            "ndcg_at_20": round(agg["ndcg_at_20"], 6),
            "ndcg_full_nondegenerate": round(float(nd["ndcg_full"].mean()), 6),
            "ndcg_full_random_tiebreak_mean": round(float(np.mean(tie_vals)), 6),
            "ndcg_full_random_tiebreak_std": round(float(np.std(tie_vals)), 6),
            "abs_det_minus_random": round(abs(agg["ndcg_full"] - float(np.mean(tie_vals))), 6),
            "ndcg_full_swapped_gain": round(sw["ndcg_full"], 6),
            "swapped_gain_delta": round(sw["ndcg_full"] - agg["ndcg_full"], 6),
            "query_count": agg["n_queries_scored"],
            "row_count": int(len(te)),
            "zero_idcg_query_count": agg["zero_idcg_query_count"],
            "zero_idcg_policy": ZERO_POLICY,
            "zero_idcg_policy_source": ZERO_POLICY_SOURCE,
            "ndcg_full_policy_zero": round(SC.aggregate(tbl, "zero")["ndcg_full"], 6),
            "ndcg_full_policy_one": round(SC.aggregate(tbl, "one")["ndcg_full"], 6),
            "candidate_pool": POOL,
            "metric_version": SC.metric_version(zero_idcg_policy=ZERO_POLICY),
            "status": "authoritative",
        })
        print(f"  {name:12s} full={agg['ndcg_full']:.6f} @10={agg['ndcg_at_10']:.6f} "
              f"@20={agg['ndcg_at_20']:.6f} nondeg={nd['ndcg_full'].mean():.6f}", flush=True)

    rows.append({
        "model": "Original order", "ndcg_full": None, "ndcg_at_10": None, "ndcg_at_20": None,
        "candidate_pool": POOL, "status": "N/A",
        "note": ("the ESCI examples table has no explicit rank/position column "
                 f"(columns: {sorted(ex_cols)}); section 12.2 forbids inferring an order "
                 "from parquet row order, so this baseline is not computable"),
        "metric_version": SC.metric_version(zero_idcg_policy=ZERO_POLICY),
    })
    rows.append({
        "model": "MLP", "ndcg_full": None, "ndcg_at_10": None, "ndcg_at_20": None,
        "candidate_pool": POOL, "status": "PENDING TASK1 RETRAIN",
        "note": ("frozen weights + normalisation statistics were fitted on large-version "
                 "feature distributions; the Task 1 features are recomputed on a different "
                 "pool, so running the old checkpoint here would be off-distribution and "
                 "not a meaningful baseline. Section 12.2 says not to spend time reproducing "
                 "the old 0.8512."),
        "metric_version": SC.metric_version(zero_idcg_policy=ZERO_POLICY),
    })
    bc = pd.DataFrame(rows)
    bc.to_csv(os.path.join(BASE, "task1_baseline_comparison.csv"), index=False)

    # 8.3 swapped-gain sensitivity
    sens = bc[bc["ndcg_full"].notna()][
        ["model", "ndcg_full", "ndcg_full_swapped_gain", "swapped_gain_delta"]].copy()
    sens = sens.rename(columns={"ndcg_full": "ndcg_full_competition",
                                "ndcg_full_swapped_gain": "ndcg_full_swapped"})
    sens["abs_delta"] = sens["swapped_gain_delta"].abs()
    sens["candidate_pool"] = POOL
    sens.to_csv(os.path.join(BASE, "gain_convention_sensitivity.csv"), index=False)
    max_sw = float(sens["abs_delta"].max())
    print(f"\n[8.3] swapped-gain max |delta| = {max_sw:.6f}", flush=True)

    # =================== 11 locale results ==================================
    print("\n[11] locale results", flush=True)
    lr = []
    for name in baselines:
        tbl = per_query[name]
        keep = tbl[tbl["idcg_full"] > 0]
        lv = qloc.reindex(keep.index)
        for lc in ["us", "es", "jp"]:
            s = keep[lv == lc]
            lr.append({"model": name, "locale": lc, "query_count": int(len(s)),
                       "ndcg_full": round(float(s["ndcg_full"].mean()), 6),
                       "ndcg_at_10": round(float(s["ndcg_at_10"].mean()), 6),
                       "ndcg_at_20": round(float(s["ndcg_at_20"].mean()), 6),
                       "candidate_pool": POOL,
                       "metric_version": SC.metric_version(zero_idcg_policy=ZERO_POLICY)})
        lr.append({"model": name, "locale": "overall", "query_count": int(len(keep)),
                   "ndcg_full": round(float(keep["ndcg_full"].mean()), 6),
                   "ndcg_at_10": round(float(keep["ndcg_at_10"].mean()), 6),
                   "ndcg_at_20": round(float(keep["ndcg_at_20"].mean()), 6),
                   "candidate_pool": POOL,
                   "metric_version": SC.metric_version(zero_idcg_policy=ZERO_POLICY)})
    lrdf = pd.DataFrame(lr)
    lrdf.to_csv(os.path.join(BASE, "locale_results.csv"), index=False)
    print(lrdf.pivot(index="model", columns="locale", values="ndcg_full")
          [["us", "es", "jp", "overall"]].to_string(), flush=True)

    # =================== 12.4 prereg check ==================================
    with open(os.path.join(BASE, "prereg_predictions.json")) as f:
        prereg = json.load(f)
    obs = float(bc.loc[bc.model == "LambdaMART", "ndcg_full"].iloc[0])
    pred = prereg["b_locale_reweight_prediction"]["predicted_ndcg_from_locale_reweight_alone"]
    if 0.76 <= obs <= 0.87:
        band, action = "within", "proceed"
    elif 0.70 <= obs < 0.76 or 0.87 < obs <= 0.90:
        band, action = "outside_soft", "note_in_report"
    else:
        band, action = "outside_hard", "STOP"
    pc = {"predicted_locale_reweight_only": pred,
          "observed_lambdamart_ndcg_full": round(obs, 6),
          "delta_vs_prediction": round(obs - pred, 6),
          "band": band, "action": action,
          "preregistered_band": [0.76, 0.87],
          "hard_stop_outside": [0.70, 0.90],
          "three_factor_decomposition": {
              "old_large_version_lambdamart_ndcg_at_10": 0.852064,
              "locale_reweight_only_prediction": pred,
              "locale_reweight_effect": round(pred - 0.852064, 6),
              "residual_after_locale_reweight": round(obs - pred, 6),
              "residual_attribution": ("harder query subset (easy queries filtered from the "
                                       "reduced version) + full-list vs @10 truncation + a "
                                       "different metric gain convention + recomputed "
                                       "pool-dependent features; not separable from these "
                                       "artifacts alone"),
          }}
    with open(os.path.join(BASE, "prereg_check.json"), "w") as f:
        json.dump(pc, f, indent=2, default=str)
    print(f"\n[12.4] prereg: observed {obs:.6f}, predicted {pred:.4f}, band={band}", flush=True)
    if action == "STOP":
        print("\n=== STOP: LambdaMART outside the hard band ===")
        sys.exit(2)

    # =================== 13 phase 2 training pools ==========================
    print("\n[13] phase 2 training pools", flush=True)
    ex = load_examples()
    arm_a, info_a = build_arm_a_pool(set(tr["query_id"]), ex)
    arm_b, info_b = build_arm_b_pool(ex)
    task1_test_q = set(te["query_id"])

    def dist(d):
        ql = d.drop_duplicates("query_id")["product_locale"].value_counts(normalize=True)
        lb = d["esci_label"].value_counts(normalize=True)
        return ({k: round(float(ql.get(k, 0)), 4) for k in ["us", "es", "jp"]},
                {k: round(100 * float(lb.get(k, 0)), 4) for k in ["E", "S", "C", "I"]})

    la, lba = dist(arm_a)
    lb_, lbb = dist(arm_b)
    pools = {
        "arm_A": {"name": "task1_only", "filter": info_a["definition"],
                  "rows": int(len(arm_a)), "queries": int(arm_a.query_id.nunique()),
                  "locale_dist": la, "label_dist": lba,
                  "exclusion_helper": info_a["helper"]},
        "arm_B": {"name": "large_train_excl_task1_test", "filter": info_b["definition"],
                  "rows": int(len(arm_b)), "queries": int(arm_b.query_id.nunique()),
                  "locale_dist": lb_, "label_dist": lbb,
                  "exclusion_helper": info_b["helper"],
                  "rows_removed_by_exclusion": info_b["rows_removed"],
                  "queries_removed_by_exclusion": info_b["queries_removed"],
                  "removed_query_ids": info_b["removed_query_ids_sample"]},
        "eval_pool": "small_version==1 AND split=='test'",
        "leakage_assert_passed": bool(
            not (set(arm_a.query_id) & task1_test_q) and not (set(arm_b.query_id) & task1_test_q)),
        "leakage_detail": {
            "arm_A_intersection_with_task1_test": len(set(arm_a.query_id) & task1_test_q),
            "arm_B_intersection_with_task1_test": len(set(arm_b.query_id) & task1_test_q)},
        "distribution_difference_note": (
            "arm B contains the 'easy' queries that the reduced version filters out, so its "
            "locale and label distributions necessarily differ from arm A. Recorded, not corrected."),
        "locale_dist_delta_B_minus_A": {k: round(lb_[k] - la[k], 4) for k in la},
        "label_dist_delta_B_minus_A": {k: round(lbb[k] - lba[k], 4) for k in lba},
        "this_phase_trains_nothing": True,
    }
    assert pools["leakage_assert_passed"], "arm pool leakage assertion FAILED"
    with open(os.path.join(BASE, "phase2_training_pools.json"), "w") as f:
        json.dump(pools, f, indent=2, default=str)
    print(f"  arm A {pools['arm_A']['rows']} rows / {pools['arm_A']['queries']} q", flush=True)
    print(f"  arm B {pools['arm_B']['rows']} rows / {pools['arm_B']['queries']} q "
          f"(removed {info_b['rows_removed']} rows)", flush=True)

    # =================== 9 terrier parity: NOT_ATTEMPTED ====================
    m = te.groupby("product_id")["product_locale"].nunique()
    parity = {
        "status": "NOT_ATTEMPTED",
        "reason": ("No Terrier, no trec_eval and no Java runtime on this machine "
                   "('Unable to locate a Java Runtime'). Resolution item 6 forbids installing "
                   "a JRE or any system-level dependency."),
        "terrier_version": None,
        "terrier_path": None,
        "exact_command": ("$1/terrier trec_eval \"${TREC_EVAL_DATA_PATH}/test.qrels\" "
                          "\"${TREC_EVAL_DATA_PATH}/hypothesis.results\" -c -J "
                          "-m 'ndcg.1=0,2=0.01,3=0.1,4=1'"),
        "command_source": "read_from_launch_script",
        "launch_script_path": "/Users/jaygao/WORKSPACE/projects/dataset/esci-data/ranking/launch-predictions-task1.sh",
        "qrels_mapping": {"I": 1, "C": 2, "S": 3, "E": 4},
        "qrels_mapping_note": ("CORRECTED. The reference helper uses {E:4,S:2,C:3,I:1}, which "
                               "with the same gain spec yields S=0.01 and C=0.1 -- the known "
                               "S/C inversion. See gain_convention_conflict.json."),
        "gain_spec": "ndcg.1=0,2=0.01,3=0.1,4=1",
        "output_decimal_places": None,
        "tolerance_used": None,
        "tests": [{"name": n, "python": None, "terrier": None, "delta": None, "pass": None}
                  for n in ["synthetic", "dev_random", "dev_bm25"]],
        "zero_idcg_policy_observed": None,
        "zero_idcg_policy_applied": ZERO_POLICY,
        "zero_idcg_policy_source": ZERO_POLICY_SOURCE,
        "input_files_generated": True,
        "input_files_dir": "trec_eval_data/",
        "how_to_complete": "scripts/run_terrier_parity.sh <path to terrier bin>",
        "notes": {
            "product_locale_multiplicity_check": {
                "expression": "task1.groupby('product_id').product_locale.nunique().max()",
                "value_on_test": int(m.max()),
                "product_ids_in_multiple_locales_on_test": int((m > 1).sum()),
                "locale_prefixed_doc_id_mandatory": bool(m.max() > 1),
                "doc_id_format_used": "{product_locale}_{product_id}",
                "divergence_from_official_helper": (
                    "prepare_trec_eval_files.py writes qrels keyed on product_id alone "
                    "(lines 121-126). With product_ids spanning up to "
                    f"{int(m.max())} locales, that keying merges distinct judgements onto one "
                    "doc_id. A locale-prefixed doc_id is therefore used in both the qrels and "
                    "the run file, diverging from the reference helper deliberately."),
            },
            "consequence_for_claims": ("Benchmark status is READY (PARITY UNVERIFIED). The "
                                       "scorer is an independent implementation whose parity "
                                       "with Terrier has NOT been checked. Do not claim "
                                       "'reproduced Terrier' or 'reproduced the official "
                                       "scorer'."),
        },
    }
    with open(os.path.join(BASE, "terrier_parity.json"), "w") as f:
        json.dump(parity, f, indent=2, default=str)

    # =================== 18 test lock manifest ==============================
    test_path = os.path.join(BASE, "test_task1.parquet")
    lockloc = te.drop_duplicates("query_id")["product_locale"].value_counts()
    manifest = {
        "test_file": "test_task1.parquet",
        "test_file_sha256": sha256(test_path),
        "query_count": int(te.query_id.nunique()),
        "row_count": int(len(te)),
        "locale_breakdown": {k: int(v) for k, v in lockloc.items()},
        "scorer_file": "scripts/kdd_task1_ndcg.py",
        "scorer_sha256": SC.scorer_sha256(),
        "metric_version": SC.metric_version(zero_idcg_policy=ZERO_POLICY),
        "gain_convention": "competition_E1_S0.1_C0.01_I0",
        "zero_idcg_policy": ZERO_POLICY,
        "zero_idcg_policy_source": "fallback_pending_gate_b",
        "manifest_revision": 1,
        "tie_break": "deterministic_product_id_asc",
        "terrier_parity_status": "NOT_ATTEMPTED",
        "degenerate_fraction": deg["test"]["degenerate_fraction"],
        "all_I_fraction": deg["test"]["all_I_fraction"],
        "git_head": "9b173e95307b6af4335d46518f703265a9608d57",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lock_rule": ("From Phase 2 onward, training and tuning may only read train_task1 and "
                      "dev_task1. test_task1 is run exactly once, after the model "
                      "configuration is frozen."),
    }
    with open(os.path.join(BASE, "test_lock_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # per-query dump
    pq = pd.DataFrame({f"{k}_ndcg_full": v["ndcg_full"] for k, v in per_query.items()})
    for k, v in per_query.items():
        pq[f"{k}_ndcg_at_10"] = v["ndcg_at_10"]
        pq[f"{k}_ndcg_at_20"] = v["ndcg_at_20"]
    pq["locale"] = qloc.reindex(pq.index)
    pq["n_candidates"] = per_query["BM25"]["n_candidates"]
    pq["idcg_full"] = per_query["BM25"]["idcg_full"]
    pq["is_degenerate"] = ~pq.index.isin(nondeg_test)
    pq.index.name = "query_id"
    pq.reset_index().to_parquet(os.path.join(BASE, "per_query_scores.parquet"), index=False)

    print(f"\nDone in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
