"""
Day 2 feature-group ablation for the 17-feature AdvancedDeepReranker.

Reuses Day 1's exclusion mechanism (reranking.advanced_features.ALL_FEATURES /
get_active_features) and Day 1's train_experiment()/evaluate_experiment()
functions from scripts/run_feature_ablation.py verbatim -- group ablation is
just passing a longer excluded_features list into the same training loop.
Nothing about Day 1's own output/ablations/ artifacts is touched or retrained.

Usage:
    python scripts/run_group_ablation.py --validate-groups
    python scripts/run_group_ablation.py --experiment full
    python scripts/run_group_ablation.py --experiment no_lexical
    python scripts/run_group_ablation.py --all
    python scripts/run_group_ablation.py --summarize
"""
import os
import sys
import json
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT_DIR
from reranking.advanced_features import ALL_FEATURES
from scripts.run_feature_ablation import train_experiment, evaluate_experiment, SEED

OUT_DIR = f"{ROOT_DIR}/output/group_ablations"
DAY1_BASELINE_PATH = f"{ROOT_DIR}/output/ablations/full.json"
DAY1_BASELINE_CKPT = f"{ROOT_DIR}/output/best_advanced_reranker.pth"
DAY1_BASELINE_STATS = f"{ROOT_DIR}/output/advanced_normalization_stats.json"

GROUPS = {
    "lexical": ["bm25_score", "word_overlap"],
    "semantic": ["semantic_score"],
    "query_understanding": ["query_length", "query_mean_idf", "query_max_idf", "user_budget", "cheap_intent"],
    "item_quality": ["log_price", "is_price_missing", "stars_clean", "log_review_count", "is_rating_missing"],
    "interaction": ["is_dominant_category", "brand_match", "color_match", "is_over_budget"],
}
EXPERIMENTS = {f"no_{group}": features for group, features in GROUPS.items()}
GROUP_OF_EXPERIMENT = {f"no_{group}": group for group in GROUPS}


def validate_groups():
    all_grouped = [f for feats in GROUPS.values() for f in feats]
    duplicates = {f for f in all_grouped if all_grouped.count(f) > 1}
    missing = set(ALL_FEATURES) - set(all_grouped)
    extra = set(all_grouped) - set(ALL_FEATURES)

    print(f"Total features covered: {len(set(all_grouped))}")
    print(f"Duplicates: {sorted(duplicates) if duplicates else 'none'}")
    print(f"Missing: {sorted(missing) if missing else 'none'}")
    if extra:
        print(f"Extra (not in ALL_FEATURES): {sorted(extra)}")

    ok = (len(all_grouped) == 17 and not duplicates and not missing and not extra)
    if not ok:
        raise AssertionError("Feature-group definition does not cleanly partition ALL_FEATURES. Fix before training.")
    print("Group definition OK: 5 groups partition all 17 features with no overlap.\n")
    return ok


def get_day1_baseline():
    """Reads Day 1's canonical full-model NDCG@10 without retraining."""
    with open(DAY1_BASELINE_PATH) as f:
        return json.load(f)["test_ndcg_at_10"]


def run_full_reference():
    """Writes output/group_ablations/full.json by copying Day 1's baseline
    number (same checkpoint, same stats, same evaluate_experiment() function
    Day 1 already used) -- does NOT retrain."""
    os.makedirs(OUT_DIR, exist_ok=True)
    baseline_ndcg = get_day1_baseline()
    with open(DAY1_BASELINE_STATS) as f:
        n_features = len(json.load(f)["features"])
    payload = {
        "experiment": "full",
        "removed_group": None,
        "removed_features": [],
        "active_features": ALL_FEATURES,
        "num_features": n_features,
        "seed": SEED,
        "best_epoch": None,
        "validation_metric": None,
        "test_ndcg_at_10": baseline_ndcg,
        "baseline_ndcg_at_10": baseline_ndcg,
        "delta_ndcg_at_10": 0.0,
        "checkpoint": DAY1_BASELINE_CKPT,
        "source": "Day 1 output/ablations/full.json (re-evaluated there via the same evaluate_experiment(); not retrained here)",
    }
    with open(f"{OUT_DIR}/full.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[full] test_ndcg_at_10 = {baseline_ndcg:.4f} (sourced from Day 1, no retrain)")
    return payload


def run_one(name):
    os.makedirs(OUT_DIR, exist_ok=True)

    if name == "full":
        return run_full_reference()

    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Options: full, {list(EXPERIMENTS)}")

    removed = EXPERIMENTS[name]
    group = GROUP_OF_EXPERIMENT[name]
    train_result = train_experiment(name, removed, out_dir=OUT_DIR)
    test_ndcg = evaluate_experiment(train_result["checkpoint"], train_result["stats_path"])

    if not os.path.exists(f"{OUT_DIR}/full.json"):
        run_full_reference()
    baseline_ndcg = get_day1_baseline()

    payload = {
        "experiment": name,
        "removed_group": group,
        "removed_features": removed,
        "active_features": train_result["active_features"],
        "num_features": train_result["num_features"],
        "seed": SEED,
        "best_epoch": train_result["best_epoch"],
        "validation_metric": train_result["best_val_loss"],
        "test_ndcg_at_10": test_ndcg,
        "baseline_ndcg_at_10": baseline_ndcg,
        "delta_ndcg_at_10": test_ndcg - baseline_ndcg,
        "checkpoint": train_result["checkpoint"],
    }
    with open(f"{OUT_DIR}/{name}.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[{name}] test_ndcg_at_10 = {test_ndcg:.4f} | baseline = {baseline_ndcg:.4f} "
          f"| delta = {payload['delta_ndcg_at_10']:+.4f} | best_epoch = {train_result['best_epoch']} "
          f"| removed_group = {group} ({len(removed)} features)")
    return payload


GROUP_LABELS = {
    "full": "None",
    "no_semantic": "Semantic",
    "no_lexical": "Lexical",
    "no_query_understanding": "Query Understanding",
    "no_item_quality": "Item / Quality",
    "no_interaction": "Interaction",
}


def summarize():
    rows = []
    for name in ["full"] + list(EXPERIMENTS):
        p = f"{OUT_DIR}/{name}.json"
        if not os.path.exists(p):
            continue
        with open(p) as f:
            r = json.load(f)
        rows.append({
            "experiment": name,
            "removed_group": GROUP_LABELS[name],
            "removed_features": ", ".join(r["removed_features"]) if r["removed_features"] else "—",
            "num_features": r["num_features"],
            "ndcg_at_10": r["test_ndcg_at_10"],
            "delta_vs_full": r["delta_ndcg_at_10"],
            "best_epoch": r["best_epoch"] if r["best_epoch"] is not None else "",
            "seed": r["seed"],
        })

    df = pd.DataFrame(rows)
    csv_path = f"{OUT_DIR}/feature_group_ablation_summary.csv"
    df.to_csv(csv_path, index=False)

    md_lines = [
        "| Removed Group | Removed Features | Remaining Features | NDCG@10 | Δ vs Full |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        delta_str = "—" if row["experiment"] == "full" else f"{row['delta_vs_full']:+.4f}"
        md_lines.append(
            f"| {row['removed_group']} | {row['removed_features']} | {row['num_features']} | "
            f"{row['ndcg_at_10']:.4f} | {delta_str} |"
        )
    md_path = f"{OUT_DIR}/feature_group_ablation_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSaved {csv_path}")
    print(f"Saved {md_path}")
    print("\n".join(md_lines))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None,
                         help="full, or one of: " + ", ".join(EXPERIMENTS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--validate-groups", action="store_true")
    args = parser.parse_args()

    validate_groups()

    if args.validate_groups:
        pass
    elif args.summarize:
        summarize()
    elif args.all:
        run_one("full")
        for exp_name in EXPERIMENTS:
            run_one(exp_name)
        summarize()
    elif args.experiment:
        run_one(args.experiment)
    else:
        parser.print_help()
