"""
Day 1 single-feature ablation for the 17-feature AdvancedDeepReranker.

Each ablation experiment removes exactly one feature from
reranking.advanced_features.ALL_FEATURES, retrains the MLP from scratch with
identical hyperparameters/splits/loss/optimizer/scheduler to
scripts/train_adv_reranker.py, and evaluates standard NDCG@10 on the test set
using the exact same candidate pool and label mapping as
evaluation/evaluate_advanced.py.

The "full" experiment does NOT retrain -- it re-evaluates the existing
output/best_advanced_reranker.pth + output/advanced_normalization_stats.json
checkpoint through the same evaluate_experiment() codepath used for the
ablations, so the baseline number is produced by the identical measurement
function rather than copied from a prior run's stdout.

Usage:
    python scripts/run_feature_ablation.py --experiment full
    python scripts/run_feature_ablation.py --experiment no_semantic_score
    python scripts/run_feature_ablation.py --all
    python scripts/run_feature_ablation.py --summarize
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMPLES_PATH, PRODUCTS_PATH, ROOT_DIR
from reranking.advanced_features import extract_advanced_features, AdvancedPairwiseDataset, get_active_features
from reranking.advanced_model import AdvancedDeepReranker
from evaluation.evaluate_advanced import extract_test_advanced_features
from evaluation.metrics import ndcg_at_k

SEED = 42
OUT_DIR = f"{ROOT_DIR}/output/ablations"
BM25_TRAIN_CSV = f"{ROOT_DIR}/output/bm25_scores_train.csv"
SEM_TRAIN_CSV = f"{ROOT_DIR}/output/two_tower_scores_train.csv"
ESCI_S_PATH = f"{ROOT_DIR}/esci-data/esci-s_dataset/esci_s_products.parquet"

# name -> list of features to remove (single-feature ablations for Day 1)
EXPERIMENTS = {
    "no_semantic_score": ["semantic_score"],
    "no_bm25_score": ["bm25_score"],
    "no_word_overlap": ["word_overlap"],
    "no_brand_match": ["brand_match"],
    "no_query_mean_idf": ["query_mean_idf"],
}

STANDARD_LABEL_MAP = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_experiment(name, excluded_features, out_dir=OUT_DIR):
    """Retrains AdvancedDeepReranker with one or more features removed. Mirrors
    scripts/train_adv_reranker.py exactly except for the active feature list.
    out_dir lets callers (e.g. scripts/run_group_ablation.py) reuse this exact
    training loop for group-level exclusions without duplicating it."""
    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== [{name}] Extracting features (excluded={excluded_features}) ===")
    df_all, feature_columns, idf_map = extract_advanced_features(
        EXAMPLES_PATH, PRODUCTS_PATH, BM25_TRAIN_CSV, SEM_TRAIN_CSV, ESCI_S_PATH,
        excluded_features=excluded_features,
    )
    expected_n = 17 - len(set(excluded_features))
    assert len(feature_columns) == expected_n, (
        f"expected {expected_n} active features, got {len(feature_columns)}: {feature_columns}"
    )
    for f in excluded_features:
        assert f not in feature_columns, f"{f} leaked back into feature_columns"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] Training on: {device}")

    unique_queries = df_all['query_id'].unique().tolist()
    train_queries, val_queries = train_test_split(unique_queries, test_size=0.15, random_state=42)
    df_train = df_all[df_all['query_id'].isin(train_queries)].copy()
    df_val = df_all[df_all['query_id'].isin(val_queries)].copy()

    train_dataset = AdvancedPairwiseDataset(df_train, feature_columns)
    val_dataset = AdvancedPairwiseDataset(df_val, feature_columns, mean=train_dataset.mean, std=train_dataset.std)

    num_workers = 4 if os.name != 'nt' else 0
    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1024, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    stats_path = f"{out_dir}/{name}_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "mean": train_dataset.mean.tolist(),
            "std": train_dataset.std.tolist(),
            "features": feature_columns,
            "idf_map": idf_map,
        }, f)

    model = AdvancedDeepReranker(input_dim=len(feature_columns)).to(device)
    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs, patience, patience_counter, best_val_loss, best_epoch = 50, 6, 0, float('inf'), -1
    ckpt_path = f"{out_dir}/{name}.pth"

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_x_pos, batch_x_neg, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x_pos, batch_x_neg, batch_y = batch_x_pos.to(device), batch_x_neg.to(device), batch_y.to(device)
            pos_scores = model(batch_x_pos).squeeze()
            neg_scores = model(batch_x_neg).squeeze()
            loss = criterion(pos_scores, neg_scores, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x_pos, batch_x_neg, batch_y in val_loader:
                batch_x_pos, batch_x_neg, batch_y = batch_x_pos.to(device), batch_x_neg.to(device), batch_y.to(device)
                pos_scores = model(batch_x_pos).squeeze()
                neg_scores = model(batch_x_neg).squeeze()
                loss = criterion(pos_scores, neg_scores, batch_y)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"[{name}] Epoch [{epoch+1:02d}/{epochs}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | {elapsed:.0f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{name}]   --> best weights updated (epoch {best_epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{name}] Early stopping at epoch {epoch+1}")
                break

    return {
        "checkpoint": ckpt_path,
        "stats_path": stats_path,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "num_features": len(feature_columns),
        "active_features": feature_columns,
    }


def evaluate_experiment(ckpt_path, stats_path):
    """Runs the exact same test-set NDCG@10 measurement evaluate_advanced.py
    uses for 'Standard Textual Relevance', parameterized by whatever feature
    list/checkpoint is passed in. Used for both ablations and the full model."""
    with open(stats_path, "r") as f:
        stats = json.load(f)
    feature_cols = stats["features"]
    train_mean = np.array(stats["mean"])
    train_std = np.array(stats["std"])
    idf_map = stats.get("idf_map", {})

    df_test = extract_test_advanced_features(idf_map)
    df_test = df_test.dropna(subset=feature_cols)

    features_raw = df_test[feature_cols].values
    features_normalized = (features_raw - train_mean) / train_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedDeepReranker(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(features_normalized, dtype=torch.float32).to(device)
        predictions = model(x_tensor).cpu().squeeze().numpy()

    df_test = df_test.copy()
    df_test['predicted_score'] = predictions
    df_test['relevance'] = df_test['esci_label'].map(STANDARD_LABEL_MAP).fillna(0.0)

    return ndcg_at_k(df_test, score_col='predicted_score', k=10)


def run_one(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    result_path = f"{OUT_DIR}/{name}.json"

    if name == "full":
        # Reuse the existing trained checkpoint -- do NOT retrain the full model.
        ckpt_path = f"{ROOT_DIR}/output/best_advanced_reranker.pth"
        stats_path = f"{ROOT_DIR}/output/advanced_normalization_stats.json"
        assert os.path.exists(ckpt_path), "output/best_advanced_reranker.pth not found"
        assert os.path.exists(stats_path), "output/advanced_normalization_stats.json not found"
        with open(stats_path) as f:
            n_features = len(json.load(f)["features"])
        test_ndcg = evaluate_experiment(ckpt_path, stats_path)
        payload = {
            "experiment": "full",
            "removed_features": [],
            "num_features": n_features,
            "active_features": None,
            "seed": SEED,
            "best_epoch": None,
            "validation_metric": None,
            "test_ndcg_at_10": test_ndcg,
            "baseline_ndcg_at_10": test_ndcg,
            "delta_ndcg_at_10": 0.0,
            "checkpoint": ckpt_path,
        }
        with open(result_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[full] test_ndcg_at_10 = {test_ndcg:.4f} (reused existing checkpoint, re-evaluated)")
        return payload

    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Options: full, {list(EXPERIMENTS)}")

    removed = EXPERIMENTS[name]
    train_result = train_experiment(name, removed)
    test_ndcg = evaluate_experiment(train_result["checkpoint"], train_result["stats_path"])

    # Baseline is always measured fresh from the full-model checkpoint via the
    # same evaluate_experiment() path, not a hardcoded literal.
    full_path = f"{OUT_DIR}/full.json"
    if not os.path.exists(full_path):
        run_one("full")
    with open(full_path) as f:
        baseline_ndcg = json.load(f)["test_ndcg_at_10"]

    payload = {
        "experiment": name,
        "removed_features": removed,
        "num_features": train_result["num_features"],
        "active_features": train_result["active_features"],
        "seed": SEED,
        "best_epoch": train_result["best_epoch"],
        "validation_metric": train_result["best_val_loss"],
        "test_ndcg_at_10": test_ndcg,
        "baseline_ndcg_at_10": baseline_ndcg,
        "delta_ndcg_at_10": test_ndcg - baseline_ndcg,
        "checkpoint": train_result["checkpoint"],
    }
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[{name}] test_ndcg_at_10 = {test_ndcg:.4f} | baseline = {baseline_ndcg:.4f} "
          f"| delta = {payload['delta_ndcg_at_10']:+.4f} | best_epoch = {train_result['best_epoch']}")
    return payload


def summarize():
    rows = []
    full_path = f"{OUT_DIR}/full.json"
    if os.path.exists(full_path):
        with open(full_path) as f:
            full = json.load(f)
        rows.append({
            "experiment": "full",
            "removed_feature": "",
            "num_features": full["num_features"],
            "ndcg_at_10": full["test_ndcg_at_10"],
            "delta_vs_full": 0.0,
            "best_epoch": "",
            "seed": full["seed"],
        })

    for name in EXPERIMENTS:
        p = f"{OUT_DIR}/{name}.json"
        if not os.path.exists(p):
            continue
        with open(p) as f:
            r = json.load(f)
        rows.append({
            "experiment": name,
            "removed_feature": ",".join(r["removed_features"]),
            "num_features": r["num_features"],
            "ndcg_at_10": r["test_ndcg_at_10"],
            "delta_vs_full": r["delta_ndcg_at_10"],
            "best_epoch": r["best_epoch"],
            "seed": r["seed"],
        })

    df = pd.DataFrame(rows)
    csv_path = f"{OUT_DIR}/feature_ablation_summary.csv"
    df.to_csv(csv_path, index=False)

    md_lines = ["| Experiment | Features | NDCG@10 | Δ vs Full |", "|---|---:|---:|---:|"]
    for _, row in df.iterrows():
        label = "Full" if row["experiment"] == "full" else f"- {row['removed_feature']}"
        delta_str = "—" if row["experiment"] == "full" else f"{row['delta_vs_full']:+.4f}"
        md_lines.append(f"| {label} | {row['num_features']} | {row['ndcg_at_10']:.4f} | {delta_str} |")
    md_path = f"{OUT_DIR}/feature_ablation_summary.md"
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
    args = parser.parse_args()

    if args.summarize:
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
