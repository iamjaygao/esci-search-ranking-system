"""
Task 4 -- Win/Loss/Tie decomposition, loss-query regression slice, and worst-20
queries, computed directly from experiments/audit/per_query_ndcg.csv (Task 3
output). No new inference, no retraining.
"""
import os
import json

import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

WIN_EPS = 1e-6


def main():
    df = pd.read_csv(f"{OUT_DIR}/per_query_ndcg.csv")

    win = df[df['delta'] > WIN_EPS]
    loss = df[df['delta'] < -WIN_EPS]
    tie = df[df['delta'].abs() <= WIN_EPS]

    n_total = len(df)
    win_loss_tie = {}
    for name, sub in [("win", win), ("loss", loss), ("tie", tie)]:
        win_loss_tie[name] = {
            "query_count": int(len(sub)),
            "percentage": 100.0 * len(sub) / n_total,
            "mean_delta": float(sub['delta'].mean()) if len(sub) else 0.0,
        }
    with open(f"{OUT_DIR}/win_loss_tie.json", "w") as f:
        json.dump(win_loss_tie, f, indent=2)
    print(json.dumps(win_loss_tie, indent=2))

    # ---- Loss-query regression slice vs all queries ----
    slice_vars = ['query_length', 'query_mean_idf', 'candidate_count', 'E_count', 'S_count']
    rows = []
    for var in slice_vars:
        all_mean, all_median = df[var].mean(), df[var].median()
        loss_mean, loss_median = loss[var].mean(), loss[var].median()
        rows.append({
            "variable": var,
            "all_queries_mean": all_mean,
            "all_queries_median": all_median,
            "loss_queries_mean": loss_mean,
            "loss_queries_median": loss_median,
            "mean_difference": loss_mean - all_mean,
            "median_difference": loss_median - all_median,
        })
    loss_slice_df = pd.DataFrame(rows)
    loss_slice_df.to_csv(f"{OUT_DIR}/loss_slice_stats.csv", index=False)
    print(loss_slice_df.to_string(index=False))

    # ---- Worst 20 queries by delta ----
    worst_20 = df.sort_values('delta', ascending=True).head(20)[
        ['query_id', 'query_text', 'bm25_ndcg10', 'model_ndcg10', 'delta',
         'candidate_count', 'E_count', 'S_count', 'C_count', 'I_count']
    ]
    worst_20.to_csv(f"{OUT_DIR}/worst_20_queries.csv", index=False)
    print(worst_20.to_string(index=False))


if __name__ == "__main__":
    main()
