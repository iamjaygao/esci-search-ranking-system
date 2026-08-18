"""
Generates interview-ready visual assets under screenshots/.
Run: python scripts/generate_interview_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "screenshots")
os.makedirs(OUT, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#4A90D9"
C_ORANGE = "#F5A623"
C_PURPLE = "#9B59B6"
C_GREEN  = "#2ECC71"
C_RED    = "#E74C3C"
C_DARK   = "#2C3E50"
C_LIGHT  = "#ECF0F1"
C_GRAY   = "#95A5A6"


# ── 1. Metrics bar chart ──────────────────────────────────────────────────────
def make_metrics_chart():
    methods = ["BM25 Baseline", "Neural Reranker"]
    values  = [0.8188, 0.8464]
    colors  = [C_ORANGE, C_PURPLE]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    bars = ax.bar(methods, values, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5, zorder=3)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=C_DARK)

    # lift annotation — arrow points left-to-right (BM25 → Reranker)
    x0 = bars[0].get_x() + bars[0].get_width() / 2
    x1 = bars[1].get_x() + bars[1].get_width() / 2
    y_ann = 0.851
    ax.annotate("", xy=(x1, y_ann), xytext=(x0, y_ann),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2,
                                mutation_scale=16))
    ax.text((x0 + x1) / 2, y_ann + 0.003,
            "+0.041 absolute gain / ~5.1% relative lift",
            ha="center", va="bottom", fontsize=10, color=C_RED, fontweight="bold")

    ax.set_ylim(0.77, 0.875)
    ax.set_ylabel("NDCG@10", fontsize=13, color=C_DARK)
    ax.set_title("NDCG@10 Improvement from Reranking",
                 fontsize=14, fontweight="bold", color=C_DARK, pad=12)
    ax.set_facecolor("#F8F9FA")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.99, 0.01, "Zoomed y-axis for readability",
            ha="right", va="bottom", fontsize=8, color=C_GRAY,
            transform=ax.transAxes, style="italic")

    plt.tight_layout()
    path = os.path.join(OUT, "retrieval_vs_reranking_metrics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ── 2. Pipeline summary (3-column infographic) ────────────────────────────────
def make_pipeline_summary():
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    columns = [
        {
            "title": "① Candidate Generation",
            "color": C_ORANGE,
            "items": [
                "BM25  —  lexical exact match\n(brand names, SKUs, model numbers)",
                "Dense retrieval  —  semantic intent\n(SentenceTransformer, FAISS IndexFlatIP)",
                "Merge top-150 candidates\nfrom both retrievers",
            ],
        },
        {
            "title": "② Neural Reranking",
            "color": C_PURPLE,
            "items": [
                "17 engineered features",
                "Query intent signals\n(budget, IDF, cheap-keywords)",
                "Item authority signals\n(price, stars, review count)",
                "Interaction signals\n(brand match, color match,\nover-budget penalty)",
                "Pairwise MarginRankingLoss\n3-layer MLP + LayerNorm + Dropout",
            ],
        },
        {
            "title": "③ Evaluation",
            "color": C_BLUE,
            "items": [
                "NDCG@10  (primary metric)",
                "Recall@K  (coverage)",
                "Business-aware NDCG\n(penalizes over-budget results,\nstar rating as tie-breaker)",
                "BM25 baseline  →  0.8188\nNeural reranker  →  0.8464\n+0.0276 / ~3.4% lift",
            ],
        },
    ]

    for ax, col in zip(axes, columns):
        ax.set_facecolor("#F8F9FA")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # column header
        header = FancyBboxPatch((0.05, 0.88), 0.90, 0.10,
                                boxstyle="round,pad=0.02",
                                facecolor=col["color"], edgecolor="none")
        ax.add_patch(header)
        ax.text(0.50, 0.932, col["title"],
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", transform=ax.transData)

        # item boxes
        n = len(col["items"])
        slot_h = 0.82 / n
        for i, item in enumerate(col["items"]):
            y_top = 0.85 - i * slot_h
            box = FancyBboxPatch((0.05, y_top - slot_h + 0.01), 0.90, slot_h - 0.02,
                                 boxstyle="round,pad=0.02",
                                 facecolor="white",
                                 edgecolor=col["color"], linewidth=1.2)
            ax.add_patch(box)
            ax.text(0.50, y_top - (slot_h - 0.01) / 2,
                    item,
                    ha="center", va="center", fontsize=9,
                    color=C_DARK, wrap=True,
                    multialignment="center",
                    transform=ax.transData)

    fig.suptitle("Hybrid Search Reranking System  —  Amazon ESCI (~2.6M products)",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.01)
    plt.tight_layout(pad=1.5)
    path = os.path.join(OUT, "search_pipeline_summary.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ── 3. UI placeholder README ──────────────────────────────────────────────────
def make_ui_placeholder():
    md = os.path.join(OUT, "README_sample_ui.md")
    content = """\
# Sample Search Results UI — Manual Capture Required

The interactive demo (`interactive_search.py`) requires:
- Trained model weights in `models/two_tower_finetuned/` and `models/adv_reranker.pt`
- Pre-built BM25 and FAISS indices in `output/`
- The full Amazon ESCI product dataset

Because these artefacts are gitignored and not present, the GUI cannot be launched
automatically. To capture a real screenshot:

```bash
# 1. Complete Setup steps 1–5 from README.md
# 2. Run the demo
python interactive_search.py

# 3. Enter a test query such as:
#      "wireless noise cancelling headphones under $100"
#      "comfortable work from home chair"
#      "iPhone 15 Pro Max case"

# 4. Screenshot the results table and save as:
#      screenshots/sample_search_results_ui.png
```

The GUI shows columns: **Rank | Score | Brand | Price | Stars | Reviews | Category | Title**
Results are diversified using MMR (λ=0.6) to reduce brand repetition.
"""
    with open(md, "w") as f:
        f.write(content)
    print(f"  saved: {md}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating interview figures …")
    make_metrics_chart()
    make_pipeline_summary()
    make_ui_placeholder()
    print("Done.")
