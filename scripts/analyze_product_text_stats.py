"""
Phase 4 (stats only, no training) -- Product text length / truncation-rate
analysis for the current item_text construction (title+description+bullets,
no field separators) against the model's max_seq_length=510 (WordPiece
tokens, from models/two_tower_finetuned/sentence_bert_config.json). Uses the
model's own tokenizer for exact token counts (not word counts). Read-only;
does not train or modify any model.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import EXAMPLES_PATH, PRODUCTS_PATH

OUT_DIR = f"{ROOT_DIR}/experiments/two_tower_v2/phase4_metadata"
MAX_SEQ_LEN = 510
LOCALE = "us"
SAMPLE_SIZE = 20000
SEED = 42


def token_len(tokenizer, texts):
    lens = []
    for i in range(0, len(texts), 256):
        batch = texts[i:i + 256]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False)
        lens.extend(len(ids) for ids in enc["input_ids"])
    return np.array(lens)


def summarize(name, lens):
    return {
        "field": name, "n": int(len(lens)),
        "mean_tokens": float(lens.mean()), "median_tokens": float(np.median(lens)),
        "p90_tokens": float(np.percentile(lens, 90)), "p99_tokens": float(np.percentile(lens, 99)),
        "max_tokens": int(lens.max()),
        "pct_exceeding_max_seq_length": float(100.0 * (lens > MAX_SEQ_LEN).mean()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(f"{ROOT_DIR}/models/two_tower_finetuned")

    df_pr = pd.read_parquet(PRODUCTS_PATH)
    df_pr = df_pr[df_pr["product_locale"] == LOCALE]
    df_pr = df_pr.sample(n=min(SAMPLE_SIZE, len(df_pr)), random_state=SEED)

    title = df_pr["product_title"].fillna("").astype(str)
    desc = df_pr["product_description"].fillna("").astype(str)
    bullets = df_pr["product_bullet_point"].fillna("").astype(str)
    combined_current = title + " " + desc + " " + bullets

    print("Tokenizing title...")
    title_lens = token_len(tokenizer, title.tolist())
    print("Tokenizing description...")
    desc_lens = token_len(tokenizer, desc.tolist())
    print("Tokenizing bullets...")
    bullet_lens = token_len(tokenizer, bullets.tolist())
    print("Tokenizing current combined (title+description+bullets)...")
    combined_lens = token_len(tokenizer, combined_current.tolist())

    stats = {
        "sample_size": len(df_pr), "locale": LOCALE, "max_seq_length": MAX_SEQ_LEN, "seed": SEED,
        "fields": [
            summarize("title", title_lens),
            summarize("description", desc_lens),
            summarize("bullet_point", bullet_lens),
            summarize("combined_current (title+description+bullets, current V0 construction)", combined_lens),
        ],
        "truncation_risk_note": (
            "Current item_text = title + ' ' + description + ' ' + bullet_point, concatenated with NO field "
            "prefixes/separators and NO field-priority ordering. Because 'title' is placed first, it is never "
            "itself truncated by max_seq_length -- but title tokens still count against the 510-token budget, "
            "and title/bullet content can be pushed out if description is long enough that combined length "
            "exceeds max_seq_length, since the tokenizer truncates from the END of the sequence by default. "
            "See pct_exceeding_max_seq_length for 'combined_current' for how often this actually happens."
        ),
    }
    with open(f"{OUT_DIR}/product_text_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
