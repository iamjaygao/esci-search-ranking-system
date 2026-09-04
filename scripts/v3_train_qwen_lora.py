"""
V3 STEP 2/4 -- LoRA fine-tune Qwen3-Reranker-0.6B on the frozen 50K subset.

Training objective (stated explicitly, per STEP 3)
--------------------------------------------------
The model's NATIVE yes/no reranker formulation is preserved: we read the
final-position logits for the tokens "yes" and "no", form a 2-way softmax, and
treat P(yes) as the relevance score -- identical to the zero-shot inference path.

Supervision uses the frozen official Task 1 gains as SOFT targets:

    E -> 1.00    S -> 0.10    C -> 0.01    I -> 0.00

Loss = soft-target cross-entropy over that 2-way softmax:

    L = -[ t*log P(yes) + (1-t)*log P(no) ],   t = official_gain(label)

So E/S/C are NOT collapsed into one binary positive: a Substitute is trained
toward P(yes)=0.10 and a Complement toward 0.01, which is exactly the graded
structure the evaluation metric rewards. The old [7,3,1,0] gain is never used.

LoRA only -- no full-parameter training. One pre-registered config, no sweep.
DEV/TEST are never read for training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CE = os.path.join(ROOT, "experiments/ranking_v2/cross_encoder_task1/_cache")
BK = os.path.join(ROOT, "experiments/ranking_v3/supervised_bakeoff")
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
GAIN = {"E": 1.00, "S": 0.10, "C": 0.01, "I": 0.00}

TASK = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
          'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
          '<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

FIELDS = [("title", "product_title"), ("brand", "product_brand"), ("color", "product_color"),
          ("bullet", "product_bullet_point"), ("description", "product_description")]


def clean(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "<na>", "null"} else " ".join(s.split())


def plain_doc(row):
    return "\n".join(f"{l}: {clean(row[c])}" for l, c in FIELDS if clean(row[c]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke test only")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--save_fracs", default="0.25,0.5,1.0")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    # ---------- frozen 50K subset ----------
    man = json.load(open(os.path.join(BK, "train50k_manifest.json")))
    keep_ex = set(np.load(os.path.join(BK, "train50k_example_ids.npy")).tolist())
    df = pd.read_parquet(os.path.join(CE, "train_text.parquet"))
    df = df[df["example_id"].isin(keep_ex)].reset_index(drop=True)
    assert len(df) == man["exact_row_count"], f"{len(df)} != {man['exact_row_count']}"
    dev_q = set(pd.read_parquet(
        os.path.join(ROOT, "experiments/ranking_v2/kdd_task1_benchmark/dev_task1.parquet"),
        columns=["query_id"])["query_id"])
    assert not (set(df["query_id"]) & dev_q), "DEV LEAK in training data"
    print(f"train50k: {len(df)} rows / {df['query_id'].nunique()} queries "
          f"(sha256 {man['sha256_example_ids'][:12]})", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pre_ids = tok.encode(PREFIX, add_special_tokens=False)
    suf_ids = tok.encode(SUFFIX, add_special_tokens=False)
    inner = args.max_length - len(pre_ids) - len(suf_ids)
    tid_yes, tid_no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")

    texts = [f"<Instruct>: {TASK}\n<Query>: {q}\n<Document>: {plain_doc(r)}"
             for q, (_, r) in zip(df["query"].astype(str), df.iterrows())]
    targets = df["esci_label"].map(GAIN).astype("float32").values
    print(f"target distribution: " +
          ", ".join(f"{k}->{v}: {int((targets==v).sum())}" for k, v in GAIN.items()), flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable:,} / {total:,} = {100*trainable/total:.3f}%", flush=True)

    idx = np.arange(len(df)); np.random.RandomState(args.seed).shuffle(idx)

    def collate(batch):
        bt = [texts[i] for i in batch]
        enc = tok(bt, padding=False, truncation="longest_first",
                  max_length=inner, return_attention_mask=False)
        enc["input_ids"] = [pre_ids + x + suf_ids for x in enc["input_ids"]]
        enc = tok.pad(enc, padding=True, return_tensors="pt")
        enc["target"] = torch.tensor([targets[i] for i in batch], dtype=torch.float32)
        return enc

    dl = DataLoader(idx.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    n_opt = (len(dl) + args.grad_accum - 1) // args.grad_accum
    if args.max_steps:
        n_opt = min(n_opt, args.max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sch = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * n_opt), n_opt)
    save_at = {int(round(float(f) * n_opt)): f for f in args.save_fracs.split(",")}
    print(f"batches={len(dl)}  optimizer steps={n_opt}  save_at={sorted(save_at)}", flush=True)

    model.train()
    run, seen, step, t0 = 0.0, 0, 0, time.time()
    hist = []
    opt.zero_grad(set_to_none=True)
    stop = False
    for i, batch in enumerate(dl):
        tgt = batch.pop("target").to(dev)
        batch = {k: v.to(dev) for k, v in batch.items()}
        lg = model(**batch).logits[:, -1, :]
        pair = torch.stack([lg[:, tid_no], lg[:, tid_yes]], dim=1).float()
        logp = torch.log_softmax(pair, dim=1)
        loss = -(tgt * logp[:, 1] + (1.0 - tgt) * logp[:, 0]).mean()
        (loss / args.grad_accum).backward()
        run += loss.item() * tgt.size(0); seen += tgt.size(0)
        if (i + 1) % args.grad_accum == 0 or (i + 1) == len(dl):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"  step {step}/{n_opt}  loss {run/seen:.4f}  lr {sch.get_last_lr()[0]:.2e}"
                      f"  {seen/el:.1f} rows/s  eta {(n_opt-step)*el/step/60:.0f} min", flush=True)
            if step in save_at:
                ck = os.path.join(args.out, f"frac_{save_at[step]}")
                model.save_pretrained(ck); tok.save_pretrained(ck)
                hist.append({"frac": save_at[step], "step": step,
                             "train_loss": round(run / seen, 6),
                             "seconds": round(time.time() - t0, 1)})
                print(f"  saved {ck}", flush=True)
            if step >= n_opt:
                stop = True
        if stop:
            break

    peak = None
    try:
        if dev == "mps":
            peak = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        elif dev == "cuda":
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    cfg = {
        "model_id": MODEL_ID, "method": "LoRA (peft)", "full_parameter_training": False,
        "objective": "soft-target cross-entropy over the native 2-way {no,yes} softmax",
        "targets": GAIN, "gain_convention": "frozen official Task 1 (NOT [7,3,1,0])",
        "labels_collapsed_to_binary": False,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": lcfg.target_modules},
        "trainable_params": int(trainable), "total_params": int(total),
        "trainable_pct": round(100 * trainable / total, 4),
        "lr": args.lr, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_length": args.max_length, "seed": args.seed, "epochs": 1,
        "optimizer_steps_planned": int(n_opt), "optimizer_steps_done": int(step),
        "rows_seen": int(seen), "train_rows_total": int(len(df)),
        "train_queries": int(df["query_id"].nunique()),
        "subset_sha256": man["sha256_example_ids"],
        "device": dev, "dtype": "float32 base + LoRA",
        "wall_time_seconds": round(time.time() - t_start, 1),
        "training_seconds": round(time.time() - t0, 1),
        "rows_per_sec": round(seen / max(time.time() - t0, 1e-9), 2),
        "peak_device_memory_gb": peak,
        "final_train_loss": round(run / max(seen, 1), 6),
        "history": hist, "is_smoke": bool(args.max_steps),
        "dev_leak_check": "PASS (asserted: no DEV query in training data)",
        "test_split_touched": False,
    }
    json.dump(cfg, open(os.path.join(args.out, "train_config.json"), "w"), indent=2, default=str)
    print(f"\ndone: {step} steps, loss {run/max(seen,1):.4f}, "
          f"{cfg['training_seconds']:.0f}s, {cfg['rows_per_sec']:.1f} rows/s, peak {peak} GB")


if __name__ == "__main__":
    main()
