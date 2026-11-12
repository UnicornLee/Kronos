#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate that the finetuned artifacts load and predict correctly.

Checks (before the 8h training finishes, so a bad artifact format is caught early):
  1. config.json of the finetuned dirs vs the pretrained ones
  2. KronosTokenizer.from_pretrained on the finetuned tokenizer
  3. Kronos.from_pretrained on the finetuned basemodel
  4. one real predict through KronosPredictor (CPU, to not touch MPS while training runs)
"""
import json
import os
import sys

ROOT = "/Volumes/ssd990pro1t/kronos"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf-cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, f"{ROOT}/Kronos")

import pandas as pd
import torch

from model import Kronos, KronosTokenizer, KronosPredictor

RUNS = f"{ROOT}/finetune/runs"
SMOKE = f"{RUNS}/smoke_btc_1d"
BTC_FT = f"{RUNS}/btc_1d_base"
PRE = {
    "tokenizer": f"{ROOT}/hf-cache/hub/models--NeoQuasar--Kronos-Tokenizer-base/snapshots/0e0117387f39004a9016484a186a908917e22426",
    "predictor": f"{ROOT}/hf-cache/hub/models--NeoQuasar--Kronos-base/snapshots/2b554741eca47781b64468546e77fef3e85130e6",
}

COLS = ["open", "high", "low", "close", "volume", "amount"]


def show_cfg(tag, path):
    p = os.path.join(path, "config.json")
    if not os.path.exists(p):
        print(f"  [{tag}] config.json 缺失: {p}")
        return None
    cfg = json.load(open(p))
    keep = ("d_model", "n_layers", "n_heads", "s1_bits", "s2_bits", "group_size",
            "hidden_size", "ffn_dim", "attn_dim", "n_kv_heads", "rope_base")
    small = {k: v for k, v in cfg.items() if k in keep}
    print(f"  [{tag}] {small}")
    return cfg


print("=== 1. config.json 对比 ===")
for tag, p in [("pretrained tokenizer", PRE["tokenizer"]),
               ("finetuned tokenizer(smoke)", f"{SMOKE}/tokenizer/best_model"),
               ("finetuned tokenizer(btc 30ep)", f"{BTC_FT}/tokenizer/best_model"),
               ("pretrained predictor", PRE["predictor"]),
               ("finetuned basemodel(smoke)", f"{SMOKE}/basemodel/best_model")]:
    show_cfg(tag, p)

print("\n=== 2/3. 加载微调产物 ===")
tok_ft = KronosTokenizer.from_pretrained(f"{SMOKE}/tokenizer/best_model")
attrs = {a: getattr(tok_ft, a) for a in ("d_model", "n_codes", "n_bins", "group_size")
         if hasattr(tok_ft, a)}
print(f"  finetuned tokenizer OK: {attrs}")
mdl_ft = Kronos.from_pretrained(f"{SMOKE}/basemodel/best_model")
n = sum(p.numel() for p in mdl_ft.parameters())
print(f"  finetuned basemodel OK: params={n:,}")

print("\n=== 4. 端到端 predict（CPU，不抢 MPS） ===")
df = pd.read_csv(f"{ROOT}/forecast/data/BTC_1D.csv")
df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)
df[COLS] = df[COLS].astype("float64")
end = len(df) - 7
x_df = df.iloc[end - 512:end][COLS].reset_index(drop=True)
x_ts = pd.Series(df["timestamps"].iloc[end - 512:end].values)
y_ts = pd.Series(df["timestamps"].iloc[end:end + 7].values)

for tag, tok, mdl in [("finetuned(smoke)", tok_ft, mdl_ft),
                      ("pretrained", KronosTokenizer.from_pretrained(PRE["tokenizer"]),
                       Kronos.from_pretrained(PRE["predictor"]))]:
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=512)
    torch.manual_seed(20261112)
    p = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                     pred_len=7, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    close = p["close"].to_numpy(dtype="float64")
    actual = df["close"].iloc[end:end + 7].to_numpy(dtype="float64")
    mape = float((abs(close - actual) / actual).mean() * 100)
    print(f"  [{tag}] pred_close={np.round(close, 2) if False else close.round(2)}")
    print(f"  [{tag}] actual    ={actual.round(2)}")
    print(f"  [{tag}] MAPE={mape:.3f}%")

print("\nALL OK")
