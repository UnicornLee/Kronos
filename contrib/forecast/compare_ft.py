#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare PRETRAINED kronos-base against the 1-epoch FINETUNED base
(finetune_csv smoke run) on the same BTC_1D windows.

Walk-forward on windows that end inside the validation period, so ground truth
exists: for each window we feed the previous 512 daily bars and compare the
7-bar forecast against the real bars (close MAPE + direction hit).
"""
import json
import os
import sys

ROOT = "/Volumes/ssd990pro1t/kronos"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf-cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, f"{ROOT}/Kronos")

import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosTokenizer, KronosPredictor

FT = f"{ROOT}/finetune/runs/smoke_btc_1d"
COLS = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK, PRED = 512, 7
SAMPLES = 8

df = pd.read_csv(f"{ROOT}/forecast/data/BTC_1D.csv")
df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)
df[COLS] = df[COLS].astype("float64")
print(f"data rows={len(df)} {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")

# window end indices inside the validation period (train split ended 2025-12-21)
train_end = int(len(df) * 0.9)
last = len(df) - PRED
ends = [last - i * 21 for i in range(8)][::-1]  # last 8 weekly windows, all after the train split
print("window ends:", [str(df['timestamps'].iloc[e]) for e in ends])


def build(tok_path, mdl_path):
    tok = KronosTokenizer.from_pretrained(tok_path)
    mdl = Kronos.from_pretrained(mdl_path)
    return KronosPredictor(mdl, tok, device="mps", max_context=512)


def forecast(pred, end_idx, seed0):
    x_df = df.iloc[end_idx - LOOKBACK:end_idx][COLS].reset_index(drop=True)
    x_ts = pd.Series(df["timestamps"].iloc[end_idx - LOOKBACK:end_idx].values)
    y_ts = pd.Series(df["timestamps"].iloc[end_idx:end_idx + PRED].values)
    paths = []
    for i in range(SAMPLES):
        torch.manual_seed(seed0 + i * 977)
        p = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                         pred_len=PRED, T=1.0, top_p=0.9, sample_count=1, verbose=False)
        paths.append(p[COLS].to_numpy(dtype="float64"))
    return np.stack(paths)          # (S, P, 6)


def score(pred, tag):
    mapes, hits = [], []
    last_close = float(df["close"].iloc[ends[0] - 1])
    for k, e in enumerate(ends):
        paths = forecast(pred, e, seed0=1000 + k * 13)
        med_close = np.median(paths[:, :, 3], axis=0)
        actual = df["close"].iloc[e:e + PRED].to_numpy(dtype="float64")
        mape = float(np.mean(np.abs(med_close - actual) / actual) * 100)
        hit = 1.0 if (med_close[-1] - last_close) * (actual[-1] - last_close) > 0 else 0.0
        mapes.append(mape)
        hits.append(hit)
        last_close = float(df["close"].iloc[e - 1])
        print(f"  [{tag}] {str(df['timestamps'].iloc[e])[:10]} "
              f"pred_end={med_close[-1]:.1f} actual_end={actual[-1]:.1f} "
              f"MAPE={mape:.3f}% hit={int(hit)}")
    return {"mape_mean": float(np.mean(mapes)), "hit_rate": float(np.mean(hits)), "mapes": mapes}


print("\n=== PRETRAINED kronos-base ===", flush=True)
pre = build("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base")
r_pre = score(pre, "pretrained")

print("\n=== FINETUNED (1 epoch, BTC_1D) ===", flush=True)
ft = build(f"{FT}/tokenizer/best_model", f"{FT}/basemodel/best_model")
r_ft = score(ft, "finetuned")

out = {"pretrained": r_pre, "finetuned": r_ft}
with open(f"{ROOT}/finetune/runs/compare_ft_btc_1d.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nSUMMARY", json.dumps(out, indent=2))

