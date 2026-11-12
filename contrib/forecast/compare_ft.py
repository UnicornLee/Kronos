#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare PRETRAINED kronos-base against a FINETUNED base on the same windows.

Walk-forward over windows that end inside the validation period, so ground
truth exists: for each window we feed the previous 512 bars and compare the
7-bar forecast against the real bars (close MAPE + direction hit).

Usable for any asset:  --symbol BTC --ft-dir .../runs/btc_1d_base
"""
import argparse
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

COLS = ["open", "high", "low", "close", "volume", "amount"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", help="BTC / ETH / XAUT")
    ap.add_argument("--timeframe", default="1D")
    ap.add_argument("--ft-dir", default=None, help="finetuned run dir (default runs/<sym>_1d_base)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=16, help="sampling paths per window")
    ap.add_argument("--windows", type=int, default=8, help="number of walk-forward windows")
    ap.add_argument("--stride", type=int, default=21, help="days between windows")
    ap.add_argument("--lookback", type=int, default=512)
    ap.add_argument("--pred", type=int, default=7)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--pretrained-tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--pretrained-predictor", default="NeoQuasar/Kronos-base")
    ap.add_argument("--dry-run", action="store_true",
                    help="only validate data + walk-forward windows, do not load models")
    return ap.parse_args()


def main():
    a = parse_args()
    sym = a.symbol.upper()
    ft_dir = a.ft_dir or f"{ROOT}/finetune/runs/{sym.lower()}_1d_base"
    out_path = a.out or f"{ROOT}/finetune/runs/compare_ft_{sym.lower()}_{a.timeframe}.json"

    df = pd.read_csv(f"{ROOT}/forecast/data/{sym}_{a.timeframe}.csv")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    df[COLS] = df[COLS].astype("float64")
    print(f"[{sym}_{a.timeframe}] rows={len(df)} {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")

    train_end = int(len(df) * 0.9)
    last = len(df) - a.pred
    ends = [last - i * a.stride for i in range(a.windows)][::-1]
    ends = [e for e in ends if e >= train_end]
    if not ends:
        raise SystemExit(f"no window inside validation period (train_end={train_end}, last={last})")
    print(f"  train_end={train_end}  windows={[str(df['timestamps'].iloc[e])[:10] for e in ends]}")

    if a.dry_run:
        print(f"[dry-run] {sym}_{a.timeframe}: {len(ends)} windows OK, "
              f"ft_dir={ft_dir} (not loaded)")
        return

    def build(tok_path, mdl_path):
        tok = KronosTokenizer.from_pretrained(tok_path)
        mdl = Kronos.from_pretrained(mdl_path)
        return KronosPredictor(mdl, tok, device=a.device, max_context=a.lookback)

    def forecast(pred, end_idx, seed0):
        x_df = df.iloc[end_idx - a.lookback:end_idx][COLS].reset_index(drop=True)
        x_ts = pd.Series(df["timestamps"].iloc[end_idx - a.lookback:end_idx].values)
        y_ts = pd.Series(df["timestamps"].iloc[end_idx:end_idx + a.pred].values)
        paths = []
        for i in range(a.samples):
            torch.manual_seed(seed0 + i * 977)
            p = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                             pred_len=a.pred, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            paths.append(p[COLS].to_numpy(dtype="float64"))
        return np.stack(paths)          # (S, P, 6)

    def score(pred, tag):
        mapes, hits, day_hits, maes = [], [], [], []
        last_close = float(df["close"].iloc[ends[0] - 1])
        for k, e in enumerate(ends):
            paths = forecast(pred, e, seed0=1000 + k * 13)
            med_close = np.median(paths[:, :, 3], axis=0)
            actual = df["close"].iloc[e:e + a.pred].to_numpy(dtype="float64")
            mape = float(np.mean(np.abs(med_close - actual) / actual) * 100)
            mae = float(np.mean(np.abs(med_close - actual)))
            hit = 1.0 if (med_close[-1] - last_close) * (actual[-1] - last_close) > 0 else 0.0
            # per-day direction accuracy (day 0 has no previous close inside the window)
            dh = float(np.mean(np.sign(np.diff(med_close)) == np.sign(np.diff(actual))) * 100)
            mapes.append(mape)
            maes.append(mae)
            hits.append(hit)
            day_hits.append(dh)
            last_close = float(df["close"].iloc[e - 1])
            print(f"  [{tag}] {str(df['timestamps'].iloc[e])[:10]} "
                  f"pred_end={med_close[-1]:.2f} actual_end={actual[-1]:.2f} "
                  f"MAPE={mape:.3f}% MAE={mae:.2f} hit={int(hit)} day_dir={dh:.1f}%", flush=True)
        return {"mape_mean": float(np.mean(mapes)),
                "mae_mean": float(np.mean(maes)),
                "hit_rate": float(np.mean(hits)),
                "day_direction_mean": float(np.mean(day_hits)),
                "mapes": mapes, "hits": hits, "day_direction": day_hits}

    print(f"\n=== PRETRAINED kronos-base ({sym}_{a.timeframe}) ===", flush=True)
    pre = build(a.pretrained_tokenizer, a.pretrained_predictor)
    r_pre = score(pre, "pretrained")

    print(f"\n=== FINETUNED ({ft_dir}) ===", flush=True)
    ft = build(f"{ft_dir}/tokenizer/best_model", f"{ft_dir}/basemodel/best_model")
    r_ft = score(ft, "finetuned")

    out = {"symbol": sym, "timeframe": a.timeframe, "ft_dir": ft_dir,
           "samples": a.samples, "windows": len(ends), "lookback": a.lookback,
           "pred_len": a.pred, "train_end": train_end,
           "pretrained": r_pre, "finetuned": r_ft,
           "delta": {"mape_mean": r_ft["mape_mean"] - r_pre["mape_mean"],
                     "hit_rate": r_ft["hit_rate"] - r_pre["hit_rate"],
                     "day_direction_mean": r_ft["day_direction_mean"] - r_pre["day_direction_mean"]}}
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSUMMARY", json.dumps(out, indent=2))
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
