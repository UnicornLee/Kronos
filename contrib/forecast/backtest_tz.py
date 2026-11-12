#!/usr/bin/env python3
"""Timezone-alignment experiment for Kronos time features.

Kronos derives 5 time features (minute/hour/weekday/day/month) from the
timestamps.  OKX labels its candles in UTC+8, but our CSVs store UTC, so daily
bars land on 16:00 and the hour-of-day feature is shifted by 8.

This script reruns a compact walk-forward backtest with the timestamps shifted
by TZ_SHIFT hours (0 = as fetched UTC, 8 = exchange-local UTC+8) and reports
which alignment the model actually scores better on.

usage: backtest_tz.py <tz_shift_hours>
"""
import json
import os
import sys
import time

ROOT = "/Volumes/ssd990pro1t/kronos"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf-cache")
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("http_proxy", "http://127.0.0.1:7897")
os.environ.setdefault("https_proxy", "http://127.0.0.1:7897")

sys.path.insert(0, f"{ROOT}/Kronos")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

TZ = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DATA = f"{ROOT}/forecast/data"
OUT = f"{ROOT}/forecast/out"
COLS = ["open", "high", "low", "close", "volume", "amount"]
MODELS = {
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
}
PLAN = {
    "1D": {"model": "base", "lookback": 500, "pred_len": 7, "samples": 16, "weeks": 8},
    "1H": {"model": "mini", "lookback": 1000, "pred_len": 168, "samples": 12, "weeks": 4},
}
SYMBOLS = ["BTC", "ETH", "XAUT"]
_P = {}


def get_pred(key):
    if key not in _P:
        mid, tid, ctx = MODELS[key]
        _P[key] = KronosPredictor(Kronos.from_pretrained(mid),
                                   KronosTokenizer.from_pretrained(tid),
                                   device="mps", max_context=ctx)
    return _P[key]


def load(sym, bar):
    df = pd.read_csv(f"{DATA}/{sym}_{bar}.csv")
    df["timestamps"] = pd.to_datetime(df["timestamps"]) + pd.Timedelta(hours=TZ)
    df = df.sort_values("timestamps").reset_index(drop=True)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[COLS] = df[COLS].astype("float64")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    res = {}
    for sym in SYMBOLS:
        for bar in PLAN:
            cfg = PLAN[bar]
            df = load(sym, bar)
            n = len(df)
            pred_len = cfg["pred_len"]
            lookback = cfg["lookback"]
            if n < lookback + pred_len * (cfg["weeks"] + 1):
                lookback = max(60, n - pred_len * (cfg["weeks"] + 1) - 1)
            pred = get_pred(cfg["model"])
            rows = []
            for k in range(1, cfg["weeks"] + 1):
                x_end = n - k * pred_len
                if x_end - lookback < 30:
                    break
                act = df.iloc[x_end:x_end + pred_len]
                if len(act) < pred_len:
                    break
                fut = df["timestamps"].iloc[x_end:x_end + pred_len].values
                lo = max(0, x_end - lookback)
                x_df = df.iloc[lo:x_end][COLS].reset_index(drop=True)
                x_ts = pd.Series(df["timestamps"].iloc[lo:x_end].values)
                paths = []
                for i in range(cfg["samples"]):
                    torch.manual_seed(900 + k * 41 + i * 977)
                    p = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(fut),
                                     pred_len=pred_len, T=1.0, top_p=0.9,
                                     sample_count=1, verbose=False)
                    paths.append(p["close"].to_numpy(dtype="float64"))
                med = np.median(np.stack(paths, axis=0), axis=0)
                act_c = act["close"].to_numpy()
                mape = float(np.mean(np.abs(med - act_c) / act_c) * 100)
                hit = bool(np.sign(med[-1] - act_c[0]) == np.sign(act_c[-1] - act_c[0]))
                flat = float(np.mean(np.abs(act_c[0] - act_c) / act_c) * 100)
                rows.append({"mape": round(mape, 3), "hit": hit, "flat": round(flat, 3),
                             "act_chg": round((act_c[-1] / act_c[0] - 1) * 100, 3),
                             "pred_chg": round((med[-1] / med[0] - 1) * 100, 3)})
                print(f"  [{sym} {bar} tz{TZ:+d}] wk-{k} mape={mape:.2f}% hit={hit} "
                      f"flat={flat:.2f}% act={rows[-1]['act_chg']}% pred={rows[-1]['pred_chg']}%",
                      flush=True)
            if rows:
                res[f"{sym}_{bar}"] = {
                    "tz_shift": TZ,
                    "n": len(rows),
                    "mean_mape_pct": round(float(np.mean([r["mape"] for r in rows])), 3),
                    "hit_rate": round(float(np.mean([r["hit"] for r in rows])), 3),
                    "flat_mape_pct": round(float(np.mean([r["flat"] for r in rows])), 3),
                    "detail": rows,
                }
                print(f"[tz{TZ:+d}] {sym} {bar} mape={res[f'{sym}_{bar}']['mean_mape_pct']}% "
                      f"hit={res[f'{sym}_{bar}']['hit_rate']} flat={res[f'{sym}_{bar}']['flat_mape_pct']}%",
                      flush=True)
    with open(f"{OUT}/tz{TZ}_all.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
