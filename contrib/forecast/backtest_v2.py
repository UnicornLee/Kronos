#!/usr/bin/env python3
"""Kronos walk-forward scorecard for BTC / ETH / XAUT (native MPS).

Improves on the first pass:
  * more backtest windows (1D: 12 weeks, 1H: 6 weeks) -> usable hit-rate estimate
  * naive baselines (flat = last close) so the model's edge is visible
  * temperature sensitivity (T=1.0 repo default vs T=0.7 conservative)
  * per-window detail dumped as JSON

Outputs: /Volumes/ssd990pro1t/kronos/forecast/out/backtest2_<SYM>_<BAR>.json
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
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

sys.path.insert(0, f"{ROOT}/Kronos")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

DATA = f"{ROOT}/forecast/data"
OUT = f"{ROOT}/forecast/out"
COLS = ["open", "high", "low", "close", "volume", "amount"]

MODELS = {
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
}
PLAN = {
    "1D": {"model": "base", "lookback": 500, "pred_len": 7, "samples": 24, "weeks": 12},
    "1H": {"model": "mini", "lookback": 1000, "pred_len": 168, "samples": 16, "weeks": 6},
}
SYMBOLS = ["BTC", "ETH", "XAUT"]
_P = {}


def get_pred(key):
    if key not in _P:
        mid, tid, ctx = MODELS[key]
        tok = KronosTokenizer.from_pretrained(tid)
        mdl = Kronos.from_pretrained(mid)
        _P[key] = KronosPredictor(mdl, tok, device="mps", max_context=ctx)
    return _P[key]


def load(sym, bar):
    df = pd.read_csv(f"{DATA}/{sym}_{bar}.csv")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[COLS] = df[COLS].astype("float64")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def paths_for(pred, df, x_end, lookback, pred_len, future_ts, samples, T, seed0):
    lo = max(0, x_end - lookback)
    x_df = df.iloc[lo:x_end][COLS].reset_index(drop=True)
    x_ts = pd.Series(df["timestamps"].iloc[lo:x_end].values)
    out = []
    for i in range(samples):
        torch.manual_seed(seed0 + i * 977)
        p = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(future_ts),
                         pred_len=pred_len, T=T, top_p=0.9, sample_count=1, verbose=False)
        out.append(p[COLS].to_numpy(dtype="float64"))
    return np.stack(out, axis=0)


def score(med_close, act_close):
    mape = float(np.mean(np.abs(med_close - act_close) / act_close) * 100)
    hit = bool(np.sign(med_close[-1] - act_close[0]) == np.sign(act_close[-1] - act_close[0]))
    return mape, hit


def run(sym, bar):
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
        act_c = act["close"].to_numpy()
        t0 = time.time()
        stack = paths_for(pred, df, x_end, lookback, pred_len, fut, cfg["samples"], 1.0, 500 + k * 31)
        med = np.median(stack, axis=0)[:, 3]
        stack07 = paths_for(pred, df, x_end, lookback, pred_len, fut, cfg["samples"], 0.7, 500 + k * 31)
        med07 = np.median(stack07, axis=0)[:, 3]
        mape, hit = score(med, act_c)
        mape07, hit07 = score(med07, act_c)
        flat = np.full_like(act_c, act_c[0])
        flat_mape, _ = score(flat, act_c)
        rows.append({
            "week_back": k,
            "window_start": pd.Timestamp(act["timestamps"].iloc[0]).isoformat(),
            "window_end": pd.Timestamp(act["timestamps"].iloc[-1]).isoformat(),
            "actual_start": round(float(act_c[0]), 4),
            "actual_end": round(float(act_c[-1]), 4),
            "actual_chg_pct": round((act_c[-1] / act_c[0] - 1) * 100, 3),
            "model_T1_end": round(float(med[-1]), 4),
            "model_T1_chg_pct": round((med[-1] / med[0] - 1) * 100, 3),
            "model_T1_mape_pct": round(mape, 3),
            "model_T1_hit": hit,
            "model_T07_end": round(float(med07[-1]), 4),
            "model_T07_chg_pct": round((med07[-1] / med07[0] - 1) * 100, 3),
            "model_T07_mape_pct": round(mape07, 3),
            "model_T07_hit": hit07,
            "flat_mape_pct": round(flat_mape, 3),
            "sec": round(time.time() - t0, 1),
        })
        print(f"  [{sym} {bar}] wk-{k} {rows[-1]['window_start'][:10]} "
              f"act={rows[-1]['actual_chg_pct']}% modelT1={rows[-1]['model_T1_chg_pct']}% "
              f"mape={mape:.2f}% hit={hit} flat_mape={flat_mape:.2f}%", flush=True)

    def agg(key_mape, key_hit):
        m = [r[key_mape] for r in rows]
        h = [r[key_hit] for r in rows]
        return round(float(np.mean(m)), 3), round(float(np.mean(h)), 3)

    m1, h1 = agg("model_T1_mape_pct", "model_T1_hit")
    m07, h07 = agg("model_T07_mape_pct", "model_T07_hit")
    fm = [r["flat_mape_pct"] for r in rows]
    summary = {
        "symbol": sym, "bar": bar, "model": cfg["model"], "lookback": lookback,
        "windows": len(rows),
        "T1.0": {"mean_mape_pct": m1, "direction_hit_rate": h1,
                 "max_mape_pct": round(max(r["model_T1_mape_pct"] for r in rows), 3)},
        "T0.7": {"mean_mape_pct": m07, "direction_hit_rate": h07,
                 "max_mape_pct": round(max(r["model_T07_mape_pct"] for r in rows), 3)},
        "flat_baseline_mean_mape_pct": round(float(np.mean(fm)), 3),
        "detail": rows,
    }
    with open(f"{OUT}/backtest2_{sym}_{bar}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[bt2] {sym} {bar} n={len(rows)} T1 mape={m1}% hit={h1} | T0.7 mape={m07}% hit={h07} "
          f"| flat mape={summary['flat_baseline_mean_mape_pct']}%", flush=True)
    return summary


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    allout = {}
    for sym in SYMBOLS:
        for bar in PLAN:
            try:
                allout[f"{sym}_{bar}"] = run(sym, bar)
            except Exception as exc:  # noqa: BLE001
                print(f"[err] {sym} {bar}: {exc}")
    with open(f"{OUT}/backtest2_all.json", "w", encoding="utf-8") as fh:
        json.dump(allout, fh, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
