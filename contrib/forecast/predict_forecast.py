#!/usr/bin/env python3
"""Kronos future-week forecast for BTC / ETH / XAUT (native MPS, no Docker).

Two views per symbol:
  * 1D  -> kronos-base (ctx 512)  : lookback 500 daily bars, pred_len 7  (next 7 days)
  * 1H  -> kronos-mini (ctx 2048) : lookback 1000 hourly bars, pred_len 168 (next 7 days)

Uncertainty: `predict()` averages samples internally, so we call it N times with
different torch seeds (sample_count=1) and keep every path -> median + p10/p90 band.

Backtest: the same procedure is re-run on the most recent past weeks that we have
real data for, to score the forecast against ground truth (close MAPE, direction hit).

Outputs (under /Volumes/ssd990pro1t/kronos/forecast/out):
  forecast_<SYM>_<BAR>.csv / .json      future path with median + band
  backtest_<SYM>_<BAR>.json             scorecard of the walk-forward test
  summary.md                            human-readable report
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
    "1D": {"model": "base", "lookback": 500, "pred_len": 7, "samples": 16, "backtest_weeks": 4},
    "1H": {"model": "mini", "lookback": 1000, "pred_len": 168, "samples": 12, "backtest_weeks": 2},
}

SYMBOLS = ["BTC", "ETH", "XAUT"]
_PREDICTORS = {}


def get_predictor(key):
    if key not in _PREDICTORS:
        model_id, tok_id, ctx = MODELS[key]
        print(f"[load] {key}: {model_id} + {tok_id} (ctx={ctx})", flush=True)
        tok = KronosTokenizer.from_pretrained(tok_id)
        mdl = Kronos.from_pretrained(model_id)
        _PREDICTORS[key] = KronosPredictor(mdl, tok, device="mps", max_context=ctx)
    return _PREDICTORS[key]


def load_csv(sym, bar):
    path = f"{DATA}/{sym}_{bar}.csv"
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[COLS] = df[COLS].astype("float64")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def step_of(df):
    d = df["timestamps"].diff().dropna()
    return d.median()


def run_paths(pred, df, x_end, lookback, pred_len, future_ts, samples, seed0):
    """Return list of (pred_len, 6) arrays, one per sampled path."""
    x_df = df.iloc[max(0, x_end - lookback):x_end][COLS].reset_index(drop=True)
    x_ts = pd.Series(df["timestamps"].iloc[max(0, x_end - lookback):x_end].values)
    paths = []
    for i in range(samples):
        torch.manual_seed(seed0 + i * 977)
        pred_df = pred.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=pd.Series(future_ts),
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        paths.append(pred_df[COLS].to_numpy(dtype="float64"))
    return paths


def summarize(paths, future_ts, last_close):
    stack = np.stack(paths, axis=0)                 # (S, P, 6)
    close = stack[:, :, 3]
    med = np.median(stack, axis=0)                  # (P, 6)
    p10 = np.percentile(close, 10, axis=0)
    p90 = np.percentile(close, 90, axis=0)
    rows = []
    for i in range(med.shape[0]):
        rows.append({
            "timestamp": pd.Timestamp(future_ts[i]).isoformat(),
            "open": round(float(med[i, 0]), 4),
            "high": round(float(med[i, 1]), 4),
            "low": round(float(med[i, 2]), 4),
            "close": round(float(med[i, 3]), 4),
            "close_p10": round(float(p10[i]), 4),
            "close_p90": round(float(p90[i]), 4),
        })
    end = med[-1, 3]
    return {
        "rows": rows,
        "last_actual_close": round(float(last_close), 4),
        "median_end_close": round(float(end), 4),
        "median_change_pct": round((end / last_close - 1) * 100, 3),
        "band_end_p10": round(float(p10[-1]), 4),
        "band_end_p90": round(float(p90[-1]), 4),
        "band_end_change_pct": [
            round((float(p10[-1]) / last_close - 1) * 100, 3),
            round((float(p90[-1]) / last_close - 1) * 100, 3),
        ],
        "prob_up": round(float((close[:, -1] > last_close).mean()), 3),
        "samples": len(paths),
    }


def forecast_symbol(sym, bar):
    cfg = PLAN[bar]
    df = load_csv(sym, bar)
    step = step_of(df)
    pred_len = cfg["pred_len"]
    n = len(df)
    if n < cfg["lookback"] + pred_len + 1:
        cfg = dict(cfg)
        cfg["lookback"] = max(60, n - pred_len - 1)
    pred = get_predictor(cfg["model"])

    last_ts = df["timestamps"].iloc[-1]
    future_ts = pd.date_range(start=last_ts + step, periods=pred_len, freq=step)
    t0 = time.time()
    paths = run_paths(pred, df, n, cfg["lookback"], pred_len, future_ts, cfg["samples"], 1234)
    summary = summarize(paths, future_ts, df["close"].iloc[-1])
    summary.update({
        "symbol": sym,
        "bar": bar,
        "model": cfg["model"],
        "lookback": cfg["lookback"],
        "pred_len": pred_len,
        "step": str(step),
        "history_from": pd.Timestamp(df["timestamps"].iloc[0]).isoformat(),
        "history_to": pd.Timestamp(last_ts).isoformat(),
        "history_rows": n,
        "elapsed_sec": round(time.time() - t0, 1),
    })
    return summary, df, cfg


def backtest(sym, bar):
    """Walk-forward: forecast k past weeks, score against the real bars."""
    cfg = PLAN[bar]
    df = load_csv(sym, bar)
    n = len(df)
    pred_len = cfg["pred_len"]
    lookback = cfg["lookback"]
    if n < lookback + pred_len * (cfg["backtest_weeks"] + 1):
        lookback = max(60, n - pred_len * (cfg["backtest_weeks"] + 1) - 1)
    pred = get_predictor(cfg["model"])
    step = step_of(df)
    results = []
    for k in range(1, cfg["backtest_weeks"] + 1):
        x_end = n - k * pred_len
        if x_end - lookback < 30:
            break
        actual = df.iloc[x_end:x_end + pred_len]
        if len(actual) < pred_len:
            break
        future_ts = df["timestamps"].iloc[x_end:x_end + pred_len].values
        paths = run_paths(pred, df, x_end, lookback, pred_len, future_ts, cfg["samples"], 4321 + k * 13)
        med = np.median(np.stack(paths, axis=0), axis=0)[:, 3]
        act = actual["close"].to_numpy()
        mape = float(np.mean(np.abs(med - act) / act) * 100)
        pred_dir = np.sign(med[-1] - act[0])
        act_dir = np.sign(act[-1] - act[0])
        results.append({
            "week_index_back": k,
            "window_start": pd.Timestamp(actual["timestamps"].iloc[0]).isoformat(),
            "window_end": pd.Timestamp(actual["timestamps"].iloc[-1]).isoformat(),
            "actual_start": round(float(act[0]), 4),
            "actual_end": round(float(act[-1]), 4),
            "pred_start": round(float(med[0]), 4),
            "pred_end": round(float(med[-1]), 4),
            "actual_change_pct": round((act[-1] / act[0] - 1) * 100, 3),
            "pred_change_pct": round((med[-1] / med[0] - 1) * 100, 3),
            "close_mape_pct": round(mape, 3),
            "direction_hit": bool(pred_dir == act_dir),
            "end_error_pct": round((med[-1] - act[-1]) / act[-1] * 100, 3),
        })
    if not results:
        return {"symbol": sym, "bar": bar, "weeks": []}
    mape_list = [r["close_mape_pct"] for r in results]
    hit = [r["direction_hit"] for r in results]
    return {
        "symbol": sym,
        "bar": bar,
        "model": cfg["model"],
        "lookback": lookback,
        "mean_close_mape_pct": round(float(np.mean(mape_list)), 3),
        "max_close_mape_pct": round(float(np.max(mape_list)), 3),
        "direction_hit_rate": round(float(np.mean(hit)), 3),
        "weeks": results,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    all_fc, all_bt = {}, {}
    for sym in SYMBOLS:
        for bar in PLAN:
            try:
                summary, df, _ = forecast_symbol(sym, bar)
            except FileNotFoundError:
                print(f"[miss] {sym} {bar}: no csv")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[err] forecast {sym} {bar}: {exc}")
                continue
            all_fc[f"{sym}_{bar}"] = summary
            tag = f"{sym}_{bar}"
            with open(f"{OUT}/forecast_{tag}.json", "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            pd.DataFrame(summary["rows"]).to_csv(f"{OUT}/forecast_{tag}.csv", index=False)
            print(f"[fc] {tag} model={summary['model']} end={summary['median_end_close']} "
                  f"chg={summary['median_change_pct']}% band={summary['band_end_change_pct']} "
                  f"p_up={summary['prob_up']} ({summary['elapsed_sec']}s)", flush=True)

            try:
                bt = backtest(sym, bar)
            except Exception as exc:  # noqa: BLE001
                print(f"[err] backtest {sym} {bar}: {exc}")
                continue
            all_bt[f"{sym}_{bar}"] = bt
            with open(f"{OUT}/backtest_{tag}.json", "w", encoding="utf-8") as fh:
                json.dump(bt, fh, indent=2)
            if bt.get("weeks"):
                print(f"[bt] {tag} mape={bt['mean_close_mape_pct']}% "
                      f"max={bt['max_close_mape_pct']}% hit={bt['direction_hit_rate']}", flush=True)

    with open(f"{OUT}/all_forecasts.json", "w", encoding="utf-8") as fh:
        json.dump({"forecasts": all_fc, "backtests": all_bt}, fh, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
