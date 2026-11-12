#!/usr/bin/env python3
"""Bias-correct the Kronos weekly forecast using its own walk-forward scorecard.

Why: the walk-forward backtest measures the model's systematic error in the
*predicted weekly change*.  If the model habitually says "+5%" while reality
comes in at "-1%", that gap is measurable and can be subtracted from the fresh
forecast.

Sign convention (kept explicit so it can't be flipped by accident):
    err_i  = model_chg_i - actual_chg_i      # per window
    bias   = mean(err_i)                    # >0 => the model is too bullish
    corrected_chg = raw_chg - bias

Inputs (Windows side):
  out/forecast_<SYM>_<BAR>.json     raw forecast from the Mac
  out/backtest2_<SYM>_<BAR>.json    walk-forward scorecard (T1.0 column)

Outputs:
  out/forecast_calibrated_<SYM>_<BAR>.json
  out/calibration_summary.json
"""
import json
import os

ROOT = r"D:\DSH\kronos-forecast"
OUT = os.path.join(ROOT, "out")

SYMBOLS = ["BTC", "ETH", "XAUT"]
BARS = ["1D", "1H"]
# The forecast is produced with T=1.0, so score against the T1.0 columns.
T_KEY = "model_T1_chg_pct"
HIT_KEY = "model_T1_hit"


def read_json(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def calibrate(sym, bar):
    fc = read_json(f"forecast_{sym}_{bar}.json")
    bt = read_json(f"backtest2_{sym}_{bar}.json")
    if not fc:
        return None
    if not bt or not bt.get("detail"):
        return {"symbol": sym, "bar": bar, "status": "no_backtest",
                "raw_change_pct": fc["median_change_pct"]}

    detail = bt["detail"]
    errs = [r[T_KEY] - r["actual_chg_pct"] for r in detail]   # model minus reality
    bias = sum(errs) / len(errs)                               # >0 => model too bullish
    raw = fc["median_change_pct"]
    corrected = raw - bias
    last = fc["last_actual_close"]
    end = last * (1 + corrected / 100.0)
    band = fc["band_end_change_pct"]
    half = (band[1] - band[0]) / 2.0

    # direction hit-rate of the corrected signal on the same windows
    hits = 0
    for r in detail:
        c = r[T_KEY] - bias
        pred_dir = 1 if c > 0 else (-1 if c < 0 else 0)
        act_dir = 1 if r["actual_chg_pct"] > 0 else (-1 if r["actual_chg_pct"] < 0 else 0)
        if pred_dir == act_dir:
            hits += 1

    # probability that the corrected end price is above the last close
    p_up = fc["prob_up"]
    prob_up_corrected = min(1.0, max(0.0, p_up + bias / 100.0))

    out = {
        "symbol": sym,
        "bar": bar,
        "model": fc["model"],
        "status": "ok",
        "windows": len(detail),
        "raw_change_pct": raw,
        "raw_end": fc["median_end_close"],
        "model_bias_pct": round(bias, 3),
        "bias_corrected_change_pct": round(corrected, 3),
        "bias_corrected_end": round(end, 4),
        "corrected_band_pct": [round(corrected - half, 3), round(corrected + half, 3)],
        "backtest_mean_mape_pct": bt["T1.0"]["mean_mape_pct"],
        "backtest_direction_hit_raw": bt["T1.0"]["direction_hit_rate"],
        "backtest_direction_hit_corrected": round(hits / len(detail), 3),
        "flat_baseline_mean_mape_pct": bt["flat_baseline_mean_mape_pct"],
        "prob_up_raw": p_up,
        "prob_up_corrected": round(prob_up_corrected, 3),
        "note": "bias from the walk-forward windows; small-sample, treat as indicative",
    }
    return out


def main():
    rows = {}
    for sym in SYMBOLS:
        for bar in BARS:
            r = calibrate(sym, bar)
            if r:
                rows[f"{sym}_{bar}"] = r
                p = os.path.join(OUT, f"forecast_calibrated_{sym}_{bar}.json")
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(r, fh, indent=2)
                print(f"[cal] {sym} {bar}: raw={r.get('raw_change_pct')}% -> "
                      f"corrected={r.get('bias_corrected_change_pct')}% "
                      f"(model bias {r.get('model_bias_pct')}%) hit "
                      f"{r.get('backtest_direction_hit_raw')} -> {r.get('backtest_direction_hit_corrected')}")
    with open(os.path.join(OUT, "calibration_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
