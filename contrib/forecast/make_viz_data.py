#!/usr/bin/env python3
"""Build viz_data.json for the in-chat interactive card (compact payload).

Short keys on purpose: the card embeds this JSON inline.
  h  history closes      m  median path      a  p10 path      b  p90 path
  t0 first history epoch (UTC)   ts history step (s)   ft0 first forecast epoch
  fs forecast step (s)   fe last forecast epoch
  l  last actual close   e  median end close  c  median change %
  d  end band [lo, hi] %  u  prob up           o  model         n  samples
  g  [bias, corrected %, corrected end]        k  [bt n, mape, hit, flat]
  w  [window labels, actual chg %, model T1.0 chg %]
"""
import json
import os

import pandas as pd

ROOT = r"D:\DSH\kronos-forecast"
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
VIZ = os.path.join(ROOT, "viz_data.json")

SYMBOLS = ["BTC", "ETH", "XAUT"]
BARS = ["1D", "1H"]
HIST_KEEP = {"1D": 40, "1H": 24 * 8}
HIST_STEP = {"1D": 1, "1H": 4}
FC_STEP = {"1D": 1, "1H": 8}


def rj(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ep(s):
    return int(pd.Timestamp(s).tz_localize("UTC").timestamp())


def main():
    out = {}
    for sym in SYMBOLS:
        out[sym] = {}
        for bar in BARS:
            fc = rj(f"forecast_{sym}_{bar}.json")
            if not fc:
                continue
            hist = pd.read_csv(os.path.join(DATA, f"{sym}_{bar}.csv"))
            hist["timestamps"] = pd.to_datetime(hist["timestamps"])
            hist = hist.sort_values("timestamps").tail(HIST_KEEP[bar]).iloc[::HIST_STEP[bar]]
            rows = fc["rows"][::FC_STEP[bar]]

            t0 = ep(hist["timestamps"].iloc[0])
            ts = int((hist["timestamps"].iloc[-1] - hist["timestamps"].iloc[0]).total_seconds()
                      / max(1, len(hist) - 1))
            cal = rj(f"forecast_calibrated_{sym}_{bar}.json") or {}
            bt = rj(f"backtest2_{sym}_{bar}.json") or {}
            b1 = (bt.get("T1.0") or {})
            det = bt.get("detail") or []

            out[sym][bar] = {
                "h": [int(round(float(v))) for v in hist["close"]],
                "m": [int(round(r["close"])) for r in rows],
                "a": [int(round(r["close_p10"])) for r in rows],
                "b": [int(round(r["close_p90"])) for r in rows],
                "t0": t0,
                "ts": ts,
                "ft0": ep(rows[0]["timestamp"]),
                "fs": int((ep(rows[-1]["timestamp"]) - ep(rows[0]["timestamp"]))
                          / max(1, len(rows) - 1)),
                "fe": ep(rows[-1]["timestamp"]),
                "l": fc["last_actual_close"],
                "e": fc["median_end_close"],
                "c": fc["median_change_pct"],
                "d": fc["band_end_change_pct"],
                "u": fc["prob_up"],
                "o": fc["model"],
                "n": fc["samples"],
                "g": [cal.get("model_bias_pct"), cal.get("bias_corrected_change_pct"),
                      cal.get("bias_corrected_end")] if cal.get("status") == "ok" else None,
                "k": [bt.get("windows"), b1.get("mean_mape_pct"),
                      b1.get("direction_hit_rate"), bt.get("flat_baseline_mean_mape_pct")],
                "w": [[str(d.get("window_start", ""))[5:10] for d in det],
                      [d.get("actual_chg_pct") for d in det],
                      [d.get("model_T1_chg_pct") for d in det]],
            }

    out["_t"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    with open(VIZ, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(VIZ, os.path.getsize(VIZ), "bytes")


if __name__ == "__main__":
    main()
