#!/usr/bin/env python3
"""Render Kronos forecast + backtest charts (PNG) on Windows from the JSON artifacts.

Inputs (copied from the Mac):
  data/<SYM>_<BAR>.csv            history OHLCV (timestamps,open,high,low,close,volume,amount)
  out/forecast_<SYM>_<BAR>.json    forecast median path + p10/p90 band
  out/backtest2_<SYM>_<BAR>.json   walk-forward scorecard (T1.0 / T0.7 / flat baseline)
  out/forecast_calibrated_<SYM>_<BAR>.json  bias-corrected headline numbers

Outputs:
  charts/forecast_<SYM>_<BAR>.png   history + median path + uncertainty band
  charts/backtest_<SYM>_<BAR>.png   per-window actual vs predicted weekly change
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = r"D:\DSH\kronos-forecast"
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
CHARTS = os.path.join(ROOT, "charts")
os.makedirs(CHARTS, exist_ok=True)

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SYMBOLS = ["BTC", "ETH", "XAUT"]
BARS = ["1D", "1H"]
LABEL = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "XAUT": "XAUT/USDT (Tether Gold)"}
HISTORY_BARS = {"1D": 180, "1H": 24 * 14}
CLR = {"BTC": "#f7931a", "ETH": "#627eea", "XAUT": "#d4a017"}


def read_json(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_history(sym, bar):
    path = os.path.join(DATA, f"{sym}_{bar}.csv")
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.sort_values("timestamps").reset_index(drop=True)


def chart_forecast(sym, bar):
    fc = read_json(f"forecast_{sym}_{bar}.json")
    if not fc:
        return None
    cal = read_json(f"forecast_calibrated_{sym}_{bar}.json") or {}
    hist = load_history(sym, bar)
    rows = pd.DataFrame(fc["rows"])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"])

    h = hist.tail(HISTORY_BARS[bar])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(h["timestamps"], h["close"], color="#4a5568", lw=1.5, label="历史收盘", zorder=3)
    ax.fill_between(rows["timestamp"], rows["close_p10"], rows["close_p90"],
                    color=CLR[sym], alpha=0.20, label="P10–P90 区间", zorder=2)
    ax.plot(rows["timestamp"], rows["close"], color=CLR[sym], lw=2.3, marker="o", ms=3.2,
            label="中位预测", zorder=4)
    ax.axhline(fc["last_actual_close"], color="#c05621", lw=1.0, ls="--",
               label=f"最新收盘 {fc['last_actual_close']:,.2f}", zorder=1)

    last_hist_ts = h["timestamps"].iloc[-1]
    ax.axvline(last_hist_ts, color="#718096", lw=1.0, ls=":")
    ax.annotate("预测起点", xy=(last_hist_ts, ax.get_ylim()[0]),
                xytext=(4, 6), textcoords="offset points", fontsize=8, color="#4a5568")

    band = fc["band_end_change_pct"]
    cal_txt = ""
    if cal.get("status") == "ok":
        ax.scatter([rows["timestamp"].iloc[-1]], [cal["bias_corrected_end"]],
                   marker="D", s=55, color="#22543d", zorder=6,
                   label=f"偏差校正末值 {cal['bias_corrected_end']:,.2f}")
        cal_txt = (f"\n偏差校正 {cal['bias_corrected_change_pct']:+.2f}%"
                   f"（模型偏差 {cal['model_bias_pct']:+.2f}%）")
    title = (f"{LABEL[sym]} · {bar} 未来一周 Kronos 预测\n"
             f"末值中位 {fc['median_end_close']:,.2f} "
             f"({fc['median_change_pct']:+.2f}%)  "
             f"区间 {band[0]:+.2f}% ~ {band[1]:+.2f}%  "
             f"上涨概率 {fc['prob_up']:.0%}  "
             f"模型 {fc['model']} · {fc['samples']} 采样{cal_txt}")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("时间 (UTC)")
    ax.set_ylabel("收盘价 (USDT)")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)
    if bar == "1D":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(CHARTS, f"forecast_{sym}_{bar}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_backtest(sym, bar):
    bt = read_json(f"backtest2_{sym}_{bar}.json")
    if not bt or not bt.get("detail"):
        return None
    detail = bt["detail"]
    labels = [d["window_start"][:10] for d in detail]
    actual = [d["actual_chg_pct"] for d in detail]
    pred = [d["model_T1_chg_pct"] for d in detail]
    pred07 = [d["model_T07_chg_pct"] for d in detail]
    mape = [d["model_T1_mape_pct"] for d in detail]
    flat = [d["flat_mape_pct"] for d in detail]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(labels))
    ax1.bar(x - 0.27, actual, width=0.26, label="实际周涨跌%", color="#38a169")
    ax1.bar(x, pred, width=0.26, label="模型 T=1.0", color="#4299e1")
    ax1.bar(x + 0.27, pred07, width=0.26, label="模型 T=0.7", color="#9f7aea")
    ax1.axhline(0, color="#4a5568", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    hit = bt["T1.0"]["direction_hit_rate"]
    ax1.set_title(f"{LABEL[sym]} {bar} 走查回测：周涨跌方向（方向命中 {hit:.0%}）", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2, axis="y", ls=":")

    ax2.bar(x - 0.2, mape, width=0.4, label=f"模型 T=1.0 MAPE {bt['T1.0']['mean_mape_pct']}%", color="#4299e1")
    ax2.bar(x + 0.2, flat, width=0.4, label=f"朴素基线 MAPE {bt['flat_baseline_mean_mape_pct']}%", color="#a0aec0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_title(f"{LABEL[sym]} {bar} 收盘价误差（%），窗口数 {len(detail)}", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2, axis="y", ls=":")
    fig.tight_layout()
    path = os.path.join(CHARTS, f"backtest_{sym}_{bar}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    made = []
    for sym in SYMBOLS:
        for bar in BARS:
            p = chart_forecast(sym, bar)
            if p:
                made.append(p)
            p = chart_backtest(sym, bar)
            if p:
                made.append(p)
    for p in made:
        print(p)


if __name__ == "__main__":
    main()
