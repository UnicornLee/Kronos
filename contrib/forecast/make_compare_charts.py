#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Charts for the finetune-vs-pretrained walk-forward comparison.

Reads compare_ft_<sym>_1D_ft.json (produced on the Mac) and renders:
  compare_mape.png      grouped bars: pretrained vs finetuned MAPE per asset
  compare_hit.png       direction hit rate + per-day direction accuracy
  compare_windows.png   per-window MAPE curves (3 assets x 2 models)
  compare_delta.png     delta bars (finetuned - pretrained)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for cand in ("Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Arial Unicode MS"):
    if any(cand.lower() == f.name.lower() for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False

IN_DIR = sys.argv[1] if len(sys.argv) > 1 else r"D:\DSH\kronos-forecast\out\compare"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else r"D:\DSH\kronos-forecast\charts"
os.makedirs(OUT_DIR, exist_ok=True)

ORDER = ["BTC", "ETH", "XAUT"]
LABEL = {"BTC": "BTC", "ETH": "ETH", "XAUT": "XAUT(黄金)"}
C_PRE, C_FT = "#8a8f98", "#e0632a"

data = {}
for sym in ORDER:
    p = os.path.join(IN_DIR, f"compare_ft_{sym.lower()}_1D_ft.json")
    if os.path.exists(p):
        data[sym] = json.load(open(p))
    else:
        print(f"missing: {p}")

if not data:
    print("no compare results found in", IN_DIR)
    sys.exit(1)

syms = [s for s in ORDER if s in data]
labels = [LABEL[s] for s in syms]
n = len(syms)
x = np.arange(n)
w = 0.36


def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(p, dpi=160, facecolor="white")
    plt.close(fig)
    print("wrote", p)


# --- 1. MAPE ---
fig, ax = plt.subplots(figsize=(7.2, 4.4))
pre = [data[s]["pretrained"]["mape_mean"] for s in syms]
ft = [data[s]["finetuned"]["mape_mean"] for s in syms]
b1 = ax.bar(x - w / 2, pre, w, label="预训练", color=C_PRE)
b2 = ax.bar(x + w / 2, ft, w, label="微调后", color=C_FT)
for r in list(b1) + list(b2):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.06,
            f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylabel("MAPE %")
ax.set_title("走查回测：预测误差（越低越好）")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
save(fig, "compare_mape.png")

# --- 2. hit rate / day direction ---
fig, ax = plt.subplots(figsize=(7.2, 4.4))
pre_h = [data[s]["pretrained"]["hit_rate"] * 100 for s in syms]
ft_h = [data[s]["finetuned"]["hit_rate"] * 100 for s in syms]
pre_d = [data[s]["pretrained"].get("day_direction_mean", 0) for s in syms]
ft_d = [data[s]["finetuned"].get("day_direction_mean", 0) for s in syms]
ax.bar(x - w / 2, pre_h, w, label="预训练 · 终点方向命中", color=C_PRE, alpha=0.95)
ax.bar(x + w / 2, ft_h, w, label="微调后 · 终点方向命中", color=C_FT)
ax.plot(x - w / 2, pre_d, "o", color="#3a3f47", label="预训练 · 逐日方向")
ax.plot(x + w / 2, ft_d, "o", color="#8a3a12", label="微调后 · 逐日方向")
for xi, v in zip(x - w / 2, pre_h):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color="#3a3f47")
for xi, v in zip(x + w / 2, ft_h):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color="#8a3a12")
ax.set_xticks(x, labels)
ax.set_ylim(0, 105)
ax.set_ylabel("%")
ax.set_title("走查回测：方向判断")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
save(fig, "compare_hit.png")

# --- 3. per-window MAPE ---
fig, ax = plt.subplots(figsize=(7.6, 4.6))
colors = {"BTC": "#2f6fd0", "ETH": "#7b52c8", "XAUT": "#c8a02f"}
for s in syms:
    p = data[s]["pretrained"]["mapes"]
    f = data[s]["finetuned"]["mapes"]
    xs = np.arange(len(p))
    ax.plot(xs, p, "--", marker="o", ms=3.5, lw=1.4, color=colors.get(s, "#666"),
            alpha=0.55, label=f"{LABEL[s]} 预训练")
    ax.plot(xs, f, "-", marker="s", ms=3.5, lw=1.8, color=colors.get(s, "#666"),
            label=f"{LABEL[s]} 微调后")
ax.set_xlabel("走查窗口序号（每 21 天一窗，窗口 1 = 最新）")
ax.set_ylabel("MAPE %")
ax.set_title("逐窗口预测误差")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)
save(fig, "compare_windows.png")

# --- 4. delta ---
fig, ax = plt.subplots(figsize=(7.2, 4.0))
d_m = [data[s]["finetuned"]["mape_mean"] - data[s]["pretrained"]["mape_mean"] for s in syms]
d_h = [(data[s]["finetuned"]["hit_rate"] - data[s]["pretrained"]["hit_rate"]) * 100 for s in syms]
ax.bar(x - w / 2, d_m, w, label="ΔMAPE（负=变好）", color="#2f8f5b")
ax.bar(x + w / 2, d_h, w, label="Δ方向命中（正=变好）", color="#2f6fd0")
ax.axhline(0, color="#333", lw=1)
for xi, v in zip(x - w / 2, d_m):
    ax.text(xi, v + (0.06 if v >= 0 else -0.16), f"{v:+.2f}", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, d_h):
    ax.text(xi, v + (1.2 if v >= 0 else -2.6), f"{v:+.0f}", ha="center", fontsize=9)
ax.set_xticks(x, labels)
ax.set_title("微调带来的增量")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
save(fig, "compare_delta.png")

print("\n--- 数值汇总 ---")
hdr = f"{'资产':<10}{'预训练MAPE':>12}{'微调MAPE':>10}{'Δ':>8}{'预训练命中':>12}{'微调命中':>10}{'Δ':>8}"
print(hdr)
for s in syms:
    p, f = data[s]["pretrained"], data[s]["finetuned"]
    print(f"{LABEL[s]:<10}{p['mape_mean']:>12.3f}{f['mape_mean']:>10.3f}"
          f"{f['mape_mean']-p['mape_mean']:>+8.3f}"
          f"{p['hit_rate']*100:>11.1f}%{f['hit_rate']*100:>9.1f}%"
          f"{(f['hit_rate']-p['hit_rate'])*100:>+8.1f}")
