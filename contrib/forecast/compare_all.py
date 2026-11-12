#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run compare_ft.py for every asset and merge the results into one table.

Usage:
    python compare_all.py                 # only run assets whose result json is missing
    python compare_all.py --force         # re-run everything
    python compare_all.py --symbols BTC ETH
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = "/Volumes/ssd990pro1t/kronos"
FC = f"{ROOT}/forecast/scripts"
RUNS = f"{ROOT}/finetune/runs"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "XAUT"])
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge-only", action="store_true", help="do not run anything, just merge")
    return ap.parse_args()


def run_one(sym, samples, windows):
    out = f"{RUNS}/compare_ft_{sym.lower()}_1D_ft.json"
    if os.path.exists(out):
        print(f"[skip] {sym} result exists -> {out}")
        return out
    cmd = [sys.executable, f"{FC}/compare_ft.py", "--symbol", sym,
           "--samples", str(samples), "--windows", str(windows)]
    print(f"[run ] {sym}: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[FAIL] {sym} exit={rc}")
        return None
    return out


def fmt(v, n=3):
    return "n/a" if v is None else f"{v:.{n}f}"


def pct(v, n=1):
    """hit_rate is a 0..1 fraction in the json -> render as a percentage."""
    return "n/a" if v is None else f"{v * 100:.{n}f}%"


def main():
    a = parse_args()
    results = {}
    if not a.merge_only:
        for sym in a.symbols:
            p = run_one(sym, a.samples, a.windows)
            if p and os.path.exists(p):
                results[sym] = json.load(open(p))

    for sym in a.symbols:
        p = f"{RUNS}/compare_ft_{sym.lower()}_1D_ft.json"
        if sym not in results and os.path.exists(p):
            results[sym] = json.load(open(p))

    # 兼容旧格式（早期冒烟结果没有 delta 段）
    for sym, r in results.items():
        if "delta" not in r:
            pre, ft = r["pretrained"], r["finetuned"]
            r["delta"] = {"mape_mean": ft["mape_mean"] - pre["mape_mean"],
                          "hit_rate": ft["hit_rate"] - pre["hit_rate"],
                          "day_direction_mean": ft.get("day_direction_mean", 0.0)
                          - pre.get("day_direction_mean", 0.0)}

    if not results:
        print("no results yet")
        return

    merged = {"assets": results, "samples": a.samples, "windows": a.windows}
    with open(f"{RUNS}/compare_all.json", "w") as fh:
        json.dump(merged, fh, indent=2)

    lines = []
    lines.append("| 资产 | 预训练 MAPE% | 微调 MAPE% | ΔMAPE | 预训练命中 | 微调命中 | "
                 "预训练逐日方向% | 微调逐日方向% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sym, r in results.items():
        pre, ft, dl = r["pretrained"], r["finetuned"], r["delta"]
        lines.append(
            f"| {sym} | {fmt(pre['mape_mean'])} | {fmt(ft['mape_mean'])} | "
            f"{fmt(dl['mape_mean'])} | {pct(pre['hit_rate'])} | {pct(ft['hit_rate'])} | "
            f"{fmt(pre['day_direction_mean'], 1)} | {fmt(ft['day_direction_mean'], 1)} |")
    table = "\n".join(lines)
    print("\n" + table)

    md = [f"# Kronos 微调 vs 预训练 —— 三资产横向对比",
          "",
          f"模型：kronos-base (102.3M)；窗口 {a.windows} 个（每 21 天一窗）；"
          f"每窗 {a.samples} 条采样路径取中位。",
          "",
          table, ""]
    for sym, r in results.items():
        md += [f"## {sym}",
               f"- 走查窗口结束日：{r.get('windows', '?')} 个，"
               f"train_end={r.get('train_end')}",
               f"- 预训练 MAPE 明细：{[round(x, 3) for x in r['pretrained']['mapes']]}",
               f"- 微调 MAPE 明细：{[round(x, 3) for x in r['finetuned']['mapes']]}",
               f"- 微调产物：{r['ft_dir']}", ""]
    with open(f"{RUNS}/compare_all.md", "w") as fh:
        fh.write("\n".join(md))

    print(f"\nwritten -> {RUNS}/compare_all.json")
    print(f"written -> {RUNS}/compare_all.md")


if __name__ == "__main__":
    main()
