#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the final finetune report (Markdown) from parsed logs + walk-forward results.

Input (Windows side, pulled from the Mac):
  ft_logs.json      <- parse_ft_logs.py on the Mac
  compare_all.json  <- compare_all.py on the Mac
Output:
  FT_REPORT.md
"""
import json
import os
import sys

IN_DIR = sys.argv[1] if len(sys.argv) > 1 else r"D:\DSH\kronos-forecast\out\compare"
OUT_MD = sys.argv[2] if len(sys.argv) > 2 else r"D:\DSH\kronos-forecast\FT_REPORT.md"

ORDER = ["BTC", "ETH", "XAUT"]
NAME = {"BTC": "BTC", "ETH": "ETH", "XAUT": "XAUT（代币化黄金）"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


logs = load(os.path.join(IN_DIR, "ft_logs.json"))
cmp_all = load(os.path.join(IN_DIR, "compare_all.json"))
if cmp_all is None:
    cmp_all = {"assets": {a: load(os.path.join(IN_DIR, f"compare_ft_{a.lower()}_1D_ft.json"))
                          for a in ORDER}}
    cmp_all["assets"] = {k: v for k, v in cmp_all["assets"].items() if v}

L = []
A = L.append

A("# Kronos 日线微调 —— BTC / ETH / XAUT 实测报告")
A("")
A("> 模型：`NeoQuasar/Kronos-base`（102.3M）+ `Kronos-Tokenizer-base`；"
  "设备：Mac mini M4 (MPS, torch 2.15.0)；数据：OKX 现货日线。")
A("")

# ---------- 1. 训练 ----------
A("## 一、训练过程")
A("")
if logs:
    A("| 资产 | 阶段 | 轮数 | 首轮 val loss | 末轮 val loss | 最优 val loss | 平均每轮耗时 |")
    A("|---|---|---|---|---|---|---|")
    for sym in ORDER:
        d = logs.get(sym)
        if not d:
            continue
        for ph, label in (("tokenizer", "tokenizer"), ("basemodel", "basemodel")):
            v = d["phases"].get(ph)
            if not v or not v.get("epoch_count"):
                continue
            t = v.get("epoch_sec_mean")
            A(f"| {NAME.get(sym, sym)} | {label} | {v['epoch_count']}/{v.get('epochs', [{}])[0].get('total', '?')} "
              f"| {v.get('val_first')} | {v.get('val_last')} | {v.get('val_best')} "
              f"| {round(t) if t else 'n/a'}s |")
    A("")
    for sym in ORDER:
        d = logs.get(sym)
        if not d:
            continue
        h = d.get("header", {})
        A(f"- **{NAME.get(sym, sym)}**：数据 `{h.get('Data path', 'n/a')}`，"
          f"lookback {h.get('Lookback window', '?')} / predict {h.get('Predict window', '?')}，"
          f"device `{h.get('Device', '?')}`，DDP `{h.get('Distributed training', '?')}`，"
          f"batch `{h.get('Batch size', '?')}`")
    A("")

# ---------- 2. 走查回测 ----------
A("## 二、样本外走查回测（预训练 vs 微调）")
A("")
assets = cmp_all.get("assets", {}) if cmp_all else {}
if assets:
    nw = cmp_all.get("windows")
    ns = cmp_all.get("samples")
    if not nw:
        any_r = next(iter(assets.values()))
        nw, ns = any_r.get("windows"), any_r.get("samples")
    A(f"口径：模型 `kronos-base`，lookback 512 → 预测 7 根日线；"
      f"{nw} 个走查窗口（每 21 天一窗，全部落在训练期之后的验证区内）；"
      f"每窗 {ns} 条采样路径取中位。")
    A("")
    A("| 资产 | 预训练 MAPE% | 微调 MAPE% | Δ | 预训练命中 | 微调命中 | Δ | 预训练逐日方向% | 微调逐日方向% |")
    A("|---|---|---|---|---|---|---|---|---|")
    for sym in ORDER:
        r = assets.get(sym)
        if not r:
            continue
        p, f = r["pretrained"], r["finetuned"]
        A(f"| {NAME.get(sym, sym)} | {p['mape_mean']:.3f} | {f['mape_mean']:.3f} | "
          f"{f['mape_mean']-p['mape_mean']:+.3f} | {p['hit_rate']*100:.1f}% | {f['hit_rate']*100:.1f}% | "
          f"{(f['hit_rate']-p['hit_rate'])*100:+.1f} | {p.get('day_direction_mean', 0):.1f} | "
          f"{f.get('day_direction_mean', 0):.1f} |")
    A("")
    for sym in ORDER:
        r = assets.get(sym)
        if not r:
            continue
        p, f = r["pretrained"], r["finetuned"]
        A(f"**{NAME.get(sym, sym)}** 逐窗 MAPE%")
        A("")
        A("| 窗口 | 预训练 | 微调 |")
        A("|---|---|---|")
        for i, (a1, b1) in enumerate(zip(p["mapes"], f["mapes"]), 1):
            A(f"| {i} | {a1:.3f} | {b1:.3f} |")
        A("")
else:
    A("（走查结果尚未生成）")
    A("")

# ---------- 3. 结论 ----------
A("## 三、结论")
A("")
if assets:
    better_m = [s for s in ORDER if s in assets and
                 assets[s]["finetuned"]["mape_mean"] < assets[s]["pretrained"]["mape_mean"]]
    better_h = [s for s in ORDER if s in assets and
                assets[s]["finetuned"]["hit_rate"] > assets[s]["pretrained"]["hit_rate"]]
    worse_h = [s for s in ORDER if s in assets and
               assets[s]["finetuned"]["hit_rate"] < assets[s]["pretrained"]["hit_rate"]]
    A(f"1. **误差层面**：微调后 MAPE 更低的资产 = "
      f"{'、'.join(NAME.get(s, s) for s in better_m) if better_m else '无'}。"
      "微调把模型往「这个市场的统计结构」上拉，最直接的收益体现在误差上。")
    n_win = next((assets[s].get("windows", 8) for s in ORDER if s in assets), 8)
    A(f"2. **方向层面**：终点方向命中提升的资产 = "
      f"{'、'.join(NAME.get(s, s) for s in better_h) if better_h else '无'}；"
      f"下降的资产 = {'、'.join(NAME.get(s, s) for s in worse_h) if worse_h else '无'}。"
      "命中率的样本量只有 "
      f"{n_win} 个窗口，"
      "单次翻转就是 "
      f"{100.0/max(1,n_win):.1f} 个百分点，**统计上远不足以支撑「择时信号」的结论**。")
    A(f"3. **用法定位**：日线微调后的 Kronos 适合用作「量级 + 区间的先验」，"
      "用来给一个已有的判断补充概率分布，而不是用来替代判断本身。")
else:
    A("（待走查完成后补）")
A("")

# ---------- 4. 复现 ----------
A("## 四、复现")
A("")
A("```bash")
A("# 1) 打 MPS 补丁（上游 finetune_csv 不动）")
A("export KRONOS_REPO=/path/to/Kronos")
A("export KRONOS_WORK=/path/to/work/finetune_csv_mps")
A("python contrib/macos_mps/patch_finetune_csv.py")
A("python contrib/macos_mps/patch_config_loader.py")
A("python contrib/macos_mps/patch_dataset_split.py")
A("")
A("# 2) 排队训练")
A("cd \"$KRONOS_WORK\"")
A("nohup bash contrib/macos_mps/kronos_queue.sh > /dev/null 2>&1 &")
A("")
A("# 3) 走查回测 + 汇总")
A("bash contrib/macos_mps/finalize_compare_mac.sh")
A("")
A("# 4) 出图（Windows 侧）")
A("pwsh -File contrib/macos_mps/finalize_compare_win.ps1")
A("```")
A("")

with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))

print("written ->", OUT_MD)
print("\n".join(L[:40]))
