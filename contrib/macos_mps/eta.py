#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETA for the Kronos finetune queue.

Reads the training logs under finetune/runs/ and reports, per asset, which
phase it is in, how many epochs are left and when the queue should finish.

Log facts (verified on the Mac):
  - phase marker lines: "... tokenizer_training_rank_0 ..." / "... basemodel_training_rank_0 ..."
  - per-epoch block:  "=== Epoch 3/20 Summary ===" / "Validation Loss: 2.5002" / "Epoch Time: 590.16 seconds"
    (the tokenizer prints "Epoch Time: 0:01:30" instead)
  - every block appears twice in the file (stdout + file handler), so epoch
    times are keyed by epoch number instead of appended to a list.

Usage:  python eta.py [runs_dir]
"""
import datetime
import os
import re
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/ssd990pro1t/kronos/finetune/runs"
ASSETS = ["btc", "eth", "xaut"]

STEP_RE = re.compile(r"\[Epoch\s+(\d+)\s*/\s*(\d+),\s*Step\s+(\d+)\s*/\s*(\d+)\]")
SUM_BLOCK_RE = re.compile(r"[-=]+\s*Epoch\s+(\d+)\s*/\s*(\d+)\s+Summary\s*[-=]+")
VAL_RE = re.compile(r"Validation Loss:\s*([0-9.eE+-]+)")
TIME_SEC_RE = re.compile(r"Epoch Time:\s*([0-9.]+)\s*seconds")
TIME_HMS_RE = re.compile(r"Epoch Time:\s*(\d+):(\d{2}):(\d{2})")


def parse(path):
    """-> {phase: {'total', 'done', 'epoch_s': {epoch: sec}, 'val': {epoch: loss}}}"""
    out = {}
    if not os.path.exists(path):
        return out
    cur = None
    last_epoch = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if "tokenizer_training" in line:
                if cur != "tokenizer":
                    cur = "tokenizer"
                    last_epoch = None
            elif "basemodel_training" in line:
                if cur != "basemodel":
                    cur = "basemodel"
                    last_epoch = None
            if cur is None:
                continue
            p = out.setdefault(cur, {"total": 0, "done": 0, "epoch_s": {}, "val": {}})
            m = STEP_RE.search(line)
            if m:
                p["total"] = int(m.group(2))
                p["done"] = max(p["done"], int(m.group(1)))
            s = SUM_BLOCK_RE.search(line)
            if s:
                n_sum = int(s.group(1))
                # a phase's summary block only counts when its epoch number fits
                # that phase (the file interleaves stdout + logger copies, so a
                # stale block from the previous phase can show up late)
                if 1 <= n_sum <= p["total"]:
                    last_epoch = n_sum
                    p["total"] = max(p["total"], int(s.group(2)))
            v = VAL_RE.search(line)
            if v and last_epoch is not None:
                p["val"][last_epoch] = float(v.group(1))
            t = TIME_SEC_RE.search(line)
            if t and last_epoch is not None:
                p["epoch_s"][last_epoch] = float(t.group(1))
                continue
            h = TIME_HMS_RE.search(line)
            if h and last_epoch is not None:
                p["epoch_s"][last_epoch] = (int(h.group(1)) * 3600
                                            + int(h.group(2)) * 60
                                            + int(h.group(3)))
    return out


def fmt_dur(sec):
    sec = int(sec)
    if sec <= 0:
        return "0m"
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


now = datetime.datetime.now()
print(f"现在: {now:%Y-%m-%d %H:%M:%S}")
print()
remaining = 0
for a in ASSETS:
    log = os.path.join(RUNS, f"{a}_1d_base.log")
    info = parse(log)
    print(f"--- {a}_1d_base ---")
    if not info:
        print("   (未开始 / 无日志)")
        continue
    for ph in ("tokenizer", "basemodel"):
        d = info.get(ph)
        if not d or not d["total"]:
            continue
        es = list(d["epoch_s"].values())
        avg = (sum(es) / len(es)) if es else 0
        left = max(0, d["total"] - d["done"])
        rem = left * avg
        remaining += rem
        vals = list(d["val"].values())
        best = min(vals) if vals else None
        print(f"   {ph:<10} epoch {d['done']}/{d['total']}  avg {avg:6.1f}s/epoch  "
              f"剩余 {left:>2} epoch ≈ {fmt_dur(rem):>7}  "
              f"last_val={vals[-1] if vals else '-'}  best_val={best if best is not None else '-'}")
    print()

print(f"=== 已开始任务的剩余总时长 ≈ {fmt_dur(remaining)} ===")
print(f"=== 预计完成: {(now + datetime.timedelta(seconds=remaining)):%Y-%m-%d %H:%M} ===")
