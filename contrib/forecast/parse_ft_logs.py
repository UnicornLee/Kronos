#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse Kronos finetune logs into structured JSON (per-epoch validation loss).

Reads runs/{sym}_1d_base.log (and any other *_1d_base.log) and extracts:
  - per-epoch validation loss + epoch time, for both phases
  - the best validation loss and where the best model was saved
  - the header block (device, epochs, batch sizes, lr, data path)

Output: finetune/runs/ft_logs.json
"""
import json
import os
import re
import sys

ROOT = "/Volumes/ssd990pro1t/kronos"
RUNS = f"{ROOT}/finetune/runs"

SUM_RE = re.compile(r"^---\s*Epoch\s+(\d+)\s*/\s*(\d+)\s*Summary\s*---")
VAL_RE = re.compile(r"^Validation Loss:\s*([0-9.eE+-]+)")
TIME_RE = re.compile(r"^Epoch Time:\s*([0-9:]+)")
TOT_RE = re.compile(r"^Total Training Time:\s*([0-9:]+)")
BEST_RE = re.compile(r"Best model saved to:\s*(\S+)\s*\(validation loss:\s*([0-9.eE+-]+)\)")
HDR_RE = re.compile(r"(Experiment [Nn]ame|Data path|Lookback window|Predict window|"
                    r"Tokenizer training epochs|Basemodel training epochs|"
                    r"Batch size|Tokenizer learning rate|Predictor learning rate|"
                    r"Device|Distributed training)\s*:\s*(.*)$", re.IGNORECASE)
STEP_RE = re.compile(r"\[Epoch\s+(\d+)\s*/\s*(\d+),\s*Step\s+(\d+)\s*/\s*(\d+)\]"
                     r".*?Loss:\s*([0-9.eE+-]+)")
MARK_RE = re.compile(r"===\s*(.*Training Started)\s*===")


def to_sec(s):
    try:
        parts = [int(x) for x in s.split(":")]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse(path):
    out = {"file": os.path.basename(path), "header": {}, "phases": {}, "steps": []}
    cur_phase = None
    seen = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = MARK_RE.search(line)
            if m:
                cur_phase = "tokenizer" if "Tokenizer" in m.group(1) else "basemodel"
                out["phases"].setdefault(cur_phase, {"epochs": []})
                continue
            m = HDR_RE.search(line)
            if m:
                k, v = m.group(1), m.group(2).strip()
                if k not in out["header"] or len(v) > len(out["header"][k]):
                    out["header"][k] = v
                continue
            m = SUM_RE.match(line.strip())
            if m:
                ep, total = int(m.group(1)), int(m.group(2))
                phase = "tokenizer" if total == 30 else ("basemodel" if total == 20 else cur_phase)
                ph = out["phases"].setdefault(phase, {"epochs": []})
                if (phase, ep) in seen:
                    continue
                rec = {"epoch": ep, "total": total}
                ph["epochs"].append(rec)
                seen.add((phase, ep))
                cur = rec
                continue
            if "Validation Loss:" in line and out["phases"].get(cur_phase, {}).get("epochs"):
                v = VAL_RE.match(line.strip())
                if v:
                    out["phases"][cur_phase]["epochs"][-1]["val_loss"] = float(v.group(1))
                    cur = out["phases"][cur_phase]["epochs"][-1]
                continue
            if line.strip().startswith("Epoch Time:") and out["phases"].get(cur_phase, {}).get("epochs"):
                t = TIME_RE.match(line.strip())
                if t:
                    out["phases"][cur_phase]["epochs"][-1]["epoch_sec"] = to_sec(t.group(1))
                continue
            if line.strip().startswith("Total Training Time:"):
                t = TOT_RE.match(line.strip())
                if t:
                    out["phases"].setdefault(cur_phase or "unknown", {})["total_sec"] = to_sec(t.group(1))
                continue
            m = BEST_RE.search(line)
            if m:
                ph = out["phases"].setdefault(cur_phase or "unknown", {})
                b = ph.get("best")
                val = float(m.group(2))
                if not b or val < b["val_loss"]:
                    ph["best"] = {"val_loss": val, "path": m.group(1)}
                continue
            m = STEP_RE.search(line)
            if m and len(out["steps"]) < 20000:
                out["steps"].append({"phase": cur_phase, "epoch": int(m.group(1)),
                                     "step": int(m.group(3)), "loss": float(m.group(5))})
    for ph in out["phases"].values():
        eps = ph.get("epochs", [])
        ph["epochs"] = sorted(eps, key=lambda e: e["epoch"])
        vals = [e["val_loss"] for e in eps if "val_loss" in e]
        ph["val_first"] = vals[0] if vals else None
        ph["val_last"] = vals[-1] if vals else None
        ph["val_best"] = min(vals) if vals else None
        ph["epoch_count"] = len(eps)
        ph["epoch_sec_mean"] = (sum(e["epoch_sec"] for e in eps if "epoch_sec" in e) /
                                max(1, sum(1 for e in eps if "epoch_sec" in e))) or None
    return out


def main():
    targets = sys.argv[1:] or ["btc_1d_base", "eth_1d_base", "xaut_1d_base"]
    result = {}
    for name in targets:
        p = f"{RUNS}/{name}.log"
        if not os.path.exists(p):
            print(f"missing log: {p}")
            continue
        d = parse(p)
        result[name.replace("_1d_base", "").upper()] = d
        print(f"\n=== {name} ===")
        print("  header:", {k: d["header"].get(k) for k in
                             ("Experiment Name", "Data path", "Device", "Batch size",
                              "Tokenizer training epochs", "Basemodel training epochs")})
        for ph, v in d["phases"].items():
            print(f"  {ph}: epochs={v.get('epoch_count')} "
                  f"val {v.get('val_first')} -> {v.get('val_last')} (best {v.get('val_best')}) "
                  f"avg {v.get('epoch_sec_mean') and round(v['epoch_sec_mean'], 1)}s/epoch")
            if v.get("best"):
                print(f"    best_model: {v['best']['path']}")

    outp = f"{RUNS}/ft_logs.json"
    with open(outp, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwritten -> {outp}")


if __name__ == "__main__":
    main()
