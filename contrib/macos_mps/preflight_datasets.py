#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-flight: verify the patched dataset can build train/val/test for every asset.

Runs on the Mac inside the patched copy (finetune_csv_mps) so the class under
test is exactly the one the training queue will use.
"""
import os
import sys

WORK = "/Volumes/ssd990pro1t/kronos/finetune/finetune_csv_mps"
ROOT = "/Volumes/ssd990pro1t/kronos"
sys.path.insert(0, WORK)
sys.path.insert(0, f"{ROOT}/Kronos")

import torch  # noqa: E402
from finetune_base_model import CustomKlineDataset  # noqa: E402

ASSETS = [
    ("BTC", f"{ROOT}/forecast/data/BTC_1D.csv"),
    ("ETH", f"{ROOT}/forecast/data/ETH_1D.csv"),
    ("XAUT", f"{ROOT}/forecast/data/XAUT_1D.csv"),
]
LOOKBACK = 512
PREDICT = 7

rc = 0
for sym, path in ASSETS:
    print(f"\n=== {sym} ({path}) ===")
    if not os.path.exists(path):
        print(f"  MISSING DATA FILE: {path}")
        rc = 1
        continue
    for dt in ("train", "val"):  # test_ratio=0 -> the trainer never builds a test split
        try:
            ds = CustomKlineDataset(path, data_type=dt, lookback_window=LOOKBACK,
                                    predict_window=PREDICT, clip=5.0, seed=42,
                                    train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)
            if dt == "train":
                x, y = ds[0]
                print(f"  sample[0] x={tuple(x.shape)} y={tuple(y.shape)} "
                      f"x.range=[{float(x.min()):.3f},{float(x.max()):.3f}]")
                xd, yd = ds[len(ds) - 1]
                print(f"  sample[last] x={tuple(xd.shape)} y={tuple(yd.shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{dt}] FAIL: {type(e).__name__}: {e}")
            rc = 1

print("\nPREFLIGHT", "OK" if rc == 0 else "FAILED")
sys.exit(rc)
