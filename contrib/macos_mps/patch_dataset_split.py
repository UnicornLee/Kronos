#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two fixes for the MPS copy of finetune_csv:

1) dataset split bug: the time split slices RAW ROWS, so the val/test slice
   can be shorter than one training window (lookback + predict + 1).
   With BTC_1D (3239 rows, lookback 512, predict 7) the val slice was
   323 rows < 520 -> n_samples = -196 -> "ValueError: __len__() should return >= 0".
   Fix: give val/test the preceding `lookback_window` rows as context so the
   first window's TARGET still lies strictly inside the val/test period
   (no leakage into the training period).

2) separate batch sizes: tokenizer uses tokenizer_batch_size,
   the base predictor uses basemodel_batch_size (base OOMs at 16 on MPS).
"""
import sys

BASE = os.environ.get("KRONOS_WORK", "/Volumes/ssd990pro1t/kronos/finetune/finetune_csv_mps")

# ---------------------------------------------------------------- patch 1 + 3
p = BASE + "/finetune_base_model.py"
text = open(p, encoding="utf-8").read()

if "CONTEXT_ROWS_FIXED" not in text:
    old_val = """        elif self.data_type == 'val':
            self.data = self.data.iloc[train_end:val_end].copy()
            self.timestamps = self.timestamps.iloc[train_end:val_end].copy()"""
    new_val = """        elif self.data_type == 'val':
            # CONTEXT_ROWS_FIXED: keep the `lookback_window` rows before the
            # split point as lookback context, otherwise the val slice can be
            # shorter than one window and n_samples goes negative.
            start = max(0, train_end - self.lookback_window)
            self.data = self.data.iloc[start:val_end].copy()
            self.timestamps = self.timestamps.iloc[start:val_end].copy()"""
    assert old_val in text, "val split block not found"
    text = text.replace(old_val, new_val)

    old_test = """        elif self.data_type == 'test':
            self.data = self.data.iloc[val_end:].copy()
            self.timestamps = self.timestamps.iloc[val_end:].copy()"""
    new_test = """        elif self.data_type == 'test':
            # CONTEXT_ROWS_FIXED: same context trick as val.
            start = max(0, val_end - self.lookback_window)
            self.data = self.data.iloc[start:].copy()
            self.timestamps = self.timestamps.iloc[start:].copy()"""
    assert old_test in text, "test split block not found"
    text = text.replace(old_test, new_test)

    old_n = """        self.n_samples = len(self.data) - self.window + 1
            
            print(f"[{data_type.upper()}] Data length: {len(self.data)}, Available samples: {self.n_samples}")"""
    old_n = old_n.replace("            print", "        print")
    new_n = """        self.n_samples = len(self.data) - self.window + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"[{self.data_type.upper()}] not enough rows after the time split: "
                f"{len(self.data)} rows left, one window needs {self.window} "
                f"(lookback {self.lookback_window} + predict {self.predict_window} + 1). "
                "Reduce lookback_window/predict_window or raise val_ratio/test_ratio."
            )

        print(f"[{data_type.upper()}] Data length: {len(self.data)}, Available samples: {self.n_samples}")"""
    assert old_n in text, "n_samples block not found"
    text = text.replace(old_n, new_n)
    open(p, "w", encoding="utf-8").write(text)
    print("[patched] finetune_base_model.py: split fix")
else:
    print("[no-op] finetune_base_model.py already patched")

text = open(p, encoding="utf-8").read()
text = text.replace("batch_size=config.batch_size,", "batch_size=config.basemodel_batch_size,")
text = text.replace('logger.info(f"Batch size: {config.batch_size}")',
                    'logger.info(f"Batch size: {config.basemodel_batch_size}")')
open(p, "w", encoding="utf-8").write(text)
print("  base_model batch_size -> basemodel_batch_size:",
      open(p, encoding="utf-8").read().count("config.basemodel_batch_size"))

# ---------------------------------------------------------------- patch 2
p = BASE + "/finetune_tokenizer.py"
text = open(p, encoding="utf-8").read()
text = text.replace("batch_size=config.batch_size,", "batch_size=config.tokenizer_batch_size,")
text = text.replace('logger.info(f"Batch size: {config.batch_size}")',
                    'logger.info(f"Batch size: {config.tokenizer_batch_size}")')
open(p, "w", encoding="utf-8").write(text)
print("  tokenizer batch_size -> tokenizer_batch_size:",
      open(p, encoding="utf-8").read().count("config.tokenizer_batch_size"))

