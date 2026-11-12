#!/usr/bin/env python3
"""Patch the copied config_loader.py so tokenizer and predictor can use
different batch sizes (upstream template has one shared `batch_size`).

  tokenizer_batch_size: 32   (4M params, cheap)
  basemodel_batch_size: 8    (base: 102M params, MPS OOM at 16)
"""
import re
import sys

PATH = os.environ.get("KRONOS_WORK", "/Volumes/ssd990pro1t/kronos/finetune/finetune_csv_mps") + "/config_loader.py"

with open(PATH, encoding="utf-8") as fh:
    text = fh.read()

if "tokenizer_batch_size" in text:
    print("[no-op] config_loader.py already patched")
    sys.exit(0)

text = text.replace(
    "        self.batch_size = training_config.get('batch_size', 160)",
    "        self.batch_size = training_config.get('batch_size', 160)\n"
    "        self.tokenizer_batch_size = training_config.get('tokenizer_batch_size', self.batch_size)\n"
    "        self.basemodel_batch_size = training_config.get('basemodel_batch_size', self.batch_size)",
)
text = text.replace(
    "            'epochs': self.tokenizer_epochs,\n            'batch_size': self.batch_size,",
    "            'epochs': self.tokenizer_epochs,\n            'batch_size': self.tokenizer_batch_size,",
)
text = text.replace(
    "            'epochs': self.basemodel_epochs,\n            'batch_size': self.batch_size,",
    "            'epochs': self.basemodel_epochs,\n            'batch_size': self.basemodel_batch_size,",
)
text = text.replace(
    '        print(f"Batch size: {self.batch_size}")',
    '        print(f"Batch size: {self.batch_size} '
    '(tokenizer={self.tokenizer_batch_size}, basemodel={self.basemodel_batch_size})")',
)

with open(PATH, "w", encoding="utf-8") as fh:
    fh.write(text)

check = open(PATH, encoding="utf-8").read()
left = re.findall(r"'batch_size': self\.batch_size", check)
print("[patched] config_loader.py")
print("  tokenizer_batch_size refs:", check.count("self.tokenizer_batch_size"))
print("  basemodel_batch_size refs:", check.count("self.basemodel_batch_size"))
print("  residual shared batch_size in config dicts:", len(left))
sys.exit(1 if left else 0)

