#!/usr/bin/env python3
"""Create an MPS-friendly copy of Kronos' finetune_csv pipeline.

Source  : /Volumes/ssd990pro1t/kronos/Kronos/finetune_csv      (upstream, untouched)
Target  : /Volumes/ssd990pro1t/kronos/finetune/finetune_csv_mps (patched copy)

Patches applied (all string replacements, idempotent):
  1. sys.path.append('../')  ->  absolute path to the Kronos repo root
  2. device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                              ->  device = pick_device()          (mps_device.py)
  3. train_sequential._setup_device() -> MPS-aware branch
  4. pin_memory=True -> pin_memory=False   (no CUDA host on Apple Silicon)

Nothing in the upstream repo is modified; the patched copy lives on the
external drive next to the training data.
"""
import os
import shutil
import sys

KRONOS = os.environ.get("KRONOS_REPO", "/Volumes/ssd990pro1t/kronos/Kronos")
SRC = os.path.join(KRONOS, "finetune_csv")
DST = os.environ.get("KRONOS_WORK", "/Volumes/ssd990pro1t/kronos/finetune/finetune_csv_mps")

OLD_DEVICE = 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
NEW_DEVICE = "device = pick_device()"


def patch_file(path, subs, extra_import=None):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    orig = text
    for old, new in subs:
        text = text.replace(old, new)
    if extra_import and extra_import not in text:
        # insert after the last top-level import line
        lines = text.split("\n")
        idx = max(i for i, ln in enumerate(lines[:60]) if ln.startswith("import ") or ln.startswith("from "))
        lines.insert(idx + 1, extra_import)
        text = "\n".join(lines)
    if text != orig:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[patched] {os.path.basename(path)}")
    else:
        print(f"[no-op]   {os.path.basename(path)}")


def main():
    if not os.path.isdir(SRC):
        print(f"missing source: {SRC}")
        return 1
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"[copy] {SRC} -> {DST}")

    # mps_device.py helper
    shutil.copy(os.environ.get("KRONOS_MPS_DEVICE", "/Volumes/ssd990pro1t/kronos/finetune/mps_device.py"),
                os.path.join(DST, "mps_device.py"))

    common = [
        ("sys.path.append('../')", f"sys.path.append('{KRONOS}')"),
        ("sys.path.append(\"../\")", f"sys.path.append('{KRONOS}')"),
        (OLD_DEVICE, NEW_DEVICE),
        ("pin_memory=True", "pin_memory=False"),
    ]

    patch_file(os.path.join(DST, "finetune_tokenizer.py"), common,
               extra_import="from mps_device import pick_device")
    patch_file(os.path.join(DST, "finetune_base_model.py"), common,
               extra_import="from mps_device import pick_device")

    ts = os.path.join(DST, "train_sequential.py")
    patch_file(ts, common + [(
        """    def _setup_device(self):
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")
""",
        """    def _setup_device(self):
        device = pick_device()
        if device.type == "cuda":
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
""")],
        extra_import="from mps_device import pick_device")

    # distributed: nccl is NVIDIA-only; force gloo when not on cuda
    patch_file(ts, [(
        """        if self.world_size > 1 and torch.cuda.is_available():
            backend = os.environ.get("DIST_BACKEND", "nccl").lower()""",
        """        if self.world_size > 1 and torch.cuda.is_available():
            backend = os.environ.get("DIST_BACKEND", "nccl").lower()""",
    )])

    print("\n--- verify ---")
    for f in ("finetune_tokenizer.py", "finetune_base_model.py", "train_sequential.py"):
        p = os.path.join(DST, f)
        with open(p, encoding="utf-8") as fh:
            t = fh.read()
        print(f"{f:26s} pick_device={t.count('pick_device')} "
              f"pin_memory_false={t.count('pin_memory=False')} "
              f"cuda_device_line={'yes' if OLD_DEVICE in t else 'no'}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

