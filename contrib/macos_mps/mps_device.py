"""Device picker for Kronos fine-tuning on Apple Silicon (MPS) / CUDA / CPU.

Used by the patched finetune_csv scripts. Resolution order:
  1. $KRONOS_DEVICE explicit override: "mps" | "cuda" | "cpu"
  2. CUDA if available
  3. MPS if available (Apple Silicon)
  4. CPU
"""
import os

import torch


def pick_device():
    pref = os.environ.get("KRONOS_DEVICE", "auto").strip().lower()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()

    if pref in ("mps", "cuda", "cpu"):
        if pref == "mps" and not mps_ok:
            raise RuntimeError("KRONOS_DEVICE=mps requested but MPS is not available")
        if pref == "cuda" and not cuda_ok:
            raise RuntimeError("KRONOS_DEVICE=cuda requested but CUDA is not available")
        return torch.device(pref)

    if cuda_ok:
        return torch.device("cuda")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")
