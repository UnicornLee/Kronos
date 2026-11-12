#!/usr/bin/env python3
"""Probe: can Kronos tokenizer + predictor train (forward+backward) on Mac MPS?

Tests, in order:
  1. MPS availability / basic ops (scatter_reduce sum, unique, index_select)
  2. KronosTokenizer.from_pretrained -> forward -> mse loss -> backward
  3. Kronos.from_pretrained        -> forward -> head.compute_loss -> backward
Each stage prints elapsed seconds and the ops that fail (if any).
"""
import os
import sys
import time
import traceback

ROOT = "/Volumes/ssd990pro1t/kronos"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf-cache")
sys.path.insert(0, f"{ROOT}/Kronos")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import Kronos, KronosTokenizer  # noqa: E402

TOK_DIR = (f"{ROOT}/hf-cache/hub/models--NeoQuasar--Kronos-Tokenizer-base/"
           "snapshots/0e0117387f39004a9016484a186a908917e22426")
MODEL_DIR = (f"{ROOT}/hf-cache/hub/models--NeoQuasar--Kronos-base/"
             "snapshots/2b554741eca47781b64468546e77fef3e85130e6")


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"[ok]   {name}  {time.time() - t0:.2f}s  -> {out}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}  {time.time() - t0:.2f}s")
        print("       " + repr(exc).replace("\n", "\n       ")[:1200])
        return False


def main():
    print("torch", torch.__version__)
    print("mps_available", torch.backends.mps.is_available())
    print("mps_built", torch.backends.mps.is_built())
    if not torch.backends.mps.is_available():
        print("RESULT: NO_MPS")
        return 2

    dev = torch.device("mps")
    results = {}

    # --- 1. raw ops used by model/module.py -------------------------------
    def op_scatter():
        zi = torch.randint(0, 1024, (4096,), device=dev)
        cnt = torch.scatter_reduce(torch.zeros(1024, device=dev), 0, zi,
                                   torch.ones_like(zi, dtype=torch.float32), "sum")
        return f"sum={float(cnt.sum()):.0f}"

    def op_unique():
        u = torch.unique(torch.randint(0, 64, (1024,), device=dev))
        return f"unique_n={u.numel()}"

    def op_index():
        idx = torch.randint(0, 1024, (512,), device=dev)
        out = torch.randn(1024, device=dev)[idx]
        return f"index_ok={out.numel()}"

    def op_finfo():
        b = torch.zeros(1, dtype=torch.bfloat16, device=dev)
        return f"bf16_ok={b.dtype}"

    results["scatter_reduce"] = step("op scatter_reduce(sum)", op_scatter)
    results["unique"] = step("op unique", op_unique)
    results["index"] = step("op index_select", op_index)
    results["bf16"] = step("op bfloat16 alloc", op_finfo)

    # --- 2. tokenizer fwd/bwd --------------------------------------------
    tok = None

    def load_tok():
        nonlocal tok
        tok = KronosTokenizer.from_pretrained(TOK_DIR).to(dev)
        n = sum(p.numel() for p in tok.parameters())
        return f"params={n:,}"

    results["tokenizer_load"] = step("load KronosTokenizer -> mps", load_tok)

    if tok is not None:
        B, T = 8, 600

        def tok_fwd():
            x = torch.randn(B, T, 6, device=dev)
            (z_pre, z), bsq, q, idx = tok(x)
            loss = F.mse_loss(z_pre, x) + F.mse_loss(z, x)
            loss.backward()
            return f"loss={float(loss):.4f} bsq={float(bsq):.4f}"

        results["tokenizer_fwd_bwd"] = step("tokenizer fwd+bwd (B=8,T=600)", tok_fwd)

        def tok_fwd_cpu():
            tokc = KronosTokenizer.from_pretrained(TOK_DIR)
            x = torch.randn(B, T, 6)
            t0 = time.time()
            (z_pre, z), bsq, q, idx = tokc(x)
            loss = F.mse_loss(z_pre, x) + F.mse_loss(z, x)
            loss.backward()
            return f"cpu_loss={float(loss):.4f} in {time.time() - t0:.2f}s"

        results["tokenizer_fwd_bwd_cpu"] = step("tokenizer fwd+bwd CPU (reference)", tok_fwd_cpu)

    # --- 3. predictor fwd/bwd -------------------------------------------
    model = None

    def load_model():
        nonlocal model
        model = Kronos.from_pretrained(MODEL_DIR).to(dev)
        n = sum(p.numel() for p in model.parameters())
        return f"params={n:,}"

    results["model_load"] = step("load Kronos -> mps", load_model)

    if model is not None and tok is not None:
        B, T = 8, 600

        def model_fwd():
            x = torch.randn(B, T, 6, device=dev)
            with torch.no_grad():
                _, _, z_idx = tok.tokenizer(tok.quant_embed(tok.embed(x)))
            s1 = z_idx[..., :10].long()
            s2 = z_idx[..., 10:].long()
            stamp = torch.zeros(B, T, 5, device=dev)
            logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1, :])
            loss = F.cross_entropy(
                logits[0].reshape(-1, logits[0].shape[-1]), s1[:, 1:].reshape(-1))
            loss.backward()
            return f"loss={float(loss):.4f} logits={tuple(logits[0].shape)}"

        results["model_fwd_bwd"] = step("predictor fwd+bwd (B=8,T=600)", model_fwd)

    print("\nSUMMARY " + repr(results))
    ok = all(results.values())
    print("RESULT:", "ALL_OK" if ok else "SOME_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
