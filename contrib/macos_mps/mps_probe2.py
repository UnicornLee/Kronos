#!/usr/bin/env python3
"""Probe v2: realistic Kronos fine-tuning step on Mac MPS (and CPU reference).

Uses the exact call path the official finetune_csv scripts use:
  tokenizer.encode(x, half=True) -> (token_seq_0, token_seq_1)
  model(s1_in, s2_in, stamp)      -> logits
  model.head.compute_loss(...)    -> loss

Measures per-step wall time and peak RSS for tokenizer and predictor at a few
batch sizes, on MPS and on CPU (CPU numbers scaled down because it is slow).
"""
import os
import sys
import time

ROOT = "/Volumes/ssd990pro1t/kronos"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf-cache")
sys.path.insert(0, f"{ROOT}/Kronos")

import torch  # noqa: E402

from model import Kronos, KronosTokenizer  # noqa: E402

TOK_DIR = (f"{ROOT}/hf-cache/hub/models--NeoQuasar--Kronos-Tokenizer-base/"
           "snapshots/0e0117387f39004a9016484a186a908917e22426")
MODEL_DIR = (f"{ROOT}/hf-cache/hub/models--NeoQuasar--Kronos-base/"
             "snapshots/2b554741eca47781b64468546e77fef3e85130e6")


def rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return -1.0


def bench_tokenizer(dev, B, T, steps=3):
    tok = KronosTokenizer.from_pretrained(TOK_DIR).to(dev)
    opt = torch.optim.AdamW(tok.parameters(), lr=1e-4)
    x = torch.randn(B, T, 6, device=dev)
    ts = []
    for i in range(steps):
        t0 = time.time()
        (z_pre, z), bsq, q, idx = tok(x)
        loss = torch.nn.functional.mse_loss(z_pre, x) + torch.nn.functional.mse_loss(z, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        ts.append(time.time() - t0)
    return sum(ts) / len(ts), ts


def bench_predictor(dev, B, T, steps=3):
    tok = KronosTokenizer.from_pretrained(TOK_DIR).to(dev)
    model = Kronos.from_pretrained(MODEL_DIR).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    x = torch.randn(B, T, 6, device=dev)
    stamp = torch.zeros(B, T, 5, device=dev)
    ts = []
    for i in range(steps):
        t0 = time.time()
        with torch.no_grad():
            s1, s2 = tok.encode(x, half=True)
        logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1, :])
        loss, l1, l2 = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        ts.append(time.time() - t0)
    return sum(ts) / len(ts), ts


def main():
    print(f"torch={torch.__version__} mps={torch.backends.mps.is_available()}")
    out = {}
    for dev_name in ("mps", "cpu"):
        dev = torch.device(dev_name)
        for label, fn, sizes in (
            ("tokenizer", bench_tokenizer, [(8, 561), (16, 561), (32, 561)]),
            ("predictor", bench_predictor, [(8, 561), (16, 561)]),
        ):
            for (B, T) in sizes:
                steps = 3 if dev_name == "mps" else 1
                try:
                    avg, ts = fn(dev, B, T, steps)
                    key = f"{dev_name}/{label}/B{B}"
                    out[key] = avg
                    print(f"[ok] {key:26s} avg={avg:.3f}s  per-step={[round(t,3) for t in ts]} "
                          f"peakRSS={rss_mb():.0f}MB")
                except Exception as exc:  # noqa: BLE001
                    print(f"[FAIL] {dev_name}/{label}/B{B}: {exc!r}"[:400])
                if dev_name == "cpu":
                    break  # one CPU config is enough
    print("\nSUMMARY " + repr({k: round(v, 3) for k, v in out.items()}))


if __name__ == "__main__":
    sys.exit(main())
