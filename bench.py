#!/usr/bin/env python3
"""Kronos benchmark: load time, cold/warm prediction latency, peak RSS, device.

Usage (inside container or native venv):
    python bench.py                      # default: NeoQuasar/Kronos-small, cpu
    BENCH_MODEL=NeoQuasar/Kronos-base BENCH_DEVICE=mps python bench.py
"""
import json
import os
import platform
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (HERE, os.path.dirname(HERE)):
    if os.path.isdir(os.path.join(_cand, "model")):
        sys.path.insert(0, _cand)
        break

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

MODEL_ID = os.environ.get("BENCH_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_ID = os.environ.get("BENCH_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
DEVICE = os.environ.get("BENCH_DEVICE", "auto")
LOOKBACK = int(os.environ.get("BENCH_LOOKBACK", "400"))
PRED_LEN = int(os.environ.get("BENCH_PRED_LEN", "120"))
SAMPLE_COUNT = int(os.environ.get("BENCH_SAMPLES", "1"))
REPEAT = int(os.environ.get("BENCH_REPEAT", "3"))


def peak_rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes
    mb = raw / 1024 if platform.system() == "Darwin" else raw / 1024
    return round(mb, 1)


def pick_device():
    if DEVICE != "auto":
        return DEVICE
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    t0 = time.perf_counter()
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    load_s = time.perf_counter() - t0

    device = pick_device()
    max_context = 2048 if "mini" in MODEL_ID else 512
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)

    rng = np.random.default_rng(0)
    n = LOOKBACK
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.1, n),
        "high": close + np.abs(rng.normal(0, 0.3, n)),
        "low": close - np.abs(rng.normal(0, 0.3, n)),
        "close": close,
        "volume": rng.integers(1_000, 10_000, n).astype(float),
        "amount": rng.integers(100_000, 1_000_000, n).astype(float),
    })
    x_ts = pd.Series(pd.date_range("2025-01-01", periods=n, freq="5min"))
    y_ts = pd.Series(pd.date_range(x_ts.iloc[-1], periods=PRED_LEN + 1, freq="5min")[1:])

    def run():
        return predictor.predict(
            df=df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=SAMPLE_COUNT,
        )

    t1 = time.perf_counter()
    pred = run()
    cold_s = time.perf_counter() - t1

    warm = []
    for _ in range(REPEAT):
        t2 = time.perf_counter()
        run()
        warm.append(time.perf_counter() - t2)

    out = {
        "model": MODEL_ID,
        "tokenizer": TOKENIZER_ID,
        "device": device,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "host": f"{platform.system()} {platform.machine()}",
        "load_seconds": round(load_s, 2),
        "predict_cold_seconds": round(cold_s, 3),
        "predict_warm_seconds": [round(x, 3) for x in warm],
        "predict_warm_avg_seconds": round(sum(warm) / len(warm), 3),
        "lookback": LOOKBACK,
        "pred_len": PRED_LEN,
        "sample_count": SAMPLE_COUNT,
        "peak_rss_mb": peak_rss_mb(),
        "pred_rows": int(len(pred)),
        "pred_close_head": [round(float(v), 3) for v in pred["close"].head(5).tolist()],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
