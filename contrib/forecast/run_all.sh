#!/usr/bin/env bash
# One-command refresh of the Kronos weekly forecast on the Mac (native MPS).
#
#   bash /Volumes/ssd990pro1t/kronos/forecast/scripts/run_all.sh [--backtest]
#
# Steps:
#   1. pull the latest OKX candles for BTC/ETH/XAUT (1H + 1D)
#   2. run the native-MPS forecast (7 daily bars + 168 hourly bars, median + P10/P90)
#   3. optional: longer walk-forward accuracy backtest
#
# Env knobs for step 2 (higher sampling = tighter band, slower):
#   KRONOS_SAMPLES_1D=32 KRONOS_SAMPLES_1H=16 bash run_all.sh
#
# Everything is written under /Volumes/ssd990pro1t/kronos/forecast/{data,out}.
set -euo pipefail

ROOT=/Volumes/ssd990pro1t/kronos
FC=$ROOT/forecast
VENV=$ROOT/native-venv/bin/python
SYS_PY=/usr/bin/python3
LOG=$FC/out

export HF_HOME=$ROOT/hf-cache
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_TELEMETRY=1
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

mkdir -p "$FC/data" "$LOG"
cd "$FC"

echo "[1/3] fetching OKX candles ..."
"$SYS_PY" scripts/fetch_okx.py

echo "[2/3] forecasting on MPS ..."
"$VENV" scripts/predict_final.py

if [ "${1:-}" = "--backtest" ]; then
  echo "[3/3] walk-forward backtest (12 weekly windows) ..."
  "$VENV" scripts/backtest_v2.py
else
  echo "[3/3] skipped backtest (pass --backtest to enable)"
fi

echo "done. artifacts in $FC/out"
ls -1 "$FC/out" | tail -20
