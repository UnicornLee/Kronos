#!/bin/bash
# 等 btc_1d_base 训练结束后，依次跑 ETH_1D、XAUT_1D 微调
FT=/Volumes/ssd990pro1t/kronos/finetune
WORK=$FT/finetune_csv_mps
PY=/Volumes/ssd990pro1t/kronos/native-venv/bin/python
QL=$FT/runs/queue.log

cd "$WORK" || exit 1

echo "[$(date '+%F %T')] 队列启动，等待 btc_1d_base 结束" >> "$QL"
while pgrep -f "train_sequential.py --config ../configs/config_btc_1d_base.yaml" > /dev/null; do
  sleep 30
done
echo "[$(date '+%F %T')] btc_1d_base 已结束，开始 ETH_1D" >> "$QL"

KRONOS_DEVICE=mps "$PY" train_sequential.py --config ../configs/config_eth_1d_base.yaml > "$FT/runs/eth_1d_base.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] ETH_1D 结束 exit=$rc" >> "$QL"

echo "[$(date '+%F %T')] 开始 XAUT_1D" >> "$QL"
KRONOS_DEVICE=mps "$PY" train_sequential.py --config ../configs/config_xaut_1d_base.yaml > "$FT/runs/xaut_1d_base.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] XAUT_1D 结束 exit=$rc" >> "$QL"

echo "[$(date '+%F %T')] 队列全部完成" >> "$QL"
