#!/bin/bash
# 一眼看清微调队列跑到哪了
FT=/Volumes/ssd990pro1t/kronos/finetune/runs
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "=== queue.log ==="
cat "$FT/queue.log" 2>/dev/null || echo "(还没有)"
echo
echo "=== 正在跑的 train_sequential ==="
ps aux | grep 'train_sequential.py --config' | grep -v grep || echo "(无)"
echo
for n in btc eth xaut; do
  L="$FT/${n}_1d_base.log"
  [ -f "$L" ] || continue
  echo "--- ${n}_1d_base ---"
  grep -E 'Epoch [0-9]+/[0-9]+, Step' "$L" | tail -1
  grep -E 'Summary' "$L" | tail -1
  grep -E 'Val Loss|Validation Loss|Loss:' "$L" | tail -1
done
echo
echo "=== 产物 ==="
for n in btc eth xaut; do
  D="$FT/${n}_1d_base"
  [ -d "$D" ] || continue
  echo -n "$n: "
  du -sh "$D" 2>/dev/null | cut -f1
  ls "$D" 2>/dev/null | tr '\n' ' '
  echo
done
echo
echo "=== ETA ==="
/Volumes/ssd990pro1t/kronos/native-venv/bin/python "$HERE/eta.py" 2>/dev/null || echo "(eta.py 不可用)"
echo
echo "=== 负载/温度 ==="
uptime
