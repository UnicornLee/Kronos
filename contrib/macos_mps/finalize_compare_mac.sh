#!/bin/bash
# 训练队列全部结束后：跑三资产走查回测 + 汇总（Mac 侧）
# 用法: bash finalize_compare_mac.sh
set -euo pipefail
ROOT=/Volumes/ssd990pro1t/kronos
PY=$ROOT/native-venv/bin/python
FC=$ROOT/forecast/scripts
FT=$ROOT/finetune/runs

echo "=== 队列状态 ==="
cat "$FT/queue.log"
echo
echo "=== 三个模型的产物 ==="
for n in btc eth xaut; do
  D="$FT/${n}_1d_base"
  if [ -d "$D" ]; then
    echo "--- $n ---"
    du -sh "$D" | cut -f1
    ls "$D"
    for sub in tokenizer basemodel; do
      if [ -d "$D/$sub/best_model" ]; then
        echo -n "  $sub/best_model: "
        du -sh "$D/$sub/best_model" | cut -f1
      else
        echo "  $sub/best_model: 缺失"
      fi
    done
  else
    echo "--- $n --- 目录不存在"
  fi
done
echo

echo "=== 训练 loss 摘要 ==="
for n in btc eth xaut; do
  L="$FT/${n}_1d_base.log"
  [ -f "$L" ] || continue
  echo "--- ${n}_1d_base ---"
  grep -E "Summary" "$L" | tail -2
  grep -E "Train Loss|Val Loss|train_loss|val_loss" "$L" | tail -4
done
echo

echo "=== 走查回测（BTC/ETH/XAUT，16 采样 × 8 窗口） ==="
"$PY" "$FC/compare_all.py" --symbols BTC ETH XAUT --samples 16 --windows 8
echo
echo "=== 汇总文件 ==="
ls -la "$FT"/compare_all.* "$FT"/compare_ft_*_1D_ft.json 2>/dev/null
echo "DONE"
