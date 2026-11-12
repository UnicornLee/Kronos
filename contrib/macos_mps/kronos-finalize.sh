#!/bin/bash
# Kronos 微调训练队列收尾（Mac 侧自包含，不依赖 Windows 在线）
# 流程：等队列结束 -> 解析训练日志 -> 样本外走查回测 -> 汇总 -> 生成报告 -> 汇总产物
# 运行方式：launchd -> ssh 127.0.0.1（launchd 直接读外置盘会被 macOS TCC 拒绝）
LOG="/Users/unicorn/Library/Logs/kronos-finalize.log"
ROOT=/Volumes/ssd990pro1t/kronos
PY=$ROOT/native-venv/bin/python
FC=$ROOT/forecast/scripts
FT=$ROOT/finetune/runs
OUT=$ROOT/forecast/out/compare

mkdir -p "/Users/unicorn/Library/Logs" "$OUT"
exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] finalize watcher 启动"

# 0) 等外置盘挂载
for i in $(seq 1 60); do
  [ -d "$ROOT" ] && break
  echo "[$(date '+%F %T')] 等待外置盘 ($i)"
  sleep 5
done
if [ ! -d "$ROOT" ]; then
  echo "[$(date '+%F %T')] 外置盘缺失，退出"
  exit 1
fi

# 1) 等训练队列结束（BTC -> ETH -> XAUT）
echo "[$(date '+%F %T')] 等待训练队列结束..."
while pgrep -f kronos_queue.sh > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] 队列已结束"
cat "$FT/queue.log" || true
echo

# 2) 解析三个训练日志 -> ft_logs.json
echo "[$(date '+%F %T')] 步骤 1/4 解析训练日志"
"$PY" "$FC/parse_ft_logs.py" btc_1d_base eth_1d_base xaut_1d_base || echo "!! 日志解析失败"
echo

# 3) 样本外走查回测（预训练 vs 微调）
echo "[$(date '+%F %T')] 步骤 2/4 走查回测（BTC/ETH/XAUT，16 采样 x 8 窗口）"
cd "$FC" || exit 1
"$PY" "$FC/compare_all.py" --symbols BTC ETH XAUT --samples 16 --windows 8 || echo "!! 走查失败"
echo

# 4) 汇总报告
echo "[$(date '+%F %T')] 步骤 3/4 生成报告"
"$PY" "$FC/make_ft_report.py" "$FT" "$ROOT/forecast/FT_REPORT.md" || echo "!! 报告失败"
echo

# 5) 汇总产物到 out/compare（Windows 侧一键拉取这个目录即可）
echo "[$(date '+%F %T')] 步骤 4/4 汇总产物"
cp -f "$FT"/compare_ft_*_1D_ft.json "$OUT"/ 2>/dev/null || true
cp -f "$FT"/compare_all.json "$OUT"/ 2>/dev/null || true
cp -f "$FT"/compare_all.md "$OUT"/ 2>/dev/null || true
cp -f "$FT"/ft_logs.json "$OUT"/ 2>/dev/null || true
cp -f "$FT"/btc_1d_base.log "$FT"/eth_1d_base.log "$FT"/xaut_1d_base.log "$OUT"/ 2>/dev/null || true
cp -f "$ROOT"/forecast/FT_REPORT.md "$OUT"/ 2>/dev/null || true
ls -la "$OUT"
echo "[$(date '+%F %T')] 收尾完成"
echo "FINALIZE_DONE"
