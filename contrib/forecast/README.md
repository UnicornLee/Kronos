# Kronos 一周走势预测（BTC / ETH / XAUT）

用 [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) 在 Mac mini M4 上做未来一周的
BTC / ETH / XAUT 走势预测。原生 PyTorch **MPS**（Apple GPU 直通）跑推理，比 Docker 容器快约 12 倍。

```
D:\DSH\kronos-forecast\
├─ REPORT.md                 最终报告（预测表 + 路径明细 + 回测准确率 + 复现步骤）
├─ viz_data.json             精简数据集（供聊天内交互卡片使用）
├─ scripts\
│  ├─ fetch_okx.py           抓 OKX 现货 K 线（BTC/ETH/XAUT × 1H/1D）
│  ├─ predict_final.py       MPS 推理（最终版，32/16 次采样 + 双周期共识 ensemble_week.json）
│  ├─ predict_forecast.py    早期版本推理脚本（保留参考）
│  ├─ backtest_v2.py         走查回测（12 周日线 / 6 周小时线，T=1.0 vs T=0.7 + 朴素基线）
│  ├─ backtest_tz.py         时区特征对齐实验（tz=0 vs tz=8）
│  ├─ calibrate.py           用回测偏差校正预测方向
│  ├─ make_charts.py         出图（matplotlib，Windows 侧）
│  ├─ make_report.py         生成 REPORT.md
│  ├─ make_viz_data.py       精简数据集（viz_data.json）
│  ├─ compare_ft.py          走查回测（预训练 vs 微调，单资产：MAPE + 方向命中）
│  ├─ compare_all.py         三资产批量走查 + 汇总 compare_all.json/md
│  ├─ make_compare_charts.py 微调对比图（MAPE / 命中 / 逐窗 / 差值）
│  ├─ parse_ft_logs.py       训练日志 → ft_logs.json（每轮 val loss / 耗时 / 产物）
│  ├─ make_ft_report.py      生成 FT_REPORT.md（训练过程 + 走查 + 结论 + 复现）
│  ├─ preflight_datasets.py  训练前预检：train/val 切分能否出样本
│  └─ run_all.sh             Mac 侧一键刷新（抓数 + 预测 + 可选回测）
├─ data\                     6 份 OHLCV CSV（从 Mac 同步）
├─ out\                      预测 JSON/CSV、回测 JSON、日志
└─ charts\                   forecast_*.png / backtest_*.png
```

## 30 秒上手（Mac 侧）

```bash
bash /Volumes/ssd990pro1t/kronos/forecast/scripts/run_all.sh            # 刷新预测
bash /Volumes/ssd990pro1t/kronos/forecast/scripts/run_all.sh --backtest  # 顺带跑 12 周回测
KRONOS_SAMPLES_1D=64 KRONOS_SAMPLES_1H=32 bash .../run_all.sh           # 采样更多，区间更窄但更慢
```

Mac 侧数据目录：`/Volumes/ssd990pro1t/kronos/forecast/{data,out}`（外置盘，不占系统盘）。

Windows 侧出图与出报告（matplotlib 装在 DSH 自带 Python 里）：

```powershell
$py = 'C:\Users\unico\.dsh\dsh-runtimes\dsh-next-primary-runtime\dependencies\python\python.exe'
& $py D:\DSH\kronos-forecast\scripts\calibrate.py     # 用回测偏差校正
& $py D:\DSH\kronos-forecast\scripts\make_charts.py   # 12 张 PNG
& $py D:\DSH\kronos-forecast\scripts\make_report.py   # REPORT.md
```

## 关键结论

- **容器内没有 MPS**：macOS 的 Docker Desktop 不透传 Metal，Kronos 在容器里只能跑 CPU
  （small 模型 22.5s / 次 vs 原生 MPS 1.9s / 次）。所以预测一律用原生 venv：
  `/Volumes/ssd990pro1t/kronos/native-venv/bin/python`。
- **Kronos 的 `predict()` 内部会把多次采样取平均**，丢掉分位数。想要不确定区间必须
  固定种子多次独立调用 `predict(sample_count=1)` 再取分位数（脚本里已这么做）。
- **WebUI 不适合出未来一周**：`/api/predict` 是「拿前 N 根预测后 M 根」并与已有数据对照，
  适合看回测效果；真正的未来一周必须自己构造未来的时间戳，脚本已实现。

## 微调效果评估（compare_* 系列）

预训练模型和微调模型跑**同一批样本外窗口**，比 close MAPE 与终点方向命中率：

```bash
# Mac：单个资产
python compare_ft.py --symbol BTC --samples 16 --windows 8
# Mac：三个资产 + 汇总
python compare_all.py --symbols BTC ETH XAUT --samples 16 --windows 8
```

口径：lookback 512 → 预测 7 根日线；窗口按每 21 天一个往前推，
全部落在训练期之后的验证区内；每窗 16 条采样路径取中位数。

结果文件：`/Volumes/ssd990pro1t/kronos/finetune/runs/compare_ft_<SYM>_1D_ft.json`
（`_ft` 后缀是必须的：macOS 文件系统大小写不敏感，`1D`/`1d` 会撞名导致资产被跳过）。
汇总：`compare_all.json` / `compare_all.md`。
Windows 侧出图与报告见 `contrib/macos_mps/finalize_compare_win.ps1`。

## 图形界面（可选）

- 原生 MPS 版（快）：http://unicornmac-mini:7071/
- 容器 CPU 版（慢、但可移植到有 CUDA 的 Linux）：http://unicornmac-mini:7070/
