# contrib — 社区贡献与本地扩展

本目录收录在 Apple Silicon（macOS / MPS）上把 Kronos 跑起来所需的补丁、
配置与实战脚本。上游 `model/`、`webui/`、`finetune/`、`finetune_csv/` 均未修改。

| 目录 | 内容 | 用途 |
|---|---|---|
| `macos_mps/` | MPS 设备补丁、配置、MPS 探针 | 让 `finetune_csv/` 在 Mac 上跑微调 |
| `forecast/` | 数据抓取、预测、走查回测、出图与报告 | BTC / ETH / XAUT 的一周走势预测全流程 |

## macos_mps 快速开始

```bash
# 1) 打补丁（生成 finetune_csv_mps 副本，上游目录不动）
export KRONOS_REPO=/path/to/Kronos
export KRONOS_WORK=/path/to/work/finetune_csv_mps
python contrib/macos_mps/patch_finetune_csv.py
python contrib/macos_mps/patch_config_loader.py
python contrib/macos_mps/patch_dataset_split.py

# 2) 冒烟（各 1 epoch）
cd "$KRONOS_WORK"
python train_sequential.py --config /path/to/contrib/macos_mps/configs/config_smoke.yaml
```

补丁做了 4 件事：

1. **设备选择**：新增 `mps_device.py`，`pick_device()` 按 `$KRONOS_DEVICE` > cuda > mps > cpu 选择；
   `finetune_tokenizer.py` / `finetune_base_model.py` / `train_sequential.py` 三处 device 全部改走它。
2. **batch size 拆分**：`config_loader.py` 支持 `tokenizer_batch_size` 与 `basemodel_batch_size`
   两个独立值（MPS 上 predictor batch 必须 ≤ 8）。
3. **pin_memory=False**（MPS 后端不支持 pinned memory）。
4. **数据集切分修正**：`CustomKlineDataset._split_data_by_time` 按时间比例切原始行，
   val/test 段行数可能小于一个窗口，触发 `ValueError: __len__() should return >= 0`。
   修法是给 val/test 段前置 `lookback_window` 行作为上下文，目标部分仍留在 val/test 内。

实测（M4 / 16GB / torch 2.15.0）：tokenizer B8 0.62s，predictor(base) B8 1.85s/step，
CPU 同口径 10.1s/step → MPS 快约 4.8×；predictor B16 会 OOM。

## forecast 快速开始

```bash
python contrib/forecast/fetch_okx.py       # 抓 OKX 现货 K 线
python contrib/forecast/predict_final.py    # 跑 base(mini) 一周预测
python contrib/forecast/backtest_v2.py      # 走查回测
python contrib/forecast/make_report.py      # 生成报告
```

注意 `KronosPredictor.predict()` 内部对采样做均值，会丢掉分位数；
脚本改为固定种子多次独立调用 `predict(sample_count=1)` 后自行统计中位 / P10 / P90。
