# Kronos 日线微调 —— BTC / ETH / XAUT 实测报告

> 模型：`NeoQuasar/Kronos-base`（102.3M）+ `Kronos-Tokenizer-base`；设备：Mac mini M4 (MPS, torch 2.15.0)；数据：OKX 现货日线。

## 一、训练过程

| 资产 | 阶段 | 轮数 | 首轮 val loss | 末轮 val loss | 最优 val loss | 平均每轮耗时 |
|---|---|---|---|---|---|---|
| BTC | tokenizer | 30/30 | 0.0059 | 0.0057 | 0.0055 | 91s |
| BTC | basemodel | 20/20 | 2.5306 | 2.6023 | 2.4562 | 589s |
| ETH | tokenizer | 30/30 | 0.0055 | 0.0047 | 0.0046 | 89s |
| ETH | basemodel | 20/20 | 2.5134 | 2.6111 | 2.416 | 589s |
| XAUT（代币化黄金） | tokenizer | 30/30 | 0.0027 | 0.0023 | 0.0022 | 46s |
| XAUT（代币化黄金） | basemodel | 20/20 | 1.8983 | 1.6153 | 1.6063 | 237s |

- **BTC**：数据 `/Volumes/ssd990pro1t/kronos/forecast/data/BTC_1D.csv`，lookback 512 / predict 7，device `mps`，DDP `False`，batch `8 (tokenizer=16, basemodel=8)`
- **ETH**：数据 `/Volumes/ssd990pro1t/kronos/forecast/data/ETH_1D.csv`，lookback 512 / predict 7，device `mps`，DDP `False`，batch `8 (tokenizer=16, basemodel=8)`
- **XAUT（代币化黄金）**：数据 `/Volumes/ssd990pro1t/kronos/forecast/data/XAUT_1D.csv`，lookback 512 / predict 7，device `mps`，DDP `False`，batch `8 (tokenizer=16, basemodel=8)`

## 二、样本外走查回测（预训练 vs 微调）

口径：模型 `kronos-base`，lookback 512 → 预测 7 根日线；8 个走查窗口（每 21 天一窗，全部落在训练期之后的验证区内）；每窗 16 条采样路径取中位。

| 资产 | 预训练 MAPE% | 微调 MAPE% | Δ | 预训练命中 | 微调命中 | Δ | 预训练逐日方向% | 微调逐日方向% |
|---|---|---|---|---|---|---|---|---|
| BTC | 3.155 | 3.693 | +0.538 | 62.5% | 62.5% | +0.0 | 39.6 | 52.1 |
| ETH | 5.170 | 4.485 | -0.685 | 62.5% | 62.5% | +0.0 | 37.5 | 41.7 |
| XAUT（代币化黄金） | 1.811 | 1.413 | -0.399 | 87.5% | 62.5% | -25.0 | 50.0 | 52.1 |

**BTC** 逐窗 MAPE%

| 窗口 | 预训练 | 微调 |
|---|---|---|
| 1 | 4.840 | 5.966 |
| 2 | 5.210 | 1.883 |
| 3 | 1.994 | 5.704 |
| 4 | 2.594 | 1.886 |
| 5 | 2.294 | 2.281 |
| 6 | 1.301 | 2.389 |
| 7 | 1.283 | 1.943 |
| 8 | 5.727 | 7.491 |

**ETH** 逐窗 MAPE%

| 窗口 | 预训练 | 微调 |
|---|---|---|
| 1 | 1.952 | 4.431 |
| 2 | 8.869 | 6.544 |
| 3 | 5.340 | 2.034 |
| 4 | 4.138 | 1.474 |
| 5 | 2.853 | 2.004 |
| 6 | 5.050 | 1.187 |
| 7 | 2.523 | 3.242 |
| 8 | 10.634 | 14.960 |

**XAUT（代币化黄金）** 逐窗 MAPE%

| 窗口 | 预训练 | 微调 |
|---|---|---|
| 1 | 1.397 | 2.170 |
| 2 | 2.905 | 0.864 |
| 3 | 2.326 | 2.556 |
| 4 | 1.290 | 0.817 |
| 5 | 1.327 | 0.981 |
| 6 | 1.443 | 1.977 |
| 7 | 2.761 | 1.114 |
| 8 | 1.041 | 0.821 |

## 三、结论

1. **误差层面**：微调后 MAPE 更低的资产 = ETH、XAUT（代币化黄金）。微调把模型往「这个市场的统计结构」上拉，最直接的收益体现在误差上。
2. **方向层面**：终点方向命中提升的资产 = 无；下降的资产 = XAUT（代币化黄金）。命中率的样本量只有 8 个窗口，单次翻转就是 12.5 个百分点，**统计上远不足以支撑「择时信号」的结论**。
3. **用法定位**：日线微调后的 Kronos 适合用作「量级 + 区间的先验」，用来给一个已有的判断补充概率分布，而不是用来替代判断本身。
4. **口径提醒（重要）**：训练配置是 `train:val = 9:1` 且没有单独留 test 段，每轮用 val 段算 val_loss 选 best_model。走查窗口全部落在 val 段内，所以**微调模型对这段数据有早停曝光**，其指标相对预训练模型是偏乐观的；预训练模型则完全没见过这段数据。两者在**同一批窗口**上对比，结论只能读作「同口径下微调是否更好」，不能读作「微调后的样本外真实水平」。要做干净的样本外评估，需要在配置里留出 test 段并把走查窗口挪到 test 段之后。

## 四、复现

```bash
# 1) 打 MPS 补丁（上游 finetune_csv 不动）
export KRONOS_REPO=/path/to/Kronos
export KRONOS_WORK=/path/to/work/finetune_csv_mps
python contrib/macos_mps/patch_finetune_csv.py
python contrib/macos_mps/patch_config_loader.py
python contrib/macos_mps/patch_dataset_split.py

# 2) 排队训练
cd "$KRONOS_WORK"
nohup bash contrib/macos_mps/kronos_queue.sh > /dev/null 2>&1 &

# 3) 走查回测 + 汇总
bash contrib/macos_mps/finalize_compare_mac.sh

# 4) 出图（Windows 侧）
pwsh -File contrib/macos_mps/finalize_compare_win.ps1
```
