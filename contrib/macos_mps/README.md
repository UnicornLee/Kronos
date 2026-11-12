# Kronos 微调（finetune_csv 路线）— Mac MPS 落地手册

实测环境：Mac mini M4 16GB / torch 2.15.0 / MPS / Python 3.11.15
仓库：`/Volumes/ssd990pro1t/kronos/Kronos`（上游未改动）
工作副本：`/Volumes/ssd990pro1t/kronos/finetune/finetune_csv_mps`
数据：`/Volumes/ssd990pro1t/kronos/forecast/data/{BTC,ETH,XAUT}_{1D,1H}.csv`
产物：`/Volumes/ssd990pro1t/kronos/finetune/runs/<exp_name>/{tokenizer,basemodel}/best_model/`

---

## 1. 官方两条路线，Mac 只能走第二条

| 路线 | 目录 | 数据 | 设备要求 | Mac |
|---|---|---|---|---|
| A：A股 Qlib 演示 | `finetune/` | qlib `csi300` 2011–2025，需 `qlib_data_preprocess.py` 生成 pkl | `train_tokenizer.py:223` / `train_predictor.py:185` 写死 `torch.device(f"cuda:{local_rank}")`，DDP + DistributedSampler，末尾 `raise RuntimeError("This script must be launched with torchrun.")` | ✗ |
| B：通用 CSV | `finetune_csv/` | CSV 列 `timestamps,open,high,low,close,volume,amount` | `finetune_base_model.py:375` / `finetune_tokenizer.py:291` = `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | ✓（改 MPS） |

路线 B 的 CSV 列名与我们已有的 OKX 数据**完全一致**，直接喂即可。

## 2. 完整流程（finetune_csv）

1. 准备 CSV：`timestamps,open,high,low,close,volume,amount`（volume/amount 可为 0）
2. 写 YAML：`data` / `training` / `model_paths` / `experiment` / `device` / `distributed` 六节
3. 跑 `train_sequential.py --config configs/xxx.yaml`（顺序执行 tokenizer → basemodel）
   - 或分开跑 `finetune_tokenizer.py` / `finetune_base_model.py`
4. 产物：`{base_path}/{exp_name}/tokenizer/best_model/`、`.../basemodel/best_model/`
5. 推理时**两个模型都要用微调版**：`KronosTokenizer.from_pretrained(tok_dir)` + `Kronos.from_pretrained(mdl_dir)`

## 3. Mac 上必须改的 4 处（补丁脚本已就绪）

| # | 位置 | 原样 | 改成 |
|---|---|---|---|
| 1 | `finetune_base_model.py:375`、`finetune_tokenizer.py:291`、`train_sequential.py:_setup_device` | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | `pick_device()`（新文件 `mps_device.py`：`$KRONOS_DEVICE` > cuda > mps > cpu） |
| 2 | `config_loader.py` | 单一 `batch_size` | 拆成 `tokenizer_batch_size` / `basemodel_batch_size` |
| 3 | `finetune_base_model.py:216/226/452`、`finetune_tokenizer.py:128/138/343` | `config.batch_size` | `config.basemodel_batch_size` / `config.tokenizer_batch_size` |
| 4 | `CustomKlineDataset._split_data_by_time` | 按时间比例切**原始行** | val/test 段前置 `lookback_window` 行作为上下文（见下） |

DDP 不用改：`use_ddp` 判定是 `dist.is_available() and dist.is_initialized()`，不起 torchrun 就是单进程；
`distributed.backend` 填 `gloo`（`nccl` 只给英伟达）；`pin_memory=True` → `False`（MPS 下无意义）。

### 4 处必须改的原因：数据集切分 bug

官方 `_split_data_by_time` 按时间比例切原始行，然后 `n_samples = len(self.data) - window + 1`。
BTC_1D 有 3239 行、lookback 512 + predict 7 + 1 = 520 行一个窗口：

```
train_end = 2915, val_end = 3238
train 段 2915 行 → 2396 个样本   ✓
val   段  323 行 → 323-520+1 = -196  ✗  ValueError: __len__() should return >= 0
```

修正：val/test 段向前借 `lookback_window` 行做 lookback 上下文，
窗口的**目标部分仍严格落在 val/test 期间内**（不泄漏到训练段）：
```
start = max(0, train_end - self.lookback_window)
val:   [start, val_end)   → 836 行 → 317 个样本 ✓
```

## 4. MPS 实测性能（Mac mini M4 16GB，窗口 512+7+1）

| 任务 | MPS | CPU | 备注 |
|---|---|---|---|
| tokenizer B8 | 0.62 s/step | 0.79 s/step | |
| tokenizer B16 | 0.44 s/step | — | 训练实际用的档位 |
| tokenizer B32 | 1.09 s/step | — | 也 OK |
| **predictor (base 102M) B8** | **1.85 s/step** | 10.14 s/step | MPS 快 4.8× |
| predictor (base) B16 | **OOM** | — | `kIOGPUCommandBufferCallbackErrorOutOfMemory` |
| predictor CPU B8 | — | 10.14 s/step | peakRSS 2456 MB |

> base 模型在 MPS 上 batch **必须 ≤ 8**，否则直接 OOM。

算子兼容性（全部通过）：`scatter_reduce(sum)`、`unique`、`index_select`、bfloat16 分配、
`KronosTokenizer` 前向+反向、`KronosPredictor` 前向。

## 5. 冒烟测试实测结果（BTC_1D / base，各 1 epoch）

```
数据      train 2396 样本 / val 317 样本（2017-12-29 → 2026-11-10）
tokenizer 149 步/epoch × 0.44 s ≈ 66 s     best val loss 见日志
basemodel 299 步/epoch × 1.85 s ≈ 9.2 min  train loss 2.2182 / val loss 2.4561
总耗时     11.69 分钟
产物      tokenizer/best_model/model.safetensors 15.8 MB
          basemodel/best_model/model.safetensors  409 MB
```

## 6. 微调有没有用？（1 epoch，走查 8 周，样本外）

窗口：2026-06-10 ~ 2026-11-04，每 21 天一个窗口，输入前 512 根日线，预测 7 根，
8 次采样取中位数，对比真实值。

| 模型 | close MAPE 均值 | 方向命中 |
|---|---|---|
| 预训练 Kronos-base | 3.398% | 5/8 = 62.5% |
| **微调 1 epoch** | **2.725%** | **6/8 = 75%** |

> 8 个窗口样本很小，方向性参考而非统计结论。脚本：`/Volumes/ssd990pro1t/kronos/forecast/scripts/compare_ft.py`
> 结果：`/Volumes/ssd990pro1t/kronos/finetune/runs/compare_ft_btc_1d.json`

## 7. 正式训练

```bash
cd /Volumes/ssd990pro1t/kronos/finetune/finetune_csv_mps
KRONOS_DEVICE=mps /Volumes/ssd990pro1t/kronos/native-venv/bin/python \
  train_sequential.py --config ../configs/config_btc_1d_base.yaml
```

当前配置（`config_btc_1d_base.yaml`）：BTC_1D / base / lookback 512 / predict 7 /
tokenizer 30 epoch × B16 / basemodel 20 epoch × B8 /
tokenizer_lr 2e-4 / predictor_lr 1e-5 / train:val = 9:1。

**预计耗时**：
- tokenizer：30 × 66 s ≈ **33 分钟**
- basemodel：20 × 9.2 min ≈ **3.1 小时**
- 合计 ≈ **3.6–4.5 小时**

日志：`/Volumes/ssd990pro1t/kronos/finetune/runs/btc_1d_base.log`

## 8. 其它标的 / 周期

| 目标 | 数据行数 | 单 epoch 步数 | 20 epoch 估算 | 建议 |
|---|---|---|---|---|
| BTC_1D | 3239 | 299 | ≈ 3.1 h | ✅ 已开跑 |
| ETH_1D | 3239 | 299 | ≈ 3.1 h | ✅ 同配置可跑 |
| XAUT_1D | 1577 | 132 | ≈ 1.4 h | ✅ 数据偏短，效果打折 |
| BTC_1H | 40139 | 4450 | ≈ **46 h** | ⚠️ 改 mini 模型 / 减 epoch / 增 batch（mini 显存余量大） |
| ETH_1H | 40139 | 4450 | ≈ 46 h | ⚠️ 同上 |
| XAUT_1H | 37845 | 4400 | ≈ 45 h | ⚠️ 同上 |

小时线要 `predict_window: 168`（一周 = 168 根小时线），窗口变成 512+168+1 = 681 行。
建议小时线用 **kronos-mini**（4.1M 参数）微调，显存和速度都宽裕得多。

## 9. 注意事项

1. 上游仓库 `Kronos/finetune_csv` **未改动**，所有补丁都在副本 `finetune_csv_mps` 里；升级上游后需重新打补丁。
2. `finetune/` 目录的代码注释是 Gemini 2.5 Pro 生成的，README.md:308 自己声明可能有误，**以代码为准**。
3. 微调数据归一化只用**窗口自身**的均值方差（`x = (x-x_mean)/(x_std+1e-5)` 后 clip），不会用到未来数据。
4. train 集每 epoch 用伪随机 stride `start_idx = (idx*9973 + (epoch+1)*104729) % (max_start+1)` 重采样，val/test 是顺序窗口。
5. `use_comet` 必须设 `false`，否则脚本要装 comet_ml 并联网。
6. 训练完推理要用**微调后的 tokenizer + 微调后的 predictor 配对**，混用预训练 tokenizer 会掉点。
7. Docker 容器内跑不了 MPS（Docker Desktop 不透传 Metal），微调只能在原生 venv 里做。
8. `pretrained_tokenizer` / `pretrained_predictor` 要填**本地 snapshot 绝对路径**（离线可用）。
