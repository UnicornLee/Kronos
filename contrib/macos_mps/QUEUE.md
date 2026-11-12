# 微调训练队列（Apple Silicon / MPS）

`kronos_queue.sh` 用于在 Mac 上串行排队跑多个资产的微调，避免 MPS 显存/内存
争抢导致 OOM。

```bash
cd /path/to/Kronos/contrib/macos_mps
cp config_btc_1d_base.yaml config_eth_1d_base.yaml config_xaut_1d_base.yaml \
   /path/to/work/finetune/configs/

nohup bash kronos_queue.sh > /dev/null 2>&1 &
```

行为：

1. 轮询 `pgrep -f "train_sequential.py --config ../configs/config_btc_1d_base.yaml"`，
   等当前训练结束后再启动下一个；
2. 依次执行 `config_eth_1d_base.yaml` → `config_xaut_1d_base.yaml`；
3. 每段日志写入 `<base_path>/runs/{eth,xaut}_1d_base.log`，
   队列总日志写入 `<base_path>/runs/queue.log`；
4. `KRONOS_DEVICE=mps` 显式指定设备（MPS 上 base 模型 predictor batch 必须 ≤ 8）。

`gen_queue_configs.py` 用于从 `config_btc_1d_base.yaml` 批量生成其它资产的配置
（只替换 data_path / exp_name / experiment.name / description）。

## 实测耗时（M4 16GB，torch 2.15.0，base 模型）

| 资产 | 数据量 | 预估时长 |
|---|---|---|
| BTC_1D | 3239 行 | ≈ 3.6h |
| ETH_1D | 3239 行 | ≈ 3.1h |
| XAUT_1D | 1577 行 | ≈ 1.4h |

1H 级别（4 万行 + predict_window 168）单轮约 46h，需要换 mini 模型，不建议在 M4 上跑。
