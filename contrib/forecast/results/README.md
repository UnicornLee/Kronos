# Kronos 日线微调实测产物（Mac mini M4 / MPS）

生成时间：2026-11-13 00:25（Mac 侧自动收尾 watcher 产出，Windows 侧汇总复刻）
对应分支：develop，训练配置 `contrib/macos_mps/configs/config_{btc,eth,xaut}_1d_base.yaml`

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `FT_REPORT.md` | 完整实测报告（路线选择、MPS 性能、训练结果、走查回测、结论与口径提醒） |
| `compare_all.md` | 三资产预训练 vs 微调横向对比摘要（可直接贴 Notion / PR） |
| `compare_all.json` | 上表的机器可读版本（供 make_compare_charts.py / push_ft.py 消费） |
| `compare_ft_btc_1D_ft.json` / `_eth_` / `_xaut_` | 单资产逐窗口走查明细（8 个窗口的 MAPE、方向命中、预测与实际序列） |
| `ft_logs.json` | 三份训练日志解析结果（逐 epoch train/val loss、耗时、产物路径） |
| `charts/compare_mape.png` | 三资产 MAPE 对比（预训练 vs 微调） |
| `charts/compare_hit.png` | 方向命中率对比 |
| `charts/compare_windows.png` | 逐窗口 MAPE 明细 |
| `charts/compare_delta.png` | MAPE 变化量（微调 − 预训练） |

## 结论速览

- BTC_1D：MAPE 3.155% → 3.693%（变差 +0.538），命中 62.5% → 62.5%
- ETH_1D：MAPE 5.170% → 4.485%（更好 −0.685），命中 62.5% → 62.5%
- XAUT_1D：MAPE 1.811% → 1.413%（更好 −0.399），命中 87.5% → 62.5%

口径提醒：train:val = 9:1 且无独立 test 段，best_model 由 val_loss 选出，8 个走查窗口全部落在 val 段内，
微调模型对该段存在早停曝光，指标相对预训练偏乐观，只能读作「同口径下微调是否更好」。
