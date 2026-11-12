# Kronos 微调夜间状态（2026-11-12 18:10 记录）

## 现在在跑什么

| 项 | 状态 |
| --- | --- |
| BTC_1D 微调 | basemodel epoch 18/20，约 18:30 结束（tokenizer 30/30 已完成） |
| ETH_1D 微调 | 队列自动接续，预计 ~22:00 结束 |
| XAUT_1D 微调 | 队列自动接续，预计 ~23:45 结束 |
| 训练队列 | `/tmp/kronos_queue.sh`（PID 76339），日志 `finetune/runs/queue.log` |
| 收尾 watcher | launchd `com.unicorn.kronos-finalize` → ssh → `~/.local/bin/kronos-finalize.sh`，已在跑 |

**本机（Windows）关机不影响以上任何一步**：训练与收尾全部在 Mac 上自包含。

## 训练队列结束后自动发生什么

`kronos-finalize.sh` 会依次：

1. `parse_ft_logs.py btc_1d_base eth_1d_base xaut_1d_base` → `finetune/runs/ft_logs.json`
2. `compare_all.py --symbols BTC ETH XAUT --samples 16 --windows 8`
   → `finetune/runs/compare_ft_{btc,eth,xaut}_1D_ft.json` + `compare_all.json` + `compare_all.md`
3. `make_ft_report.py` → `forecast/FT_REPORT.md`
4. 汇总到 `forecast/out/compare/`（含三份训练日志）

日志：`~/Library/Logs/kronos-finalize.log`，结束时会打一行 `FINALIZE_DONE`。

## 明天回来怎么收尾（Windows 一条命令）

```powershell
pwsh -File D:\DSH\kronos-forecast\finalize_all.ps1          # 拉产物 + 出对比图 + 出 FT_REPORT.md
pwsh -File D:\DSH\kronos-forecast\finalize_all.ps1 -Notion  # 再归档到 Notion
```

如果提示 `compare_all.json 不存在`，说明训练还没跑完，先看
`ssh mac "tail -20 ~/Library/Logs/kronos-finalize.log"`。

## 已推送的内容

- 仓库：git@github.com:UnicornLee/Kronos.git，分支 `develop`，最新提交 `fdcea98`
- `contrib/macos_mps/`：MPS 补丁、探针、配置、队列、ETA、收尾脚本、launchd watcher
- `contrib/forecast/`：抓数、预测、走查回测、出图、出报告、Windows 一键收尾、Notion 归档脚本
- Notion 子页 `415d9f93-8b29-81ca-97fa-e1e5d0430ee9` 已追加「五、Kronos 日线微调实测」方法论章节（19 块）；
  最终走查结果与横向对比结论待训练结束后以「五（续）」补上。

## 待办（下次会话）

1. 跑 `finalize_all.ps1 -Notion`，读出三资产横向对比结论并补 Notion 的「五（续）」章节。
2. 把最终产物（`FT_REPORT.md`、对比图、`compare_all.json`）提交到 develop。
3. 结论口径提醒：训练 train:val=9:1 且无独立 test 段，best_model 由 val_loss 选出，
   而走查窗口全落在 val 段内 → 微调模型对该段有早停曝光，指标相对预训练偏乐观；
   只能读作「同口径下微调是否更好」，不是「样本外真实水平」。
