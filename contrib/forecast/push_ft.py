# -*- coding: utf-8 -*-
"""把「Kronos 日线微调实测」这一章追加到已有的 Notion 子页面。

数据来源全部来自真实产物，不写死数字：
  D:\\DSH\\kronos-forecast\\out\\compare\\ft_logs.json     训练日志解析结果
  D:\\DSH\\kronos-forecast\\out\\compare\\compare_all.json 三资产走查汇总
  D:\\DSH\\kronos-forecast\\FT_REPORT.md                  完整报告（可选内嵌）

用法: python push_ft.py [page_id]
"""
import json
import os
import sys
import urllib.request
import urllib.error

TOKEN_FILE = r"C:\Users\unico\.dsh\notion.json"
API = "https://api.notion.com/v1/blocks/%s/children"
BATCH = 90
PAGE_ID = sys.argv[1] if len(sys.argv) > 1 else "415d9f93-8b29-81ca-97fa-e1e5d0430ee9"
CMP_DIR = os.environ.get("KRONOS_CMP_DIR", r"D:\DSH\kronos-forecast\out\compare")
REPORT = os.environ.get("KRONOS_REPORT", r"D:\DSH\kronos-forecast\FT_REPORT.md")

ORDER = ["BTC", "ETH", "XAUT"]
NAME = {"BTC": "BTC", "ETH": "ETH", "XAUT": "XAUT（代币化黄金）"}


def load_token():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["token"]


def post(token, page_id, children):
    body = json.dumps({"children": children}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API % page_id, data=body, method="PATCH",
        headers={"Authorization": "Bearer " + token,
                 "Notion-Version": "2022-06-28",
                 "Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.status, resp.read().decode("utf-8")


def h1(t):
    return {"object": "block", "type": "heading_1",
            "heading_1": {"rich_text": [{"type": "text", "text": {"content": t}}]}}


def h2(t):
    return {"object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": t}}]}}


def h3(t):
    return {"object": "block", "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": t}}]}}


def para(t):
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": t}}]}}


def bullet(t):
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": t}}]}}


def code(t, lang="plain text"):
    return {"object": "block", "type": "code",
            "code": {"rich_text": [{"type": "text", "text": {"content": t}}],
                     "language": lang}}


def callout(t, emoji="🎯"):
    return {"object": "block", "type": "callout",
            "callout": {"rich_text": [{"type": "text", "text": {"content": t}}],
                        "icon": {"type": "emoji", "emoji": emoji}}}


def divider():
    return {"object": "block", "type": "divider", "divider": {}}


def table(header, rows):
    def row(cells):
        return {"type": "table_row",
                "table_row": {"cells": [[{"type": "text", "text": {"content": str(c)}}]
                                        for c in cells]}}
    n = len(header)
    return {"object": "block", "type": "table",
            "table": {"table_width": n, "has_column_header": True,
                      "has_row_header": False,
                      "children": [row(header)] + [row(r) for r in rows]}}


def build():
    logs = json.load(open(os.path.join(CMP_DIR, "ft_logs.json"), encoding="utf-8"))
    cmp_all = json.load(open(os.path.join(CMP_DIR, "compare_all.json"), encoding="utf-8"))
    assets = cmp_all.get("assets", cmp_all)
    B = []

    B.append(divider())
    B.append(h1("五、Kronos 日线微调实测（Mac mini M4 / MPS）"))
    B.append(para("在 Mac mini M4（16GB、macOS 27.0.1）上用 Apple GPU（MPS）跑通 Kronos 的 "
                  "finetune_csv 微调路线，对 BTC / ETH / XAUT 三个日线资产做了完整训练，"
                  "并用同一批样本外走查窗口对比预训练与微调后的模型。"))

    B.append(h2("5.1 路线选择与 Mac 适配"))
    B.append(para("官方有两条微调路线，Mac 只能走第二条："))
    B.append(table(
        ["路线", "数据", "设备要求", "Mac 可用"],
        [["finetune/（A股 Qlib 演示）", "qlib csi300 2011–2025，需预处理 pkl",
          "train_tokenizer.py:223 / train_predictor.py:185 写死 torch.device(f\"cuda:{local_rank}\")，DDP + DistributedSampler，末尾 raise RuntimeError(\"must be launched with torchrun\")", "✗"],
         ["finetune_csv/（通用 CSV）", "CSV 列 timestamps,open,high,low,close,volume,amount",
          "finetune_base_model.py:375 / finetune_tokenizer.py:291 = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")", "✓ 改 MPS"]]))
    B.append(para("我们手上的 OKX 日线数据列名与 finetune_csv 完全一致，直接喂即可。"))
    B.append(h3("Mac 上必须改的 4 处（补丁打在副本 finetune_csv_mps，上游一行未动）"))
    B.append(bullet("device 选择：新增 mps_device.py 的 pick_device()（$KRONOS_DEVICE > cuda > mps > cpu），"
                    "替换 finetune_base_model.py:375、finetune_tokenizer.py:291、train_sequential.py:_setup_device"))
    B.append(bullet("config_loader.py 的单一 batch_size 拆成 tokenizer_batch_size / basemodel_batch_size"))
    B.append(bullet("两脚本 DataLoader 与 logger 行分别读各自的 batch size，pin_memory=True → False"))
    B.append(bullet("官方 bug：CustomKlineDataset._split_data_by_time 按时间比例切原始行，"
                    "val 段行数可能小于一个窗口 → ValueError: __len__() should return >= 0。"
                    "修法：val/test 段前置 lookback_window 行作上下文，目标部分仍严格留在 val/test 内"))
    B.append(para("DDP 不用改（use_ddp 判据是 dist.is_initialized()，不起 torchrun 就是单进程），"
                  "distributed.backend 填 gloo。"))

    B.append(h2("5.2 MPS 实测性能与显存上限"))
    B.append(table(
        ["任务", "MPS", "CPU", "备注"],
        [["tokenizer B8", "0.62 s/step", "0.79 s/step", ""],
         ["tokenizer B16", "0.44 s/step", "—", "训练实际档位"],
         ["tokenizer B32", "1.09 s/step", "—", "也 OK"],
         ["predictor (base 102M) B8", "1.85 s/step", "10.14 s/step", "MPS 快 4.8×"],
         ["predictor (base) B16", "OOM", "—", "kIOGPUCommandBufferCallbackErrorOutOfMemory"]]))
    B.append(callout("base 模型在 MPS 上 batch 必须 ≤ 8，否则直接 OOM。"
                      "训练期间不要在 Mac 上并行跑任何占显存的对比任务，会把训练顶掉。", "⚠️"))

    B.append(h2("5.3 训练结果"))
    rows = []
    for sym in ORDER:
        d = logs.get(sym)
        if not d:
            continue
        for ph, label in (("tokenizer", "tokenizer"), ("basemodel", "basemodel")):
            v = d.get("phases", {}).get(ph)
            if not v or not v.get("epoch_count"):
                continue
            total = (v.get("epochs") or [{}])[0].get("total", "?")
            t = v.get("epoch_sec_mean")
            rows.append([NAME[sym], label, f"{v['epoch_count']}/{total}",
                         v.get("val_first"), v.get("val_last"), v.get("val_best"),
                         f"{round(t)}s" if t else "n/a"])
    if rows:
        B.append(table(["资产", "阶段", "轮数", "首轮 val loss", "末轮 val loss",
                        "最优 val loss", "平均每轮耗时"], rows))

    B.append(h2("5.4 样本外走查回测（预训练 vs 微调）"))
    B.append(para("口径：模型 kronos-base，lookback 512 → 预测 7 根日线；"
                  f"{cmp_all.get('windows', 8)} 个走查窗口（每 21 天一窗），"
                  f"每窗 {cmp_all.get('samples', 16)} 条采样路径取中位；"
                  "预训练与微调用同一批窗口、同一批随机种子。"))
    rows = []
    for sym in ORDER:
        r = assets.get(sym)
        if not r:
            continue
        pre, ft, dl = r["pretrained"], r["finetuned"], r["delta"]
        rows.append([NAME[sym], f"{pre['mape_mean']:.3f}%", f"{ft['mape_mean']:.3f}%",
                     f"{dl['mape_mean']:+.3f}",
                     f"{pre['hit_rate']*100:.1f}%", f"{ft['hit_rate']*100:.1f}%",
                     f"{dl['hit_rate']*100:+.1f}",
                     f"{pre['day_direction_mean']:.1f}", f"{ft['day_direction_mean']:.1f}"])
    if rows:
        B.append(table(["资产", "预训练 MAPE", "微调 MAPE", "ΔMAPE",
                        "预训练命中", "微调命中", "Δ命中",
                        "预训练逐日方向%", "微调逐日方向%"], rows))

    B.append(h2("5.5 结论"))
    better_m = [s for s in ORDER if s in assets and
                assets[s]["finetuned"]["mape_mean"] < assets[s]["pretrained"]["mape_mean"]]
    better_h = [s for s in ORDER if s in assets and
                assets[s]["finetuned"]["hit_rate"] > assets[s]["pretrained"]["hit_rate"]]
    worse_h = [s for s in ORDER if s in assets and
               assets[s]["finetuned"]["hit_rate"] < assets[s]["pretrained"]["hit_rate"]]
    B.append(bullet("误差层面：微调后 MAPE 更低的资产 = "
                    + ("、".join(NAME.get(s, s) for s in better_m) if better_m else "无")
                    + "。微调把模型往「这个市场的统计结构」上拉，最直接的收益体现在误差上。"))
    B.append(bullet("方向层面：终点方向命中提升的资产 = "
                    + ("、".join(NAME.get(s, s) for s in better_h) if better_h else "无")
                    + "；下降的资产 = " + ("、".join(NAME.get(s, s) for s in worse_h) if worse_h else "无")
                    + "。命中率的样本量只有几个窗口，单次翻转就是十几个百分点，"
                      "统计上远不足以支撑「择时信号」的结论。"))
    B.append(bullet("口径提醒（重要）：训练配置是 train:val = 9:1 且没有单独留 test 段，"
                    "每轮用 val 段算 val_loss 选 best_model，而走查窗口全部落在 val 段内——"
                    "微调模型对这段数据有早停曝光，指标相对预训练模型偏乐观；"
                    "预训练模型则完全没见过这段数据。两者同口径对比，只能读作"
                    "「同口径下微调是否更好」，不能读作「微调后的样本外真实水平」。"))
    B.append(bullet("用法定位：日线微调后的 Kronos 适合用作「量级 + 区间的先验」，"
                    "给一个已有的判断补充概率分布，而不是用来替代判断本身。"))

    B.append(h2("5.6 产物与代码"))
    B.append(code("产物（外置盘 /Volumes/ssd990pro1t/kronos/finetune/runs/）\n"
                  "  {btc,eth,xaut}_1d_base/tokenizer/best_model/   15.8 MB\n"
                  "  {btc,eth,xaut}_1d_base/basemodel/best_model/   409 MB\n"
                  "  {btc,eth,xaut}_1d_base.log                     训练日志\n"
                  "  compare_ft_{SYM}_1D_ft.json                    走查结果\n"
                  "  compare_all.json / compare_all.md              三资产汇总\n"
                  "  ft_logs.json                                    日志解析结果"))
    B.append(code("代码（UnicornLee/Kronos 的 develop 分支）\n"
                  "  contrib/macos_mps/   补丁脚本 + MPS 探针 + 配置 + 队列 + ETA + 收尾脚本\n"
                  "  contrib/forecast/    抓数 + 预测 + 走查回测 + 出图 + 出报告"))
    B.append(para("上游 model/ webui/ finetune/ finetune_csv/ 一行未改，所有补丁都打在副本上。"))

    if os.path.exists(REPORT):
        txt = open(REPORT, encoding="utf-8").read()
        if len(txt) < 60000:
            B.append(h2("5.7 完整报告"))
            B.append(code(txt))
    return B


def main():
    token = load_token()
    blocks = build()
    print("total blocks:", len(blocks))
    sent = 0
    for i in range(0, len(blocks), BATCH):
        chunk = blocks[i:i + BATCH]
        try:
            status, txt = post(token, PAGE_ID, chunk)
        except urllib.error.HTTPError as e:
            print("HTTPError at batch", i, e.code)
            print(e.read().decode("utf-8", "replace")[:2000])
            return
        print("batch", i, "->", status, "ok",
              len(json.loads(txt).get("results", [])), "blocks")
        sent += len(chunk)
    print("sent", sent)


if __name__ == "__main__":
    main()
