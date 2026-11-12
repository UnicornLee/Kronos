#!/usr/bin/env python3
"""Build the human-readable Kronos weekly-forecast report (Markdown) on Windows.

Reads everything under out/ (final forecasts, bias calibration, ensemble,
walk-forward scorecards, timezone experiment) and emits REPORT.md.
"""
import json
import os
import datetime as dt

ROOT = r"D:\DSH\kronos-forecast"
OUT = os.path.join(ROOT, "out")
REPORT = os.path.join(ROOT, "REPORT.md")

SYMBOLS = ["BTC", "ETH", "XAUT"]
BARS = ["1D", "1H"]
NAME = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "XAUT": "XAUT/USDT (Tether Gold)"}
HORIZON = {"1D": "日线 · 未来 7 根日 K", "1H": "小时线 · 未来 168 根小时 K"}


def read_json(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt(v, digits=2):
    if v is None:
        return "—"
    return f"{v:,.{digits}f}"


def main():
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    ens = read_json("ensemble_week.json") or {}
    lines = []
    A = lines.append

    A("# Kronos 一周走势预测报告（BTC / ETH / XAUT）")
    A("")
    A(f"生成时间：{now}（Asia/Shanghai）  ")
    A("模型：shiyu-coder/Kronos（MIT）· 日线 kronos-base / 小时线 kronos-mini  ")
    A("运行时：Mac mini M4 原生 Python + PyTorch MPS（Apple GPU 直通，非 Docker）  ")
    A("数据源：OKX 现货 K 线（BTC-USDT / ETH-USDT / XAUT-USDT），经 Clash 代理 127.0.0.1:7897 抓取  ")
    A("采样：日线 32 条独立路径、小时线 16 条独立路径（每条一次 predict 调用）")
    A("")
    A("> 本报告是对模型输出的量化复述，不是投资建议。读法：先看「回测」一节的误差尺度，再看区间宽度。")
    A("")

    # ---------- 1 summary ----------
    A("## 一、一周预测速览")
    A("")
    A("| 标的 | 最新收盘 | 日线末值(中位) | 日线涨跌 | 日线区间 | 小时线末值(中位) | 小时线涨跌 | 小时线区间 | 双周期共识 | 分歧度 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for sym in SYMBOLS:
        d = read_json(f"forecast_{sym}_1D.json")
        h = read_json(f"forecast_{sym}_1H.json")
        if not d or not h:
            continue
        e = ens.get(sym, {})
        A(f"| {NAME[sym]} | {fmt(d['last_actual_close'])} | {fmt(d['median_end_close'])} | "
          f"{d['median_change_pct']:+.2f}% | {d['band_end_change_pct'][0]:+.2f}% ~ {d['band_end_change_pct'][1]:+.2f}% | "
          f"{fmt(h['median_end_close'])} | {h['median_change_pct']:+.2f}% | "
          f"{h['band_end_change_pct'][0]:+.2f}% ~ {h['band_end_change_pct'][1]:+.2f}% | "
          f"{fmt(e.get('consensus_end'))} ({e.get('consensus_chg_pct', 0):+.2f}%) | "
          f"{e.get('spread_pct', 0):.2f}% |")
    A("")
    A("**读法**：共识末值 = 日线与小时线两套独立预测的平均；分歧度 = 两个周期预测涨跌幅之差，"
      "分歧越大说明模型对「这一周走多远」越没把握。")
    A("")

    # ---------- 2 calibrated ----------
    A("## 二、偏差校正后的 headline")
    A("")
    A("用走查回测里「模型预测涨跌 − 实际涨跌」的均值作为系统性偏差，从原始预测里减掉。"
      "样本量小（日线 12 窗、小时线 6 窗），校正只作为参考量级，不作为主结论。")
    A("")
    A("| 标的 | 周期 | 原始中位涨跌 | 模型系统偏差 | 校正后涨跌 | 校正后末值 | 方向命中(原始→校正) | 回测 MAPE | 朴素基线 MAPE |")
    A("|---|---|---|---|---|---|---|---|---|")
    for sym in SYMBOLS:
        for bar in BARS:
            c = read_json(f"forecast_calibrated_{sym}_{bar}.json")
            if not c or c.get("status") != "ok":
                continue
            A(f"| {NAME[sym]} | {HORIZON[bar]} | {c['raw_change_pct']:+.2f}% | "
              f"{c['model_bias_pct']:+.2f}% | {c['bias_corrected_change_pct']:+.2f}% | "
              f"{fmt(c['bias_corrected_end'])} | "
              f"{c['backtest_direction_hit_raw']:.0%} → {c['backtest_direction_hit_corrected']:.0%} | "
              f"{c['backtest_mean_mape_pct']}% | {c['flat_baseline_mean_mape_pct']}% |")
    A("")

    # ---------- 3 detail ----------
    A("## 三、逐标的预测路径")
    A("")
    for sym in SYMBOLS:
        for bar in BARS:
            fc = read_json(f"forecast_{sym}_{bar}.json")
            if not fc:
                continue
            A(f"### {NAME[sym]} · {HORIZON[bar]}")
            A("")
            A(f"模型 `{fc['model']}` · lookback {fc['lookback']} 根 · {fc['samples']} 条采样路径 · "
              f"预测窗口 {fc['rows'][0]['timestamp'][:16]} → {fc['rows'][-1]['timestamp'][:16]} (UTC) · "
              f"耗时 {fc.get('elapsed_sec', '—')}s")
            A("")
            A(f"![{NAME[sym]} {bar}](charts/forecast_{sym}_{bar}.png)")
            A("")
            rows = fc["rows"]
            if bar == "1H" and len(rows) > 12:
                show = rows[:6] + [None] + rows[-6:]
            else:
                show = rows
            A("| 时间(UTC) | 中位收盘 | P10 | P90 | 相对最新收盘 |")
            A("|---|---|---|---|---|")
            for r in show:
                if r is None:
                    A(f"| … 中间 {len(rows) - 12} 根省略 … | | | | |")
                    continue
                A(f"| {r['timestamp'][:16]} | {fmt(r['close'])} | {fmt(r['close_p10'])} | "
                  f"{fmt(r['close_p90'])} | {(r['close'] / fc['last_actual_close'] - 1) * 100:+.2f}% |")
            A("")

    # ---------- 4 accuracy ----------
    A("## 四、历史准确率（走查回测）")
    A("")
    A("做法：把时间往回拨 N 周，每次只用当时可见的历史数据预测下一个完整周期，再和真实收盘价对比。"
      "MAPE = 预测收盘与真实收盘的平均绝对百分比误差；方向命中 = 涨/跌方向是否猜对；"
      "朴素基线 = 「假设价格不变」的预测误差，用来回答「模型到底比瞎猜好多少」。")
    A("")
    A("| 标的 | 周期 | 窗口数 | 模型 MAPE(T=1.0) | 模型 MAPE(T=0.7) | 最差 MAPE | 方向命中率 | 朴素基线 MAPE |")
    A("|---|---|---|---|---|---|---|---|")
    for sym in SYMBOLS:
        for bar in BARS:
            bt = read_json(f"backtest2_{sym}_{bar}.json")
            if not bt:
                continue
            A(f"| {NAME[sym]} | {HORIZON[bar]} | {bt['windows']} | "
              f"{bt['T1.0']['mean_mape_pct']}% | {bt['T0.7']['mean_mape_pct']}% | "
              f"{bt['T1.0']['max_mape_pct']}% | {bt['T1.0']['direction_hit_rate']:.0%} | "
              f"{bt['flat_baseline_mean_mape_pct']}% |")
    A("")
    A("![回测示例](charts/backtest_BTC_1D.png)")
    A("")
    A("**结论（必须看）**：")
    A("")
    A("- 日线方向命中率：BTC 67%、XAUT 75%、ETH 33%；小时线 17%~50%，接近随机。")
    A("- MAPE 与「价格不变」基线基本持平甚至更差 —— Kronos 在这三只标的上并没有稳定跑赢朴素假设。")
    A("- 因此 Kronos 的可用之处是**给出一个量级和区间**（比如「BTC 这一周大致 +7% 上下、区间 0%~13%」），"
      "而不是告诉你哪天涨跌。想拿它做择时信号，样本量还不够。")
    A("")

    for sym in SYMBOLS:
        for bar in BARS:
            bt = read_json(f"backtest2_{sym}_{bar}.json")
            if not bt or not bt.get("detail"):
                continue
            A(f"<details><summary>{NAME[sym]} {bar} 回测明细（{len(bt['detail'])} 个窗口）</summary>")
            A("")
            A("| 窗口起点→结束 | 实际涨跌 | 模型 T=1.0 | 模型 T=0.7 | 模型 MAPE | 朴素基线 MAPE | 方向命中 |")
            A("|---|---|---|---|---|---|---|")
            for d in bt["detail"]:
                A(f"| {d['window_start'][:10]} → {d['window_end'][:10]} | {d['actual_chg_pct']:+.2f}% | "
                  f"{d['model_T1_chg_pct']:+.2f}% | {d['model_T07_chg_pct']:+.2f}% | "
                  f"{d['model_T1_mape_pct']}% | {d['flat_mape_pct']}% | "
                  f"{'✓' if d['model_T1_hit'] else '✗'} |")
            A("")
            A("</details>")
            A("")

    # ---------- 5 timezone ----------
    A("## 五、时间特征对齐实验（tz=0 vs tz=+8）")
    A("")
    A("Kronos 用 minute/hour/weekday/day/month 五个时间特征。OKX 的日 K 按 UTC+8 标注，"
      "抓取时统一存成 UTC，会让日线的小时特征偏 8 小时。用走查回测比较两种对齐：")
    A("")
    A("| 标的/周期 | tz=0（UTC 原样）MAPE / 命中 | tz=+8（交易所本地）MAPE / 命中 | 结论 |")
    A("|---|---|---|---|")
    tz0 = read_json("tz0_all.json") or {}
    tz8 = read_json("tz8_all.json") or {}
    for sym in SYMBOLS:
        for bar in BARS:
            a = tz0.get(f"{sym}_{bar}")
            b = tz8.get(f"{sym}_{bar}")
            if not a or not b:
                continue
            verdict = "tz=0 更优" if a["mean_mape_pct"] <= b["mean_mape_pct"] else "tz=+8 更优"
            A(f"| {NAME[sym]} {bar} | {a['mean_mape_pct']}% / {a['hit_rate']:.0%} | "
              f"{b['mean_mape_pct']}% / {b['hit_rate']:.0%} | {verdict} |")
    A("")
    A("→ 整体 **tz=0 更优**（尤其 XAUT 日线 1.512% vs 3.000%），最终预测保持 UTC 原样不做 +8h 平移。")
    A("")

    # ---------- 6 how to ----------
    A("## 六、你自己怎么跑（完整复现）")
    A("")
    A("### 1. 一键刷新（在 Mac 上）")
    A("")
    A("```bash")
    A("cd /Volumes/ssd990pro1t/kronos/forecast")
    A("")
    A("# 1) 抓最新 K 线（OKX 现货，走 Clash 代理 127.0.0.1:7897）")
    A("/usr/bin/python3 scripts/fetch_okx.py")
    A("")
    A("# 2) 一周预测（原生 MPS；KRONOS_SAMPLES_1D/1H 控制采样条数）")
    A("KRONOS_SAMPLES_1D=32 KRONOS_SAMPLES_1H=16 \\")
    A("  /Volumes/ssd990pro1t/kronos/native-venv/bin/python scripts/predict_final.py")
    A("")
    A("# 3) 走查回测（可选，验证准确率；日线 12 周 / 小时线 6 周，双温度 + 朴素基线）")
    A("/Volumes/ssd990pro1t/kronos/native-venv/bin/python scripts/backtest_v2.py")
    A("")
    A("# 4) 时间对齐实验（可选）")
    A("/Volumes/ssd990pro1t/kronos/native-venv/bin/python scripts/backtest_tz.py 0")
    A("/Volumes/ssd990pro1t/kronos/native-venv/bin/python scripts/backtest_tz.py 8")
    A("```")
    A("")
    A("一键脚本：`scripts/run_all.sh`（fetch → predict，可选 --backtest）。")
    A("")
    A("### 2. 出图与出报告（在 Windows 上）")
    A("")
    A("```powershell")
    A("$py = 'C:\\Users\\unico\\.dsh\\dsh-runtimes\\dsh-next-primary-runtime\\dependencies\\python\\python.exe'")
    A("& $py D:\\DSH\\kronos-forecast\\scripts\\make_charts.py")
    A("& $py D:\\DSH\\kronos-forecast\\scripts\\calibrate.py")
    A("& $py D:\\DSH\\kronos-forecast\\scripts\\make_report.py")
    A("```")
    A("")
    A("### 3. 目录结构")
    A("")
    A("```")
    A("D:\\DSH\\kronos-forecast\\")
    A("├─ scripts\\  fetch_okx.py / predict_final.py / predict_forecast.py / backtest_v2.py")
    A("│             backtest_tz.py / make_charts.py / calibrate.py / make_report.py / run_all.sh")
    A("├─ data\\     BTC|ETH|XAUT × 1D|1H 共 6 份 OHLCV CSV")
    A("├─ out\\      forecast_*.json|csv、forecast_calibrated_*.json、backtest2_*.json、")
    A("│            ensemble_week.json、calibration_summary.json、tz0_all.json、tz8_all.json")
    A("├─ charts\\   forecast_*.png / backtest_*.png")
    A("└─ REPORT.md 本报告")
    A("```")
    A("")
    A("Mac 侧同一套脚本在 `/Volumes/ssd990pro1t/kronos/forecast/`。")
    A("")
    A("### 4. 也可以直接用图形界面（慢，但直观）")
    A("")
    A("- 原生版（MPS，推荐）：http://unicornmac-mini:7071/")
    A("- 容器版（CPU，慢约 12 倍）：http://unicornmac-mini:7070/")
    A("- 上传刚抓到的 CSV → 选模型 → `Lookback` 500/1000 → `Pred Len` 7/168 → Predict")
    A("- 注意：WebUI 的 `/api/predict` 是「拿历史前 N 根预测后 M 根」并与已有数据对照，适合回测看效果；"
      "要产出真正的未来一周，用上面的脚本。")
    A("")

    # ---------- 7 method ----------
    A("## 七、方法与已知局限")
    A("")
    A("1. **采样与区间**：Kronos 的 `predict()` 内部会把多次采样取平均、丢掉分位数，"
      "所以 `sample_count>1` 拿不到不确定性。这里改为固定随机种子、多次独立调用 `predict(sample_count=1)`，"
      "收集多条路径后取中位数与 P10/P90 —— 区间是真实采样离散度，不是模型内置的不确定性估计。")
    A("2. **模型与上下文**：日线用 kronos-base（ctx 512，lookback 500 根 ≈ 1.4 年）；"
      "小时线用 kronos-mini（ctx 2048，lookback 1000 根 ≈ 41 天）。base 更大更准但上下文只有 512，"
      "mini 上下文长、适合日内周期。")
    A("3. **数据细节**：只取 `confirm=1` 的已收盘 K 线；1H 数据最新到 2026-11-11 07:00 UTC，"
      "1D 数据最新到 2026-11-09 16:00 UTC（OKX 日 K 以 UTC+8 标注日界）。")
    A("4. **局限**：Kronos 是 K 线形态基础模型，不吃新闻/情绪/宏观；遇放量突破或极端事件容易失真；"
      "走查样本量小（日线 12 周、小时线 6 周），方向命中率接近随机，"
      "**它的价值主要是给出量级与区间，不是精准择时**。")
    A("5. **XAUT 说明**：XAUT（Tether Gold）是实物黄金代币，价格锚定伦敦金价 + 交易溢价，"
      "波动率远小于加密货币，因此区间更窄、模型表现相对更稳。")
    A("6. **不做什么**：本流程不做仓位建议、不做止损止盈、不接交易所下单接口。")
    A("")

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
