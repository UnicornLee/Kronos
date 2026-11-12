import pathlib

BASE = pathlib.Path("/Volumes/ssd990pro1t/kronos/finetune/configs")
src = (BASE / "config_btc_1d_base.yaml").read_text(encoding="utf-8")

specs = [
    ("ETH_1D", "eth_1d_base", "kronos_eth_1d_finetune",
     "ETH 1D fine-tune on OKX spot history (2017-12-29 .. 2026-11-10), 3239 rows"),
    ("XAUT_1D", "xaut_1d_base", "kronos_xaut_1d_finetune",
     "XAUT 1D fine-tune on OKX spot history (2022-07-18 .. 2026-11-10), 1577 rows"),
]

for csvname, exp, name, desc in specs:
    slug = csvname[:-3].lower()          # ETH_1D -> eth_1d
    t = src
    t = t.replace("BTC_1D.csv", csvname + ".csv")
    t = t.replace('exp_name: "btc_1d_base"', f'exp_name: "{exp}"')
    t = t.replace('name: "kronos_btc_1d_finetune"', f'name: "{name}"')
    t = t.replace('description: "BTC 1D fine-tune on OKX spot history (2017-12-29 .. 2026-11-10), 3239 rows"',
                  f'description: "{desc}"')
    t = t.replace("Kronos 微调配置 —— BTC 日线 / base 模型",
                  f"Kronos 微调配置 —— {csvname[:-3]} 日线 / base 模型")
    t = t.replace("config_btc_1d_base.yaml", f"config_{slug}_1d_base.yaml")
    out = BASE / f"config_{slug}_1d_base.yaml"
    out.write_text(t, encoding="utf-8")
    print("wrote", out)

    # 打印关键字段自检
    for line in out.read_text(encoding="utf-8").splitlines():
        if any(k in line for k in ("data_path", "exp_name", "name:", "description:", "lookback_window",
                                   "predict_window", "basemodel_batch_size", "tokenizer_batch_size")):
            print("   ", line.strip())
    print()
