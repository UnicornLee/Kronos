import json, os, pandas as pd
ROOT=r"D:\DSH\kronos-forecast"
out={}
for sym in ["BTC","ETH","XAUT"]:
    e={}
    for bar in ["1D","1H"]:
        with open(os.path.join(ROOT,"out",f"forecast_{sym}_{bar}.json"),encoding="utf-8") as fh:
            fc=json.load(fh)
        h=pd.read_csv(os.path.join(ROOT,"data",f"{sym}_{bar}.csv"))
        h["timestamps"]=pd.to_datetime(h["timestamps"])
        hh=h.tail(120 if bar=="1H" else 90)
        r=lambda v: round(float(v),2)
        e[bar]={
          "hts":[t.strftime("%m-%d %H:%M") if bar=="1H" else t.strftime("%m-%d") for t in hh["timestamps"]],
          "hcl":[r(v) for v in hh["close"]],
          "fts":[x["timestamp"][:16] for x in fc["rows"]],
          "med":[r(x["close"]) for x in fc["rows"]],
          "p10":[r(x["close_p10"]) for x in fc["rows"]],
          "p90":[r(x["close_p90"]) for x in fc["rows"]],
          "last":r(fc["last_actual_close"]),
          "end":r(fc["median_end_close"]),
          "chg":fc["median_change_pct"],
          "band":fc["band_end_change_pct"],
          "pup":fc["prob_up"],
          "model":fc["model"],
          "n":fc["samples"],
          "sec":fc.get("elapsed_sec"),
        }
    out[sym]=e
with open(os.path.join(ROOT,"viz_data.json"),"w",encoding="utf-8") as fh:
    json.dump(out,fh,separators=(",",":"))
print("bytes",os.path.getsize(os.path.join(ROOT,"viz_data.json")))
