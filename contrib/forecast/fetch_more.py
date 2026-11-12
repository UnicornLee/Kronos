#!/usr/bin/env python3
"""Fetch longer OKX spot history (BTC/ETH/XAUT) for Kronos fine-tuning.

Same network path as forecast/scripts/fetch_okx.py (Clash proxy 127.0.0.1:7897,
system CA bundle /etc/ssl/cert.pem). Writes into forecast/data/<SYM>_<BAR>.csv
with columns timestamps,open,high,low,close,volume,amount (UTC naive).
"""
import datetime as dt
import json
import os
import ssl
import time
import urllib.parse
import urllib.request

PROXY = os.environ.get("KRONOS_PROXY", "http://127.0.0.1:7897")
BASE = "https://www.okx.com/api/v5/market"
OUT = "/Volumes/ssd990pro1t/kronos/forecast/data"

SYMS = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "XAUT": "XAUT-USDT"}
WANT = {"1D": 3000, "1H": 40000}

ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}),
    urllib.request.HTTPSHandler(context=ctx),
)
opener.addheaders = [
    ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
    ("Accept", "application/json, text/plain, */*"),
    ("Referer", "https://www.okx.com/"),
]


def fetch_page(inst, bar, after=None, history=False, tries=4):
    path = "history-candles" if history else "candles"
    q = {"instId": inst, "bar": bar, "limit": "300"}
    if after:
        q["after"] = str(after)
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(q)
    last = None
    for a in range(tries):
        try:
            raw = opener.open(url, timeout=30).read().decode("utf-8")
            d = json.loads(raw)
            if d.get("code") != "0":
                last = RuntimeError(f"code={d.get('code')} msg={d.get('msg')}")
                time.sleep(1.0 + a)
                continue
            return d.get("data") or []
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 + a)
    raise RuntimeError(f"fail {inst} {bar} after={after}: {last}")


def collect(inst, bar, want):
    rows, after, use_history, empty = {}, None, False, 0
    while len(rows) < want and empty < 3:
        data = fetch_page(inst, bar, after, use_history)
        if not data:
            if use_history:
                empty += 1
                break
            use_history = True
            empty += 1
            time.sleep(0.4)
            continue
        empty = 0
        oldest = None
        for r in data:
            ts = int(r[0])
            rows[ts] = r
            if oldest is None or ts < oldest:
                oldest = ts
        if after is not None and oldest is not None and oldest >= after:
            break
        after = oldest
        time.sleep(0.25)
    return rows


def fmt(ms):
    return dt.datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def main():
    summary = {}
    for sym, inst in SYMS.items():
        for bar, want in WANT.items():
            rows = collect(inst, bar, want)
            closed = [rows[t] for t in sorted(rows) if len(rows[t]) > 8 and rows[t][8] == "1"]
            path = os.path.join(OUT, f"{sym}_{bar}.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("timestamps,open,high,low,close,volume,amount\n")
                for r in closed:
                    fh.write(",".join([fmt(int(r[0])), r[1], r[2], r[3], r[4], r[5], r[7]]) + "\n")
            info = {"rows": len(closed), "from": fmt(int(closed[0][0])), "to": fmt(int(closed[-1][0]))}
            summary[f"{sym}_{bar}"] = info
            print(f"[ok] {sym:4s} {bar} rows={len(closed):6d} {info['from']} -> {info['to']}")
    with open(os.path.join(OUT, "_fetch_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("SUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
