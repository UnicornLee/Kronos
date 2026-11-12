#!/usr/bin/env python3
"""Fetch OKX spot candles (BTC/ETH/XAUT) into CSV on the external drive.

Runs with the Mac system python3.11 (native-venv has no urllib3/requests).
Network path: Clash mixed proxy 127.0.0.1:7897 (direct egress to okx.com is blocked).

Output: /Volumes/ssd990pro1t/kronos/forecast/data/<SYM>_<BAR>.csv
Columns: timestamps,open,high,low,close,volume,amount   (timestamps = UTC naive)
"""
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

PROXY = os.environ.get("KRONOS_PROXY", "http://127.0.0.1:7897")
BASE = "https://www.okx.com/api/v5/market"
OUT = "/Volumes/ssd990pro1t/kronos/forecast/data"

SYMS = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "XAUT": "XAUT-USDT"}
# bars -> how many candles we want (1H: ~70 days, 1D: ~2.4 years)
BARS = {"1H": 1700, "1D": 900}

ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}),
    urllib.request.HTTPSHandler(context=ctx),
)
opener.addheaders = [
    ("User-Agent",
     "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
    ("Accept", "application/json, text/plain, */*"),
    ("Accept-Language", "en-US,en;q=0.9"),
    ("Referer", "https://www.okx.com/"),
]


def fetch_page(inst, bar, after=None, history=False, tries=4):
    path = "history-candles" if history else "candles"
    q = {"instId": inst, "bar": bar, "limit": "300"}
    if after:
        q["after"] = str(after)
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(q)
    last_err = None
    for attempt in range(tries):
        try:
            raw = opener.open(url, timeout=30).read().decode("utf-8")
            data = json.loads(raw)
            if data.get("code") != "0":
                last_err = RuntimeError(f"OKX code={data.get('code')} msg={data.get('msg')}")
                time.sleep(1.0 + attempt)
                continue
            return data.get("data") or []
        except Exception as exc:  # noqa: BLE001 - network flake tolerance
            last_err = exc
            time.sleep(1.5 + attempt)
    raise RuntimeError(f"fetch failed {inst} {bar} after={after}: {last_err}")


def collect(inst, bar, want):
    """Paginate backwards in time until we have `want` closed candles."""
    rows = {}
    after = None
    use_history = False
    empty_streak = 0
    while len(rows) < want and empty_streak < 2:
        data = fetch_page(inst, bar, after, history=use_history)
        if not data:
            if use_history:
                empty_streak += 1
                break
            # regular candles exhausted -> switch to the history endpoint
            use_history = True
            empty_streak += 1
            time.sleep(0.4)
            continue
        empty_streak = 0
        oldest = None
        for r in data:
            ts = int(r[0])
            rows[ts] = r
            if oldest is None or ts < oldest:
                oldest = ts
        if after is not None and oldest is not None and oldest >= after:
            break  # no progress
        after = oldest
        time.sleep(0.3)
    return rows


def fmt_ts(ms):
    import datetime as dt
    return dt.datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for sym, inst in SYMS.items():
        for bar, want in BARS.items():
            rows = collect(inst, bar, want)
            closed = [rows[t] for t in sorted(rows) if len(rows[t]) > 8 and rows[t][8] == "1"]
            if not closed:
                print(f"[skip] {sym} {bar}: no closed candles")
                continue
            path = os.path.join(OUT, f"{sym}_{bar}.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("timestamps,open,high,low,close,volume,amount\n")
                for r in closed:
                    fh.write(",".join([
                        fmt_ts(int(r[0])), r[1], r[2], r[3], r[4], r[5], r[7],
                    ]) + "\n")
            summary[f"{sym}_{bar}"] = {
                "rows": len(closed),
                "from": fmt_ts(int(closed[0][0])),
                "to": fmt_ts(int(closed[-1][0])),
                "last_close": closed[-1][4],
                "path": path,
            }
            print(f"[ok] {sym:4s} {bar} rows={len(closed):5d} "
                  f"{fmt_ts(int(closed[0][0]))} -> {fmt_ts(int(closed[-1][0]))} "
                  f"last_close={closed[-1][4]}  {path}")
    with open(os.path.join(OUT, "_fetch_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nSUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())
