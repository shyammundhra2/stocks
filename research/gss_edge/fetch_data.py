"""Pull daily ADJUSTED closes from Yahoo query1 for the GSS ETF universe + IEF/SHY/BIL (+^VIX
for the capitulation trigger), 2006-01-01 -> today. Writes prices.csv. Read-only; touches nothing live."""
import json, sys, time, ssl, urllib.request, urllib.parse
import certifi
CTX = ssl.create_default_context(cafile=certifi.where())
import pandas as pd
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

P1, P2 = 1136073600, int(time.time()) + 86400  # 2006-01-01 ..
tick = list(TREND_ASSETS) + ["IEF", "SHY", "BIL", "^VIX"]
out = {}
for t in tick:
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(t) +
           f"?period1={P1}&period2={P2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.load(urllib.request.urlopen(req, timeout=30, context=CTX))["chart"]["result"][0]
            ts = pd.to_datetime(j["timestamp"], unit="s").normalize()
            adj = j["indicators"].get("adjclose", [{}])[0].get("adjclose") or j["indicators"]["quote"][0]["close"]
            out[t] = pd.Series(adj, index=ts).groupby(level=0).last()
            break
        except Exception as e:
            err = e; time.sleep(1.5)
    else:
        print("FAIL", t, err)
df = pd.DataFrame(out).sort_index()
df.to_csv("prices.csv")
print(df.shape, df.index[0].date(), df.index[-1].date())
print("first valid:", {c: str(df[c].first_valid_index().date()) for c in ["BIL", "IEF", "SHY", "^VIX"]})
print("missing:", [t for t in tick if t not in df.columns])
