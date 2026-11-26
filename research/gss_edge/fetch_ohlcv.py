"""Daily split/dividend-ADJUSTED OHLCV from Yahoo query1 (O/H/L scaled by adjclose/close) for everything
get_trends needs, 2005-01 -> today. Output: ohlcv.pkl = DataFrame with (field, ticker) columns, like the live store."""
import json, sys, time, ssl, urllib.request, urllib.parse
import certifi, pandas as pd
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
CTX = ssl.create_default_context(cafile=certifi.where())
P1, P2 = 1104537600, int(time.time()) + 86400
tick = list(dict.fromkeys(list(TREND_ASSETS) + ["SPY", "IEF", "SHY", "BIL", "^VIX", "^VIX3M"]))
cols = {}
for t in tick:
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(t) +
           f"?period1={P1}&period2={P2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.load(urllib.request.urlopen(req, timeout=30, context=CTX))["chart"]["result"][0]
            ix = pd.to_datetime(j["timestamp"], unit="s").normalize(); q = j["indicators"]["quote"][0]
            adj = j["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
            d = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]}, index=ix)
            d = d[~d.index.duplicated(keep="last")]; a = pd.Series(adj, index=ix)[~pd.Index(ix).duplicated(keep="last")]
            f = (a / d["Close"]).where(d["Close"] > 0)
            d["Open"], d["High"], d["Low"] = d["Open"] * f, d["High"] * f, d["Low"] * f; d["Close"] = a
            cols[t] = d; break
        except Exception as e:
            err = e; time.sleep(1.5)
    else:
        print("FAIL", t, err)
df = pd.concat(cols, axis=1).swaplevel(axis=1).sort_index(axis=1)
df = df.sort_index().ffill(limit=5)  # bridge short null gaps (Yahoo)
df.to_pickle("ohlcv.pkl"); print(df.shape, df.index[0].date(), df.index[-1].date(), "missing:", [t for t in tick if t not in cols])
