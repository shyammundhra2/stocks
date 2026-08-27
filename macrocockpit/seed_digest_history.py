"""One-time backfill of the digest history from real FRED/market data, so the
week & month deltas populate immediately instead of accumulating over days."""
import warnings; warnings.filterwarnings("ignore")
import os, json
import pandas as pd, pandas_datareader.data as web, yfinance as yf
from macrocockpit.macro_board import _STORE_DIR

HIST = os.path.join(_STORE_DIR, "digest_history.json")
today = pd.Timestamp.today().normalize()

fred = web.DataReader(["T10Y2Y","BAMLH0A0HYM2","NFCI","MORTGAGE30US","DGS10","T5YIE",
                       "FEDFUNDS","DTWEXBGS","UNRATE","MSACSR","CSUSHPINSA"], "fred", "2023-01-01").ffill()
mk = yf.download(["^VIX","^MOVE"], start="2024-01-01", end=str((today+pd.Timedelta(days=1)).date()),
                 progress=False, auto_adjust=True)["Close"]

def asof(s, dt):
    x = s[s.index <= dt].dropna()
    return float(x.iloc[-1]) if len(x) else None

snaps = []
for db in [35, 28, 21, 14, 7]:
    dt = today - pd.Timedelta(days=db)
    m30, d10 = asof(fred["MORTGAGE30US"], dt), asof(fred["DGS10"], dt)
    cs = fred["CSUSHPINSA"]; cs_now, cs_yr = asof(cs, dt), asof(cs, dt - pd.DateOffset(months=12))
    sig = {
        "10Y-2Y Spread": asof(fred["T10Y2Y"], dt),
        "HY OAS (norm 4.5)": asof(fred["BAMLH0A0HYM2"], dt),
        "Fin Conditions (NFCI)": asof(fred["NFCI"], dt),
        "Mortgage Spread (norm 1.80)": round(m30 - d10, 2) if (m30 and d10) else None,
        "5y Inflation Expectations": asof(fred["T5YIE"], dt),
        "Fed Funds": asof(fred["FEDFUNDS"], dt),
        "USD Index (broad)": asof(fred["DTWEXBGS"], dt),
        "Unemployment Rate": asof(fred["UNRATE"], dt),
        "Months supply": asof(fred["MSACSR"], dt),
        "Case-Shiller momentum": round((cs_now / cs_yr - 1) * 100, 1) if (cs_now and cs_yr) else None,
        "VIX": asof(mk["^VIX"], dt),
        "MOVE (bond vol)": asof(mk["^MOVE"], dt),
    }
    snaps.append({"ts": dt.timestamp(), "date": str(dt.date()),
                  "signals": {k: v for k, v in sig.items() if v is not None}, "reads": {}})

with open(HIST, "w") as f:
    json.dump(snaps, f)
print(f"backfilled {len(snaps)} snapshots: {snaps[0]['date']} .. {snaps[-1]['date']}")
print("sample (7d ago):", {k: round(v, 2) for k, v in snaps[-1]["signals"].items()})
