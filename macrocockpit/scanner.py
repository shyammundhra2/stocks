"""
ETF trend/momentum scanner.

Ranks a broad, curated, NON-leveraged liquid-ETF universe by the honest momentum
recipe: 12-1 momentum (skip the last month - it reverses) + trend QUALITY (R2 of
the log-price fit + 52-week-high proximity), gated to names currently above their
200DMA and liquid. Flags which names you already hold (TREND_ASSETS) so the output
is an IDEA / diversification watchlist, not a re-list of the book.

Idea generation only: the scan tells you WHERE momentum is; your engine's gate
tells you whether to actually hold it. Stale-while-revalidate cached (12h) so it
can back a dashboard tab without blocking.
"""
import os
import json
import time
import threading

import numpy as np
import pandas as pd
import yfinance as yf

from macrocockpit.macro_board import _STORE_DIR
from macro.constants import TREND_ASSETS

_CACHE = os.path.join(_STORE_DIR, "scanner.json")
_TTL = 12 * 3600
_lock = threading.Lock()

# Curated liquid, non-leveraged ETFs across categories (no 2x/3x, no inverse).
UNIVERSE = {
    # broad
    "SPY": "Broad", "QQQ": "Broad", "IWM": "Broad", "RSP": "Broad", "MDY": "Broad", "VTI": "Broad",
    # sectors
    "XLK": "Sector", "XLF": "Sector", "XLE": "Sector", "XLI": "Sector", "XLB": "Sector",
    "XLV": "Sector", "XLY": "Sector", "XLP": "Sector", "XLU": "Sector", "XLRE": "Sector", "XLC": "Sector",
    # tech / semis / themes
    "SMH": "Semis", "SOXX": "Semis", "IGV": "Software", "CIBR": "Cyber", "FDN": "Internet",
    "SKYY": "Cloud", "XBI": "Biotech", "IBB": "Biotech", "ITA": "Defense", "PPA": "Defense",
    "XAR": "Aerospace", "PAVE": "Infra", "GRID": "Grid", "TAN": "Solar", "ICLN": "CleanEnergy",
    "JETS": "Airlines", "IYT": "Transport", "KIE": "Insurance", "FINX": "Fintech",
    # financials / housing
    "KRE": "Banks", "KBE": "Banks", "XHB": "Homebuild", "ITB": "Homebuild", "XRT": "Retail",
    # commodities / miners / energy
    "GLD": "Gold", "SLV": "Silver", "GDX": "GoldMiners", "GDXJ": "GoldMiners", "SIL": "SilverMiners",
    "COPX": "Copper", "CPER": "Copper", "URA": "Uranium", "URNM": "Uranium", "LIT": "Lithium",
    "REMX": "RareEarth", "XME": "Metals", "PICK": "Miners", "DBC": "Commodities", "PDBC": "Commodities",
    "DBA": "Agri", "MOO": "Agri", "WOOD": "Timber", "GNR": "NatRes",
    "XOP": "Energy", "OIH": "OilSvcs", "AMLP": "Midstream", "MLPX": "Midstream", "FCG": "NatGas", "CRAK": "Refiners",
    # countries / regions
    "EEM": "EM", "EFA": "Intl", "FXI": "China", "MCHI": "China", "INDA": "India", "EWY": "Korea",
    "EWJ": "Japan", "EWT": "Taiwan", "EWZ": "Brazil", "EWW": "Mexico", "ARGT": "Argentina",
    "GREK": "Greece", "EWU": "UK", "EWG": "Germany", "EPOL": "Poland", "VNM": "Vietnam", "ILF": "LatAm",
    # factors
    "MTUM": "Factor", "QUAL": "Factor", "VLUE": "Factor", "USMV": "Factor", "SPLV": "Factor",
    # bonds / rates
    "TLT": "Bonds", "IEF": "Bonds", "LQD": "Credit", "HYG": "HYCredit", "TIP": "TIPS", "EMB": "EMDebt",
    # alt / real assets
    "IBIT": "Crypto", "DBMF": "MgdFutures", "KMLM": "MgdFutures", "VNQ": "REIT", "IGF": "Infra",
}


def _trend_r2(logp):
    n = len(logp)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, logp, 1)
    fit = slope * x + intercept
    sst = np.sum((logp - logp.mean()) ** 2)
    r2 = 1 - np.sum((logp - fit) ** 2) / sst if sst > 0 else 0.0
    return float(slope * 252), float(max(0.0, r2))   # annualized slope, R2


def _compute():
    tickers = list(UNIVERSE)
    raw = yf.download(tickers, start=(pd.Timestamp.today() - pd.DateOffset(days=700)),
                      end=pd.Timestamp.today() + pd.Timedelta(days=1),
                      progress=False, auto_adjust=True, group_by="column")
    close = raw["Close"].ffill()
    vol = raw["Volume"] if "Volume" in raw else None
    held = set(TREND_ASSETS)
    rows = []
    for t in tickers:
        if t not in close.columns:
            continue
        c = close[t].dropna()
        if len(c) < 260:
            continue
        px = float(c.iloc[-1])
        mom = float(c.iloc[-21] / c.iloc[-252] - 1) * 100          # 12-1 momentum %
        ma200 = float(c.rolling(200).mean().iloc[-1])
        above = px > ma200
        _ma = c.rolling(200).mean()
        _v = _ma.notna()                      # only days where the 200DMA exists
        pct_above = float((c[_v] > _ma[_v]).tail(252).mean()) * 100 if _v.any() else 0.0
        slope, r2 = _trend_r2(np.log(c.tail(126).values))
        hi52 = float(px / c.tail(252).max())                       # 52w-high proximity
        vol63 = float(c.pct_change().tail(63).std() * np.sqrt(252)) * 100
        dvol = float((c * vol[t]).tail(60).mean()) / 1e6 if vol is not None and t in vol else np.nan
        rows.append({"etf": t, "cat": UNIVERSE[t], "px": round(px, 2),
                     "mom": round(mom, 1), "r2": round(r2, 2), "hi52": round(hi52, 2),
                     "pct200": round(pct_above, 0), "above": above, "vol": round(vol63, 0),
                     "dvol": round(dvol, 0) if dvol == dvol else None, "held": t in held})
    df = pd.DataFrame(rows)
    # eligible = liquid + currently trending up
    df["liquid"] = df["dvol"].fillna(0) >= 5
    elig = df[df["above"] & df["liquid"]].copy()
    # composite trend score: momentum + cleanliness (R2) + 52w-high proximity
    for col, w in [("mom", 0.5), ("r2", 0.3), ("hi52", 0.2)]:
        elig[col + "_r"] = elig[col].rank(pct=True)
    elig["score"] = (elig["mom_r"] * 0.5 + elig["r2_r"] * 0.3 + elig["hi52_r"] * 0.2)
    elig = elig.sort_values("score", ascending=False)
    ranking = [{k: r[k] for k in ["etf", "cat", "mom", "r2", "hi52", "pct200", "vol", "held"]}
               for _, r in elig.iterrows()]
    return {"asof": str(close.index[-1].date()), "n_scanned": len(df),
            "n_trending": int(df["above"].sum()), "ranking": ranking}


def _refresh():
    if not _lock.acquire(blocking=False):
        return
    def run():
        try:
            d = _compute()
            with open(_CACHE + ".tmp", "w") as f:
                json.dump({"ts": time.time(), **d}, f)
            os.replace(_CACHE + ".tmp", _CACHE)
        except Exception as e:
            print(f"scanner refresh error: {e}")
        finally:
            _lock.release()
    threading.Thread(target=run, daemon=True).start()


def get_scan():
    out = {"ranking": [], "asof": "?", "refreshing": _lock.locked(), "stale": True}
    try:
        with open(_CACHE) as f:
            c = json.load(f)
        out.update({k: c.get(k) for k in ["ranking", "asof", "n_scanned", "n_trending"]})
        out["stale"] = (time.time() - c.get("ts", 0)) > _TTL
    except Exception:
        out["stale"] = True
    if out["stale"]:
        _refresh()
        out["refreshing"] = True
    return out


if __name__ == "__main__":
    d = _compute()
    print(f"scanned {d['n_scanned']} ETFs, {d['n_trending']} above 200DMA, as of {d['asof']}\n")
    print(f"{'#':>2} {'etf':5s} {'category':12s} {'12-1%':>7s} {'R2':>5s} {'52wH':>5s} {'%>200':>6s} {'vol%':>5s}  held")
    for i, r in enumerate(d["ranking"][:30], 1):
        print(f"{i:>2} {r['etf']:5s} {r['cat']:12s} {r['mom']:>+6.1f} {r['r2']:>5.2f} "
              f"{r['hi52']:>5.2f} {r['pct200']:>5.0f}% {r['vol']:>4.0f}%  {'HELD' if r['held'] else ''}")
