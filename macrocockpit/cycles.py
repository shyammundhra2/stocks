"""
Validated cycle context for the Macro Cockpit - ONLY the cycles with real
out-of-sample power (the stock sine/wavelet cycles were retired: 45% hit vs 80%
base, no lead). Two blocks:

  recession: leading indicators that actually precede NBER recessions -
     yield curve 10y-3m (inverts ~13mo ahead, 7/7 in FRED era; P[rec<=12mo] 41%
     inverted vs 6% normal), credit spread Baa-10y, Sahm rule (confirmation).
  realestate: housing has genuine autocorrelation (+0.75) unlike stocks - so
     momentum (Case-Shiller 12m) + months'-supply (best leading, corr -0.74 to
     fwd prices) give a real near-term lean.

Descriptive context, not trade signals. FRED, disk-cached + incremental.
"""
import os
import time

import numpy as np
import pandas as pd

from macrocockpit.macro_board import ttl_cache, _STORE_DIR

CYC_FRED = ["T10Y3M", "BAA10Y", "UNRATE", "CSUSHPISA", "PERMIT", "MSACSR", "MORTGAGE30US"]
_CACHE = os.path.join(_STORE_DIR, "cycles_fred.parquet")
_TTL = 12 * 3600
# NBER recession starts (for the historical lead stats, shown as context)
LEAD_TXT = "10y-3m curve inverts ~13mo before recessions (7/7 in FRED era)"


def _load():
    try:
        if os.path.exists(_CACHE) and time.time() - os.path.getmtime(_CACHE) < _TTL:
            return pd.read_parquet(_CACHE)
    except Exception:
        pass
    import pandas_datareader.data as web
    old = None
    try:
        old = pd.read_parquet(_CACHE) if os.path.exists(_CACHE) else None
    except Exception:
        old = None
    start = (old.index[-1] - pd.Timedelta(days=75)) if old is not None and len(old) else pd.Timestamp("1970-01-01")
    try:
        new = web.DataReader(CYC_FRED, "fred", start)
    except Exception:
        return old if old is not None else pd.DataFrame()
    comb = pd.concat([old, new]) if old is not None else new
    comb = comb[~comb.index.duplicated(keep="last")].sort_index()
    try:
        comb.to_parquet(_CACHE)
    except Exception:
        pass
    return comb


def _ser(s, tail=120, freq="ME"):
    s = s.dropna().resample(freq).last().dropna().tail(tail)
    return {"x": [d.strftime("%Y-%m") for d in s.index], "y": [round(float(v), 3) for v in s.values]}


@ttl_cache(3600)
def get_cycles():
    out = {"recession": {}, "realestate": {}}
    f = _load()
    if not len(f):
        return out
    g = {c: f[c].dropna() for c in CYC_FRED if c in f.columns}

    # ---- recession block ----
    if "T10Y3M" in g:
        curve = g["T10Y3M"]; c = float(curve.iloc[-1])
        ur = g.get("UNRATE", pd.Series(dtype=float))
        sahm = (ur.rolling(3).mean() - ur.rolling(12).min())
        sh = float(sahm.iloc[-1]) if len(sahm.dropna()) else np.nan
        credit = float(g["BAA10Y"].iloc[-1]) if "BAA10Y" in g else np.nan
        inverted = c < 0
        # validated calibration (economic_cycle_honest.py): inverted 41% vs normal 6%
        risk = 41 if inverted else (14 if (sh == sh and sh > 0.3) else 6)
        warn = int(inverted) + int(credit == credit and credit > 3.0) + int(sh == sh and sh >= 0.5)
        out["recession"] = {
            "curve": round(c, 2), "curve_state": "INVERTED" if inverted else "normal",
            "sahm": round(sh, 2) if sh == sh else None, "sahm_trig": bool(sh == sh and sh >= 0.5),
            "credit": round(credit, 2) if credit == credit else None,
            "risk_pct": risk, "warnings": warn, "lead": LEAD_TXT,
            "asof": str(curve.dropna().index[-1].date()),
            "chart": _ser(curve, 180),
        }

    # ---- real-estate block ----
    if "CSUSHPISA" in g:
        lh = np.log(g["CSUSHPISA"]); mom = (lh - lh.shift(12)); fwd = (lh.shift(-12) - lh)
        supply = g.get("MSACSR", pd.Series(dtype=float))
        permit = g.get("PERMIT", pd.Series(dtype=float)); mort = g.get("MORTGAGE30US", pd.Series(dtype=float))
        # live-recomputed validation corr (honest, current). Align everything to
        # month-END - FRED's Case-Shiller is month-START indexed, supply month-END.
        d = pd.concat([mom.resample("ME").last().rename("m"),
                       fwd.resample("ME").last().rename("f"),
                       supply.resample("ME").last().rename("s")], axis=1).dropna()
        mom_corr = round(float(d["m"].corr(d["f"])), 2) if len(d) > 24 else None
        sup_corr = round(float(d["s"].corr(d["f"])), 2) if len(d) > 24 else None
        m = float(mom.dropna().iloc[-1]) * 100
        sup = float(supply.dropna().iloc[-1]) if len(supply.dropna()) else np.nan
        pyoy = float(permit.pct_change(12).dropna().iloc[-1]) * 100 if len(permit.dropna()) > 12 else np.nan
        warn = int(sup == sup and sup > 6) + int(pyoy == pyoy and pyoy < 0)
        lean = "UP" if (m > 0 and warn == 0) else ("COOLING" if m > 0 else "DOWN")
        out["realestate"] = {
            "momentum": round(m, 1), "supply": round(sup, 1) if sup == sup else None,
            "permit_yoy": round(pyoy, 0) if pyoy == pyoy else None, "lean": lean, "warnings": warn,
            "mom_corr": mom_corr, "sup_corr": sup_corr,
            "asof": str(g["CSUSHPISA"].index[-1].date()),
            "chart_cs": _ser(g["CSUSHPISA"], 180), "chart_mom": _ser(mom * 100, 180),
        }
    return out
