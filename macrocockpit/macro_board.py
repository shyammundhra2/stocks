"""
Deterministic macro RoRo dashboard - the real-data version of prompts/macro_gemini.txt.
Pulls the actual numbers (FRED + yfinance) and computes RoRo deterministically
instead of asking an LLM to hallucinate 'latest 2026 values'.

Lazy + cached: only the /macro route calls this (the trading tab never does, so
its load is unaffected). FRED (slow-moving macro) is disk-cached and fetched
INCREMENTALLY - on refresh it pulls only new points and merges; market data
(VIX/MOVE/sectors) refreshes on a shorter TTL. Returns chart series for the UI.

Gaps flagged honestly: ISM PMI (INDPRO proxy - real ISM paywalled since 2016),
CFTC net-JPY (needs a COT feed - shown N/A).
"""
import os
import time
import tempfile
import functools
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Self-contained: no import from the trading engine (macro/). Own disk-cache dir
# and a tiny in-process ttl_cache so this package stands alone.
_STORE_DIR = os.environ.get("GSS_MACRO_CACHE", os.path.join(tempfile.gettempdir(), "gss_macro_cache"))
os.makedirs(_STORE_DIR, exist_ok=True)


def ttl_cache(ttl_seconds=3600):
    def deco(fn):
        store = {}
        @functools.wraps(fn)
        def wrap(*a, **k):
            key = (a, tuple(sorted(k.items())))
            now = time.time()
            if key in store and now - store[key][0] < ttl_seconds:
                return store[key][1]
            v = fn(*a, **k); store[key] = (now, v)
            return v
        return wrap
    return deco

FRED = ["GDPC1", "INDPRO", "BUSLOANS", "TTLCONS", "UNRATE", "CPIAUCSL", "FEDFUNDS",
        "T10Y2Y", "BAMLH0A0HYM2", "M2SL", "WALCL", "RRPONTSYD", "WTREGEN", "DTWEXBGS"]
SECTORS = ["XLK", "XLF", "XLI", "XLY", "XLE", "XLV", "XLP", "XLU", "XLB", "XLRE", "XLC"]
SECNAME = {"XLK": "Tech", "XLF": "Financials", "XLI": "Industrials", "XLY": "Discr",
           "XLE": "Energy", "XLV": "Health", "XLP": "Staples", "XLU": "Utilities",
           "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Comm Svc"}
_FRED_CACHE = os.path.join(_STORE_DIR, "macro_fred.parquet")
_MKT_CACHE = os.path.join(_STORE_DIR, "macro_mkt.parquet")
_FRED_TTL = 12 * 3600       # macro data updates <= daily; no intraday refetch
_MKT_TTL = 3600             # market data hourly is plenty for a macro board


def _load_fred():
    """Disk-cached, INCREMENTAL FRED pull - refetches only new points on refresh."""
    try:
        if os.path.exists(_FRED_CACHE) and time.time() - os.path.getmtime(_FRED_CACHE) < _FRED_TTL:
            return pd.read_parquet(_FRED_CACHE)
    except Exception:
        pass
    import pandas_datareader.data as web
    old = None
    try:
        old = pd.read_parquet(_FRED_CACHE) if os.path.exists(_FRED_CACHE) else None
    except Exception:
        old = None
    start = (old.index[-1] - pd.Timedelta(days=75)) if old is not None and len(old) else pd.Timestamp("2005-01-01")
    try:
        new = web.DataReader(FRED, "fred", start)
    except Exception:
        return old if old is not None else pd.DataFrame()
    comb = pd.concat([old, new]) if old is not None else new
    comb = comb[~comb.index.duplicated(keep="last")].sort_index()
    try:
        comb.to_parquet(_FRED_CACHE)
    except Exception:
        pass
    return comb


def _load_mkt():
    try:
        if os.path.exists(_MKT_CACHE) and time.time() - os.path.getmtime(_MKT_CACHE) < _MKT_TTL:
            return pd.read_parquet(_MKT_CACHE)
    except Exception:
        pass
    try:
        raw = yf.download(["^VIX", "^MOVE", "DX-Y.NYB", "JPY=X", "SPY"] + SECTORS,
                          period="3y", auto_adjust=True, progress=False)["Close"]
        raw.to_parquet(_MKT_CACHE)
        return raw
    except Exception:
        try:
            return pd.read_parquet(_MKT_CACHE)
        except Exception:
            return pd.DataFrame()


def _z(series, win=120):
    s = series.dropna()
    if len(s) < 24:
        return 0.0
    mu = s.tail(win).mean(); sd = s.tail(win).std()
    return float(np.clip((s.iloc[-1] - mu) / sd, -2.5, 2.5)) if sd > 0 else 0.0


def _yoy(series, periods=12):
    s = series.dropna()
    return float(s.iloc[-1] / s.iloc[-1 - periods] - 1) * 100 if len(s) > periods else np.nan


def _ser(series, tail=60, freq="ME"):
    """Down-sampled (date, value) lists for a Plotly line."""
    s = series.dropna().resample(freq).last().dropna().tail(tail)
    return {"x": [d.strftime("%Y-%m") for d in s.index], "y": [round(float(v), 3) for v in s.values]}


@ttl_cache(3600)
def get_macro_dashboard():
    out = {"asof": "?", "roro_score": 50, "regime": "Neutral", "categories": [],
           "sector_strength": {"strong": [], "weak": []}, "analysis": [], "breakdown": [],
           "charts": {}}
    f = _load_fred(); mk_df = _load_mkt()
    fred = {c: f[c].dropna() for c in FRED if c in f.columns} if len(f) else {}
    mk = {c: mk_df[c].dropna() for c in mk_df.columns} if len(mk_df) else {}
    if fred:
        out["asof"] = str(f.index[-1].date())
    else:
        out["analysis"].append("FRED unavailable; macro rows blank.")

    def fv(k): return fred[k].iloc[-1] if k in fred and len(fred[k]) else np.nan
    def mv(k): return mk[k].iloc[-1] if k in mk and len(mk[k]) else np.nan

    gdp_yoy = _yoy(fred.get("GDPC1", pd.Series(dtype=float)), 4)
    cpi_yoy = _yoy(fred.get("CPIAUCSL", pd.Series(dtype=float)))
    m2_yoy = _yoy(fred.get("M2SL", pd.Series(dtype=float)))
    loan_yoy = _yoy(fred.get("BUSLOANS", pd.Series(dtype=float)))
    cons_yoy = _yoy(fred.get("TTLCONS", pd.Series(dtype=float)))
    indpro_yoy = _yoy(fred.get("INDPRO", pd.Series(dtype=float)))
    netliq_ser = pd.Series(dtype=float)
    if all(k in fred for k in ("WALCL", "RRPONTSYD", "WTREGEN")):
        common = fred["WALCL"].resample("W").last()
        netliq_ser = (fred["WALCL"].resample("W").last()
                      - fred["RRPONTSYD"].resample("W").last() * 1000
                      - fred["WTREGEN"].resample("W").last()).dropna()
    netliq = netliq_ser.iloc[-1] if len(netliq_ser) else np.nan
    jpy_vol = float(mk["JPY=X"].pct_change().tail(21).std() * np.sqrt(252) * 100) if "JPY=X" in mk else np.nan

    # RoRo scoring
    bd = []
    def add(name, z): bd.append((name, round(z, 2)))
    if "^VIX" in mk:            add("VIX (low=on)", -_z(mk["^VIX"], 252))
    if "^MOVE" in mk:           add("MOVE (low=on)", -_z(mk["^MOVE"], 252))
    if "BAMLH0A0HYM2" in fred:  add("HY OAS (tight=on)", -_z(fred["BAMLH0A0HYM2"]))
    if "T10Y2Y" in fred:        add("Curve 10-2 (steep=on)", _z(fred["T10Y2Y"]))
    if "DX-Y.NYB" in mk:        add("USD (weak=on)", -_z(mk["DX-Y.NYB"], 252))
    if "M2SL" in fred:          add("M2 growth (on)", _z(fred["M2SL"].pct_change(12).dropna()))
    if "UNRATE" in fred:        add("Unemployment (falling=on)", -_z(fred["UNRATE"].diff(6).dropna()))
    if "CPIAUCSL" in fred:      add("Inflation (low=on)", -_z(fred["CPIAUCSL"].pct_change(12).dropna()))
    if "FEDFUNDS" in fred:      add("Fed funds (cutting=on)", -_z(fred["FEDFUNDS"].diff(6).dropna()))
    have_sec = [s for s in SECTORS if s in mk and len(mk[s]) > 200]
    breadth = np.nan
    if have_sec:
        breadth = float(np.mean([mk[s].iloc[-1] > mk[s].rolling(200).mean().iloc[-1] for s in have_sec])) * 100
        add("Breadth >200DMA (on)", (breadth - 50) / 25)
    out["breakdown"] = bd
    if bd:
        avg_z = float(np.mean([z for _, z in bd]))
        score = int(round(100 * norm.cdf(avg_z * 0.9)))
        out["roro_score"] = max(1, min(100, score))
        out["regime"] = "Risk-On" if score >= 61 else "Risk-Off" if score <= 40 else "Neutral"

    if have_sec:
        perf = {s: float(mk[s].iloc[-1] / mk[s].iloc[-63] - 1) * 100 for s in have_sec if len(mk[s]) > 63}
        ordered = sorted(perf.items(), key=lambda x: -x[1])
        out["sector_strength"]["strong"] = [(SECNAME.get(s, s), round(v, 1)) for s, v in ordered[:3]]
        out["sector_strength"]["weak"] = [(SECNAME.get(s, s), round(v, 1)) for s, v in ordered[-3:]]
        out["charts"]["sectors"] = {"names": [SECNAME.get(s, s) for s, _ in ordered],
                                    "vals": [round(v, 1) for _, v in ordered]}

    # chart series
    if "T10Y2Y" in fred:   out["charts"]["curve"] = _ser(fred["T10Y2Y"], 120)
    if "BAMLH0A0HYM2" in fred: out["charts"]["hyoas"] = _ser(fred["BAMLH0A0HYM2"], 120)
    if "^VIX" in mk:       out["charts"]["vix"] = _ser(mk["^VIX"], 36)
    if "^MOVE" in mk:      out["charts"]["move"] = _ser(mk["^MOVE"], 36)
    if len(netliq_ser):    out["charts"]["netliq"] = _ser(netliq_ser / 1e6, 60)  # $tn
    out["charts"]["breakdown"] = {"names": [n for n, _ in bd], "vals": [z for _, z in bd]}

    def fmt(v, suf="", d=1): return f"{v:.{d}f}{suf}" if v == v and v is not None else "N/A"
    curve = fv("T10Y2Y")
    out["categories"] = [
        ("I. Economic Growth", [
            ("Real GDP (YoY)", fmt(gdp_yoy, "%"), "Expansion" if gdp_yoy > 1 else "Slowing"),
            ("Industrial Prod YoY (ISM proxy)", fmt(indpro_yoy, "%"), "Firm" if indpro_yoy > 0 else "Contracting"),
            ("Bank Loan Growth (YoY)", fmt(loan_yoy, "%"), "Credit growing" if loan_yoy > 0 else "Credit tight"),
            ("Construction Spend (YoY)", fmt(cons_yoy, "%"), ""),
        ]),
        ("II. Labor", [("Unemployment Rate", fmt(fv("UNRATE"), "%"), "Tight" if fv("UNRATE") < 4.5 else "Loosening")]),
        ("III. Inflation", [("CPI (YoY)", fmt(cpi_yoy, "%"), "Hot" if cpi_yoy > 3 else "Cooling")]),
        ("IV. Monetary & Risk", [
            ("Fed Funds", fmt(fv("FEDFUNDS"), "%"), ""),
            ("10Y-2Y Spread", fmt(curve, "%", 2), "Inverted" if curve < 0 else "Bull steepening" if curve > 0.3 else "Flat"),
            ("VIX", fmt(mv("^VIX")), "Calm" if mv("^VIX") < 18 else "Stressed"),
            ("MOVE (bond vol)", fmt(mv("^MOVE")), "Elevated" if mv("^MOVE") > 110 else "Contained"),
            ("HY OAS", fmt(fv("BAMLH0A0HYM2"), "%", 2), "Tight" if fv("BAMLH0A0HYM2") < 3.5 else "Widening"),
            ("USD Index (broad)", fmt(fv("DTWEXBGS")), ""),
        ]),
        ("V. Liquidity", [
            ("M2 (YoY)", fmt(m2_yoy, "%"), "Expanding" if m2_yoy > 2 else "Tight"),
            ("Net Liquidity ($tn)", fmt(netliq / 1e6, "T", 2) if netliq == netliq else "N/A", "Fed BS - RRP - TGA"),
        ]),
        ("VI. Carry & FX", [
            ("USD/JPY", fmt(mv("JPY=X"), "", 1), ""),
            ("USD/JPY Realized Vol", fmt(jpy_vol, "%"), "Crowded-short risk" if jpy_vol == jpy_vol and jpy_vol > 12 else ""),
        ]),
        ("VII. Breadth", [("% Sectors > 200DMA", fmt(breadth, "%", 0), "Broad" if breadth == breadth and breadth > 60 else "Narrow")]),
    ]
    if mv("^MOVE") == mv("^MOVE") and mv("^MOVE") > 110:
        out["analysis"].append(f"Bond volatility elevated (MOVE {mv('^MOVE'):.0f}) - rate-driven risk; watch duration.")
    if jpy_vol == jpy_vol and jpy_vol > 12:
        out["analysis"].append(f"USD/JPY realized vol {jpy_vol:.0f}% - carry-unwind risk if it spikes (2024-08 style).")
    if curve == curve and curve < 0:
        out["analysis"].append("Curve inverted - historical recession lead ~13mo (see cycles dashboard).")
    if not out["analysis"]:
        out["analysis"].append("No acute bond-vol or carry stress flagged.")
    return out
