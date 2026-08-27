"""
Deterministic macro RoRo cockpit - the real-data version of prompts/macro_gemini.txt.
Pulls actual numbers (FRED + yfinance) and computes RoRo deterministically instead
of letting an LLM hallucinate 'latest values'. Lazy (only /macro calls it) + disk-
cached + incremental FRED, so the trading tab is never delayed.

Arranged for a QUICK read: a HEADLINE strip (RoRo / financial conditions / mfg
pulse), a LEADING-indicator row (forward-looking, color-coded), then the detail
table. Manufacturing 'PMI' = composite of the FREE regional Fed diffusion surveys
(Empire/Philly/Dallas) - real ISM is paywalled since 2016; this tracks it far
better than Industrial Production.
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

_STORE_DIR = os.environ.get("GSS_MACRO_CACHE", os.path.join(tempfile.gettempdir(), "gss_macro_cache"))
os.makedirs(_STORE_DIR, exist_ok=True)


def ttl_cache(ttl_seconds=3600):
    def deco(fn):
        store = {}
        @functools.wraps(fn)
        def wrap(*a, **k):
            key = (a, tuple(sorted(k.items()))); now = time.time()
            if key in store and now - store[key][0] < ttl_seconds:
                return store[key][1]
            v = fn(*a, **k); store[key] = (now, v)
            return v
        return wrap
    return deco


FRED = ["GDPC1", "BUSLOANS", "TTLCONS", "UNRATE", "ICSA", "JTSJOL", "RSAFS", "UMCSENT",
        "CPIAUCSL", "PCEPILFE", "T5YIE", "FEDFUNDS", "T10Y2Y", "BAMLH0A0HYM2", "NFCI",
        "M2SL", "WALCL", "RRPONTSYD", "WTREGEN", "DTWEXBGS", "USINFO",
        "DGS10", "MORTGAGE30US",
        "GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI", "BACTSAMFRBDAL"]   # regional Fed mfg
REG_FED = ["GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI", "BACTSAMFRBDAL"]
SECTORS = ["XLK", "XLF", "XLI", "XLY", "XLE", "XLV", "XLP", "XLU", "XLB", "XLRE", "XLC"]
SECNAME = {"XLK": "Tech", "XLF": "Financials", "XLI": "Industrials", "XLY": "Discr",
           "XLE": "Energy", "XLV": "Health", "XLP": "Staples", "XLU": "Utilities",
           "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Comm Svc"}
_FRED_CACHE = os.path.join(_STORE_DIR, "macro_fred.parquet")
_MKT_CACHE = os.path.join(_STORE_DIR, "macro_mkt.parquet")
_FRED_TTL, _MKT_TTL = 12 * 3600, 3600


def _load_fred():
    try:
        if os.path.exists(_FRED_CACHE) and time.time() - os.path.getmtime(_FRED_CACHE) < _FRED_TTL:
            cached = pd.read_parquet(_FRED_CACHE)
            if all(c in cached.columns for c in FRED):   # schema still matches
                return cached
    except Exception:
        pass
    import pandas_datareader.data as web
    try:
        old = pd.read_parquet(_FRED_CACHE) if os.path.exists(_FRED_CACHE) else None
    except Exception:
        old = None
    # schema changed (new series added) -> full refetch, not incremental
    if old is not None and not all(c in old.columns for c in FRED):
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
            cached = pd.read_parquet(_MKT_CACHE)
            if "HG=F" in cached.columns and "CL=F" in cached.columns:   # schema matches
                return cached
    except Exception:
        pass
    try:
        raw = yf.download(["^VIX", "^MOVE", "DX-Y.NYB", "JPY=X", "SPY", "HG=F", "CL=F"] + SECTORS,
                          period="3y", auto_adjust=True, progress=False)["Close"]
        raw.to_parquet(_MKT_CACHE)
        return raw
    except Exception:
        try:
            return pd.read_parquet(_MKT_CACHE)
        except Exception:
            return pd.DataFrame()


def _z(s, win=120):
    s = s.dropna()
    if len(s) < 24:
        return 0.0
    mu, sd = s.tail(win).mean(), s.tail(win).std()
    return float(np.clip((s.iloc[-1] - mu) / sd, -2.5, 2.5)) if sd > 0 else 0.0


def _yoy(s, p=12):
    s = s.dropna()
    return float(s.iloc[-1] / s.iloc[-1 - p] - 1) * 100 if len(s) > p else np.nan


def _ser(s, tail=120, freq="ME"):
    s = s.dropna().resample(freq).last().dropna().tail(tail)
    return {"x": [d.strftime("%Y-%m") for d in s.index], "y": [round(float(v), 3) for v in s.values]}


def _col(risk_on):   # risk_on z -> color
    return "green" if risk_on > 0.4 else "red" if risk_on < -0.4 else "yellow"


@ttl_cache(3600)
def get_macro_dashboard():
    out = {"asof": "?", "roro_score": 50, "regime": "Neutral", "headline": [], "leading": [],
           "categories": [], "sector_strength": {"strong": [], "weak": []}, "analysis": [],
           "breakdown": [], "charts": {}}
    f = _load_fred(); mkdf = _load_mkt()
    fr = {c: f[c].dropna() for c in FRED if c in f.columns} if len(f) else {}
    mk = {c: mkdf[c].dropna() for c in mkdf.columns} if len(mkdf) else {}
    if fr:
        out["asof"] = str(f.index[-1].date())
    else:
        out["analysis"].append("FRED unavailable; rows blank.")

    def fv(k): return fr[k].iloc[-1] if k in fr and len(fr[k]) else np.nan
    def mv(k): return mk[k].iloc[-1] if k in mk and len(mk[k]) else np.nan
    def fmt(v, s="", d=1): return f"{v:.{d}f}{s}" if v == v and v is not None else "N/A"

    gdp_yoy = _yoy(fr.get("GDPC1", pd.Series(dtype=float)), 4)
    cpi_yoy = _yoy(fr.get("CPIAUCSL", pd.Series(dtype=float)))
    pce_yoy = _yoy(fr.get("PCEPILFE", pd.Series(dtype=float)))
    m2_yoy = _yoy(fr.get("M2SL", pd.Series(dtype=float)))
    loan_yoy = _yoy(fr.get("BUSLOANS", pd.Series(dtype=float)))
    cons_yoy = _yoy(fr.get("TTLCONS", pd.Series(dtype=float)))
    retail_yoy = _yoy(fr.get("RSAFS", pd.Series(dtype=float)))
    claims4 = fr["ICSA"].rolling(4).mean() if "ICSA" in fr else pd.Series(dtype=float)
    claims = float(claims4.dropna().iloc[-1]) / 1000 if len(claims4.dropna()) else np.nan  # thousands
    jolts = fv("JTSJOL") / 1000 if "JTSJOL" in fr else np.nan   # millions
    # Information-sector payrolls YoY = honest tech-labor proxy (no forecast).
    info_yoy = fr["USINFO"].pct_change(12).dropna().iloc[-1] * 100 if "USINFO" in fr else np.nan
    conf = fv("UMCSENT"); nfci = fv("NFCI"); be5 = fv("T5YIE")
    # regional-Fed manufacturing 'PMI' composite (diffusion, centered ~0)
    pmi_ser = pd.concat([fr[c].resample("ME").last() for c in REG_FED if c in fr], axis=1).mean(axis=1).dropna() if any(c in fr for c in REG_FED) else pd.Series(dtype=float)
    pmi = float(pmi_ser.iloc[-1]) if len(pmi_ser) else np.nan
    netliq_ser = pd.Series(dtype=float)
    if all(k in fr for k in ("WALCL", "RRPONTSYD", "WTREGEN")):
        netliq_ser = (fr["WALCL"].resample("W").last() - fr["RRPONTSYD"].resample("W").last() * 1000
                      - fr["WTREGEN"].resample("W").last()).dropna()
    netliq = netliq_ser.iloc[-1] if len(netliq_ser) else np.nan
    jpy_vol = float(mk["JPY=X"].pct_change().tail(21).std() * np.sqrt(252) * 100) if "JPY=X" in mk else np.nan
    cu_3m = float(mk["HG=F"].iloc[-1] / mk["HG=F"].iloc[-63] - 1) * 100 if "HG=F" in mk and len(mk["HG=F"]) > 63 else np.nan
    oil_3m = float(mk["CL=F"].iloc[-1] / mk["CL=F"].iloc[-63] - 1) * 100 if "CL=F" in mk and len(mk["CL=F"]) > 63 else np.nan

    # ---- RoRo (add the leading ones: claims, breakevens, NFCI, PMI) ----
    bd = []
    def add(name, z): bd.append((name, round(z, 2)))
    if "^VIX" in mk:  add("VIX (low=on)", -_z(mk["^VIX"], 252))
    if "^MOVE" in mk: add("MOVE (low=on)", -_z(mk["^MOVE"], 252))
    if "BAMLH0A0HYM2" in fr: add("HY OAS (tight=on)", -_z(fr["BAMLH0A0HYM2"]))
    if "T10Y2Y" in fr:       add("Curve (steep=on)", _z(fr["T10Y2Y"]))
    if "DX-Y.NYB" in mk:     add("USD (weak=on)", -_z(mk["DX-Y.NYB"], 252))
    if "NFCI" in fr:         add("Fin conditions (loose=on)", -float(np.clip(nfci, -2.5, 2.5)) if nfci == nfci else 0.0)
    if len(pmi_ser) > 24:    add("Mfg PMI (expand=on)", _z(pmi_ser))
    if "ICSA" in fr:         add("Jobless claims (low=on)", -_z(claims4))
    if "T5YIE" in fr:        add("Inflation exp (low=on)", -_z(fr["T5YIE"]))
    if "M2SL" in fr:         add("M2 growth (on)", _z(fr["M2SL"].pct_change(12).dropna()))
    if "UNRATE" in fr:       add("Unemployment (falling=on)", -_z(fr["UNRATE"].diff(6).dropna()))
    have_sec = [s for s in SECTORS if s in mk and len(mk[s]) > 200]
    breadth = np.nan
    if have_sec:
        breadth = float(np.mean([mk[s].iloc[-1] > mk[s].rolling(200).mean().iloc[-1] for s in have_sec])) * 100
        add("Breadth >200DMA (on)", (breadth - 50) / 25)
    out["breakdown"] = bd
    if bd:
        avg = float(np.mean([z for _, z in bd])); sc = int(round(100 * norm.cdf(avg * 0.9)))
        out["roro_score"] = max(1, min(100, sc))
        out["regime"] = "Risk-On" if sc >= 61 else "Risk-Off" if sc <= 40 else "Neutral"

    # ---- HEADLINE strip ----
    nfci_z = -float(np.clip(nfci, -2.5, 2.5)) if nfci == nfci else 0.0
    pmi_z = _z(pmi_ser) if len(pmi_ser) > 24 else 0.0
    out["headline"] = [
        {"label": "Financial Conditions", "value": fmt(nfci, "", 2),
         "state": "Loose" if nfci < 0 else "Tight", "color": _col(nfci_z)},
        {"label": "Mfg Pulse (PMI proxy)", "value": fmt(pmi, "", 1),
         "state": "Expanding" if pmi > 0 else "Contracting", "color": _col(pmi_z)},
        {"label": "Consumer Confidence", "value": fmt(conf, "", 1),
         "state": "Weak" if conf == conf and conf < 70 else "Firm", "color": "red" if conf == conf and conf < 70 else "green"},
    ]

    # ---- LEADING row (forward-looking, color-coded) ----
    def lead(name, val, z): return {"name": name, "value": val, "color": _col(z)}
    out["leading"] = [
        lead("Yield curve 10-2", fmt(fv("T10Y2Y"), "", 2), _z(fr["T10Y2Y"]) if "T10Y2Y" in fr else 0),
        lead("Jobless claims (4wk)", fmt(claims, "k", 0), -_z(claims4)),
        lead("5y inflation exp", fmt(be5, "%", 2), -_z(fr["T5YIE"]) if "T5YIE" in fr else 0),
        lead("Copper 3mo", fmt(cu_3m, "%"), np.clip(cu_3m / 10, -2, 2) if cu_3m == cu_3m else 0),
        lead("Fin conditions (NFCI)", fmt(nfci, "", 2), nfci_z),
        lead("Job openings (JOLTS)", fmt(jolts, "M", 1), _z(fr["JTSJOL"]) if "JTSJOL" in fr else 0),
    ]

    # ---- mortgage spread: the one honestly-forecastable rate signal ----
    # 30yr mortgage = 10yr Treasury + a mean-reverting spread. The spread reverts
    # (level->fwd-change rank IC -0.54); it averages ~1.80%, blew out to 3.27% in
    # 2022-23. So the forecastable part of mortgages is spread NORMALIZATION, not
    # the (near-random-walk) 10yr. Relevant to origination/RKT.
    mort30 = fv("MORTGAGE30US"); dgs10 = fv("DGS10")
    mspread = (mort30 - dgs10) if (mort30 == mort30 and dgs10 == dgs10) else float("nan")
    MORT_NORM = 1.80
    if   mspread != mspread: _mtag = ""
    elif mspread > 2.30:     _mtag = "Elevated - room to compress (easing tailwind)"
    elif mspread > 1.95:     _mtag = "Mildly elevated - normalizing"
    elif mspread < 1.60:     _mtag = "Tight (rich)"
    else:                    _mtag = "Near normal"

    # ---- credit spread (HY OAS) forward read ----
    # HY OAS mean-reverts strongly around a ~4.5 long-run median, and the LEVEL is
    # a contrarian equity signal: tight = complacent / risk building (spreads have
    # room to widen); wide = stress already priced, equity historically cheap
    # ahead. (Live FRED series row-caps to 3yr, so 4.5 is a fixed reference.)
    HYOAS_NORM = 4.5
    _oas = fv("BAMLH0A0HYM2")
    if   _oas != _oas: _oastag = ""
    elif _oas < 3.2:   _oastag = "Tight vs 4.5 norm - complacent, room to widen (risk building)"
    elif _oas < 5.5:   _oastag = "Near 4.5 norm"
    elif _oas < 8.0:   _oastag = "Wide - stress priced; equity historically cheap ahead"
    else:              _oastag = "Blown out - crisis (contrarian equity buy)"

    # ---- detail table ----
    curve = fv("T10Y2Y")
    out["categories"] = [
        ("I. Growth", [
            ("Real GDP (YoY)", fmt(gdp_yoy, "%"), "Expansion" if gdp_yoy > 1 else "Slowing"),
            ("Mfg survey (PMI proxy)", fmt(pmi, "", 1), "Expanding" if pmi > 0 else "Contracting"),
            ("Bank Loan Growth (YoY)", fmt(loan_yoy, "%"), "Credit growing" if loan_yoy > 0 else "Tight"),
            ("Construction (YoY)", fmt(cons_yoy, "%"), ""),
            ("Copper (Dr. Copper) 3mo", fmt(cu_3m, "%"), "Demand firm" if cu_3m > 0 else "Soft"),
            ("Oil (WTI) 3mo", fmt(oil_3m, "%"), ""),
        ]),
        ("II. Labor", [
            ("Unemployment Rate", fmt(fv("UNRATE"), "%"), "Tight" if fv("UNRATE") < 4.5 else "Loosening"),
            ("Initial Claims (4wk avg)", fmt(claims, "k", 0), "Low" if claims == claims and claims < 240 else "Rising"),
            ("Job Openings (JOLTS)", fmt(jolts, "M", 1), ""),
            ("Tech jobs (Info-sector YoY)", fmt(info_yoy, "%"), "Contracting" if info_yoy == info_yoy and info_yoy < 0 else "Growing"),
        ]),
        ("III. Consumer", [
            ("Retail Sales (YoY)", fmt(retail_yoy, "%"), "Solid" if retail_yoy > 2 else "Soft"),
            ("Consumer Confidence", fmt(conf, "", 1), "Weak" if conf == conf and conf < 70 else "Firm"),
        ]),
        ("IV. Inflation", [
            ("CPI (YoY)", fmt(cpi_yoy, "%"), "Hot" if cpi_yoy > 3 else "Cooling"),
            ("Core PCE (YoY, Fed target)", fmt(pce_yoy, "%"), "Above 2%" if pce_yoy > 2 else "At target"),
            ("5y Inflation Expectations", fmt(be5, "%", 2), "Anchored" if be5 == be5 and be5 < 2.6 else "Rising"),
        ]),
        ("V. Monetary & Risk", [
            ("Fed Funds", fmt(fv("FEDFUNDS"), "%"), ""),
            ("10Y-2Y Spread", fmt(curve, "%", 2), "Inverted" if curve < 0 else "Bull steepening" if curve > 0.3 else "Flat"),
            ("Fin Conditions (NFCI)", fmt(nfci, "", 2), "Loose" if nfci < 0 else "Tight"),
            ("VIX", fmt(mv("^VIX")), "Calm" if mv("^VIX") < 18 else "Stressed"),
            ("MOVE (bond vol)", fmt(mv("^MOVE")), "Elevated" if mv("^MOVE") > 110 else "Contained"),
            ("HY OAS (norm 4.5)", fmt(_oas, "%", 2), _oastag),
            ("USD Index (broad)", fmt(fv("DTWEXBGS")), ""),
        ]),
        ("VI. Liquidity", [
            ("M2 (YoY)", fmt(m2_yoy, "%"), "Expanding" if m2_yoy > 2 else "Tight"),
            ("Net Liquidity ($tn)", fmt(netliq / 1e6, "T", 2) if netliq == netliq else "N/A", "Fed BS - RRP - TGA"),
        ]),
        ("VII. Carry & FX", [
            ("USD/JPY", fmt(mv("JPY=X"), "", 1), ""),
            ("USD/JPY Realized Vol", fmt(jpy_vol, "%"), "Crowded-short risk" if jpy_vol == jpy_vol and jpy_vol > 12 else ""),
        ]),
        ("VIII. Breadth", [("% Sectors > 200DMA", fmt(breadth, "%", 0), "Broad" if breadth == breadth and breadth > 60 else "Narrow")]),
        ("IX. Mortgage & Housing Rates", [
            ("30yr Mortgage", fmt(mort30, "%", 2), ""),
            ("10yr Treasury", fmt(dgs10, "%", 2), ""),
            (f"Mortgage Spread (norm {MORT_NORM:.2f}%)", fmt(mspread, "%", 2), _mtag),
        ]),
    ]

    if have_sec:
        perf = {s: float(mk[s].iloc[-1] / mk[s].iloc[-63] - 1) * 100 for s in have_sec if len(mk[s]) > 63}
        od = sorted(perf.items(), key=lambda x: -x[1])
        out["sector_strength"]["strong"] = [(SECNAME.get(s, s), round(v, 1)) for s, v in od[:3]]
        out["sector_strength"]["weak"] = [(SECNAME.get(s, s), round(v, 1)) for s, v in od[-3:]]
        out["charts"]["sectors"] = {"names": [SECNAME.get(s, s) for s, _ in od], "vals": [round(v, 1) for _, v in od]}

    out["charts"]["breakdown"] = {"names": [n for n, _ in bd], "vals": [z for _, z in bd]}
    if "T10Y2Y" in fr:       out["charts"]["curve"] = _ser(fr["T10Y2Y"], 120)
    if "NFCI" in fr:         out["charts"]["nfci"] = _ser(fr["NFCI"], 120, "W")
    if "ICSA" in fr:         out["charts"]["claims"] = _ser(claims4 / 1000, 104, "W")
    if "BAMLH0A0HYM2" in fr: out["charts"]["hyoas"] = _ser(fr["BAMLH0A0HYM2"], 120)
    if len(pmi_ser):         out["charts"]["pmi"] = _ser(pmi_ser, 120)
    if len(netliq_ser):      out["charts"]["netliq"] = _ser(netliq_ser / 1e6, 60)
    if "^VIX" in mk:         out["charts"]["vix"] = _ser(mk["^VIX"], 36)
    if "^MOVE" in mk:        out["charts"]["move"] = _ser(mk["^MOVE"], 36)
    if "MORTGAGE30US" in fr and "DGS10" in fr:
        _msp = (fr["MORTGAGE30US"].resample("W").last() - fr["DGS10"].resample("W").last()).dropna()
        out["charts"]["mortspread"] = _ser(_msp, 156, "W")

    # analysis
    if mv("^MOVE") == mv("^MOVE") and mv("^MOVE") > 110:
        out["analysis"].append(f"Bond volatility elevated (MOVE {mv('^MOVE'):.0f}) - rate-driven risk.")
    if jpy_vol == jpy_vol and jpy_vol > 12:
        out["analysis"].append(f"USD/JPY realized vol {jpy_vol:.0f}% - carry-unwind risk (2024-08 style).")
    if mspread == mspread and mspread > 2.20:
        out["analysis"].append(f"Mortgage spread {mspread:.2f}% vs ~{MORT_NORM:.2f}% norm - elevated; room to compress = easing tailwind for mortgage rates independent of the 10yr (origination-relevant).")
    if _oas == _oas and _oas < 3.2:
        out["analysis"].append(f"Credit spread (HY OAS {_oas:.2f}%) is TIGHT vs its ~{HYOAS_NORM:.1f}% norm - credit priced for perfection. Mean-reverts wider = risk building; the cheap time to buy tail protection.")
    elif _oas == _oas and _oas > 6.0:
        out["analysis"].append(f"Credit spread (HY OAS {_oas:.2f}%) is WIDE vs ~{HYOAS_NORM:.1f}% norm - stress priced in; historically equity is cheap 12m forward from here.")
    if conf == conf and conf < 70:
        out["analysis"].append(f"Consumer confidence weak ({conf:.0f}) - demand headwind.")
    if pmi == pmi and pmi < 0:
        out["analysis"].append(f"Regional Fed mfg surveys contracting (PMI proxy {pmi:.0f}).")
    if not out["analysis"]:
        out["analysis"].append("No acute stress flagged.")
    return out
