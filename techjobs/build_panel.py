"""
Tech labour market: pooled monthly panel of tech sub-industries + macro leading indicators.

WHY A PANEL. Forecasting one series ("tech jobs") from its own history means fitting a
handful of cycles - four since 1990, and the model just memorises them. Pooling six
distinct tech sub-industries into one cross-sectional panel gives ~2,500 rows instead
of ~430, which is the same fix housing/predict_housing.py applied to Case-Shiller metros
and the one architecture this repo has actually validated.

WHY THESE SERIES. The tech-specific series people actually care about (Indeed software-dev
postings, IHLIDXUSTPSOFTDEVE) starts 2020-02 and contains exactly one boom and one bust -
n=1, useless for validation. It is carried here for the lead-lag diagnostic only, never as
a model feature. The CES industry employment series go back to 1990 and cover the dot-com
bust, the GFC, COVID, and the 2022-24 contraction.

TWO HONESTY FLAGS baked into the data, both unfixable without a lot more work:
  * REVISIONS. FRED serves the latest vintage of CES, not what was known at the time.
    Real-time vintages live in ALFRED. This gives the model a small look-ahead advantage
    on recent history, so treat measured skill as a mild upper bound.
  * PUBLICATION LAG. CES for month t prints in early t+1. Features here are lagged one
    month so a forecast dated t uses only data a forecaster could plausibly hold at t.
"""
import os

import numpy as np
import pandas as pd
import pandas_datareader.data as web

DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_CSV = os.path.join(DIR, "techjobs_panel.csv")

# Six tech sub-industries, chosen to be non-overlapping NAICS so the panel isn't
# six copies of the same aggregate. Broader roll-ups (USINFO, USPBS, prof/sci/tech)
# are deliberately excluded - they contain these and would leak.
INDUSTRIES = {
    "CES6054130001": "Computer systems design",
    "CES5051200001": "Software publishers",
    "CES5051800001": "Data processing & hosting",
    "CES5051700001": "Telecommunications",
    "CES3133440001": "Semiconductors",
    "CES5051900001": "Other information services",
}

# Macro leading indicators, all available from 1990 so the panel keeps its full span.
# JOLTS (2001+) and Indeed postings (2020+) are fetched separately for diagnostics only.
MACRO = {
    "TEMPHELPS": "temphelp",     # temp help - classic 3-6mo leader for the labour cycle
    "ICSA": "claims",            # initial claims, weekly
    "T10Y3M": "curve",           # recession-signal yield curve
    "NASDAQCOM": "nasdaq",       # tech equity - funding conditions for tech hiring
    "AWHMAN": "hours",           # avg weekly hours, mfg - employers cut hours before heads
}
DIAG = {
    "JTSJOL": "openings", "JTSQUL": "quits", "JTSLDL": "layoffs", "JTSHIL": "hires",
    "IHLIDXUSTPSOFTDEVE": "postings_swe",
}

FEATS = ["mom3", "mom6", "mom12", "vs_ma36", "vol12",
         "temphelp_yoy", "claims_yoy", "curve", "nasdaq_12m", "hours_chg"]
START = "1990-01-01"


def _fred(ids, start=START):
    return {v: web.DataReader(k, "fred", start).iloc[:, 0].dropna() for k, v in ids.items()}


def fetch_macro() -> pd.DataFrame:
    """Monthly macro features, each transformed to a stationary form."""
    m = _fred(MACRO)
    mo = {k: v.resample("MS").mean() for k, v in m.items()}   # weekly/daily -> month
    f = pd.DataFrame(index=mo["temphelp"].index)
    f["temphelp_yoy"] = mo["temphelp"].pct_change(12) * 100
    f["claims_yoy"] = mo["claims"].pct_change(12) * 100
    f["curve"] = mo["curve"]
    f["nasdaq_12m"] = mo["nasdaq"].pct_change(12) * 100
    f["hours_chg"] = mo["hours"].diff(12)
    return f


def fetch_industries() -> pd.DataFrame:
    """Wide frame of industry employment levels (thousands), monthly."""
    d = _fred(INDUSTRIES)
    return pd.DataFrame({k: v.resample("MS").last() for k, v in d.items()})


def fetch_diagnostics() -> pd.DataFrame:
    """JOLTS + Indeed postings, for the lead-lag diagnostic only. Never model features."""
    out = {}
    for sid, name in DIAG.items():
        try:
            out[name] = web.DataReader(sid, "fred", "2000-01-01").iloc[:, 0].dropna().resample("MS").mean()
        except Exception:
            pass
    return pd.DataFrame(out)


def build_panel(horizon: int = 12) -> pd.DataFrame:
    """Long panel: one row per (industry, month) with features and a forward-h target.

    Macro features are lagged one month (publication lag). The target is the forward
    h-month log change in employment, in percent.
    """
    ind, mac = fetch_industries(), fetch_macro()
    mac = mac.shift(1)                        # publication lag

    rows = []
    for sid, name in INDUSTRIES.items():
        s = ind[name].dropna()
        if len(s) < 180:
            continue
        lp = np.log(s)
        d = pd.DataFrame(index=s.index)
        d["industry"] = name
        d["emp"] = s
        d["mom3"] = lp.diff(3) * 100
        d["mom6"] = lp.diff(6) * 100
        d["mom12"] = lp.diff(12) * 100
        d["vs_ma36"] = (lp - lp.rolling(36).mean()) * 100
        d["vol12"] = (lp.diff().rolling(12).std()) * 100
        d["target"] = lp.shift(-horizon).sub(lp) * 100     # forward h-month growth, %
        d["naive_mom"] = lp.diff(horizon) * 100            # baseline: trailing = forward
        rows.append(d.join(mac))

    p = pd.concat(rows).reset_index().rename(columns={"index": "date", "DATE": "date"})
    p = p.dropna(subset=FEATS)
    return p.sort_values(["date", "industry"]).reset_index(drop=True)


def load(horizon: int = 12, refresh: bool = False) -> pd.DataFrame:
    csv = PANEL_CSV.replace(".csv", f"_h{horizon}.csv")
    if refresh or not os.path.exists(csv):
        p = build_panel(horizon)
        p.to_csv(csv, index=False)
        return p
    p = pd.read_csv(csv, parse_dates=["date"])
    return p


def main():
    for h in (6, 12):
        p = build_panel(h)
        p.to_csv(PANEL_CSV.replace(".csv", f"_h{h}.csv"), index=False)
        tr = p.dropna(subset=["target"])
        print(f"h={h:>2}m  panel {len(p):,} rows  ({len(tr):,} with matured target)  "
              f"{p.date.min().date()} -> {p.date.max().date()}  "
              f"{p.industry.nunique()} industries")
    print()
    latest = build_panel(12).sort_values("date").groupby("industry").tail(1)
    print("latest employment by industry (thousands):")
    for _, r in latest.sort_values("emp", ascending=False).iterrows():
        print(f"  {r['industry']:28s} {r['emp']:>8,.1f}   12m {r['mom12']:>+6.1f}%")
    print(f"\nwrote {PANEL_CSV.replace('.csv','_h{6,12}.csv')}")


if __name__ == "__main__":
    main()
