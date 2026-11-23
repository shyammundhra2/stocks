"""
Live ETF/trend-asset ranker - the TREND_ASSETS universe this tilt was
ACTUALLY validated on (backtest_trend_dd_tilt.py), not the S&P 500 stock
sleeve. predict_top_stocks.py's own tilt (wired in 2026-11-23) was an
extrapolation to a different, un-tested universe; this script is the one
that matches the validated design directly.

Rule, reproduced exactly from backtest_trend_dd_tilt.py (no un-tested
extras added - no SPY-200DMA gate, no sector cap, no vol overlay; none of
those were part of what was actually tested here, unlike the stock
sleeve's separately-validated gate/cap/overlay):

  Score = 12-1 month momentum x 2.0 if the asset is currently 5-15% off its
  own 252d high, else x1.0 (backtest_trend_vs_mr_by_dd_phase.py: trend's
  forward-return IC peaks specifically in that "climbing out, not yet at
  highs" zone). Hold the TOP_N by that score, equal-weight, capped at 15%
  per name; only candidates with positive raw momentum are eligible.
  Remainder of the sleeve sits in cash.

Full-period backtest (2007-26, net@5bps) at tilt=2.0: Sharpe 0.64 / CAGR
13.4% / maxDD -50.8%, vs 0.61/12.7%/-53.5% untilted and 0.60/10.5%/-51.2%
for equal-weight buy-and-hold. Improves (does not just trade off) Sharpe,
CAGR, and maxDD together across 2008 GFC, the full period, and the 2020-26
regime window - see backtest_trend_dd_tilt.py for the full window table.

HONEST CAVEATS: this is a NEW live script with no track record yet (the
backtest is the only evidence). No book-notional is assumed here - output
is weights, not dollar sizing; apply whatever capital allocation you
intend separately. Costs are NOT deducted live (the 5bps-net backtest
figure already accounts for them historically; this script reports the
gross target weights for today).
"""
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))
from macro.constants import TREND_ASSETS

OUTPUT_CSV = os.path.join(_DIR, "trend_etf_ranking.csv")

MOM_LOOKBACK = 252      # 12 months
MOM_SKIP = 21           # skip last month (12-1 momentum)
DD_LOOKBACK = 252
DD_TILT_ZONE = (-0.15, -0.05)
DD_TILT_FACTOR = 2.0
TOP_N = 8
CAP = 0.15


def momentum_12_1(close):
    if len(close) < MOM_LOOKBACK + 1:
        return np.nan
    p_now = close.iloc[-1 - MOM_SKIP]
    p_then = close.iloc[-MOM_LOOKBACK]
    if not (np.isfinite(p_now) and np.isfinite(p_then)) or p_then <= 0:
        return np.nan
    return float(p_now / p_then - 1.0)


def infer_today():
    print("🚀 Trend-asset ranker: 12-1 momentum x drawdown-phase tilt\n")
    tickers = list(TREND_ASSETS.keys())
    end = datetime.now()
    start = end - timedelta(days=650)
    print(f"📅 {start.date()} → {end.date()}")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close_all = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    rows = []
    for t in tickers:
        if t not in close_all.columns:
            continue
        close = close_all[t].dropna()
        if len(close) < MOM_LOOKBACK + 1:
            continue
        mom = momentum_12_1(close)
        if not np.isfinite(mom):
            continue
        high_252 = close.rolling(DD_LOOKBACK, min_periods=60).max().iloc[-1]
        dd = float(close.iloc[-1] / high_252 - 1.0) if np.isfinite(high_252) and high_252 > 0 else np.nan
        in_zone = np.isfinite(dd) and (DD_TILT_ZONE[0] < dd <= DD_TILT_ZONE[1])
        score = mom * (DD_TILT_FACTOR if in_zone else 1.0) if mom > 0 else -np.inf
        rows.append({
            "Ticker": t, "Name": TREND_ASSETS.get(t, t),
            "Mom_12_1": round(mom, 4), "DD_From_High": round(dd, 4) if np.isfinite(dd) else np.nan,
            "Tilt_Applied": bool(in_zone), "Score": round(score, 4) if np.isfinite(score) else np.nan,
            "Current_Price": round(float(close.iloc[-1]), 2),
            "Eligible": bool(mom > 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ no valid assets"); return df

    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    elig = df[df["Eligible"]].head(TOP_N).copy()
    if len(elig):
        tot = elig["Score"].sum()
        elig["Weight"] = (elig["Score"] / tot).clip(upper=CAP)
        elig["Weight"] = elig["Weight"] / elig["Weight"].sum()
    df["Weight"] = 0.0
    df.loc[elig.index, "Weight"] = elig["Weight"]
    df["Signal"] = np.where(df["Weight"] > 0, "HOLD", "-")

    out_cols = ["Ticker", "Name", "Score", "Mom_12_1", "DD_From_High", "Tilt_Applied",
                "Signal", "Weight", "Current_Price"]
    df[out_cols].to_csv(OUTPUT_CSV, index=False)

    print("=" * 110)
    print("TOP TREND-ASSET HOLDS (tilt-weighted)")
    print("=" * 110)
    print(df[out_cols].head(TOP_N + 5).to_string(index=False))
    print(f"\n✅ saved → {OUTPUT_CSV}")
    cash_w = 1.0 - df["Weight"].sum()
    print(f"\n📊 {len(df)} ranked | {len(elig)} held | deployed {df['Weight'].sum():.0%} | cash {cash_w:.0%}")
    return df


if __name__ == "__main__":
    infer_today()
