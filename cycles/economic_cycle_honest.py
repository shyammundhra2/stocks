"""
Honest, information-only economic-cycle view.

Fixes the old module's three flaws (backtest_cycle_oos.py): the self-referential
S&P sine forecast has NO recession lead (hit 45% vs 80% base), the 'cycle' length
wanders 3-15yr (not real), and the CWT/HP reading revises ~44% in real time.

This version:
  1. CAUSAL cycle position: S&P log-price minus its trailing 10y mean (one-sided,
     so today's expansion/contraction reading never revises with future data).
     DESCRIPTIVE only - where price sits vs its own trend. No forward projection.
  2. LEADING recession gauge: the yield-curve slope (10y - 3m), which actually
     leads recessions - and we VALIDATE that lead against the real NBER recession
     dates instead of claiming it. Recession-risk = calibrated logistic on the
     curve, not an arbitrary amplitude CDF.

Nothing here is a trade signal (the shipped 200DMA gate + VIX throttle REACT to
downturns, which beats predicting them). This is context, honestly labeled.
"""
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# NBER US recession windows (month granularity)
RECESSIONS = [("1973-11", "1975-03"), ("1980-01", "1980-07"), ("1981-07", "1982-11"),
              ("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06"),
              ("2020-02", "2020-04")]


def main():
    t0 = time.time()
    px = yf.download("^GSPC", start="1960-01-01", progress=False, auto_adjust=True)["Close"]
    tnx = yf.download("^TNX", start="1960-01-01", progress=False, auto_adjust=True)["Close"]
    irx = yf.download("^IRX", start="1960-01-01", progress=False, auto_adjust=True)["Close"]
    for s in (px, tnx, irx):
        if isinstance(s, pd.DataFrame):
            s.columns = [0]
    m = np.log(pd.Series(np.asarray(px).ravel(), index=px.index).resample("ME").last().ffill())
    t10 = pd.Series(np.asarray(tnx).ravel(), index=tnx.index).resample("ME").last()
    t3 = pd.Series(np.asarray(irx).ravel(), index=irx.index).resample("ME").last()
    curve = (t10 - t3).dropna()                       # 10y - 3m yield-curve slope

    # 1. causal cycle position
    cyc = (m - m.rolling(120).mean())
    idx = cyc.dropna().index

    # recession label per month
    rec = pd.Series(0, index=cyc.index)
    for a, b in RECESSIONS:
        rec.loc[a:b] = 1
    # target: recession STARTS within next 12 months (for lead validation)
    starts = pd.Series(0, index=cyc.index)
    for a, _ in RECESSIONS:
        d = pd.Timestamp(a)
        starts.loc[(starts.index >= d - pd.DateOffset(months=12)) & (starts.index < d)] = 1

    print(f"\n=== HONEST ECONOMIC-CYCLE VIEW (info only, causal) ===\n")

    # 2. validate the yield curve's recession LEAD
    print("YIELD-CURVE (10y-3m) recession lead validation:")
    cw = curve.reindex(cyc.index).dropna()
    n_lead, leads = 0, []
    for a, _ in RECESSIONS:
        d = pd.Timestamp(a)
        win = cw[(cw.index >= d - pd.DateOffset(months=24)) & (cw.index < d)]
        if len(win) and (win < 0).any():
            n_lead += 1
            first_inv = win[win < 0].index[0]
            leads.append((d - first_inv).days / 30.0)
    cov = len([1 for a, _ in RECESSIONS if pd.Timestamp(a) >= cw.index.min()])
    print(f"   recessions in curve-data era: {cov} | preceded by an inversion (<24mo): {n_lead}/{cov}")
    if leads:
        print(f"   median lead from first inversion to recession start: {np.median(leads):.0f} months "
              f"(range {min(leads):.0f}-{max(leads):.0f})")

    # calibrated recession-in-12mo risk from the curve (in-sample; honest caveat)
    d = pd.concat([cw.rename("curve"), starts.rename("y")], axis=1).dropna()
    lr = LogisticRegression().fit(d[["curve"]].values, d["y"].values)
    d["risk"] = lr.predict_proba(d[["curve"]].values)[:, 1]
    # calibration by curve sign
    inv = d[d["curve"] < 0]; norm = d[d["curve"] >= 0]
    print(f"   P(recession starts within 12mo):  curve INVERTED {inv['y'].mean():.0%}  "
          f"vs NORMAL {norm['y'].mean():.0%}   (base {d['y'].mean():.0%})")

    # 3. current readings
    last_cyc = cyc.dropna().iloc[-1]; last_curve = cw.iloc[-1]
    cur_risk = float(lr.predict_proba([[last_curve]])[0, 1])
    pos = "ABOVE trend (late/expansion)" if last_cyc > 0 else "BELOW trend (early/contraction)"
    print(f"\nCURRENT ({cyc.dropna().index[-1].date()}):")
    print(f"   S&P cycle position: {last_cyc:+.1%} vs 10y trend  -> {pos}")
    print(f"   yield curve (10y-3m): {last_curve:+.2f}%  ({'INVERTED' if last_curve<0 else 'normal'})")
    print(f"   curve-implied P(recession within 12mo): {cur_risk:.0%}")
    print(f"\n   [descriptive context only - NOT a forecast/trade signal. The live")
    print(f"    200DMA gate + VIX throttle react to downturns; they don't predict them.]")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
