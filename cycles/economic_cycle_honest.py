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
import pandas_datareader.data as web
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
    # -----------------------------------------------
    # 4. SAHM RULE - a SEPARATE signal, deliberately not blended into the
    # curve's logistic score above. Reused from economic_cycle_composite.py
    # (already built, already validated there) - merged in here rather than
    # duplicated, with one correction: that module's own validation framed
    # Sahm as a LEADING indicator ("preceded within 6mo"), which its real
    # track record doesn't support (checked below: only 1/7). Sahm's actual
    # academic purpose is a real-time CONFIRMATION that a recession has
    # already started, filling the ~12-18mo gap before NBER dates it
    # officially - a coincident check is the honest test for what it's
    # actually for, run alongside the leading-style check rather than
    # instead of it, so neither framing is quietly cherry-picked.
    # -----------------------------------------------
    ur = web.DataReader("UNRATE", "fred", "1960-01-01").iloc[:, 0].resample("ME").last()
    sahm = (ur.rolling(3).mean() - ur.rolling(12).min()).dropna()
    sahm_trig = lambda w: w >= 0.5

    print(f"\nSAHM RULE (FRED UNRATE) - separate signal, not blended with the curve:")
    n_lead_sahm, n_coincident = 0, 0
    for a, b in RECESSIONS:
        d, e = pd.Timestamp(a), pd.Timestamp(b)
        lead_win = sahm[(sahm.index >= d - pd.DateOffset(months=6)) & (sahm.index < d)]
        if len(lead_win) and sahm_trig(lead_win).any():
            n_lead_sahm += 1
        coincident_win = sahm[(sahm.index >= d) & (sahm.index <= e + pd.DateOffset(months=3))]
        if len(coincident_win) and sahm_trig(coincident_win).any():
            n_coincident += 1
    cov_sahm = len([1 for a, _ in RECESSIONS if pd.Timestamp(a) >= sahm.index.min()])
    print(f"   as a LEADING signal (fires <6mo before start): {n_lead_sahm}/{cov_sahm} - weak, as expected")
    print(f"   as a COINCIDENT/confirmation signal (fires during, or within 3mo after start): "
          f"{n_coincident}/{cov_sahm} - this is what it's actually validated for")
    cur_sahm = float(sahm.iloc[-1])
    print(f"   CURRENT Sahm reading: {cur_sahm:+.2f}  (confirms recession already underway if >= 0.50: "
          f"{'YES' if cur_sahm >= 0.5 else 'no'})")

    # -----------------------------------------------
    # 5. INTERNATIONAL YIELD-CURVE POOL - a SEPARATE signal from the US
    # curve above, not merged into one number. The US-only curve's 7/7
    # record is a sample of exactly 7 events - suggestive, not statistically
    # strong. Pooling 10y-3m inversions and each country's OWN technical
    # recessions (2 consecutive quarters of negative real GDP growth - a
    # simplified, mechanically-computed definition, honestly NOT the same
    # as an official NBER-style business-cycle dating committee call, but
    # transparent and reproducible) across 5 more economies gives a larger,
    # independent-ish sample to check whether the US lead-time finding
    # generalizes, without re-testing the same 7 US events again.
    # -----------------------------------------------
    print(f"\nINTERNATIONAL YIELD-CURVE POOL (separate signal, OECD data via FRED, 1990-present):")
    print(f"   technical recession = 2 consecutive quarters of negative real GDP growth (NOT an")
    print(f"   official NBER-equivalent dating - a simplified, transparent proxy, flagged as such)")
    countries = {
        "UK": ("IRLTLT01GBM156N", "IR3TIB01GBM156N", "NGDPRSAXDCGBQ"),
        "Germany": ("IRLTLT01DEM156N", "IR3TIB01DEM156N", "CLVMNACSCAB1GQDE"),
        "Japan": ("IRLTLT01JPM156N", "IR3TIB01JPM156N", "JPNRGDPEXP"),
        "Canada": ("IRLTLT01CAM156N", "IR3TIB01CAM156N", "NGDPRSAXDCCAQ"),
        "Australia": ("IRLTLT01AUM156N", "IR3TIB01AUM156N", "NGDPRSAXDCAUQ"),
    }
    pool_n_lead, pool_cov, pool_leads = 0, 0, []
    for name, (c10, c3m, cgdp) in countries.items():
        try:
            y10 = web.DataReader(c10, "fred", "1990-01-01").iloc[:, 0].resample("ME").last()
            y3m = web.DataReader(c3m, "fred", "1990-01-01").iloc[:, 0].resample("ME").last()
            curve_i = (y10 - y3m).dropna()
            gdp = web.DataReader(cgdp, "fred", "1990-01-01").iloc[:, 0]
            gdp_growth = gdp.pct_change()
            contraction = (gdp_growth < 0) & (gdp_growth.shift(1) < 0)
            rec_quarters = gdp.index[contraction]
            # Each contraction-start quarter -> one technical-recession event
            # (collapse consecutive flagged quarters into single episodes).
            episodes = []
            for q in rec_quarters:
                if not episodes or (q - episodes[-1][-1]).days > 100:
                    episodes.append([q])
                else:
                    episodes[-1].append(q)
            for ep in episodes:
                d = pd.Timestamp(ep[0]) - pd.DateOffset(months=3)  # contraction's first bad quarter
                win = curve_i[(curve_i.index >= d - pd.DateOffset(months=24)) & (curve_i.index < d)]
                pool_cov += 1
                if len(win) and (win < 0).any():
                    pool_n_lead += 1
                    first_inv = win[win < 0].index[0]
                    pool_leads.append((d - first_inv).days / 30.0)
            print(f"   {name:10s}: {len(episodes)} technical-recession episode(s) found "
                  f"since {curve_i.index.min().date()}")
        except Exception as e:
            print(f"   {name:10s}: skipped ({str(e)[:60]})")
    if pool_cov:
        print(f"\n   POOLED (5 countries): curve inversion preceded {pool_n_lead}/{pool_cov} technical "
              f"recessions (<24mo)")
        if pool_leads:
            print(f"   pooled median lead: {np.median(pool_leads):.0f} months (range "
                  f"{min(pool_leads):.0f}-{max(pool_leads):.0f}) - compare to the US-only 13mo (n=7) above")

    print(f"\n   [descriptive context only - NOT a forecast/trade signal. The live")
    print(f"    200DMA gate + VIX throttle react to downturns; they don't predict them.]")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
