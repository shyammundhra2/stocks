"""
Does a volatility-risk-premium (VRP) ETF sleeve help? Test the accessible
listed VRP harvesters standalone and blended with equity:
  QYLD/XYLD  covered-call (ATM, Nasdaq/S&P)   2013+
  PUTW       systematic put-write (S&P ATM)     2016+
  DIVO       selective covered calls            2016+
  JEPI/JEPQ  smart covered call                 2020/2022+
  SVOL       short-vol w/ tail hedge            2021+

Reports per ETF over its OWN history (total return): CAGR, Sharpe, maxDD,
correlation to SPY (diversifying or just equity beta?), and the two tail
windows (2018 Q4, 2020 COVID) - the VRP's price. Then blends 80% SPY + 20% VRP
vs 100% SPY over the common window, to see if the sleeve adds anything.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

VRP = ["QYLD", "XYLD", "PUTW", "DIVO", "JEPI", "JEPQ", "SVOL"]
BENCH = ["SPY", "QQQ"]
END = "2026-08-21"
TAILS = [("2018 Q4", "2018-09-01", "2019-01-31"), ("2020 COVID", "2020-02-01", "2020-04-30"),
         ("2022 bear", "2022-01-01", "2022-12-31")]


def stats(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 30 or r.std() == 0:
        return np.nan, np.nan, np.nan
    sh = (r.mean() * 252) / (r.std() * np.sqrt(252))
    eq = np.cumprod(1 + r); cg = eq[-1] ** (252 / len(r)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def wret(r, dates, lo, hi):
    m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
    x = np.asarray(r, float)[m.values if hasattr(m, "values") else m]; x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    eq = np.cumprod(1 + x)
    return eq[-1] - 1, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    dl = VRP + BENCH
    print(f"Downloading {dl} ...")
    raw = yf.download(dl, start="2013-01-01", end=END, progress=False, auto_adjust=True)["Close"]
    rets = raw.pct_change()
    spy = rets["SPY"]

    print(f"\n{'ETF':>6s} {'from':>10s} {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'corr SPY':>9s} "
          f"{'2018Q4':>8s} {'COVID':>8s} {'2022':>7s}")
    print("-" * 82)
    def row(t):
        s = raw[t].dropna()
        r = s.pct_change().dropna()
        sh, cg, dd = stats(r.values)
        cor = r.corr(spy.reindex(r.index))
        d = r.index
        w18 = wret(r.values, d, "2018-09-01", "2019-01-31")[0]
        wcv = wret(r.values, d, "2020-02-01", "2020-04-30")[0]
        w22 = wret(r.values, d, "2022-01-01", "2022-12-31")[0]
        print(f"{t:>6s} {str(s.index.min().date()):>10s} {cg:>7.1%} {sh:>7.2f} {dd:>7.1%} {cor:>9.2f} "
              f"{w18:>8.1%} {wcv:>8.1%} {w22:>7.1%}")
    for t in BENCH + VRP:
        try:
            row(t)
        except Exception as e:
            print(f"{t:>6s}  err {e}")

    # Blend test: 80% SPY + 20% VRP vs 100% SPY, common window per ETF
    print(f"\nBLEND 80% SPY + 20% VRP vs 100% SPY (each over the VRP's history):")
    print(f"{'VRP':>6s} {'window from':>12s} {'SPY CAGR':>9s} {'SPY Sh':>7s} {'SPY DD':>7s} | "
          f"{'blend CAGR':>10s} {'blend Sh':>8s} {'blend DD':>8s}")
    for t in VRP:
        try:
            r = pd.concat([spy, rets[t]], axis=1).dropna()
            r.columns = ["spy", "vrp"]
            blend = 0.8 * r["spy"] + 0.2 * r["vrp"]
            sh0, cg0, dd0 = stats(r["spy"].values)
            shb, cgb, ddb = stats(blend.values)
            print(f"{t:>6s} {str(r.index.min().date()):>12s} {cg0:>9.1%} {sh0:>7.2f} {dd0:>7.1%} | "
                  f"{cgb:>10.1%} {shb:>8.2f} {ddb:>8.1%}")
        except Exception as e:
            print(f"{t:>6s}  err {e}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
