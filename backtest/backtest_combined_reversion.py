"""
The momentum overlay failed to diversify the (momentum-driven) long book -
same factor, corr ~0.55. This tests the only genuinely different diversifier:
a short-term REVERSAL sleeve (anti-momentum), which should be low/negatively
correlated with the trend book.

  Sleeve A  production long book (imported from backtest_trend_following).
  Sleeve R  cross-sectional 1-week reversal: long bottom-quintile 5d-return
            (recent losers), short top-quintile (recent winners), equal-weight,
            SPY-beta-hedged, rebalanced weekly, costs charged on turnover.

A's daily returns are cached to CSV so repeat runs skip the ~2min walk-forward.
Reports corr(A,R) and combined Sharpe / maxDD across allocation scales.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.indicators import _trend_stats  # noqa: F401 (kept for parity)
from backtest.backtest_trend_following import (
    download_all, walk_forward, build_equity_curve, TEST_START, END, CASH_TICKER,
)

REBAL_EVERY = 5                  # weekly (reversal is a fast signal)
REV_LB = 5
Q = 5
COST_BPS = 5.0
HARD_TO_SHORT = {"URNM", "COAL", "IBIT", "FCG", "RKT", "SLX", "WOOD"}
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
A_CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sleeveA_returns.csv"


def perf(r):
    r = pd.Series(r).dropna().values
    if len(r) < 20:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    yrs = len(r) / 252
    eq = np.cumprod(1 + r)
    dd = eq / np.maximum.accumulate(eq) - 1
    vol = r.std() * np.sqrt(252)
    return dict(cagr=eq[-1] ** (1 / yrs) - 1, vol=vol,
                sharpe=(r.mean() * 252) / vol if vol > 0 else np.nan, maxdd=dd.min())


def get_A(raw, close_all):
    if os.path.exists(A_CACHE):
        print("Loading cached sleeve A ...")
        s = pd.read_csv(A_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
        return s
    print("Running sleeve A (production long book) ...")
    daily_df, _ = walk_forward(raw)
    _, A_log = build_equity_curve(daily_df, close_all)
    A = np.expm1(A_log)
    os.makedirs(os.path.dirname(A_CACHE), exist_ok=True)
    A.to_frame("A").to_csv(A_CACHE)
    return A


def build_reversal(close_all, syms):
    dates = close_all.index[(close_all.index >= TEST_START) & (close_all.index <= END)]
    simple = close_all[syms].pct_change()
    spy = close_all["SPY"].pct_change()
    var_spy = spy.rolling(63).var()
    betas = simple.rolling(63).cov(spy).div(var_spy, axis=0)

    W = pd.DataFrame(0.0, index=dates, columns=syms)
    cur = pd.Series(0.0, index=syms); prev = pd.Series(0.0, index=syms)
    turnover = pd.Series(0.0, index=dates)
    for k, dt in enumerate(dates):
        if k % REBAL_EVERY == 0:
            r5 = {}
            for s in syms:
                h = close_all[s].loc[:dt].dropna()
                if len(h) > REV_LB:
                    r5[s] = np.log(h.iloc[-1] / h.iloc[-1 - REV_LB])
            if len(r5) >= Q * 2:
                ser = pd.Series(r5).sort_values()      # ascending: losers first
                n = len(ser) // Q
                longs = list(ser.index[:n])            # buy recent losers
                shorts = [s for s in ser.index[::-1] if s not in HARD_TO_SHORT][:n]  # short winners
                cur = pd.Series(0.0, index=syms)
                if longs:
                    cur[longs] = 1.0 / len(longs)
                if shorts:
                    cur[shorts] = -1.0 / len(shorts)
            turnover[dt] = float((cur - prev).abs().sum()); prev = cur.copy()
        W.loc[dt] = cur.values

    fwd = simple.reindex(dates).shift(-1)
    spy_fwd = spy.reindex(dates).shift(-1)
    net_beta = (W * betas.reindex(dates)).sum(axis=1)
    ret = (W * fwd).sum(axis=1) - net_beta * spy_fwd - turnover * (COST_BPS / 1e4)
    return ret.iloc[:-1]


def main():
    t0 = time.time()
    raw = download_all()
    close_all = raw["Close"].ffill()
    if "Adj Close" in raw.columns.get_level_values(0) and CASH_TICKER in raw["Adj Close"].columns:
        close_all[CASH_TICKER] = raw["Adj Close"][CASH_TICKER].ffill()

    A = get_A(raw, close_all)
    from macro.constants import TREND_ASSETS
    syms = [s for s in TREND_ASSETS if s in close_all.columns]
    print(f"Running sleeve R (weekly reversal, beta-hedged, {len(syms)} names) ...")
    R = build_reversal(close_all, syms)

    df = pd.concat({"A": A, "R": R}, axis=1).dropna()
    pa, pr = perf(df["A"]), perf(df["R"])
    print(f"\nPeriods: {len(df)}  ({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"corr(A, R) = {df['A'].corr(df['R']):+.3f}   (want ~0 or negative)\n")

    print(f"{'book':22s} {'CAGR':>7s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>7s}")
    print("-" * 54)
    print(f"{'A: long book alone':22s} {pa['cagr']:>7.1%} {pa['vol']:>7.1%} {pa['sharpe']:>7.2f} {pa['maxdd']:>7.1%}")
    print(f"{'R: reversal alone':22s} {pr['cagr']:>7.1%} {pr['vol']:>7.1%} {pr['sharpe']:>7.2f} {pr['maxdd']:>7.1%}")
    print("-" * 54)
    best = (0.0, pa["sharpe"])
    for sc in SCALES:
        p = perf(df["A"] + sc * df["R"])
        flag = "  <- higher Sharpe, DD not worse" if (sc > 0 and p["sharpe"] > pa["sharpe"] + 1e-6 and p["maxdd"] >= pa["maxdd"] - 1e-4) else ""
        if p["sharpe"] > best[1]:
            best = (sc, p["sharpe"])
        print(f"{'A + ' + format(sc, '.2f') + '*R':22s} {p['cagr']:>7.1%} {p['vol']:>7.1%} {p['sharpe']:>7.2f} {p['maxdd']:>7.1%}{flag}")
    print(f"\n-> best Sharpe {best[1]:.2f} at scale {best[0]} (A-alone {pa['sharpe']:.2f})")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
