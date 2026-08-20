"""
Does adding an uncorrelated market-neutral sleeve raise Sharpe WITHOUT adding
drawdown? Combines two sleeves on the same capital:

  Sleeve A  your production long book (get_trends + Kelly optimizer), ~38% net
            long, idle cash in SGOV. Imported verbatim from
            backtest_trend_following - this is the real strategy.
  Sleeve B  market-neutral slope*r2 overlay: every 2 weeks, long the top
            slope*r2 quintile / short the bottom quintile, equal-weight,
            DOLLAR-NEUTRAL (net beta ~ 0). Funded from idle cash (margin).
            Hard-to-short names excluded from the short leg. Transaction
            costs charged on turnover.

Total return = A + scale * B. We sweep scale, and report the realized
correlation of A vs B plus combined CAGR / vol / Sharpe / max drawdown.
The claim to test: corr(A,B) ~ 0, so combined Sharpe > A alone with a max
drawdown no worse than A's.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.indicators import _trend_stats
# reuse the exact production sleeve-A machinery
from backtest.backtest_trend_following import (
    download_all, walk_forward, build_equity_curve,
    TEST_START, END, CASH_TICKER,
)

REBAL_EVERY = 10                 # 2-week hold, matches sleeve A
TRW, TRSCALE = 20, 10
Q = 5                            # quintiles
COST_BPS = 5.0                   # one-way transaction cost per unit turnover
HARD_TO_SHORT = {"URNM", "COAL", "IBIT", "FCG", "RKT", "SLX", "WOOD"}
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]


def perf(simple_rets):
    r = simple_rets.dropna().values
    if len(r) < 20:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    years = len(r) / 252
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / vol if vol > 0 else np.nan
    dd = eq / np.maximum.accumulate(eq) - 1
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd.min())


def build_sleeve_B(close_all, syms):
    """Daily simple returns of the slope*r2 overlay, net of costs. Returns both
    the dollar-neutral version and a BETA-NEUTRAL version that hedges each day's
    net market exposure with SPY (beta to SPY over a trailing 63d window)."""
    dates = close_all.index[(close_all.index >= TEST_START) & (close_all.index <= END)]
    simple = close_all[syms].pct_change()
    spy = close_all["SPY"].pct_change()

    # trailing 63d beta of each name to SPY
    var_spy = spy.rolling(63).var()
    betas = simple.rolling(63).cov(spy).div(var_spy, axis=0)

    W = pd.DataFrame(0.0, index=dates, columns=syms)
    cur = pd.Series(0.0, index=syms)
    prev_w = pd.Series(0.0, index=syms)
    turnover = pd.Series(0.0, index=dates)
    for k, dt in enumerate(dates):
        if k % REBAL_EVERY == 0:
            sr = {}
            for s in syms:
                hist = close_all[s].loc[:dt].dropna()
                if len(hist) < TRW + 5:
                    continue
                sl, r2 = _trend_stats(hist, TRW, TRSCALE)
                sr[s] = sl * r2
            if len(sr) >= Q * 2:
                ser = pd.Series(sr).sort_values()
                n = len(ser) // Q
                shorts = [s for s in ser.index if s not in HARD_TO_SHORT][:n]
                longs = list(ser.index[-n:])
                cur = pd.Series(0.0, index=syms)
                if longs:
                    cur[longs] = 1.0 / len(longs)
                if shorts:
                    cur[shorts] = -1.0 / len(shorts)
            turnover[dt] = float((cur - prev_w).abs().sum())
            prev_w = cur.copy()
        W.loc[dt] = cur.values

    fwd = simple.reindex(dates).shift(-1)
    spy_fwd = spy.reindex(dates).shift(-1)
    beta_al = betas.reindex(dates)

    gross_ret = (W * fwd).sum(axis=1)
    cost = turnover * (COST_BPS / 1e4)
    net_beta = (W * beta_al).sum(axis=1)                 # portfolio beta each day
    hedged_ret = gross_ret - net_beta * spy_fwd          # subtract market exposure

    B_dollar = (gross_ret - cost).iloc[:-1]
    B_beta = (hedged_ret - cost).iloc[:-1]
    return B_dollar, B_beta, float(net_beta.mean())


def main():
    t0 = time.time()
    raw = download_all()
    close_all = raw["Close"].ffill()
    if "Adj Close" in raw.columns.get_level_values(0) and CASH_TICKER in raw["Adj Close"].columns:
        close_all[CASH_TICKER] = raw["Adj Close"][CASH_TICKER].ffill()

    print("Running sleeve A (production long book, 2-week hold) ...")
    daily_df, _ = walk_forward(raw)
    _, A_log = build_equity_curve(daily_df, close_all)
    A = np.expm1(A_log)                         # simple daily returns

    from macro.constants import TREND_ASSETS
    syms = [s for s in TREND_ASSETS if s in close_all.columns]
    print(f"Running sleeve B (slope*r2 overlay, {len(syms)} names, "
          f"cost {COST_BPS}bps, ex-short {len(HARD_TO_SHORT)}) ...")
    B_dollar, B_beta, avg_beta = build_sleeve_B(close_all, syms)

    df = pd.concat({"A": A, "Bd": B_dollar, "Bh": B_beta}, axis=1).dropna()
    pa = perf(df["A"])
    print(f"\nPeriods: {len(df)}  ({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"avg net beta of dollar-neutral book: {avg_beta:+.2f}  (that residual is the problem)")
    print(f"corr(A, dollar-neutral B) = {df['A'].corr(df['Bd']):+.3f}")
    print(f"corr(A, beta-neutral  B) = {df['A'].corr(df['Bh']):+.3f}   (want ~0)\n")

    def report(bcol, title):
        pb = perf(df[bcol])
        print(f"== {title} ==")
        print(f"{'book':22s} {'CAGR':>7s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>7s}")
        print("-" * 54)
        print(f"{'A: long book alone':22s} {pa['cagr']:>7.1%} {pa['vol']:>7.1%} "
              f"{pa['sharpe']:>7.2f} {pa['maxdd']:>7.1%}")
        print(f"{'B alone':22s} {pb['cagr']:>7.1%} {pb['vol']:>7.1%} "
              f"{pb['sharpe']:>7.2f} {pb['maxdd']:>7.1%}")
        print("-" * 54)
        best = (0.0, pa["sharpe"])
        for sc in SCALES:
            p = perf(df["A"] + sc * df[bcol])
            flag = ""
            if sc > 0 and p["sharpe"] > pa["sharpe"] + 1e-6 and p["maxdd"] >= pa["maxdd"] - 1e-4:
                flag = "  <- higher Sharpe, DD not worse"
            if p["sharpe"] > best[1]:
                best = (sc, p["sharpe"])
            print(f"{'A + ' + format(sc, '.2f') + '*B':22s} {p['cagr']:>7.1%} {p['vol']:>7.1%} "
                  f"{p['sharpe']:>7.2f} {p['maxdd']:>7.1%}{flag}")
        print(f"-> best Sharpe {best[1]:.2f} at scale {best[0]} (A-alone {pa['sharpe']:.2f})\n")

    report("Bd", "DOLLAR-NEUTRAL overlay (broken: net-long beta)")
    report("Bh", "BETA-NEUTRAL overlay (SPY-hedged fix)")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
