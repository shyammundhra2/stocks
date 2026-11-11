"""
Find an edge in MR: transfer the VALIDATED capitulation trigger to the pool.

Everything mined so far says MR's only real content is timing: the eligible
pool bounces after broad stress, and no z-score picks WHICH name bounces. The
repo already owns one validated MR timing trade - the VIX capitulation SPY
buyer (backtest_gss_capitulation, 2007-26, +0.06 Sharpe / +1.1%/yr on a 20%
reserve): enter on VIX z>2 AND SPY RSI14<30, exit when SPY recovers to within
5% of its 252d high.

PRE-COMMITTED HYPOTHESIS: the same frozen trigger earns more per episode when
deployed into the chop pool (equal-weight CHOP names above their 200DMA) or
QQQ than into SPY, because beta-to-the-bounce is higher. Trigger and exit are
FROZEN from the prior validation - nothing in this test is fit to this data.

PRE-COMMITTED BAR: the alternative target must beat SPY-deployment on total
return across episodes AND not lose in a majority of episodes AND not worsen
the worst episode materially (>2pp). n(episodes) is small and is reported per
episode - this is an indicative transfer test, not a tunable backtest.

Honest wrinkle handled explicitly: at capitulation, many names are BELOW their
200DMA, so the "eligible pool" can be thin or empty exactly when the trigger
fires. Basket rule: CHOP & >200DMA; if <3 names, widen to all >200DMA; if
still <3, the episode is UNFILLABLE for the pool target and says so.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

Z_ENTER, RSI_ENTER, DD_EXIT = 2.0, 30.0, -0.05       # frozen (capitulation backtest)


def wilder_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)


def main():
    t0 = time.time()
    syms = sorted(set(TREND_ASSETS) | {"SPY", "QQQ", "^VIX"})
    px = yf.download(syms, period="15y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    spy, vix = px["SPY"].dropna(), px["^VIX"].dropna()

    z = ((vix - vix.rolling(50).mean()) / vix.rolling(50).std()).reindex(spy.index)
    rsi = wilder_rsi(spy)
    dd = spy / spy.rolling(252, min_periods=60).max() - 1.0

    # episode state machine, identical shape to get_vix_signal's walk
    episodes, deployed, start = [], False, None
    for t in spy.index:
        if deployed and dd.loc[t] > DD_EXIT:
            episodes.append((start, t)); deployed = False
        elif not deployed and z.loc[t] > Z_ENTER and rsi.loc[t] < RSI_ENTER:
            deployed, start = True, t
    if deployed:
        episodes.append((start, spy.index[-1]))

    assets = [s for s in TREND_ASSETS if s in px.columns]
    ret = px[assets].pct_change()

    def basket_at(t):
        """CHOP & >200DMA; widen to >200DMA if <3; else unfillable."""
        held = []
        for s in assets:
            c = px[s].loc[:t].dropna()
            if len(c) < 210:
                continue
            above = c.iloc[-1] > c.rolling(200).mean().iloc[-1]
            net = abs(c.iloc[-1] - c.iloc[-21])
            path = c.diff().abs().iloc[-20:].sum()
            er = net / path if path > 0 else 0
            if above and er <= 0.35:
                held.append(s)
        if len(held) < 3:
            held = [s for s in assets
                    if len(px[s].loc[:t].dropna()) >= 210
                    and px[s].loc[:t].dropna().iloc[-1]
                    > px[s].loc[:t].dropna().rolling(200).mean().iloc[-1]]
        return held if len(held) >= 3 else None

    print(f"\nFrozen trigger: VIX z>{Z_ENTER:g} & SPY RSI14<{RSI_ENTER:g}; "
          f"exit SPY within {-DD_EXIT:.0%} of 252d high")
    print(f"{len(episodes)} episodes found\n")
    print(f"{'enter':>11} {'exit':>11} {'days':>5} {'SPY':>8} {'QQQ':>8} {'POOL':>8} {'n':>3}")
    tot = {"SPY": 1.0, "QQQ": 1.0, "POOL": 1.0}
    wins = {"QQQ": 0, "POOL": 0}
    worst = {"SPY": 0.0, "QQQ": 0.0, "POOL": 0.0}
    n_ok = 0
    for a, b in episodes:
        r_spy = float(spy.loc[b] / spy.loc[a] - 1)
        qqq = px["QQQ"].dropna()
        r_qqq = float(qqq.asof(b) / qqq.asof(a) - 1)
        bk = basket_at(a)
        if bk:
            win = ret[bk].loc[a:b].iloc[1:]
            r_pool = float((1 + win.mean(axis=1)).prod() - 1)
            npool = len(bk)
        else:
            r_pool, npool = np.nan, 0
        print(f"{str(a.date()):>11} {str(b.date()):>11} {(b-a).days:>5} "
              f"{r_spy:>+8.1%} {r_qqq:>+8.1%} "
              f"{(f'{r_pool:+8.1%}' if npool else '   unfil'):>8} {npool:>3}")
        tot["SPY"] *= 1 + r_spy
        tot["QQQ"] *= 1 + r_qqq
        worst["SPY"] = min(worst["SPY"], r_spy)
        worst["QQQ"] = min(worst["QQQ"], r_qqq)
        if npool:
            n_ok += 1
            tot["POOL"] *= 1 + r_pool
            worst["POOL"] = min(worst["POOL"], r_pool)
            wins["POOL"] += r_pool > r_spy
        wins["QQQ"] += r_qqq > r_spy

    print(f"\ncompounded across episodes: SPY {tot['SPY']-1:+.1%}   "
          f"QQQ {tot['QQQ']-1:+.1%}   POOL {tot['POOL']-1:+.1%} "
          f"(pool fillable {n_ok}/{len(episodes)})")
    print(f"beats SPY per episode:  QQQ {wins['QQQ']}/{len(episodes)}   "
          f"POOL {wins['POOL']}/{n_ok}")
    print(f"worst single episode:   SPY {worst['SPY']:+.1%}  QQQ {worst['QQQ']:+.1%}  "
          f"POOL {worst['POOL']:+.1%}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
