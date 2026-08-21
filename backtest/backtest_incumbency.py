"""
Does an incumbency preference (favor held names when the vol cap binds) cut
turnover and raise net-of-cost Sharpe?

Reuses the exact production pipeline from backtest_trend_following (2-week hold)
but sweeps _kelly_covariance_optimizer's new incumbency_bonus. The expensive
per-rebalance trend signals are computed ONCE, then the optimizer is replayed
cheaply for each bonus level (bonus only affects allocation, not the signals).

For each bonus: annualized one-way turnover, and Sharpe gross and net of a
per-turnover transaction cost, plus CAGR / maxDD. Idle cash earns SGOV.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.indicators import _kelly_covariance_optimizer, get_regime_scalar
from backtest.backtest_trend_following import (
    download_all, compute_asset_row, compute_regime,
    TREND_SYMS, REGIME_TICKERS, PORTFOLIO_VALUE, TEST_START, END,
    REBAL_EVERY, CASH_TICKER,
)

BONUSES = [0.0, 0.10, 0.25, 0.50, 1.00]
COST_BPS = [5.0, 10.0]           # one-way cost per unit turnover, for net Sharpe


def precompute(raw, close_all, test_dates):
    """Per rebalance date: (dt, trends, regime, regime_scalar). Expensive, once."""
    rebs = []
    t0 = time.time()
    for i, dt in enumerate(test_dates):
        if i % REBAL_EVERY:
            continue
        shared_t = raw.loc[:dt]
        regime = compute_regime(close_all.loc[:dt][REGIME_TICKERS + ["SPY"]].dropna(how="all"))
        rs = get_regime_scalar(regime)
        trends = []
        for sym in TREND_SYMS:
            try:
                df_sym = shared_t.xs(sym, level=1, axis=1).dropna()
            except KeyError:
                continue
            if not df_sym.empty:
                row = compute_asset_row(sym, df_sym)
                if row is not None:
                    trends.append(row)
        rebs.append((i, dt, trends, regime, rs))
        if (len(rebs)) % 10 == 0:
            print(f"  precomputed {len(rebs)} rebalances ({time.time()-t0:.0f}s)")
    return rebs


def run_book(rebs, raw, test_dates, syms, bonus):
    """Replay the optimizer with a given incumbency bonus. Returns a daily
    weights DataFrame and a per-rebalance one-way turnover Series."""
    W = pd.DataFrame(0.0, index=test_dates, columns=syms)
    turnover = pd.Series(0.0, index=test_dates)
    cur = {}                                   # sym -> weight fraction (held book)
    reb_by_i = {i: (dt, trends, regime, rs) for (i, dt, trends, regime, rs) in rebs}
    held = {}
    for i, dt in enumerate(test_dates):
        if i in reb_by_i:
            _, trends, regime, rs = reb_by_i[i]
            if trends:
                optimized, _ = _kelly_covariance_optimizer(
                    trends, raw.loc[:dt],
                    portfolio_value=PORTFOLIO_VALUE, max_single=0.25,
                    max_risk_contribution=0.35, min_portfolio_vol=0.08,
                    max_portfolio_vol=0.135, regime_scalar=rs, regime=regime,
                    current_weights=cur, incumbency_bonus=bonus,
                )
                new = {t["sym"]: t.get("weight_pct", 0.0) / 100.0
                       for t in optimized if t.get("weight_pct", 0.0) > 0}
                allsyms = set(new) | set(cur)
                turnover[dt] = sum(abs(new.get(s, 0.0) - cur.get(s, 0.0)) for s in allsyms)
                cur = new
                held = new
        for s, w in held.items():
            if s in W.columns:
                W.at[dt, s] = w
    return W, turnover


def perf(net):
    r = net.dropna()
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(252)
    return (eq.iloc[-1] ** (1 / yrs) - 1,                      # CAGR
            (r.mean() * 252) / vol if vol > 0 else np.nan,     # Sharpe
            (eq / eq.cummax() - 1).min())                      # maxDD


def main():
    t0 = time.time()
    raw = download_all()
    close_all = raw["Close"].ffill()
    if "Adj Close" in raw.columns.get_level_values(0) and CASH_TICKER in raw["Adj Close"].columns:
        close_all[CASH_TICKER] = raw["Adj Close"][CASH_TICKER].ffill()
    test_dates = close_all.index[(close_all.index >= TEST_START) & (close_all.index <= END)]
    syms = [s for s in TREND_SYMS if s in raw["Close"].columns]

    print("Precomputing rebalance signals once ...")
    rebs = precompute(raw, close_all, test_dates)
    ppy = 252 / REBAL_EVERY

    simple = raw["Close"][syms].pct_change()
    fwd = simple.reindex(test_dates).shift(-1)
    sgov_fwd = close_all[CASH_TICKER].pct_change().reindex(test_dates).shift(-1)

    print(f"\n{'bonus':>6s} {'turnover/yr':>11s} {'CAGR':>7s} {'grossSh':>8s} "
          f"{'net@5bp':>8s} {'net@10bp':>9s} {'maxDD':>7s}")
    print("-" * 62)
    for b in BONUSES:
        W, turn = run_book(rebs, raw, test_dates, syms, b)
        gross = (W * fwd).sum(axis=1)
        cash_w = (1.0 - W.sum(axis=1)).clip(lower=0.0)
        gross = gross + cash_w * sgov_fwd.fillna(0.0)
        ann_turn = turn[turn > 0].mean() * ppy if (turn > 0).any() else 0.0
        cagr, gsh, gdd = perf(gross.iloc[:-1])
        nets = []
        for c in COST_BPS:
            net = (gross - turn * (c / 1e4)).iloc[:-1]
            nets.append(perf(net)[1])
        print(f"{b:>6.2f} {ann_turn:>10.0%} {cagr:>7.1%} {gsh:>8.2f} "
              f"{nets[0]:>8.2f} {nets[1]:>9.2f} {gdd:>7.1%}")

    print(f"\n(turnover = mean one-way per rebalance x {ppy:.0f} rebalances/yr)")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
