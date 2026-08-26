"""
Cross-sectional single-stock edge hunt on the S&P 500. Unlike ETFs (broad sort
fails - shared beta, no dispersion), single names have idiosyncratic dispersion,
so cross-sectional anomalies can be real. Test the ones that survive OOS in the
literature, walk-forward (signals use only past data, no fitting), monthly
rebalance, net of costs.

Signals (cross-sectional rank each month-end):
  mom12_1  12-month return skipping last month   (Jegadeesh-Titman momentum)
  mom6_1   6-month skip 1
  rev1m    NEGATIVE last-month return            (short-term reversal)
  lowvol   NEGATIVE 126d realized vol            (low-volatility anomaly)
  hi52     price / 252d high                     (George-Hwang)

For each: LONG top quintile (equal-wt, hold 1mo) and LONG-SHORT (top - bottom
quintile). Benchmarks: EQUAL-WEIGHT universe (the FAIR benchmark - absorbs the
survivorship-bias inflation the current-membership universe carries) and SPY.

HONEST CAVEATS: (1) survivorship bias - current S&P 500 only; inflates absolute
returns but affects all signals equally, so the RELATIVE read (which signal wins,
and vs EW-universe) is what's trustworthy. (2) 10bps one-way costs (single stocks).
A signal only 'survives' if it beats EW-universe risk-adjusted in MULTIPLE
sub-periods.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
START, END = "2004-01-01", "2026-08-21"
COST = 10.0          # bps one-way
QUINTILE = 0.20


def load_universe():
    syms = pd.read_csv("/Users/riddhisiddhi/stocks/stockselection/sp500.csv")["Symbol"].tolist()
    syms = [s.replace(".", "-") for s in syms]        # yfinance uses BRK-B etc.
    if os.path.exists(CACHE):
        px = pd.read_parquet(CACHE)
        print(f"loaded cache {px.shape}")
        return px
    print(f"downloading {len(syms)} tickers ...")
    raw = yf.download(syms + ["SPY"], start=START, end=END, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(how="all", axis=1)
    try:
        close.to_parquet(CACHE)
    except Exception as e:
        print("cache write failed:", e)
    print(f"downloaded {close.shape}")
    return close


def perf(r, lo, hi):
    d = r[(r.index >= lo) & (r.index <= hi)].dropna().values
    if len(d) < 12 or d.std() == 0:
        return (np.nan,) * 3
    sh = d.mean() / d.std() * np.sqrt(12)
    eq = np.cumprod(1 + d); cg = eq[-1] ** (12 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    close = load_universe()
    spy = close["SPY"] if "SPY" in close.columns else None
    stocks = [c for c in close.columns if c != "SPY"]
    close = close[stocks]

    # month-end rebalance dates
    me = close.resample("ME").last().index
    me = [d for d in me if d in close.index or True]
    # map each month-end to the last available trading day <= it
    idx = close.index
    reb = []
    for d in me:
        pos = idx.searchsorted(d, side="right") - 1
        if pos > 260:
            reb.append(idx[pos])
    reb = sorted(set(reb))

    # precompute signal frames (daily), sampled at reb dates
    logret = np.log(close).diff()
    sig_daily = {
        "mom12_1": close.shift(21) / close.shift(252) - 1.0,
        "mom6_1":  close.shift(21) / close.shift(126) - 1.0,
        "rev1m":   -(close / close.shift(21) - 1.0),
        "lowvol":  -(logret.rolling(126).std()),
        "hi52":    close / close.rolling(252).max(),
    }

    monret = {}   # signal -> Series of monthly long-only & LS returns
    results = {}
    for name, sd in sig_daily.items():
        lo_ret, ls_ret, dates = [], [], []
        prev_long, prev_short = set(), set()
        for k in range(len(reb) - 1):
            d0, d1 = reb[k], reb[k + 1]
            s = sd.loc[d0]
            fwd = close.loc[d1] / close.loc[d0] - 1.0
            valid = s.notna() & fwd.notna() & np.isfinite(s) & np.isfinite(fwd)
            s, fwd = s[valid], fwd[valid]
            if len(s) < 50:
                continue
            n_q = max(int(len(s) * QUINTILE), 5)
            order = s.sort_values()
            longs = set(order.index[-n_q:]); shorts = set(order.index[:n_q])
            r_long = fwd[list(longs)].mean()
            r_short = fwd[list(shorts)].mean()
            # turnover cost
            to_long = len(longs ^ prev_long) / max(len(longs) + len(prev_long), 1)
            to_ls = to_long + len(shorts ^ prev_short) / max(len(shorts) + len(prev_short), 1)
            lo_ret.append(r_long - to_long * COST / 1e4)
            ls_ret.append((r_long - r_short) - to_ls * COST / 1e4)
            dates.append(d1)
            prev_long, prev_short = longs, shorts
        monret[name + "_LONG"] = pd.Series(lo_ret, index=dates)
        monret[name + "_LS"] = pd.Series(ls_ret, index=dates)

    # benchmarks: equal-weight universe (all valid names), SPY
    ew, spm, dts = [], [], []
    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        fwd = close.loc[d1] / close.loc[d0] - 1.0
        fwd = fwd[fwd.notna() & np.isfinite(fwd)]
        if len(fwd) < 50:
            continue
        ew.append(fwd.mean()); dts.append(d1)
        spm.append((spy.loc[d1] / spy.loc[d0] - 1.0) if spy is not None else np.nan)
    monret["EW-universe"] = pd.Series(ew, index=dts)
    monret["SPY"] = pd.Series(spm, index=dts)

    wins = [("2005-2012", "2005-01-01", "2012-12-31"),
            ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"),
            ("FULL", "2005-01-01", "2026-08-21")]
    order = ["mom12_1_LONG", "mom6_1_LONG", "rev1m_LONG", "lowvol_LONG", "hi52_LONG",
             "EW-universe", "SPY",
             "mom12_1_LS", "mom6_1_LS", "rev1m_LS", "lowvol_LS", "hi52_LS"]
    print("\nCross-sectional S&P 500, monthly, top/bottom quintile, net@10bps")
    print("Sharpe / CAGR / maxDD per sub-period\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==       {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for nm in order:
            if nm not in monret:
                continue
            sh, cg, dd = perf(monret[nm], pd.Timestamp(lo), pd.Timestamp(hi))
            tag = "  <-LS" if nm.endswith("_LS") else ""
            print(f"  {nm:16s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}{tag}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
