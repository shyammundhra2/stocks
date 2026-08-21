"""
Head-to-head: should U-fit REPLACE trend strength (slope*R2) in sizing, or
just augment it? Build a fully-invested long book among BUY-eligible names,
rebalanced monthly (non-overlapping), 2007-26, and compare weighting schemes:

  ew            equal weight all eligible BUYs         (naive benchmark)
  current       weight ~ slope*R2                      (today's conviction)
  u_only        weight only the U-reversal BUYs (eq)   ("replace" - U as sizer)
  combine       weight ~ slope*R2 * (1 + BONUS if U)   ("augment")

Weights sum to 1 each period (isolates the weighting effect, not deployment;
if no eligible names / no U names for u_only, that period sits in cash = 0).
Reports annualized return / vol / Sharpe / maxDD and avg breadth.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from backtest.backtest_u_pattern import roll_slope_r2, is_U, PATH, R2_LO

START, END = "2007-01-01", "2026-08-21"
FWD = 21           # ~1 month hold / rebalance (matches the U edge horizon)
BONUS = 0.5        # combine: conviction multiplier for U-reversal BUYs


def u_fit(sp, rp):
    """Continuous [0,1] 'U-reversal fit' of the trailing (slope,R2) path:
    started as a downtrend (negative slope), broke through ~(0,0) (R2 collapse
    mid), and is turning up now. Meant to REPLACE the strength factor."""
    L = len(sp)
    if L < 15 or not (np.isfinite(sp).all() and np.isfinite(rp).all()):
        return 0.0
    a, b = L // 5, (4 * L) // 5
    down = np.clip(-sp[0] / 3.0, 0.0, 1.0)                    # how negative the start leg
    brk = np.clip((R2_LO - rp[a:b].min()) / R2_LO, 0.0, 1.0)  # how deep the mid R2 collapse
    up = 1.0 if sp[-1] > 0 else 0.0                           # must be turning up now
    return float(down * brk * up)


def perf(rets, ppy):
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return (np.nan,) * 4
    eq = np.cumprod(1 + r)
    yrs = len(r) / ppy
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ppy)
    sh = (r.mean() * ppy) / vol if vol > 0 else np.nan
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return cagr, vol, sh, dd


def wmean(fret, w):
    tot = sum(w.values())
    if tot <= 0:
        return 0.0
    return sum(fret[c] * w[c] for c in w) / tot


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers {START}..{END} ...")
    raw = yf.download(tickers, start=START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index
    n = len(idx)
    print(f"Universe with >260 bars: {len(present)}\n")

    print("Precomputing slope/R2/BUY/U/u_fit per asset ...")
    slope_df, r2_df, buy_df, u_df, uf_df = ({} for _ in range(5))
    for c in present:
        s = close[c]
        v = s.dropna()
        sl, r2 = roll_slope_r2(v.values)
        sl = pd.Series(sl, index=v.index).reindex(idx).values
        r2 = pd.Series(r2, index=v.index).reindex(idx).values
        ma50 = s.rolling(50).mean().values
        ma200 = s.rolling(200).mean().values
        pv = s.values
        buy = (pv > ma200) & (pv > ma50) & (sl > 0) & (r2 > 0.6)
        uflag = np.zeros(n, bool)
        ufit = np.zeros(n)
        for j in range(PATH, n):
            sp, rp = sl[j - PATH + 1:j + 1], r2[j - PATH + 1:j + 1]
            if np.isfinite(sp).all() and np.isfinite(rp).all():
                uflag[j] = is_U(sp, rp)
                ufit[j] = u_fit(sp, rp)
        slope_df[c], r2_df[c], buy_df[c], u_df[c], uf_df[c] = sl, r2, np.nan_to_num(buy), uflag, ufit

    # References + a FLOOR sweep of  slope*r2*(floor + (1-floor)*u_fit):
    #   floor=0.00 -> pure u_fit (concentrates);  floor=1.00 -> slope*r2 (no tilt)
    FLOORS = [0.00, 0.15, 0.30, 0.50, 0.70, 1.00]
    schemes = ["ew", "current"] + [f"fl{f:.2f}" for f in FLOORS]
    rets = {s: [] for s in schemes}
    breadth = {s: [] for s in schemes}
    pv = {c: close[c].values for c in present}

    for i in range(max(220, PATH), n - FWD, FWD):
        elig = [c for c in present if buy_df[c][i] and np.isfinite(pv[c][i]) and np.isfinite(pv[c][i + FWD])]
        if not elig:
            for s in schemes:
                rets[s].append(0.0)
            continue
        fret = {c: pv[c][i + FWD] / pv[c][i] - 1.0 for c in elig}
        # 'current' = slope*r2*strength, strength = tanh(slope*r2/8) (hurst/p_stop
        # discount omitted - too slow per day - so it's a degraded proxy).
        sr = {c: max(slope_df[c][i] * r2_df[c][i], 0.0) for c in elig}
        rets["ew"].append(wmean(fret, {c: 1.0 for c in elig})); breadth["ew"].append(len(elig))
        rets["current"].append(wmean(fret, {c: sr[c] * np.tanh(sr[c] / 8.0) for c in elig}))
        breadth["current"].append(len(elig))
        for f in FLOORS:
            w = {c: sr[c] * (f + (1.0 - f) * uf_df[c][i]) for c in elig}
            tot = sum(w.values())
            rets[f"fl{f:.2f}"].append(wmean(fret, w) if tot > 0 else 0.0)
            breadth[f"fl{f:.2f}"].append(sum(1 for c in elig if w[c] > 0))

    ppy = 252 / FWD
    npd = len(rets["ew"])
    print(f"Periods: {npd} ({idx[max(220, PATH)].date()} -> {idx[max(220, PATH) + (npd-1)*FWD].date()}), "
          f"{ppy:.1f}/yr\n")
    print(f"{'scheme':10s} {'CAGR':>7s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s} {'avg names':>9s}")
    print("-" * 54)
    for s in schemes:
        cagr, vol, sh, dd = perf(rets[s], ppy)
        nm = np.mean(breadth[s]) if breadth[s] else float("nan")
        print(f"{s:10s} {cagr:>7.1%} {vol:>7.1%} {sh:>7.2f} {dd:>8.1%} {nm:>9.1f}")
    print("\nfl0.00 = pure u_fit (concentrates) ... fl1.00 = slope*r2 (no U tilt).")
    print("Sharpe vs breadth vs drawdown across the floor - pick your point.")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
