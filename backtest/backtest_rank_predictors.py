"""
Can 'top sector / country / commodity' prediction be improved? The current
XGB models have: bfill() price lookahead, elite-target selection bias, hyper-
params tuned to one 80/20 split, and treat a RANKING problem as 11-way
classification. This tests whether simple, honestly-evaluable cross-sectional
rankers do the same job - predicting the 63d-forward winner - with no model
at all. (Ranking is the one edge shape that survived this session.)

Rankers (computed per universe from its own closes only):
  mom63, mom126     - medium-term momentum (classic cross-sectional)
  slope_r2_20/63    - OLS slope*r2 (the validated GSS signal, 2 windows)
  sharpe63          - vol-adjusted momentum
  prev_winner       - persistence baseline (last 63d top stays top)

Universes: SECTORS (11 XL*), COUNTRIES (10), COMMODITIES (futures with
enough history). Downloaded SEPARATELY (futures/equity calendar mix is the
known corruption bug). Eval: every 21d from 2007, forward 63d horizon
(overlapping x3 - noted). Metrics:
  IC      - mean cross-sectional Spearman(signal rank, fwd 63d return rank)
  top1 xs - mean excess of the top-ranked name vs universe median, 63d fwd
  top1 hit- % of dates the top pick beats the median
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import COUNTRIES, COMMODITIES

SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
DATA_START, EVAL_START, END = "2004-01-01", "2007-01-01", "2026-08-21"
H, STRIDE = 63, 21


def roll_sr(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(p > 0, p, np.nan))
    if n < win or sliding_window_view is None:
        return sl, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn; pr = s[:, None] * x[None, :] + (ym - s * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    sl[win - 1:] = s * 1000; r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return sl, r2


def build_signals(close):
    """close: DataFrame (dates x names). Returns dict name->DataFrame of signal."""
    sig = {}
    sig["mom63"] = close.pct_change(63)
    sig["mom126"] = close.pct_change(126)
    vol63 = close.pct_change().rolling(63).std()
    sig["sharpe63"] = close.pct_change(63) / (vol63 * np.sqrt(63) + 1e-9)
    for win, name in [(20, "slope_r2_20"), (63, "slope_r2_63")]:
        out = {}
        for c in close.columns:
            s, r = roll_sr(close[c].values, win)
            out[c] = s * r
        sig[name] = pd.DataFrame(out, index=close.index)
    sig["prev_winner"] = close.pct_change(63)          # same as mom63 rank -> baseline is its lag
    return sig


def evaluate(close, label):
    close = close.dropna(how="all")
    fwd = close.pct_change(H).shift(-H)                # forward 63d return
    sig = build_signals(close)
    dates = close.index[(close.index >= EVAL_START)]
    dates = dates[::STRIDE]
    rows = []
    for name, S in sig.items():
        ics = []; top_xs = []; top_hit = []
        for d in dates:
            if d not in S.index or d not in fwd.index:
                continue
            s_row = S.loc[d]; f_row = fwd.loc[d]
            m = s_row.notna() & f_row.notna()
            if m.sum() < 5:
                continue
            s_v = s_row[m]; f_v = f_row[m]
            if name == "prev_winner":
                # baseline: rank by momentum measured 63d AGO (does leadership persist?)
                d_idx = S.index.get_loc(d)
                if d_idx < 63:
                    continue
                s_prev = S.iloc[d_idx - 63][m]
                if s_prev.isna().any():
                    continue
                s_v = s_prev
            ic = s_v.rank().corr(f_v.rank())
            ics.append(ic)
            top = s_v.idxmax()
            xs = f_v[top] - f_v.median()
            top_xs.append(xs); top_hit.append(xs > 0)
        if ics:
            rows.append((name, np.mean(ics), np.mean(top_xs) * 100, np.mean(top_hit) * 100, len(ics)))
    print(f"== {label} ({close.shape[1]} names) ==")
    print(f"  {'ranker':>13s} {'IC':>7s} {'top1 xs%':>9s} {'top1 hit%':>9s} {'n':>5s}")
    for name, ic, xs, hit, n in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:>13s} {ic:>7.3f} {xs:>8.2f}% {hit:>8.0f}% {n:>5d}")
    print()


def main():
    t0 = time.time()
    print("Downloading sectors ...")
    sec = yf.download(SECTORS + ["SPY"], start=DATA_START, end=END, progress=False,
                      auto_adjust=True)["Close"][SECTORS]
    print("Downloading countries ...")
    cty = yf.download(list(COUNTRIES.keys()), start=DATA_START, end=END, progress=False,
                      auto_adjust=True)["Close"]
    print("Downloading commodities (separate - futures calendar) ...")
    cmd_all = yf.download(list(COMMODITIES.keys()), start=DATA_START, end=END, progress=False,
                          auto_adjust=True)["Close"]
    cmd = cmd_all[[c for c in cmd_all.columns if cmd_all[c].notna().sum() > 2000]]
    print(f"commodities with history: {list(cmd.columns)}\n")

    evaluate(sec, "SECTORS")
    evaluate(cty, "COUNTRIES")
    evaluate(cmd, "COMMODITIES")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
