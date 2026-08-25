"""
Find uncorrelated liquid ETFs to add for breadth, then test if they improve
the book. Breadth (not signal or risk limits) has been the binding constraint
all session - the book deploys ~56%, realizes ~7% vol, can't reach its 13.5%
cap. Hypothesis: adding genuinely uncorrelated liquid ETFs lets it deploy more
-> more return at similar Sharpe.

Step 1 - correlation screen: for each candidate, avg |corr| of daily returns
         to the existing 36-name universe (last 5y). Low = real diversifier.
Step 2 - backtest: current universe vs current + full-history low-corr adds,
         shipped adaptive config, deployment/Sharpe/CAGR/DD, 2007-26 + dev.
         Defensive adds (bonds/dollar) join the gate-exempt set so they can be
         held in risk-off (the point of a crisis diversifier).

Candidates (liquid, distinct drivers): curve/credit bonds, dollar, more
countries, ag. Managed-futures ETFs (DBMF/KMLM) are noted but too new for the
full backtest (report their correlation only).
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
from macro.constants import TREND_ASSETS, DEFENSIVE_ASSETS

DATA_START, TRADE_START, END = "2004-01-01", "2007-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY = 20, 0.40, 0.35, 15.0
MOM_CAP, REV_CAP, COST, BUF, VCAP = 0.075, 0.05, 5.0, 0.03, 0.135

BASE = list(TREND_ASSETS.keys())
# candidate breadth adds - liquid, distinct return drivers
CANDIDATES = ["IEF", "SHY", "LQD", "HYG", "EMB", "TIP",           # curve + credit bonds
              "UUP",                                              # US dollar
              "FXI", "EWG", "EWU", "EFA", "EWT", "EWW", "EWA", "EWC", "EWH",  # more countries
              "DBA", "PDBC",                                      # ag / broad commodity
              "DBMF", "KMLM"]                                     # managed futures (new)
# adds that are defensive (held through risk-off): bonds + dollar + mgd futures
NEW_DEFENSIVE = {"IEF", "SHY", "LQD", "HYG", "EMB", "TIP", "UUP", "DBMF", "KMLM"}


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


def rsi(p, k):
    d = np.diff(p, prepend=p[0]); up = pd.Series(np.where(d > 0, d, 0.)); dn = pd.Series(np.where(d < 0, -d, 0.))
    return (100 - 100 / (1 + up.rolling(k).mean() / dn.rolling(k).mean().replace(0, np.nan))).values


def er(p, L):
    s = pd.Series(p); return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def perf(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15 or d.std() == 0:
        return (np.nan,) * 4
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    dn_ = d[d < 0]
    so = (d.mean() * 252) / (np.sqrt((dn_ ** 2).mean()) * np.sqrt(252)) if len(dn_) > 4 else np.nan
    return sh, so, cg, dd


def run_book(close, universe, defensive, si, idx, n, bil, spy, spy200, rd, rv21, rv_med):
    fin = np.isfinite
    present = [t for t in universe if t in close.columns and close[t].notna().sum() > 260]
    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].reindex(idx).values
        m50[c] = close[c].rolling(50).mean().reindex(idx).values
        m200[c] = close[c].rolling(200).mean().reindex(idx).values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().reindex(idx).values

    def scalar(j):
        if not fin(rv21[j]) or not fin(rv_med[j]) or rv21[j] <= 0:
            return 1.0
        return float(np.clip(rv_med[j] / rv21[j], 0.25, 1.0))

    def route(c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return None
        a = pv[c][i] > m200[c][i]
        if e >= ER_HI and a and pv[c][i] > m50[c][i] and SL[c][i] > 0:
            return "MOM"
        if e <= ER_LO and a and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return "REV"
        return None

    w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
    dep = []; nn = []
    for j in range(si, n - 1):
        if j in rd:
            if fin(spy200[j]):
                th = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                riskoff = spy[j] < th
            rs = scalar(j); sel = {}; kind = {}
            for c in present:
                if riskoff and c not in defensive:
                    continue
                sg = route(c, j)
                if sg and fin(VOL[c][j]) and VOL[c][j] > 0 and fin(pv[c][j]):
                    sel[c] = 1.0 / VOL[c][j]; kind[c] = sg
            tot = sum(sel.values()); tg = {}
            if tot > 0:
                for c in sel:
                    tg[c] = min(sel[c] / tot, REV_CAP if kind[c] == "REV" else MOM_CAP)
                s2 = sum(tg.values())
                if s2 > 0:
                    tg = {c: v / s2 for c, v in tg.items()}
                    tg = {c: min(v, REV_CAP if kind[c] == "REV" else MOM_CAP) for c, v in tg.items()}
                tg = {c: (v * rs if kind[c] == "MOM" else v) for c, v in tg.items()}
            full = {c: tg.get(c, 0.) for c in present}
            turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
        dep.append(sum(w.values())); nn.append(sum(1 for v in w.values() if v > 0.001))
        s = 0.
        for c in present:
            if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
        ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
    sc = slice(si + 1, n)
    return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values, np.mean(dep), np.mean(nn)


def main():
    t0 = time.time()
    dl = sorted(set(BASE + CANDIDATES + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx)

    # --- Step 1: correlation screen (last 5y daily returns) ---
    rets = close.pct_change().tail(1260)
    base_present = [t for t in BASE if t in close.columns and rets[t].notna().sum() > 200]
    print("\nCORRELATION SCREEN (avg |corr| to existing 36-name book, last 5y):")
    print(f"  {'candidate':>10s} {'avg|corr|':>9s} {'hist from':>10s}")
    screen = []
    for cand in CANDIDATES:
        if cand not in close.columns or rets[cand].notna().sum() < 200:
            print(f"  {cand:>10s} {'no data':>9s}")
            continue
        cors = [abs(rets[cand].corr(rets[b])) for b in base_present if b != cand]
        avg = np.nanmean(cors)
        first = close[cand].dropna().index.min()
        screen.append((cand, avg, first))
        print(f"  {cand:>10s} {avg:>9.2f} {str(first.date()):>10s}")

    # full-history low-corr adds (avg|corr| < 0.5 AND history back to <=2008)
    adds = [c for c, avg, first in screen
            if avg < 0.55 and first <= pd.Timestamp("2008-06-01")]
    print(f"\nFull-history low-corr adds (|corr|<0.55, data pre-2008): {adds}")

    # --- Step 2: backtest current vs expanded ---
    fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values
    spy = close["SPY"].reindex(idx).values; spy200 = close["SPY"].rolling(200).mean().reindex(idx).values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}
    rv21 = close["SPY"].pct_change().rolling(21).std().values * np.sqrt(252)
    rv_med = pd.Series(rv21).rolling(252, min_periods=60).median().values

    configs = {
        "current (36)": (BASE, DEFENSIVE_ASSETS),
        "expanded": (BASE + adds, DEFENSIVE_ASSETS | (NEW_DEFENSIVE & set(adds))),
    }
    results = {}
    for name, (uni, deff) in configs.items():
        results[name] = run_book(close, uni, deff, si, idx, n, bil, spy, spy200, rd, rv21, rv_med)

    print("\n" + "=" * 60)
    for name in configs:
        _, _, dep, nm = results[name]
        print(f"{name:>14s}: avg deployment {dep:.0%}, avg {nm:.1f} names, universe {len(configs[name][0])}")
    print()
    for lab, lo, hi in [("2008 GFC", "2007-10-01", "2009-06-30"),
                        ("2022 bear", "2022-01-01", "2022-12-31"),
                        ("FULL 2007-26", TRADE_START, END), ("DEV 2020-26", "2020-01-01", END)]:
        print(f"== {lab} ==   {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'maxDD':>7s}")
        for name in configs:
            r, dts, _, _ = results[name]; sh, so, cg, dd = perf(r, dts, lo, hi)
            print(f"  {name:14s} {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
