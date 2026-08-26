"""
Can Trendguard's info predict the ODDS of continuation (for sizing), even though
the ML binary classifier failed? "Size by odds" only needs a MONOTONIC feature ->
continuation relationship. Test each feature univariately: continuation rate (y)
and avg 21d forward return by feature quintile, on the pooled Trendguard entries.

A feature is useful for sizing IFF its top-vs-bottom-quintile spread in
continuation% (or fwd return) is large and monotonic. Flat = no sizing signal.
Contrast the one thing that IS predictable (vol -> for inverse-vol sizing).
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/tg_ohlc.parquet"
HOLD = 21


def rsi(s, k=14):
    d = s.diff(); up = d.clip(lower=0).rolling(k).mean(); dn = (-d.clip(upper=0)).rolling(k).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    t0 = time.time()
    raw = pd.read_parquet(CACHE)
    close = raw["Close"]; high = raw["High"]; low = raw["Low"]
    spy = close["SPY"]; spy_ret = spy.pct_change()
    names = [s for s in TREND_ASSETS if s in close.columns and close[s].notna().sum() > 400]
    rows = []
    for c in names:
        px = close[c]; hi = high[c]; lo = low[c]
        sma50 = px.rolling(50).mean(); sma200 = px.rolling(200).mean()
        high20 = hi.rolling(20).max().shift(1)
        tr = pd.concat([hi - lo, (hi - px.shift()).abs(), (lo - px.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean(); r = px.pct_change()
        beta = r.rolling(126).cov(spy_ret) / spy_ret.rolling(126).var(); resid = r - beta * spy_ret
        feat = pd.DataFrame({
            "M63": px / px.shift(63) - 1, "M126": px / px.shift(126) - 1, "M252": px / px.shift(252) - 1,
            "TrendStr": sma50 / sma200 - 1, "BO20": px / high20 - 1,
            "ATRpx": atr / px, "vol63": r.rolling(63).std() * np.sqrt(252),
            "RSI": rsi(px, 14), "rmom": resid.rolling(126).sum(), "rvol": resid.rolling(126).std(),
            "MAX21": r.rolling(21).max(),
        })
        tg = ((sma50 > sma200) & (px > high20) & (atr < 2 * atr.rolling(126).mean()) & (rsi(px, 14) < 70))
        surv = pd.Series(True, index=px.index)
        for k in range(1, HOLD + 1):
            surv &= (px.shift(-k) / sma50.shift(-k) > 0.99)
        feat["y"] = (surv & (px.shift(-HOLD) / px - 1 > 0)).astype(float)
        feat["fwd"] = px.shift(-HOLD) / px - 1
        feat["fwdvol"] = r.shift(-HOLD).rolling(HOLD).std().shift(-1) * np.sqrt(252)  # realized fwd vol proxy
        rows.append(feat[tg & feat.notna().all(axis=1)])
    panel = pd.concat(rows).dropna()
    print(f"\nTrendguard entries: {len(panel):,}   base continuation {panel['y'].mean():.1%}   "
          f"avg 21d ret {panel['fwd'].mean():+.2%}\n")

    fcols = ["M63", "M126", "M252", "TrendStr", "BO20", "ATRpx", "vol63", "RSI", "rmom", "rvol", "MAX21"]
    print("Univariate: continuation% and avg 21d return by feature QUINTILE (Q1 low -> Q5 high)")
    print(f"{'feature':9s} {'contin Q1->Q5':>28s}   {'Q5-Q1':>6s} | {'fwd-ret Q1->Q5':>30s}")
    print("-" * 95)
    for f in fcols:
        q = pd.qcut(panel[f], 5, labels=False, duplicates="drop")
        cy = panel.groupby(q)["y"].mean(); cf = panel.groupby(q)["fwd"].mean()
        cy_s = " ".join(f"{v:.0%}" for v in cy); cf_s = " ".join(f"{v:+.1%}" for v in cf)
        print(f"{f:9s} {cy_s:>28s}   {cy.iloc[-1]-cy.iloc[0]:>+6.0%} | {cf_s:>30s}")

    # the predictable contrast: does current vol predict FORWARD vol? (for sizing)
    print("\nContrast - what IS predictable (for sizing): current vol -> forward 21d vol")
    q = pd.qcut(panel["vol63"], 5, labels=False, duplicates="drop")
    fv = panel.groupby(q)["fwdvol"].mean()
    print("  fwd-vol by current-vol quintile:", " ".join(f"{v:.0%}" for v in fv),
          f"  (Q5-Q1 {fv.iloc[-1]-fv.iloc[0]:+.0%}, monotonic = predictable)")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
