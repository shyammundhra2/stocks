"""
Is there a turn-of-month effect, and if so what does it imply for SELL timing?

Classic finding (Lakonishok & Smidt 1988, Ariel 1987, McConnell & Xu 2008):
equity returns concentrate in the last ~1-2 and first ~3 trading days of the
month; the rest of the month is close to flat on average. Tests this directly
on SPY and the TREND_ASSETS universe (equal-weight, the sleeve's own book),
bucketed by trading-day offset from month-end, with the actionable question
for a SELLER: which offset carries the excess return, so you know whether
"last 3 days of the month" is a day to SELL INTO or a day to AVOID selling on
(because the rally hasn't happened yet / is still happening).

PRE-COMMITTED BAR: an offset bucket is credited only if its mean daily return
is outside the whole-sample daily mean by >1.5x the daily std's standard error
AND the sign is consistent in both halves of the sample.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

YEARS = 25
WINDOW = range(-6, 7)   # trading days relative to month-end (0 = last trading day of month)


def offset_table(ret):
    """ret: daily return series.

    Two offsets, because they answer different questions and my first version
    only computed one:
      offset     = trading days SINCE the prior month-end (0=month-end, +1=
                   first day of new month). This is what "turn of month" means
                   in the literature - it can only be >=0 by construction.
      days_to_me = trading days UNTIL the NEXT month-end (0=month-end, -1=day
                   before, -2=two days before...). THIS is what "last 3 days of
                   the month" literally asks: it needs the backward distance to
                   an upcoming month-end, which forward-filling the prior
                   month-end position cannot produce.
    """
    idx = ret.index
    is_month_end = idx.to_series().groupby(idx.to_period("M")).transform("max") == idx
    pos = pd.Series(np.arange(len(idx)), index=idx)

    me_pos_prior = pd.Series(np.where(is_month_end, pos, np.nan), index=idx).ffill()
    offset = pd.Series(np.nan, index=idx)
    offset[me_pos_prior.notna()] = (pos - me_pos_prior)[me_pos_prior.notna()]

    me_pos_next = pd.Series(np.where(is_month_end, pos, np.nan), index=idx).bfill()
    days_to_me = pd.Series(np.nan, index=idx)
    days_to_me[me_pos_next.notna()] = (pos - me_pos_next)[me_pos_next.notna()]

    return pd.DataFrame({"ret": ret, "offset": offset.clip(-15, 15),
                         "days_to_me": days_to_me.clip(-15, 15)})


def main():
    t0 = time.time()
    syms = sorted(set(TREND_ASSETS) | {"SPY"})
    px = yf.download(syms, period=f"{YEARS}y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")

    spy = px["SPY"].dropna()
    spy_ret = spy.pct_change().dropna()
    assets = [s for s in TREND_ASSETS if s in px.columns]
    ew_ret = px[assets].pct_change().mean(axis=1).dropna()   # equal-weight book proxy

    def scan(d, col, label2, whole_mean, half):
        print(f"\n--- {label2} ---")
        print(f"{'offset':>7} {'n':>6} {'mean ret':>10} {'t-stat':>7} {'h1 mean':>9} "
              f"{'h2 mean':>9} {'sign stable':>12}")
        credited = []
        for off in WINDOW:
            b = d[d[col] == float(off)]
            if len(b) < 50:
                continue
            m = b.ret.mean()
            se = b.ret.std() / np.sqrt(len(b))
            t = (m - whole_mean) / se if se > 0 else np.nan
            b1, b2 = b[b.index <= half], b[b.index > half]
            m1 = b1.ret.mean() if len(b1) > 20 else np.nan
            m2 = b2.ret.mean() if len(b2) > 20 else np.nan
            stable = (np.sign(m1 - whole_mean) == np.sign(m2 - whole_mean) == np.sign(m - whole_mean)
                      if np.isfinite(m1) and np.isfinite(m2) else False)
            cred = abs(t) > 1.5 and stable
            if cred:
                credited.append(off)
            tag = " <-MONTH-END" if off == 0 else ""
            print(f"{off:>+7} {len(b):>6} {m:>+10.4%} {t:>+7.2f} {m1:>+9.4%} {m2:>+9.4%} "
                  f"{'y' if stable else 'n':>12}{tag}  {'<-- CREDITED' if cred else ''}")
        print(f"credited: {credited or 'none'}")
        return credited

    for label, ret in [("SPY", spy_ret), ("TREND_ASSETS equal-weight", ew_ret)]:
        d = offset_table(ret).dropna(subset=["ret"])
        half = d.index[len(d) // 2]
        print(f"\n=== {label}, {d.index[0].date()}..{d.index[-1].date()} "
              f"({len(d):,} trading days) ===")
        whole_mean = d.ret.mean()
        print(f"whole-sample mean daily return: {whole_mean:+.4%}")
        scan(d.dropna(subset=["offset"]), "offset",
             "SINCE prior month-end (classic TOM framing; 0=month-end, +1=new month)",
             whole_mean, half)
        scan(d.dropna(subset=["days_to_me"]), "days_to_me",
             "TO the upcoming month-end (what \"last 3 days of the month\" means; "
             "-3/-2/-1=3rd/2nd/last day before month-end, 0=month-end itself)",
             whole_mean, half)

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
