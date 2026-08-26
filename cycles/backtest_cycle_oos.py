"""
Vipaksa test of the cycle-forecasting method: does the projected dominant-cycle
have ANY out-of-sample power, computed CAUSALLY (only past data), and is the
detected 'cycle' stable enough to be real? Three checks:

1. OOS predictive power: at each month T (walk-forward), detect the dominant
   cycle (periodogram) and fit+project a sine using ONLY data <= T; does the
   projected 12m cycle direction predict the ACTUAL next-12m S&P return?
2. Cycle stability: how much does the 'dominant cycle length' jump around across
   T? A real cycle is stable; band-pass-filtered noise wanders.
3. Look-ahead: how much does the LATEST cyclical-deviation value revise between
   its real-time (causal) estimate and the hindsight (full-sample) estimate -
   i.e., is the actionable current reading trustworthy in real time?

Baseline to beat: the unconditional 12m up-rate (S&P is up most of the time).
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import periodogram


def dominant_period_years(dev, fs=12, lo=2, hi=15):
    f, p = periodogram(dev - dev.mean(), fs=fs)
    per = 1 / f[f > 0]; ps = p[f > 0]
    m = (per >= lo) & (per <= hi)
    if not m.any():
        return np.nan
    return per[m][np.argmax(ps[m])]


def fit_project(dev, period_m, ahead=24):
    x = np.arange(len(dev))
    w = 2 * np.pi / period_m
    A = np.c_[np.sin(w * x), np.cos(w * x), np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, dev, rcond=None)
    xf = np.arange(len(dev), len(dev) + ahead)
    Af = np.c_[np.sin(w * xf), np.cos(w * xf), np.ones_like(xf)]
    return Af @ coef


def main():
    t0 = time.time()
    df = yf.download("^GSPC", start="1960-01-01", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    m = np.log(df["Close"].resample("ME").last().ffill())
    # causal cyclical deviation: log price minus 120-month TRAILING mean
    dev_causal = (m - m.rolling(120).mean())
    fwd12 = m.shift(-12) - m            # actual next-12m log return

    idx = m.index
    start = idx.get_loc(m.dropna().index[0]) + 120 + 60   # need trailing mean + some history
    end = len(m) - 12
    periods, proj_dir, act_ret, act_dir = [], [], [], []
    for T in range(start, end, 1):
        d = dev_causal.iloc[:T + 1].dropna()
        if len(d) < 180:
            continue
        pk = dominant_period_years(d.values)
        if not np.isfinite(pk):
            continue
        proj = fit_project(d.values, pk * 12, ahead=12)
        pdir = np.sign(proj[-1] - proj[0])          # projected 12m cycle direction
        r = fwd12.iloc[T]
        if not np.isfinite(r):
            continue
        periods.append(pk); proj_dir.append(pdir); act_ret.append(r); act_dir.append(np.sign(r))
    periods = np.array(periods); proj_dir = np.array(proj_dir)
    act_ret = np.array(act_ret); act_dir = np.array(act_dir)

    print(f"\nWalk-forward cycle OOS test  ({len(periods)} monthly forecasts, "
          f"{idx[start].date()} -> {idx[end-1].date()})\n")
    print("1. OOS PREDICTIVE POWER (projected 12m cycle direction vs actual 12m S&P return)")
    valid = proj_dir != 0
    hit = np.mean(proj_dir[valid] == act_dir[valid])
    base = np.mean(act_dir > 0)
    corr = np.corrcoef(proj_dir[valid], act_ret[valid])[0, 1]
    # when cycle says UP vs DOWN
    up = act_ret[valid][proj_dir[valid] > 0]; dn = act_ret[valid][proj_dir[valid] < 0]
    print(f"   directional hit-rate {hit:.1%}   vs base up-rate {base:.1%}   corr(dir,ret) {corr:+.3f}")
    print(f"   avg 12m return when cycle says UP {up.mean():+.1%} (n={len(up)})  "
          f"vs DOWN {dn.mean():+.1%} (n={len(dn)})   spread {up.mean()-dn.mean():+.1%}")

    print("\n2. CYCLE STABILITY (detected dominant cycle length over time)")
    print(f"   mean {periods.mean():.1f}yr  std {periods.std():.1f}yr  "
          f"range {periods.min():.1f}-{periods.max():.1f}yr")
    # how often the detected period jumps > 2yr month-to-month
    jumps = np.mean(np.abs(np.diff(periods)) > 2.0)
    print(f"   share of months where detected period JUMPS >2yr: {jumps:.1%} "
          f"(a real cycle wouldn't jump)")

    print("\n3. LOOK-AHEAD (real-time vs hindsight cyclical-deviation)")
    dev_hind = (m - m.rolling(240, center=True, min_periods=60).mean())   # centered = uses future
    both = pd.concat([dev_causal.rename("causal"), dev_hind.rename("hindsight")], axis=1).dropna()
    rt_corr = both["causal"].corr(both["hindsight"])
    # sign disagreement (expansion vs contraction call flips)
    disagree = np.mean(np.sign(both["causal"]) != np.sign(both["hindsight"]))
    print(f"   corr(real-time, hindsight) {rt_corr:+.2f};  sign DISAGREES {disagree:.0%} of months")
    print(f"   -> the 'current' expansion/contraction reading flips {disagree:.0%} of the time once")
    print(f"      future data arrives; the live chart is not the chart you'd have seen.")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
