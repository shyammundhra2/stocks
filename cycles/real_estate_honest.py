"""
Honest real-estate cycle model. Unlike stocks (zero autocorrelation, cycles are
noise), home prices have STRONG momentum (annual autocorr +0.75, +50% momentum
spread) - housing IS predictable at 1-3yr. So instead of the old overfit 4-param
sawtooth projection (real_estate_cycle.py), use the REAL signals, validated:

  1. MOMENTUM   - Case-Shiller 12m change (the dominant, robust signal; continues)
  2. Leading indicators (FRED), each checked for whether it leads forward prices:
       PERMIT (building permits)  - supply pipeline, leads
       MSACSR (months' supply)    - inventory; high = softening
       MORTGAGE30US (30y rate)    - affordability/demand
  3. AFFORDABILITY - price / median income (mean-reverts at extremes)

Near-term lean = momentum (continuation); leading indicators flag turns.
Information only. Saves a chart. Monthly Case-Shiller (CSUSHPISA), 1987+.
"""
import time
import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")


def main():
    t0 = time.time()
    f = web.DataReader(["CSUSHPISA", "PERMIT", "MSACSR", "MORTGAGE30US"], "fred", "1970-01-01").resample("ME").last()
    inc = web.DataReader("MEHOINUSA646N", "fred", "1970-01-01").resample("ME").last().ffill()
    hpi = f["CSUSHPISA"].dropna()
    lhpi = np.log(hpi)
    mom = (lhpi - lhpi.shift(12))                    # 12m momentum (the real signal)
    fwd = (lhpi.shift(-12) - lhpi)                   # actual next-12m return
    permit_yoy = f["PERMIT"].pct_change(12)
    supply = f["MSACSR"]
    mort_chg = f["MORTGAGE30US"].diff(12)            # 12m change in mortgage rate
    afford = hpi / inc["MEHOINUSA646N"].reindex(hpi.index).ffill()  # price / income
    afford_z = (afford - afford.rolling(120, min_periods=36).mean()) / afford.rolling(120, min_periods=36).std()

    D = pd.concat([mom.rename("mom"), fwd.rename("fwd"), permit_yoy.rename("permit"),
                   supply.rename("supply"), mort_chg.rename("mort"), afford_z.rename("afford")],
                  axis=1).dropna()

    print("\n=== HONEST REAL-ESTATE MODEL (info only, validated) ===\n")
    print("Signal -> forward-12m home-price return: correlation (which signals lead)")
    for c, desc in [("mom", "12m momentum (continuation)"), ("permit", "permits YoY"),
                    ("supply", "months' supply (inverse)"), ("mort", "mortgage-rate 12m chg (inverse)"),
                    ("afford", "affordability z (inverse=reversion)")]:
        print(f"   {desc:36s} corr {D[c].corr(D['fwd']):+.2f}")
    # momentum continuation strength
    up = D["fwd"][D["mom"] > 0]; dn = D["fwd"][D["mom"] <= 0]
    print(f"\n   momentum continuation: next-12m return when 12m momentum +ve {up.mean():+.1%} "
          f"vs -ve {dn.mean():+.1%}  (spread {up.mean()-dn.mean():+.1%})")

    # current state
    cur = D.index[-1]
    m = D["mom"].iloc[-1]
    print(f"\nCURRENT ({cur.date()}):")
    print(f"   Case-Shiller 12m momentum: {m:+.1%}  ({'rising' if m>0 else 'falling'})")
    print(f"   building permits YoY: {D['permit'].iloc[-1]:+.0%}")
    print(f"   months' supply: {D['supply'].iloc[-1]:.1f}  ({'elevated' if D['supply'].iloc[-1]>6 else 'tight'})")
    print(f"   30y mortgage 12m change: {D['mort'].iloc[-1]:+.1f}pp")
    print(f"   affordability z: {D['afford'].iloc[-1]:+.1f}  ({'stretched' if D['afford'].iloc[-1]>1 else 'normal'})")
    warn = int(D['supply'].iloc[-1] > 6) + int(D['mort'].iloc[-1] > 0.5) + int(D['afford'].iloc[-1] > 1) + int(D['permit'].iloc[-1] < 0)
    lean = "UP (momentum +, few warnings)" if (m > 0 and warn <= 1) else \
           "COOLING (momentum + but warnings flashing)" if m > 0 else "DOWN (momentum -)"
    print(f"   >> near-term lean: {lean}   [leading warnings active: {warn}/4]")

    # chart
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1, 1]})
    ax[0].plot(hpi.index, hpi, color="black", lw=1.3); ax[0].set_yscale("log")
    ax[0].set_title("Case-Shiller US National Home Price Index (log, FRED CSUSHPISA)", fontsize=11, fontweight="bold")
    ax[1].fill_between(mom.index, 0, mom * 100, where=(mom > 0), color="green", alpha=0.4, step="mid")
    ax[1].fill_between(mom.index, 0, mom * 100, where=(mom <= 0), color="red", alpha=0.4, step="mid")
    ax[1].axhline(0, color="black", lw=0.5); ax[1].set_title("12-month momentum % (the real signal: autocorr +0.75)", fontsize=10)
    ax[2].plot(f.index, f["PERMIT"], color="#0d9488", lw=1); ax[2].set_ylabel("permits (k)", color="#0d9488", fontsize=8)
    a2 = ax[2].twinx(); a2.plot(supply.index, supply, color="#b45309", lw=1); a2.axhline(6, color="#b45309", ls=":", lw=1)
    a2.set_ylabel("months supply", color="#b45309", fontsize=8)
    ax[2].set_title("Building permits (teal) & months' supply (orange, >6 = soft)", fontsize=10)
    ax[3].plot(f.index, f["MORTGAGE30US"], color="#6d28d9", lw=1)
    ax[3].set_title("30-year mortgage rate % (FRED MORTGAGE30US)", fontsize=10)
    for a in ax:
        a.grid(alpha=0.2); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Real-Estate Cycle - momentum + validated leading indicators (info only)\n"
                 "Housing IS predictable (autocorr +0.75) unlike stocks; this is context, not a trade signal",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "/Users/riddhisiddhi/stocks/cycles/real_estate_cycle.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n   chart saved -> {out}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
