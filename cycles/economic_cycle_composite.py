"""
Honest, information-only economic-cycle dashboard (composite leading indicators).
Replaces the self-referential S&P sine forecast (no recession lead - see
backtest_cycle_oos.py) with three VALIDATED, CAUSAL signals, each checked against
the real NBER recessions:

  1. Yield curve  (FRED T10Y3M)  - LEADS ~13mo (inverts before recessions)
  2. Credit spread (FRED BAA10Y) - widens going into stress
  3. Sahm rule    (FRED UNRATE)  - 3mo-avg unemployment minus trailing-12mo low
                                    >= 0.5  = real-time recession trigger

Progression = early warning (curve) -> mid (credit) -> confirmation (Sahm).
Descriptive context ONLY - not a trade signal (the live 200DMA gate + VIX
throttle REACT to downturns; this doesn't try to time them). Saves a chart
labeled as such.
"""
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

RECESSIONS = [("1973-11", "1975-03"), ("1980-01", "1980-07"), ("1981-07", "1982-11"),
              ("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06"),
              ("2020-02", "2020-04")]


def main():
    t0 = time.time()
    px = yf.download("^GSPC", start="1960-01-01", progress=False, auto_adjust=True)["Close"]
    m = np.log(pd.Series(np.asarray(px).ravel(), index=px.index).resample("ME").last().ffill())
    fred = web.DataReader(["T10Y3M", "BAA10Y", "UNRATE"], "fred", "1960-01-01").resample("ME").last()
    curve = fred["T10Y3M"]; credit = fred["BAA10Y"]; ur = fred["UNRATE"]
    sahm = ur.rolling(3).mean() - ur.rolling(12).min()          # Sahm rule
    cyc = (m - m.rolling(120).mean())                            # causal cycle position

    df = pd.concat([m.rename("lp"), cyc.rename("cyc"), curve.rename("curve"),
                    credit.rename("credit"), sahm.rename("sahm")], axis=1)
    rec = pd.Series(0, index=df.index)
    for a, b in RECESSIONS:
        rec.loc[a:b] = 1
    df["rec"] = rec
    # recession starts within next 12mo (target for lead validation)
    y = pd.Series(0, index=df.index)
    for a, _ in RECESSIONS:
        d = pd.Timestamp(a); y.loc[(y.index >= d - pd.DateOffset(months=12)) & (y.index < d)] = 1
    df["y12"] = y

    print("\n=== COMPOSITE ECONOMIC-CYCLE DASHBOARD (info only, causal, validated) ===\n")
    # per-indicator validation
    def lead_check(sig, trig, label, window=24):
        n = 0; leads = []
        for a, _ in RECESSIONS:
            d = pd.Timestamp(a)
            w = sig[(sig.index >= d - pd.DateOffset(months=window)) & (sig.index < d)].dropna()
            if len(w) and trig(w).any():
                n += 1; first = w[trig(w)].index[0]; leads.append((d - first).days / 30)
        cov = len([1 for a, _ in RECESSIONS if pd.Timestamp(a) >= sig.dropna().index.min()])
        med = np.median(leads) if leads else np.nan
        print(f"   {label:22s} preceded {n}/{cov} recessions (<{window}mo); median lead {med:.0f}mo")
    lead_check(curve, lambda w: w < 0, "Yield curve inverts")
    lead_check(credit, lambda w: w > w.expanding().quantile(0.80), "Credit spread > 80pct")
    lead_check(sahm, lambda w: w >= 0.5, "Sahm >= 0.5", window=6)

    # transparent composite: fraction of the 3 warnings active, calibrated by base rates
    d2 = df.dropna(subset=["curve", "credit", "sahm"])
    warn = pd.DataFrame({
        "curve": (d2["curve"] < 0).astype(int),
        "credit": (d2["credit"] > d2["credit"].rolling(120, min_periods=36).quantile(0.80)).astype(int),
        "sahm": (d2["sahm"] >= 0.5).astype(int),
    }, index=d2.index)
    comp = warn.mean(axis=1)                     # 0..1 share of signals flashing
    print(f"\n   composite 'signals active' vs recession-start-within-12mo:")
    for lvl in [0.0, 1 / 3, 2 / 3, 1.0]:
        sub = df["y12"].reindex(comp.index)[np.isclose(comp, lvl)]
        if len(sub):
            print(f"     {int(lvl*3)}/3 active (n={len(sub):>4d}): P(recession<=12mo) {sub.mean():.0%}")

    # current
    cur = d2.index[-1]
    print(f"\nCURRENT ({cur.date()}):")
    print(f"   S&P cycle position: {df['cyc'].dropna().iloc[-1]:+.0%} vs 10y trend "
          f"({'above/late' if df['cyc'].dropna().iloc[-1]>0 else 'below/early'})")
    print(f"   yield curve 10y-3m: {d2['curve'].iloc[-1]:+.2f}  ({'INVERTED' if d2['curve'].iloc[-1]<0 else 'normal'})")
    print(f"   credit spread Baa-10y: {d2['credit'].iloc[-1]:.2f}  (warn={'Y' if warn['credit'].iloc[-1] else 'n'})")
    print(f"   Sahm rule: {d2['sahm'].iloc[-1]:+.2f}  (trigger>=0.5: {'Y' if warn['sahm'].iloc[-1] else 'n'})")
    print(f"   >> composite warnings active: {int(comp.iloc[-1]*3)}/3")

    # ---- chart ----
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                           gridspec_kw={"height_ratios": [1.4, 1, 1, 1]})
    def shade(a):
        for s, e in RECESSIONS:
            a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="gray", alpha=0.2)
    ax[0].plot(df.index, np.exp(df["lp"]), color="black", lw=1); ax[0].set_yscale("log")
    ax[0].set_title("S&P 500 (log) with NBER recessions shaded", fontsize=11, fontweight="bold")
    ax[1].plot(curve.index, curve, color="#0d9488", lw=1.2); ax[1].axhline(0, color="red", ls="--", lw=1)
    ax[1].set_title("Yield curve 10y-3m (FRED T10Y3M) - inverts ~13mo before recessions", fontsize=10)
    ax[2].plot(credit.index, credit, color="#b45309", lw=1.2)
    ax[2].set_title("Credit spread Baa-10y (FRED BAA10Y)", fontsize=10)
    ax2b = ax[2].twinx(); ax2b.plot(sahm.index, sahm, color="#6d28d9", lw=1, alpha=0.7)
    ax2b.axhline(0.5, color="#6d28d9", ls=":", lw=1); ax2b.set_ylabel("Sahm", color="#6d28d9", fontsize=8)
    ax[3].fill_between(comp.index, 0, comp * 100, color="#dc2626", alpha=0.4, step="mid")
    ax[3].set_title("Composite: share of 3 leading warnings active (%)", fontsize=10); ax[3].set_ylim(0, 100)
    for a in ax:
        shade(a); a.grid(alpha=0.2); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Economic-Cycle Dashboard - DESCRIPTIVE leading indicators, not a forecast\n"
                 "(validated against NBER recessions; the live 200DMA gate + VIX throttle react, not predict)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "/Users/riddhisiddhi/stocks/cycles/economic_cycle_forecast.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n   chart saved -> {out}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
