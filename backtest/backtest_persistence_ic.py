"""
Walk-forward information-coefficient (IC) backtest for the trend-vs-mean-revert
persistence classifier (macro/indicators._persistence_classify).

Question: does the 0.4/0.4/0.2 Hurst+VR+autocorr blend actually rank assets by
future persistence better than VR alone, Hurst alone, or the blend without the
autocorr term?

Design (no lookahead):
  * ONE bulk yfinance download of the whole TREND_ASSETS universe; tickers with
    no data are skipped.
  * Rebalance every STEP trading days over the trailing 2 years. On each date t
    each predictor is computed from the trailing LOOKBACK bars (data up to and
    including t only), reusing the exact production functions.
  * Ground truth = Kaufman efficiency ratio over the NEXT FWD trading days:
        ER = |sum(log returns)| / sum(|log returns|)  in [0,1]
    ER is a pure path-geometry measure of trendiness (1 = clean trend, 0 =
    chop). It is deliberately method-neutral - not VR, not Hurst, not an
    autocorrelation - so it does not structurally favor any predictor.
  * Metric = cross-sectional Spearman IC per date (rank corr of predictor vs
    forward ER across the universe that day), then averaged with a t-stat, plus
    a pooled Spearman over all (asset, date) observations.

Higher, more stable IC = better predictor of which ETFs will trend.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators import _hurst_exponent, _variance_ratio, _return_autocorr

END = pd.Timestamp("2026-08-20")
TEST_START = END - pd.DateOffset(years=2)
LOOKBACK = 252   # trailing bars used to score (matches production "1y")
FWD = 63         # forward horizon for realized persistence (~3 months)
STEP = 21        # rebalance cadence (~monthly)
MIN_NAMES = 8    # min cross-sectional breadth to compute an IC that date


def eff_ratio(prices):
    lr = np.diff(np.log(prices))
    denom = np.sum(np.abs(lr))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return abs(np.sum(lr)) / denom


def sub_scores(h, vr, ac):
    sh = np.clip((h - 0.5) / 0.15, -1.0, 1.0)
    svr = np.clip((vr - 1.0) / 0.5, -1.0, 1.0)
    sac = np.clip(ac / 0.2, -1.0, 1.0)
    return sh, svr, sac


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    data_start = (TEST_START - pd.DateOffset(days=420)).date()  # ~252 trading-day burn-in
    print(f"Downloading {len(tickers)} tickers {data_start} -> {END.date()} ...")
    raw = yf.download(tickers, start=str(data_start), end=str((END + pd.Timedelta(days=1)).date()),
                      progress=False, group_by="column", auto_adjust=True)

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.dropna(axis=1, how="all")
    present = [t for t in tickers if t in close.columns]
    skipped = [t for t in tickers if t not in present]
    if skipped:
        print(f"Skipped (no data): {', '.join(skipped)}")
    print(f"Universe: {len(present)} tickers with data\n")

    idx = close.index
    # rebalance positions: need LOOKBACK bars behind and FWD bars ahead
    positions = [i for i in range(LOOKBACK, len(idx) - FWD)
                 if idx[i] >= TEST_START]
    positions = positions[::STEP]

    variants = ["vr_only", "hurst_only", "blend", "blend_no_ac"]
    per_date_ic = {v: [] for v in variants}
    pooled = {v: [] for v in variants}
    pooled_er = []

    for i in positions:
        preds = {v: [] for v in variants}
        ers = []
        for sym in present:
            s = close[sym].iloc[i - LOOKBACK:i + 1].dropna()   # trailing, incl. t
            if len(s) < 120:
                continue
            fwd = close[sym].iloc[i:i + FWD + 1].dropna()       # from t forward
            if len(fwd) < FWD - 5:
                continue
            er = eff_ratio(fwd.values)
            if not np.isfinite(er):
                continue

            h = _hurst_exponent(s)
            vr = _variance_ratio(s, q=21)
            ac = _return_autocorr(s)
            sh, svr, sac = sub_scores(h, vr, ac)

            preds["vr_only"].append(vr)
            preds["hurst_only"].append(h)
            preds["blend"].append(0.4 * sh + 0.4 * svr + 0.2 * sac)
            preds["blend_no_ac"].append(0.5 * sh + 0.5 * svr)
            ers.append(er)

        if len(ers) < MIN_NAMES:
            continue
        ers = np.array(ers)
        for v in variants:
            arr = np.array(preds[v])
            rho, _ = spearmanr(arr, ers)
            if np.isfinite(rho):
                per_date_ic[v].append(rho)
            pooled[v].extend(arr.tolist())
        pooled_er.extend(ers.tolist())

    print(f"Rebalance dates evaluated: {len(per_date_ic['blend'])}  "
          f"(from {idx[positions[0]].date()} to {idx[positions[-1]].date()})")
    print(f"Pooled observations: {len(pooled_er)}\n")

    print(f"{'variant':14s} {'mean IC':>8s} {'std':>7s} {'t-stat':>7s} "
          f"{'hit>0':>6s} {'pooled IC':>10s}")
    print("-" * 62)
    rows = []
    for v in variants:
        ics = np.array(per_date_ic[v])
        m, sd = ics.mean(), ics.std(ddof=1)
        t = m / sd * np.sqrt(len(ics)) if sd > 0 else float("nan")
        hit = np.mean(ics > 0)
        pooled_rho, _ = spearmanr(np.array(pooled[v]), np.array(pooled_er))
        rows.append((v, m, t, pooled_rho))
        print(f"{v:14s} {m:>8.3f} {sd:>7.3f} {t:>7.2f} {hit:>6.0%} {pooled_rho:>10.3f}")

    best = max(rows, key=lambda r: r[1])
    print(f"\nBest mean cross-sectional IC: {best[0]} ({best[1]:.3f}, t={best[2]:.2f})")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
