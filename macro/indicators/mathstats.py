import yfinance as yf
import numpy as np
import pandas as pd
import threading
import time
import math
import os
import json
import hashlib
import tempfile
from functools import wraps, lru_cache
from scipy.optimize import minimize

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:  # numpy < 1.20
    sliding_window_view = None

from macro.helpers import compute_RSI, compute_ATR
from macro.constants import (
    SECTOR_NAMES, SECTORS, COUNTRIES, COMMODITIES,
    CURRENCIES, TREND_ASSETS, ML_MACRO_TICKERS
)
from predict import predict_assets, predict_commodities
from macro.paths import model_path


# =========================
# Internal Math Helpers
# =========================
def _safe_r2(y, coeffs):
    # lru_cache removed (2026-06-11): float tuples from market data never
    # repeat, so the cache was pure overhead (hash of 20-tuple per call).
    y = np.asarray(y)
    coeffs = np.asarray(coeffs)
    if len(y) < 2:
        return 0.0
    y_hat = np.polyval(coeffs, np.arange(len(y)))
    ss_res = np.sum(np.square(y - y_hat))
    y_mean = np.mean(y)
    ss_tot = np.sum(np.square(y - y_mean))
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def _ols_slope_intercept(y):
    # Closed-form degree-1 least squares over x = 0..n-1 (2026-06-17).
    # Replaces np.polyfit(x, y, 1), which routes through SVD/lstsq and is
    # far heavier than necessary for a straight line. Returns the SAME
    # [slope, intercept] to floating precision; every caller rounds to
    # 2dp downstream, so outputs are bit-identical (verified vs polyfit
    # across thousands of random walks).
    n = len(y)
    if n < 2:
        return 0.0, float(np.mean(y)) if n else 0.0
    x_mean = (n - 1) / 2.0
    xc = np.arange(n) - x_mean
    denom = float(xc @ xc)
    if denom == 0.0:
        return 0.0, float(np.mean(y))
    y_mean = float(np.mean(y))
    slope = float(xc @ (y - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _trend_stats(series, window, scale):
    c = series.dropna().tail(window)
    if len(c) < 5:
        return 0.0, 0.0
    y = np.log(c.values)
    slope_c, intercept_c = _ols_slope_intercept(y)
    coeffs = np.array([slope_c, intercept_c])   # same shape as polyfit deg-1
    slope = float(slope_c) * scale * 100
    r2 = _safe_r2(y, coeffs)
    return round(slope, 2), round(r2, 2)


def _compute_gradient(series, window=5, slice_len=10, scale=1.0):
    if len(series) < window + slice_len:
        return 0.0
    slope_now, r2_now = _trend_stats(series.tail(window), window, scale)
    slope_prev, r2_prev = _trend_stats(series.tail(window + 5).iloc[:-5], window, scale)
    dx_raw = slope_now - slope_prev
    dy_raw = r2_now - r2_prev
    slope_scale = max(abs(slope_now), abs(slope_prev), 1e-6) * 2
    r2_scale = 2.0
    dx = dx_raw / slope_scale
    dy = dy_raw / r2_scale
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return round(angle_deg, 1)


def _compute_slope_change(series, window=20):
    if len(series) < window + 5:
        return 0.0
    slope_now, r2_now = _trend_stats(series.tail(window), window, window)
    slope_prev, r2_prev = _trend_stats(series.tail(window + 5).iloc[:-5], window, window)
    dx_raw = slope_now - slope_prev
    dy_raw = r2_now - r2_prev
    slope_scale = max(abs(slope_now), abs(slope_prev), 1e-6) * 2
    r2_scale = 2.0
    dx = dx_raw / slope_scale
    dy = dy_raw / r2_scale
    return round(math.sqrt(dx ** 2 + dy ** 2), 4)


def _compute_delta_slope(series, window=20):
    if len(series) < window + 5:
        return 0.0
    slope_now, _ = _trend_stats(series.tail(window), window, window)
    slope_prev, _ = _trend_stats(series.tail(window + 5).iloc[:-5], window, window)
    return round(slope_now - slope_prev, 4)


# =========================
# Hurst Exponent - R/S with Anis-Lloyd-Peters correction (2026-06-18)
#
#   H > 0.55  -> trending     - momentum persists, trend system has edge
#   H 0.45-0.55 -> random walk - no structural edge, reduce sizing
#   H < 0.45  -> mean-reverting - trend system is fighting the series
#
# Method: Rescaled Range (R/S) analysis over log returns, de-biased with the
# Anis-Lloyd (1976) / Peters (1994) expected R/S of an i.i.d. series.
#
# WHY the correction (2026-06-18): plain R/S over a short, narrow lag range is
# biased HIGH and the bias does NOT vanish with more data - an i.i.d. random
# walk (true H=0.5) read ~0.61 on the old code (verified by Monte Carlo at
# n=252 and n=1000). That mis-centred the gates above: the 0.55 "trending"
# threshold fired for random walks and the 0.45 "mean-reverting" threshold
# essentially never fired. The correction regresses
# log(R/S_n) - log(E[R/S_n]) on log(n); the slope estimates H-0.5 directly, so
# an i.i.d. series now recovers ~0.5 (random walk 0.61->0.52, anti-persistent
# 0.50->0.41, persistent 0.74->0.65; ordering preserved, centre fixed).
# NOTE: Hurst measures return AUTOCORRELATION, not trend strength - a strong
# drift with i.i.d. noise is correctly ~0.5 here.
# =========================
@lru_cache(maxsize=512)
def _expected_rs(n):
    """Anis-Lloyd/Peters expected R/S of an i.i.d. series of length n (the null)."""
    if n < 2:
        return float('nan')
    idx = np.arange(1, n)
    tail = float(np.sum(np.sqrt((n - idx) / idx)))
    if n <= 340:
        from scipy.special import gammaln
        front = float(np.exp(gammaln((n - 1) / 2.0) - gammaln(n / 2.0)) / np.sqrt(np.pi))
    else:
        front = 1.0 / np.sqrt(n * np.pi / 2.0)
    return front * tail


def _hurst_exponent(series, max_lag=40):
    """
    Hurst via rescaled range (R/S), Anis-Lloyd-Peters corrected.
    Returns float in [0, 1]. 0.5 = random walk default on error.
    """
    try:
        c = np.array(series.dropna()) if hasattr(series, 'dropna') else np.array(series)
        if len(c) < max_lag + 10:
            return 0.5

        log_returns = np.diff(np.log(c))
        if len(log_returns) < max_lag:
            return 0.5

        used_lags, rs_values, ers_values = [], [], []

        for lag in range(10, max_lag):
            n_windows = len(log_returns) // lag
            if n_windows < 2:
                continue

            rs_window = []
            for i in range(n_windows):
                window = log_returns[i * lag:(i + 1) * lag]
                mean_adj = window - window.mean()
                cumdev = np.cumsum(mean_adj)
                r = cumdev.max() - cumdev.min()
                s = window.std(ddof=1)
                if s > 0:
                    rs_window.append(r / s)

            if rs_window:
                rs_values.append(np.mean(rs_window))
                ers_values.append(_expected_rs(lag))
                used_lags.append(lag)

        if len(rs_values) < 5:
            return 0.5

        log_lags = np.log(used_lags)
        # Anis-Lloyd-Peters: subtract the i.i.d. expected R/S so the slope
        # estimates H-0.5 directly (de-biases the finite-sample R/S inflation).
        log_excess = np.log(rs_values) - np.log(ers_values)
        slope = np.polyfit(log_lags, log_excess, 1)[0]
        h = 0.5 + float(slope)

        return round(float(np.clip(h, 0.0, 1.0)), 3)

    except Exception:
        return 0.5


# =========================
# Trend vs. Mean-Reversion persistence (2026-08-20)
#
# Three complementary, near-orthogonal views of whether a series persists
# (trends) or reverts, each normalized so +1 = strongly trending and
# -1 = strongly mean-reverting, then blended. Read as a ~2-3 month
# characterization: the persistence *regime* (does this asset trend or
# chop?) is far more stable across a quarter than its direction, so the
# label is a forward posture, not a price forecast.
#
#   Hurst (R/S, above)        long-range dependence   H>0.5 trend, <0.5 revert
#   Variance ratio (q=21)     Lo-MacKinlay VR         VR>1 trend, <1 revert
#   Weekly return autocorr    lag-1 of 5-day blocks   +ve momentum, -ve reversal
# =========================
def _variance_ratio(series, q=21):
    """
    Lo-MacKinlay overlapping variance ratio at horizon q, with the
    unbiased (heteroskedasticity-consistent) denominator. Returns 1.0
    (random-walk null) on error or insufficient data. VR>1 => positive
    serial correlation (trending); VR<1 => mean reversion.
    """
    try:
        c = np.asarray(series.dropna() if hasattr(series, 'dropna') else series, dtype=float)
        r = np.diff(np.log(c))
        n = len(r)
        if n < q * 3:
            return 1.0
        mu = r.mean()
        var1 = np.sum((r - mu) ** 2) / (n - 1)
        if var1 <= 0:
            return 1.0
        # overlapping q-period returns = rolling sums of q consecutive 1-day returns
        rq = np.convolve(r, np.ones(q), 'valid')
        # Lo-MacKinlay unbiased scaling m = q(n-q+1)(1 - q/n); the q factor in m
        # makes varq the *per-period* variance of q-day returns, so VR = varq/var1.
        m = q * (n - q + 1) * (1.0 - q / n)
        if m <= 0:
            return 1.0
        varq = np.sum((rq - q * mu) ** 2) / m
        vr = varq / var1
        return round(float(vr), 3)
    except Exception:
        return 1.0


def _return_autocorr(series, block=5, lag=1):
    """
    Lag-1 autocorrelation of non-overlapping block (weekly, 5-day) returns.
    Positive => momentum/continuation, negative => short-term reversal.
    Returns 0.0 (no signal) on error or too few blocks.
    """
    try:
        c = np.asarray(series.dropna() if hasattr(series, 'dropna') else series, dtype=float)
        r = np.diff(np.log(c))
        k = len(r) // block
        if k < 8:
            return 0.0
        rb = r[:k * block].reshape(k, block).sum(axis=1)
        rb = rb - rb.mean()
        denom = np.sum(rb ** 2)
        if denom <= 0:
            return 0.0
        num = np.sum(rb[:-lag] * rb[lag:])
        return round(float(num / denom), 3)
    except Exception:
        return 0.0


def _persistence_classify(hurst, vr, autocorr):
    """
    Blend the three persistence views into one score in [-1, 1] and a
    label/arrow. +1 = strongly trending, -1 = strongly mean-reverting.
    Weights: Hurst 0.4, variance ratio 0.4, weekly autocorr 0.2.
    """
    s_h = np.clip((hurst - 0.5) / 0.15, -1.0, 1.0)      # H 0.65->+1, 0.35->-1
    s_vr = np.clip((vr - 1.0) / 0.5, -1.0, 1.0)          # VR 1.5->+1, 0.5->-1
    s_ac = np.clip(autocorr / 0.2, -1.0, 1.0)            # ac +/-0.2 -> +/-1
    score = float(np.clip(0.4 * s_h + 0.4 * s_vr + 0.2 * s_ac, -1.0, 1.0))
    if score > 0.15:
        label, arrow = "TREND", "↑"
    elif score < -0.15:
        label, arrow = "MEAN-REVERT", "↻"
    else:
        label, arrow = "NEUTRAL", "↔"
    return {
        "persistence_score": round(score, 3),
        "persistence_label": label,
        "persistence_arrow": arrow,
    }


def _u_fit(slope_path, r2_path):
    """Continuous [0,1] 'U-reversal fit' of the trailing (slope, R2) path
    (oldest -> today): started as a downtrend (negative slope), R2 collapsed
    through the middle (trend broke near the origin), and is turning up now.
    Backtest (2007-26): as an optimizer sizing factor it tilts toward freshly
    born trends. Returns 0 for anything that did not reverse (incl. steady
    established uptrends), so it concentrates the book."""
    sp = np.asarray(slope_path, dtype=float)
    rp = np.asarray(r2_path, dtype=float)
    L = len(sp)
    if L < 15 or not (np.isfinite(sp).all() and np.isfinite(rp).all()):
        return 0.0
    a, b = L // 5, (4 * L) // 5
    down = np.clip(-sp[0] / 3.0, 0.0, 1.0)                  # how negative the start leg
    brk = np.clip((0.30 - rp[a:b].min()) / 0.30, 0.0, 1.0)  # how deep the mid-R2 collapse
    up = 1.0 if sp[-1] > 0 else 0.0                          # must be turning up now
    return float(round(down * brk * up, 3))


# =========================
# Stop Hit Probability - CORRECTED (2026-06-11)
#
# P(hit stop at log-distance a<0 before target at b>0 | BM with drift mu):
#
#   Scale function s(x) = 1 - exp(-2*mu*x/sigma^2):
#       P(stop first) = (1 - exp(-2mu*b/s2)) / (exp(-2mu*a/s2) - exp(-2mu*b/s2))
#
#   Sanity limits:
#     mu -> 0:     P -> b / (b + |a|)   (NEARER barrier wins more often)
#     mu -> +inf:  P -> 0
#     mu -> -inf:  P -> 1
#
# Previous version had +2mu exponents and |a|/(|a|+|b|) driftless fallback
# - both inverted. It passed the infinite-drift limit checks but was wrong
# at every finite drift, systematically UNDERSTATING p_stop (e.g. driftless
# 2.5xATR stop with rr=2: old said 0.33, truth is 0.67). Validated against
# Monte Carlo GBM 2026-06-11.
#
# Drift units also corrected: slope = 1000 * daily log return, so
# annual drift = slope/1000 * 252 = slope * 0.252 (old code used slope/100,
# ~25x understated).
#
# Interpretation (with honest values these gates now actually bind):
#   p_stop < 0.25   -> strong geometry - full size
#   p_stop 0.25-0.40 -> standard sizing
#   p_stop > 0.40   -> reduce size
#   p_stop > 0.55   -> unfavorable - do not enter / exit existing
# =========================
def _stop_hit_probability(price, stop, target, slope, atr, projection_days=63):
    """
    Probability that price hits stop before target under GBM with drift.
    Returns float in [0,1]; 0.5 on degenerate inputs.

    Note: projection_days is UNUSED - vol is annualised directly from ATR
    and the first-passage probability is horizon-free (barriers absorb
    whenever hit). Parameter retained for signature compatibility only.
    """
    try:
        if price <= 0 or stop >= price or target <= price:
            return 0.5
        if atr <= 0:
            return 0.5

        # slope = 1000 * daily log return -> annualised = slope * 0.252
        annual_drift = slope * 0.252
        annual_vol = (atr / price) * np.sqrt(252)
        if annual_vol <= 0:
            return 0.5

        a = np.log(stop / price)    # < 0
        b = np.log(target / price)  # > 0

        mu = annual_drift - 0.5 * annual_vol ** 2   # Ito-corrected drift
        sigma2 = annual_vol ** 2

        if abs(mu) < 1e-10:
            # Driftless: optional stopping - nearer barrier hit more often
            p_stop = b / (b + abs(a))
        else:
            exp_arg_a = float(np.clip(-2.0 * mu * a / sigma2, -500.0, 500.0))
            exp_arg_b = float(np.clip(-2.0 * mu * b / sigma2, -500.0, 500.0))
            exp_a = np.exp(exp_arg_a)
            exp_b = np.exp(exp_arg_b)
            denom = exp_a - exp_b

            if abs(denom) < 1e-12:
                p_stop = b / (b + abs(a))
            else:
                p_stop = (1.0 - exp_b) / denom

        return round(float(np.clip(p_stop, 0.0, 1.0)), 3)

    except Exception:
        return 0.5


# =========================
# Dynamic Vol Cap - REWEIGHTED (2026-06-11)
#
# Fast model retired from this calculation: holdout AUC 0.516 (coin flip).
# A vol ceiling modulated 50% by noise was adding randomness, not crash
# detection. New weighting:
#   regime_scalar 0.60 - six observable technical conditions (validated
#                        by construction: hard facts, no inference)
#   ml_slow       0.40 - SPY slow model, the one ML signal that survived
#                        OOS validation (0.625 strided AUC on SPY)
#
# Min 8% / Max 15% band unchanged. In continuous-Kelly terms this band IS
# the Kelly fraction now: full Kelly vol for the strategy equals its
# Sharpe, so 8-15% against a plausible Sharpe ~0.7 is ~1/5 to 1/9 Kelly.
# =========================
def _effective_vol_cap(regime, base_min=0.08, base_max=0.15):
    """
    Compute dynamic portfolio vol ceiling.
    Signature unchanged; ml_fast no longer used (dead signal OOS).
    """
    ml_slow = regime.get("ml_slow", 50.0) / 100.0        # 0.0 - 1.0
    regime_scalar = get_regime_scalar(regime)             # 0.0 - 1.0

    combined = (
        regime_scalar * 0.60   # observable macro conditions - primary
        + ml_slow     * 0.40   # validated SPY structural signal
    )

    vol_cap = base_min + combined * (base_max - base_min)
    return round(max(base_min, min(vol_cap, base_max)), 4)



def get_regime_scalar(regime):
    passes = sum(1 for d in regime["details"] if d["pass"])
    scalars = {6: 1.0, 5: 0.85, 4: 0.70, 3: 0.50, 2: 0.30, 1: 0.15, 0: 0.0}
    return scalars.get(passes, 0.0)

__all__ = [
    "_safe_r2",
    "_ols_slope_intercept",
    "_trend_stats",
    "_compute_gradient",
    "_compute_slope_change",
    "_compute_delta_slope",
    "_expected_rs",
    "_hurst_exponent",
    "_variance_ratio",
    "_return_autocorr",
    "_persistence_classify",
    "_u_fit",
    "_stop_hit_probability",
    "_effective_vol_cap",
    "get_regime_scalar",
]
