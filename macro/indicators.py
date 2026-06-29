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


# =========================
# TTL Cache
# =========================
def ttl_cache(ttl_seconds=30):
    def decorator(func):
        cache = {}
        lock = threading.Lock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            with lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl_seconds:
                        return value
                expired = [k for k, (_, ts) in cache.items() if now - ts >= ttl_seconds]
                for k in expired:
                    del cache[k]
            result = func(*args, **kwargs)
            with lock:
                cache[key] = (result, now)
            return result

        return wrapper
    return decorator


# =========================
# Persistent Incremental Market-Data Store (added 2026-06-18)
#
# Replaces the previous 30s whole-window parquet cache. Daily bars are
# immutable once a session closes, so instead of re-downloading a full year
# every refresh we keep history on disk and fetch only
# [last stored date - trailing window .. today], overwrite the trailing
# window (to absorb Yahoo's retroactive split/dividend revisions), and
# append new rows. Cold-start and the periodic full resync still pull the
# whole period and are byte-identical to the old download.
#
# Timing knobs (env-overridable):
#   GSS_STORE_TTL        (s)  serve store without ANY network for this long
#                             (daily bars don't change intraday; safe to set
#                             to minutes/hours). Default 30 (prev behaviour).
#   GSS_TRAILING_DAYS    (d)  trailing calendar window refetched+overwritten.
#   GSS_FULL_RESYNC_DAYS (d)  force a full re-download this often so a deep
#                             retroactive adjustment self-heals. Default 30.
#   GSS_STORE_MAX_ROWS   (n)  cap on stored history length.
#   GSS_INCREMENTAL      0/1  kill switch -> exact original full-download path.
#   GSS_CACHE_DIR             on-disk location (shared with predict.py store).
#
# Behaviour notes:
#   - Per-date VALUES are identical to a full download (same Yahoo bars);
#     verified bit-for-bit against a synthetic feed incl. trailing/deep
#     revisions and the monthly resync.
#   - The only consumer sensitive to the window's FRONT boundary is the
#     HMM telemetry block (it trains on the whole returned window). Its
#     output may shift slightly vs a raw period= call. It is telemetry-only,
#     refit weekly, and already non-deterministic on the rolling window;
#     set GSS_INCREMENTAL=0 if byte-identical HMM is ever required.
#   - ANY read/write failure (no parquet engine, IO error) falls back to the
#     live network path silently. No public signature changes.
# =========================
_STORE_DIR = os.environ.get(
    "GSS_CACHE_DIR", os.path.join(tempfile.gettempdir(), "gss_market_cache")
)
_STORE_TTL = float(os.environ.get("GSS_STORE_TTL", os.environ.get("GSS_PARQUET_TTL", "30")))
_TRAILING_DAYS = int(os.environ.get("GSS_TRAILING_DAYS", "7"))
_FULL_RESYNC_SECS = float(os.environ.get("GSS_FULL_RESYNC_DAYS", "30")) * 86400.0
_STORE_MAX_ROWS = int(os.environ.get("GSS_STORE_MAX_ROWS", "520"))
_INCREMENTAL = os.environ.get("GSS_INCREMENTAL", "1") != "0"
_STORE_WARNED = {"done": False}


def _store_base(label, tickers):
    payload = f"{label}|{','.join(sorted(map(str, tickers)))}"
    digest = hashlib.md5(payload.encode()).hexdigest()[:16]
    return os.path.join(_STORE_DIR, f"store_{label}_{digest}")


def _store_read(base):
    """Return (df, spec) rebuilding the column MultiIndex, or (None, None)."""
    try:
        pq, meta = base + ".parquet", base + ".meta"
        if not (os.path.exists(pq) and os.path.exists(meta)):
            return None, None
        with open(meta) as f:
            spec = json.load(f)
        df = pd.read_parquet(pq)
        n = len(spec["cols"])
        df = df[[str(i) for i in range(n)]]
        if spec["is_mi"]:
            df.columns = pd.MultiIndex.from_tuples([tuple(c) for c in spec["cols"]])
        else:
            df.columns = pd.Index(spec["cols"])
        df.index = pd.to_datetime(df.index)
        df.index.name = spec.get("index_name")
        return df, spec
    except Exception:
        return None, None


def _store_write(base, df, ts, full_ts):
    """Atomically persist df + meta. Silent no-op on any failure."""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        os.makedirs(_STORE_DIR, exist_ok=True)
        is_mi = isinstance(df.columns, pd.MultiIndex)
        cols = [list(t) for t in df.columns] if is_mi else list(df.columns)
        out = df.copy()
        out.columns = [str(i) for i in range(len(df.columns))]
        pid = os.getpid()
        pq, meta = base + ".parquet", base + ".meta"
        tmp_pq, tmp_meta = f"{pq}.tmp.{pid}", f"{meta}.tmp.{pid}"
        out.to_parquet(tmp_pq)
        os.replace(tmp_pq, pq)
        with open(tmp_meta, "w") as f:
            json.dump({"ts": ts, "full_ts": full_ts, "is_mi": is_mi,
                       "cols": cols, "index_name": df.index.name}, f)
        os.replace(tmp_meta, meta)
    except Exception as e:
        if not _STORE_WARNED["done"]:
            print(f"Market-data store disabled (falling back to network): {e}")
            _STORE_WARNED["done"] = True


def _slice_period(df, period):
    """Tail window matching a yfinance-style period string ('1y', '300d', '6mo')."""
    if df is None or df.empty:
        return df
    anchor = pd.Timestamp(df.index.max()).normalize()
    p = str(period)
    if p.endswith("mo"):
        cutoff = anchor - pd.DateOffset(months=int(p[:-2]))
    elif p.endswith("y"):
        cutoff = anchor - pd.DateOffset(years=int(p[:-1]))
    elif p.endswith("d"):
        cutoff = anchor - pd.Timedelta(days=int(p[:-1]))
    else:
        return df
    return df[df.index >= cutoff]


def _merge_trailing(stored, fresh, start):
    """Keep stored rows before `start`; overwrite [start..] with fresh bars."""
    fresh = fresh.reindex(columns=stored.columns)
    combined = pd.concat([stored[stored.index < start], fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    if len(combined) > _STORE_MAX_ROWS:
        combined = combined.iloc[-_STORE_MAX_ROWS:]
    return combined


def _yf_dl(tickers, period=None, start=None, end=None, group_by="column"):
    kw = dict(progress=False, auto_adjust=False, threads=True, group_by=group_by)
    if period is not None:
        return yf.download(tickers, period=period, **kw)
    return yf.download(tickers, start=start, end=end, **kw)


def _persistent_fetch(label, tickers, period, group_by="column",
                      download_fn=_yf_dl, now_fn=time.time):
    """Incremental period fetch. Returns the same bars a full period= call would."""
    if not _INCREMENTAL:
        return download_fn(tickers, period=period, group_by=group_by)

    base = _store_base(label, tickers)
    stored, spec = _store_read(base)
    now = now_fn()

    # Fast path: store fresh enough -> no network at all.
    if stored is not None and spec is not None and (now - float(spec.get("ts", 0)) < _STORE_TTL):
        return _slice_period(stored, period)

    need_full = (
        stored is None or stored.empty or spec is None
        or (now - float(spec.get("full_ts", 0)) > _FULL_RESYNC_SECS)
    )
    if need_full:
        raw = download_fn(tickers, period=period, group_by=group_by)
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            _store_write(base, raw, ts=now, full_ts=now)
            return raw
        if stored is not None and not stored.empty:
            return _slice_period(stored, period)   # network failed -> serve stale
        return raw

    # Incremental refresh: trailing window + any new sessions.
    last_date = pd.Timestamp(stored.index.max()).normalize()
    start = last_date - pd.Timedelta(days=_TRAILING_DAYS)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    try:
        fresh = download_fn(tickers, start=start, end=end, group_by=group_by)
    except Exception:
        fresh = None
    if not isinstance(fresh, pd.DataFrame) or fresh.empty:
        _store_write(base, stored, ts=now, full_ts=float(spec.get("full_ts", now)))
        return _slice_period(stored, period)

    combined = _merge_trailing(stored, fresh, start)
    _store_write(base, combined, ts=now, full_ts=float(spec.get("full_ts", now)))
    return _slice_period(combined, period)


# =========================
# Risk-regime history cache (added 2026-06-18)
#
# get_risk_regime's sparkline shows 5 strided predictions (predict_assets at
# as_of_date = today, -20, -40, -60, -80 sessions). We persist EVERY computed
# point as {date: conf} in a tiny JSON sidecar (scalars, so JSON is lighter
# than parquet) and recompute only dates not already stored. Because the
# 20-session stride passes back over each date on four later days, persisting
# the newest point too means each date is computed once (the day it first
# appears) and read from disk on every later appearance. Net effect:
#   - intraday reloads recompute nothing (all 5 are cache hits)
#   - a new trading day adds ~1 point (today's); older strides are reads
#   - the sparkline survives transient yfinance/DNS failures (served from
#     disk instead of re-predicting against the network)
# This is display-only telemetry (history / ml_fast never enter the composite,
# sizing, or status), so freezing the newest point at its first-computed value
# for the day is intended; the live RISK-ON/RISK-OFF signal is separate.
# =========================
def _risk_history_base(tickers):
    payload = "riskhist|" + ",".join(sorted(map(str, tickers)))
    digest = hashlib.md5(payload.encode()).hexdigest()[:16]
    return os.path.join(_STORE_DIR, f"riskhist_{digest}.json")


def _risk_history_load(base):
    try:
        with open(base) as f:
            return json.load(f)
    except Exception:
        return {}


def _risk_history_save(base, cache):
    try:
        os.makedirs(_STORE_DIR, exist_ok=True)
        tmp = f"{base}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, base)
    except Exception:
        pass


# =========================
# SHARED DATA MANAGER
# =========================
@ttl_cache(30)
def _get_shared_market_data():
    all_tickers = list(set(
        list(SECTORS.keys()) +
        list(COUNTRIES.keys()) +
        list(COMMODITIES.keys()) +
        list(CURRENCIES.keys()) +
        list(TREND_ASSETS.keys()) +
        list(SECTOR_NAMES.keys()) +
        ['SPY', 'RSP', '^VIX', '^MOVE', 'HYG', 'IEF', '^TNX', '^IRX',
         'JPY=X', 'HG=F', 'GC=F', 'QQQ']
    ))
    return _persistent_fetch("shared", all_tickers, "1y", group_by="column")


@ttl_cache(30)
def _get_extended_data():
    macro_proxies = ['HYG', 'IEF', '^TNX', '^IRX', 'JPY=X', 'HG=F', 'GC=F']
    risk_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys())))
    all_tickers = list(set(risk_tickers + macro_proxies + ['^VIX', '^MOVE']))
    raw = _persistent_fetch("extended", all_tickers, "300d", group_by="column")
    return raw, risk_tickers


def _get_close(shared_data, ticker):
    if shared_data.empty or 'Close' not in shared_data:
        return pd.Series(dtype=float)
    close = shared_data['Close']
    if isinstance(close, pd.Series):
        return close.dropna()
    if not isinstance(close.columns, pd.MultiIndex):
        if ticker in close.columns:
            return close[ticker].dropna()
        return pd.Series(dtype=float)
    try:
        if ticker in close.columns.get_level_values(0):
            return close[ticker].dropna()
        return close.xs(ticker, level=1, axis=1).squeeze().dropna()
    except Exception:
        return pd.Series(dtype=float)


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


# =========================
# Regime HMM - TELEMETRY ONLY (added 2026-06-15)
#
# 3-state Gaussian HMM on [SPY return, SPY realized vol, VIX Z-score].
# Outputs P(crisis-like state) and a most-likely state label, used ONLY
# to qualify the existing RISK-ON/RISK-OFF status (via _qualify_regime)
# for dashboard display.
#
# STATUS: telemetry + qualifier only. Does NOT feed _effective_vol_cap,
# sizing, or status logic. composite_score, is_risk_on, regime_scalar,
# and every downstream consumer of get_risk_regime() are unaffected by
# this block - `hmm` and `regime_qualifier` are additive keys only.
#
# Promotion criterion: if, over a sample of regime transitions,
# p_crisis is observed to rise BEFORE regime_scalar/composite move -
# i.e. it leads rather than confirms - it becomes a candidate for a
# depth-2 integration (e.g. multiplicative vol-cap dampener). If it only
# confirms (moves at the same time or later than the existing composite),
# it is redundant with the already-validated signal and should remain
# telemetry/qualifier only.
#
# Honest limitations:
#   - Refit weekly (_HMM_REFIT_INTERVAL), not per-call. 252 obs and
#     ~15-20 free params for 3 diagonal-covariance Gaussian states -
#     refitting on every 30-second cache tick would add fit-to-fit
#     noise from `random_state` interacting with marginal data updates,
#     without any real information gain.
#   - State labels are NOT fixed across refits. _fit_regime_hmm always
#     re-sorts by mean return (column 0) ascending after fitting -
#     lowest mean return = "crisis", highest = "trending_calm".
#   - Will not catch a single-day shock on day one - same structural
#     limitation as the 6-condition composite (the pattern needs to
#     accumulate across the observation window before posterior state
#     probabilities shift materially).
#   - Requires hmmlearn: pip install hmmlearn --break-system-packages
#     If unavailable, get_regime_hmm_state() returns None and
#     _qualify_regime() / the dashboard template degrade gracefully
#     (regime["hmm"] = None, regime["regime_qualifier"] = None).
# =========================
_hmm_cache = {"model": None, "labels": None, "fitted_at": None}
_HMM_REFIT_INTERVAL = 7 * 24 * 3600  # 7 days


def _fit_regime_hmm(X, n_states=3, n_iter=100):
    from hmmlearn import hmm

    if len(X) < 60:
        return None, None

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42,
    )

    model.fit(X)

    if n_states == 3:
        means = model.means_

        scores = {}

        for i in range(n_states):
            ret_3m = means[i, 0]
            vol_z  = means[i, 1]
            vix_z  = means[i, 2]

            # higher return good
            # higher vol/fear bad
            score = ret_3m - 0.5 * vol_z - 0.5 * vix_z

            scores[i] = score

        order = sorted(scores, key=scores.get)

        labels = {
            int(order[0]): "crisis",
            int(order[1]): "choppy",
            int(order[2]): "trending_calm",
        }

    else:
        labels = {i: f"state_{i}" for i in range(n_states)}

    return model, labels

@ttl_cache(30)
def get_regime_hmm_state(n_states=3):
    """
    Returns current HMM regime probabilities.

    Features:
        ret     = 21-session log return (trend)
        vol_z   = realized volatility z-score (turbulence)
        vix_z   = VIX z-score (fear)

    Returns:
        {
            "state_probs": {...},
            "most_likely": "...",
            "p_crisis": 0.12,
            "fitted_at": ...,
            "fitted_ago": "...",
            "state_means": {
                ...
            }
        }
    """
    try:
        shared_data = _get_shared_market_data()

        spy = _get_close(shared_data, "SPY")
        vix = _get_close(shared_data, "^VIX")

        # --------------------------------------------------
        # Feature 1: trend (21-session log return)
        # 2026-06-18: was a 63-day return, whose lag-1 autocorrelation is
        # ~0.997 - so over-smoothed it's nearly constant, carries almost no
        # frame-to-frame information, and grossly violates the Gaussian HMM's
        # conditional-independence assumption. With it near-constant, state
        # assignment was driven by vol/VIX noise and the decoded regime
        # switched almost every day (dwell ~1 obs). A 21-session return
        # (autocorr ~0.984, aligned with the 20-day vol window) carries real
        # trend information and yields materially more persistent regimes
        # (state dwell ~4 obs) while still separating crises ~5:1.
        # --------------------------------------------------

        daily_ret = np.log(spy / spy.shift(1))

        ret_trend = np.log(spy / spy.shift(63))

        # --------------------------------------------------
        # Feature 2: Relative volatility
        # --------------------------------------------------

        vol = daily_ret.rolling(20).std() * np.sqrt(252)

        vol_mean = vol.rolling(126).mean()
        vol_std = vol.rolling(126).std()

        vol_z = (vol - vol_mean) / vol_std

        # --------------------------------------------------
        # Feature 3: Relative fear
        # --------------------------------------------------

        vix_mean = vix.rolling(50).mean()
        vix_std = vix.rolling(50).std()

        vix_z = (vix - vix_mean) / vix_std

        # --------------------------------------------------
        # Build feature matrix
        # --------------------------------------------------

        df = pd.DataFrame({
            "ret": ret_trend,
            "vol_z": vol_z,
            "vix_z": vix_z,
        }).dropna()

        if len(df) < 60:
            return None

        X = df[["ret", "vol_z", "vix_z"]].values

        now = time.time()

        needs_refit = (
            _hmm_cache["model"] is None
            or _hmm_cache["fitted_at"] is None
            or (now - _hmm_cache["fitted_at"]) > _HMM_REFIT_INTERVAL
        )

        # --------------------------------------------------
        # Fit / refresh model
        # --------------------------------------------------

        if needs_refit:
            model, labels = _fit_regime_hmm(X, n_states)

            if model is None:
                return None

            _hmm_cache["model"] = model
            _hmm_cache["labels"] = labels
            _hmm_cache["fitted_at"] = now

        model = _hmm_cache["model"]
        labels = _hmm_cache["labels"]

        # --------------------------------------------------
        # Current probabilities
        # --------------------------------------------------

        state_probs_seq = model.predict_proba(X)
        current_probs = state_probs_seq[-1]

        state_probs = {
            labels[i]: round(float(p), 3)
            for i, p in enumerate(current_probs)
        }

        most_likely_idx = int(np.argmax(current_probs))

        p_crisis = state_probs.get("crisis", 0.0)

        # --------------------------------------------------
        # Fit age
        # --------------------------------------------------

        fitted_at = _hmm_cache["fitted_at"]

        if fitted_at:
            secs = time.time() - fitted_at

            fitted_ago = (
                f"{secs/60:.0f}m ago"
                if secs < 3600 else
                f"{secs/3600:.0f}h ago"
                if secs < 86400 else
                f"{secs/86400:.0f}d ago"
            )
        else:
            fitted_ago = "-"

        # --------------------------------------------------
        # State diagnostics
        # --------------------------------------------------

        means = model.means_

        state_means = {}

        for idx, lab in labels.items():
            ret_trend, volz, vixz = means[idx]

            state_means[lab] = {
                # NOTE: this is now the 21-session trend mean (feature 1 was
                # shortened from 63d). Key name kept as-is for template
                # compatibility; rename to "ret_trend_pct" in the template
                # and here together if you want the label to match.
                "ret_3m_pct": round(float(ret_trend) * 100, 1),
                "vol_z": round(float(volz), 2),
                "vix_z": round(float(vixz), 2),
            }

        return {
            "state_probs": state_probs,
            "most_likely": labels[most_likely_idx],
            "p_crisis": p_crisis,
            "fitted_at": fitted_at,
            "fitted_ago": fitted_ago,
            "state_means": state_means,
        }

    except ImportError:
        print(
            "Regime HMM: hmmlearn not installed - "
            "run `pip install hmmlearn --break-system-packages`"
        )
        return None

    except Exception as e:
        print(f"Regime HMM Error: {e}")
        return None


def _qualify_regime(is_risk_on, hmm_state):
    """
    Qualifies the RISK-ON/RISK-OFF status with the HMM's read on current
    market character. Purely descriptive - does not affect `status`,
    `composite_score`, `regime_scalar`, or anything used in sizing.

    Args:
        is_risk_on: bool, the existing composite-derived RISK-ON flag.
        hmm_state:  output of get_regime_hmm_state(), or None.

    Returns:
        One of "confirmed", "fragile", "transitional", or None
        (None if hmm_state is None - e.g. insufficient data or
        hmmlearn not installed; template should omit the qualifier
        entirely in this case).

    Logic:
        RISK-ON  + HMM "trending_calm"        -> "confirmed"
        RISK-ON  + HMM "choppy"/"crisis"      -> "fragile"
            (composite says on, HMM sees instability underneath -
             the more consequential mismatch, since it means
             currently-deployed capital may be exposed to a
             pattern the composite hasn't caught yet)
        RISK-OFF + HMM "crisis"               -> "confirmed"
        RISK-OFF + HMM "trending_calm"/"choppy" -> "transitional"
            (composite is cautious but HMM doesn't see crisis-level
             pattern - possibly an early-stage flip, possibly nothing)
    """
    if hmm_state is None:
        return None

    most_likely = hmm_state["most_likely"]

    if is_risk_on:
        return "confirmed" if most_likely == "trending_calm" else "fragile"
    else:
        return "confirmed" if most_likely == "crisis" else "transitional"


# =========================
# I. Risk Regime
# =========================
@ttl_cache(30)
def get_risk_regime():
    try:
        raw, risk_tickers = _get_extended_data()
        data = raw['Close'].ffill()

        # -----------------------------------------------
        # ML Risk Model - 20-day stride history (5 points)
        # RETAINED FOR DASHBOARD SPARKLINE ONLY (2026-06-11):
        # fast model holdout AUC 0.516 - no longer enters the
        # composite or any sizing/risk decision.
        # -----------------------------------------------
        history_points = []
        recent_dates = data.index[::-20][:5][::-1]

        # Display-only sparkline: persist EVERY computed point keyed by date
        # and only compute the ones not already stored. Because the 20-session
        # stride passes back over each date on four later days, persisting the
        # newest point too means each date is computed once (the day it first
        # appears) and read from disk thereafter -> steady state is ~1
        # predict_assets per new trading day, zero on intraday reloads.
        #
        # Consequence (intended): the most-recent point freezes at its
        # first-computed value for the day rather than refreshing each reload.
        # This is telemetry only (ml_fast / history are display-only and never
        # enter the composite, sizing, or status), so a frozen sparkline point
        # is fine. To force the newest point live again, delete its key from
        # the riskhist_*.json sidecar (or clear the file).
        _rh_base = _risk_history_base(risk_tickers)
        _rh_cache = _risk_history_load(_rh_base)
        _rh_dirty = False

        for ts in recent_dates:
            _key = pd.Timestamp(ts).normalize().strftime('%Y-%m-%d')

            if _key in _rh_cache:
                history_points.append(_rh_cache[_key])
                continue

            ml_res = predict_assets(
                model_path="risk_model.joblib",
                tickers=risk_tickers,
                friendly_names={},
                model_type="risk",
                as_of_date=ts
            )
            probs = ml_res.get('probabilities', {})
            conf_val = round(probs.get('Class 1', 0) * 100, 1)
            history_points.append(conf_val)
            _rh_cache[_key] = conf_val
            _rh_dirty = True

        if _rh_dirty:
            if len(_rh_cache) > 400:        # bound growth; keep most recent
                for _k in sorted(_rh_cache)[:-400]:
                    del _rh_cache[_k]
            _risk_history_save(_rh_base, _rh_cache)

        ml_fast_conf = history_points[-1]

        # -----------------------------------------------
        # SPY Slow ML Prediction - the validated signal
        # (strided OOS AUC 0.625 on SPY, the asset it trained on)
        # -----------------------------------------------
        try:
            from macro.ml_engine import get_dual_ml_confidence_for_kelly

            spy_df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].xs(
                'SPY', level=1, axis=1
            ).dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()

            spy_dual = get_dual_ml_confidence_for_kelly(spy_df)
            ml_slow_conf = spy_dual['slow']
            spy_regime = spy_dual['regime']
            spy_divergence = spy_dual['divergence']

        except Exception as e:
            print(f"SPY slow ML error: {e}")
            ml_slow_conf = 50.0   # neutral - do NOT fall back to fast (noise)
            spy_regime = "Unknown"
            spy_divergence = 0.0

        # -----------------------------------------------
        # Technical Conditions (6 independent checks)
        # -----------------------------------------------
        last_vals = data.iloc[-1]
        ma50 = data.rolling(50).mean().iloc[-1]

        credit_ratio = data['HYG'] / data['IEF']
        credit_pass = bool(
            last_vals['HYG'] / last_vals['IEF']
            > credit_ratio.rolling(50).mean().iloc[-1]
        )

        curve_spread = last_vals['^TNX'] - last_vals['^IRX']
        curve_pass = bool(curve_spread > 0)

        jpy_ret = data['JPY=X'].pct_change()
        jpy_vol = jpy_ret.rolling(20).std().iloc[-1] * np.sqrt(252)
        carry_pass = bool(
            last_vals['JPY=X'] > ma50['JPY=X'] and jpy_vol < 0.15
        )

        spy_ma200 = data['SPY'].rolling(200).mean().iloc[-1]
        spy_trend = bool(last_vals['SPY'] > spy_ma200)

        vix_low = bool(
            last_vals['^VIX'] < 20 and last_vals['^MOVE'] < 110
        )

        rsp_spy_ratio = data['RSP'] / data['SPY']
        breadth_pass = bool(
            rsp_spy_ratio.iloc[-1]
            > rsp_spy_ratio.rolling(50).mean().iloc[-1]
        )

        details = [
            {"label": "Trend (SPY > 200MA)",      "pass": spy_trend},
            {"label": "Fear (VIX/MOVE Low)",       "pass": vix_low},
            {"label": "Breadth (RSP/SPY > 50MA)",  "pass": breadth_pass},
            {"label": "Credit (HYG/IEF Ratio)",    "pass": credit_pass},
            {"label": "Curve (10Y-3M Spread)",     "pass": curve_pass},
            {"label": "Carry (JPY Weak/Stable)",   "pass": carry_pass},
        ]

        # -----------------------------------------------
        # Composite RISK-ON / RISK-OFF Signal - REWEIGHTED (2026-06-11)
        #
        #   55% technical - 6 observable macro conditions, hard facts
        #   45% ML slow   - SPY structural trend, OOS-validated (0.625)
        #    0% ML fast   - RETIRED: holdout AUC 0.516 = coin flip.
        #                   Kept in the return dict for the dashboard
        #                   sparkline only.
        #
        # Threshold 0.55: requires combined signal above neutral.
        # -----------------------------------------------
        passes = sum(1 for d in details if d["pass"])
        technical_score = passes / 6.0          # 0.0 to 1.0
        ml_slow_score = ml_slow_conf / 100.0    # 0.0 to 1.0

        composite_score = (
            technical_score * 0.55
            + ml_slow_score * 0.45
        )

        is_risk_on = composite_score > 0.55

        # HMM telemetry + qualifier - see module docstring above.
        # Additive only: does not affect composite_score, is_risk_on,
        # or anything computed above this point.
        hmm_state = get_regime_hmm_state()
        regime_qualifier = _qualify_regime(is_risk_on, hmm_state)

        return {
            "status":           "RISK-ON" if is_risk_on else "RISK-OFF",
            "regime_qualifier": regime_qualifier,   # "confirmed"/"fragile"/"transitional"/None
            "confidence":       round(composite_score * 100, 1),
            "ml_slow":          round(ml_slow_conf, 1),
            "ml_fast":          round(ml_fast_conf, 1),   # display only - not in composite
            "composite":        round(composite_score * 100, 1),
            "spy_regime":       spy_regime,
            "spy_divergence":   round(spy_divergence, 1),
            "history":          history_points,
            "details":          details,
            "hmm":              hmm_state,   # full state probs, for drill-down - may be None
        }

    except Exception as e:
        print(f"Risk Regime Error: {e}")
        return {
            "status":           "ERROR",
            "regime_qualifier": None,
            "confidence":       0,
            "ml_slow":          0,
            "ml_fast":          0,
            "composite":        0,
            "spy_regime":       "Error",
            "spy_divergence":   0,
            "history":          [],
            "details":          [],
            "hmm":              None,
        }


def get_regime_scalar(regime):
    passes = sum(1 for d in regime["details"] if d["pass"])
    scalars = {6: 1.0, 5: 0.85, 4: 0.70, 3: 0.50, 2: 0.30, 1: 0.15, 0: 0.0}
    return scalars.get(passes, 0.0)


# =========================
# II. ML Asset Predictions
# =========================
@ttl_cache(30)
def get_ml_sector_prediction():
    try:
        tickers = list(SECTOR_NAMES.keys()) + ML_MACRO_TICKERS
        res = predict_assets("sector_model.joblib", tickers, SECTOR_NAMES, "sector")
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": SECTOR_NAMES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Sector Error: {e}")
        return {"top": {"name": "N/A", "ticker": "N/A", "confidence": 0}, "all": []}


@ttl_cache(30)
def get_ml_country_prediction():
    try:
        tickers = list(COUNTRIES.keys()) + ML_MACRO_TICKERS + ['SPY']
        res = predict_assets("country_model.joblib", tickers, COUNTRIES, "country")
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": COUNTRIES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Country Error: {e}")
        return {"top": {"name": "N/A", "confidence": 0}, "all": []}


@ttl_cache(30)
def get_ml_commodity_prediction():
    try:
        res = predict_commodities(
            sector_model_path="commodity_sector_model.joblib",
            commodity_model_path="commodity_model.joblib",
            friendly_names=COMMODITIES,
            use_cache=True,
            top_n_sectors=5,
            top_n_per_sector=5
        )
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": COMMODITIES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Commodity Error: {e}")
        return {"top": {"name": "N/A", "confidence": 0}, "all": []}


# =========================
# III. Traditional Technical Signals
# =========================
@ttl_cache(30)
def get_vix_signal():
    try:
        shared_data = _get_shared_market_data()
        vix = _get_close(shared_data, '^VIX')
        rsp = _get_close(shared_data, 'RSP')
        spy = _get_close(shared_data, 'SPY')

        vix_tail = vix.tail(50)
        v_last = float(vix.iloc[-1])
        vix_mean = vix_tail.mean()
        vix_std = vix_tail.std()
        z = float((v_last - vix_mean) / vix_std)

        ratio = rsp / spy
        current_ratio = float(ratio.iloc[-1])
        ma50_ratio = float(ratio.tail(50).mean())
        breadth_failing = current_ratio < ma50_ratio

        spy_ma50 = float(spy.rolling(50).mean().iloc[-1])
        spy_ma200 = float(spy.rolling(200).mean().iloc[-1])
        spy_last = float(spy.iloc[-1])
        spy_in_uptrend = spy_last > spy_ma50 and spy_last > spy_ma200

        spy_slope, spy_r2 = _trend_stats(spy, 10, 10)
        spy_trending = spy_slope > 0 and spy_r2 > 0.6

        if z > 4.0:
            signal = "AGGRESSIVE_BUY" if spy_trending else "SCALE_IN"
        elif z > 2.0:
            signal = "SCALE_IN" if spy_in_uptrend else "HOLD"
        elif z < -1.5:
            signal = "AGGRESSIVE_TRIM"
        elif z < -1.0 and breadth_failing:
            signal = "CAUTIOUS_TRIM"
        else:
            signal = "HOLD"

        return {"vix": round(v_last, 2), "z": round(z, 2), "signal": signal}
    except Exception as e:
        print(f"VIX Signal Error: {e}")
        return {"vix": 0, "z": 0, "signal": "ERROR"}


def get_mean_reversion():
    try:
        shared_data = _get_shared_market_data()
        c = _get_close(shared_data, 'QQQ')
        if c.empty:
            raise ValueError("QQQ data empty")

        rsi2 = float(compute_RSI(c, 2).iloc[-1])
        if pd.isna(rsi2):
            raise ValueError("RSI(2) returned NaN - insufficient data")

        price = float(c.iloc[-1])
        sma200 = float(c.rolling(200, min_periods=200).mean().iloc[-1])

        if pd.isna(sma200):
            raise ValueError("SMA200 returned NaN - insufficient data (need 200 bars)")

        # Exit is intentionally RSI(2)-only. The unused dma10 computation and
        # its mismatched "DMA5" error message were removed (2026-06-11).
        if price < sma200:
            signal = "RISK OFF"
        elif rsi2 <= 10:
            signal = "BUY"
        elif rsi2 >= 70:
            signal = "EXIT"
        else:
            signal = "HOLD"

        return {"price": round(price, 2), "rsi2": round(rsi2, 1), "signal": signal}

    except Exception as e:
        print(f"Mean Reversion Error: {e}")
        return {"price": 0, "rsi2": 0, "signal": "ERROR"}


# =========================
# IV. Rotation
# =========================
@ttl_cache(30)
def get_sector_rotation():
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        sector_data = data[list(SECTORS.keys())]
        rel = sector_data.div(data["SPY"], axis=0).pct_change(63).iloc[-1]
        out = []
        for t, name in SECTORS.items():
            slope, r2 = _trend_stats(data[t], 20, 20)
            gradient = _compute_gradient(data[t].tail(20))
            slope_change = _compute_slope_change(data[t])
            out.append({
                "name": name, "gain": f"{rel[t]:+.2%}",
                "is_positive": slope > 0, "r2": r2, "slope": slope,
                "gradient": gradient, "slope_change": slope_change
            })
        return {"all_ranked": sorted(out, key=lambda x: x["slope"], reverse=True)}
    except Exception as e:
        print(f"Sector Rotation Error: {e}")
        return {"all_ranked": []}


@ttl_cache(30)
def get_country_rotation():
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        results = []
        for s, n in COUNTRIES.items():
            slope, r2 = _trend_stats(data[s], 20, 20)
            gradient = _compute_gradient(data[s].tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(data[s])
            results.append({"sym": s, "name": n, "slope": slope, "r2": r2,
                            "gradient": gradient, "slope_change": slope_change})
        return results
    except Exception as e:
        print(f"Country Rotation Error: {e}")
        return []


@ttl_cache(30)
def get_commodity_rotation():
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        results = []
        for s, n in COMMODITIES.items():
            slope, r2 = _trend_stats(data[s], 20, 20)
            gradient = _compute_gradient(data[s].tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(data[s])
            results.append({"sym": s, "name": n, "slope": slope, "r2": r2,
                            "gradient": gradient, "slope_change": slope_change})
        return results
    except Exception as e:
        print(f"Commodity Rotation Error: {e}")
        return []


@ttl_cache(30)
def get_currency_rotation():
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        invert_set = {"EURUSD=X", "GBPUSD=X", "AUDUSD=X"}
        results = []
        for s, n in CURRENCIES.items():
            c = data[s].dropna().tail(60)
            if s not in invert_set:
                c = (1 / c).replace([np.inf, -np.inf], np.nan).dropna()
            if len(c) < 5:
                continue
            slope, r2 = _trend_stats(c, 60, 60)
            gradient = _compute_gradient(c.tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(c)
            results.append({"sym": s, "name": n, "slope": round(slope, 4), "r2": r2,
                            "gradient": gradient, "slope_change": slope_change})
        return results
    except Exception as e:
        print(f"Currency Rotation Error: {e}")
        return []


# =========================
# V. Portfolio Optimizer
# =========================
_portfolio_summary = {}


def get_portfolio_summary():
    return _portfolio_summary


def _compute_covariance_matrix(shared_data, tickers, window=60):
    prices = shared_data['Close'][tickers].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    cov_matrix = log_returns.tail(window).cov() * 252
    corr_matrix = log_returns.tail(window).corr()
    return cov_matrix, corr_matrix


def _compute_portfolio_var(weights, cov_matrix):
    return float(weights @ cov_matrix.values @ weights)


def _compute_risk_contribution(weights, cov_matrix):
    """
    Risk contributions for REPORTING ONLY.

    The epsilon (1e-9) makes this safe at w~0 for dashboard display, but
    non-homogeneous - do NOT use inside optimizer constraints; use the
    homogeneous formulation in _kelly_covariance_optimizer instead.
    """
    port_var = _compute_portfolio_var(weights, cov_matrix)
    marginal = cov_matrix.values @ weights
    return weights * marginal / (port_var + 1e-9)


def _conviction_score(item):
    # 2026-06-11: ml_conf replaced by strength (validated trend-quality
    # multiplier). ml_conf per-asset was OOS noise (transfer AUC 0.47).
    # Fallback to r2 keeps the function safe on items without strength.
    return max(item["slope"] * item["r2"] * item.get("strength", item["r2"]), 0.0)


def _kelly_covariance_optimizer(
        trends,
        shared_data,
        portfolio_value=500000,
        max_single=0.15,
        max_risk_contribution=0.30,
        min_portfolio_vol=0.08,
        max_portfolio_vol=0.135,
        regime_scalar=1.0,
        conviction_threshold=0.5,
        regime=None,
):
    """
    Covariance portfolio optimizer with dynamic vol cap.
    (Name retains 'kelly' for signature stability; raw weights are now
    vol-targeted - continuous Kelly under the equal-Sharpe assumption.)

    Note: conviction_threshold is UNUSED - quality gating happens upstream
    in _compute_kelly_size (slope/r2/strength gates produce zero
    dollar_amount, which zeroes the upper bound here). Parameter retained
    for signature compatibility only.

    Risk-contribution constraint notes (2026-06-10 fix):

    1. FEASIBILITY - Contributions sum to 1, so max contribution >= 1/n.
       effective_max_rc = max(max_risk_contribution, 1/n + 0.05).

    2. HOMOGENEITY - Constraint formulated as
       effective_max_rc * (wTSw) - wi(Sw)i >= 0 (degree-2 homogeneous),
       removing the epsilon-degenerate attractor near w=0.

    3. FALLBACK - On SLSQP failure: clip per-name at max_single, rescale
       whole vector to the effective vol cap.
    """
    empty_summary = {
        "total_allocated": 0.0, "portfolio_vol": 0.0, "n_positions": 0,
        "max_risk_contributor": "N/A", "optimization_success": False,
        "regime_scalar": round(regime_scalar, 2),
        # 2026-06-11: percent units, consistent with all other return paths
        "vol_cap": round(max_portfolio_vol * 100, 1),
        "max_rc": round(max_risk_contribution * 100, 1),
        "fallback_used": False,
    }

    if regime is not None:
        effective_vol_cap = _effective_vol_cap(
            regime,
            base_min=min_portfolio_vol,
            base_max=max_portfolio_vol,
        )
    else:
        effective_vol_cap = max_portfolio_vol

    active_signals = [
        t for t in trends
        if any(s in t["status"] for s in ["BUY", "STRONG BUY", "BREAKOUT", "HOLD"])
    ]

    if not active_signals:
        return trends, {**empty_summary, "vol_cap": round(effective_vol_cap * 100, 1)}

    tickers = [t["sym"] for t in active_signals]
    available = [t for t in tickers if t in shared_data['Close'].columns]

    if not available:
        return trends, {**empty_summary, "vol_cap": round(effective_vol_cap * 100, 1)}

    # Single asset bypass path
    if len(available) < 2:
        sym = available[0]
        item = next(t for t in trends if t["sym"] == sym)
        w = min(max(item["dollar_amount"] / portfolio_value, 0.02) * regime_scalar, max_single)
        dollar_amount = round(w * portfolio_value, 2)
        updated_trends = []
        for t in trends:
            if t["sym"] == sym:
                t = {**t, "pos_size": f"${dollar_amount:,.0f}", "dollar_amount": dollar_amount,
                     "weight_pct": round(w * 100, 1), "risk_contribution": 100.0,
                     "top_correlations": {}}
            else:
                t = {**t, "weight_pct": 0.0, "risk_contribution": 0.0, "top_correlations": {}}
            updated_trends.append(t)
        return updated_trends, {
            "total_allocated": round(w * 100, 1), "portfolio_vol": 0.0,
            "n_positions": 1, "max_risk_contributor": sym,
            "optimization_success": True,
            "regime_scalar": round(regime_scalar, 2),
            "vol_cap": round(effective_vol_cap * 100, 1),
            "max_rc": 100.0,
            "fallback_used": False,
            }

    cov_matrix, corr_matrix = _compute_covariance_matrix(shared_data, available)
    cov_values = cov_matrix.values

    # Raw weights from sizing (vol-targeted), regime-scaled.
    # Variable name kept for diff stability.
    kelly_weights = np.array([
        next(t["dollar_amount"] for t in active_signals if t["sym"] == sym)
        / portfolio_value * regime_scalar
        for sym in available
    ])

    # Instruments with zero raw weight get zero upper bound -
    # prevents optimizer allocating to instruments below quality threshold
    def upper_bound(kelly_w):
        if kelly_w > 0:
            return min(kelly_w, max_single)
        return 0.0

    bounds = [(0.0, upper_bound(kelly_weights[i])) for i in range(len(available))]
    x0 = np.array([w if w > 0 else 0.0 for w in kelly_weights])

    conviction_raw = np.array([
        _conviction_score(next(t for t in active_signals if t["sym"] == sym))
        for sym in available
    ])

    conviction_sum = conviction_raw.sum()
    if conviction_sum == 0:
        return trends, {**empty_summary, "vol_cap": round(effective_vol_cap * 100, 1)}

    conviction = conviction_raw / conviction_sum
    n = len(available)

    effective_max_rc = max(max_risk_contribution, 1.0 / n + 0.05)

    def neg_objective(weights):
        return -float(weights @ conviction)

    def rc_constraint_factory(i):
        def rc_constraint(w):
            port_var = float(w @ cov_values @ w)
            marginal_i = float(cov_values[i] @ w)
            return effective_max_rc * port_var - w[i] * marginal_i
        return rc_constraint

    constraints = [
        {"type": "ineq", "fun": lambda w: 1.0 - w.sum()},
        {"type": "ineq",
         "fun": lambda w: effective_vol_cap ** 2 - _compute_portfolio_var(w, cov_matrix)},
        *[
            {"type": "ineq", "fun": rc_constraint_factory(i)}
            for i in range(n)
        ]
    ]

    result = minimize(
        neg_objective, x0=x0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 1000, "ftol": 1e-9}
    )

    fallback_used = False
    if result.success:
        optimized_weights = result.x
    else:
        # Vol-capped fallback: clip per-name, rescale vector to vol cap.
        fallback_used = True
        fb = np.minimum(kelly_weights, max_single)
        fb_vol = float(np.sqrt(_compute_portfolio_var(fb, cov_matrix)))
        if fb_vol > effective_vol_cap and fb_vol > 0:
            fb = fb * (effective_vol_cap / fb_vol)
        optimized_weights = fb

    port_vol = np.sqrt(_compute_portfolio_var(optimized_weights, cov_matrix))
    risk_contribs = _compute_risk_contribution(optimized_weights, cov_matrix)

    ticker_to_weight = dict(zip(available, optimized_weights))
    ticker_to_risk = dict(zip(available, risk_contribs))
    ticker_to_corr = {t: corr_matrix[t].to_dict() for t in available}

    updated_trends = []
    for item in trends:
        if item["sym"] in ticker_to_weight:
            w = ticker_to_weight[item["sym"]]
            dollar_amount = round(w * portfolio_value, 2)
            top_corr = dict(list({
                k: round(v, 2)
                for k, v in sorted(
                    ticker_to_corr[item["sym"]].items(),
                    key=lambda x: abs(x[1]), reverse=True
                )
                if k != item["sym"]
            }.items())[:3])
            item = {
                **item,
                "pos_size": f"${dollar_amount:,.0f}",
                "dollar_amount": dollar_amount,
                "weight_pct": round(w * 100, 1),
                "risk_contribution": round(ticker_to_risk[item["sym"]] * 100, 1),
                "top_correlations": top_corr,
            }
        else:
            item = {**item, "weight_pct": 0.0, "risk_contribution": 0.0,
                    "top_correlations": {}}
        updated_trends.append(item)

    return updated_trends, {
        "total_allocated": round(float(optimized_weights.sum()) * 100, 1),
        "portfolio_vol": round(port_vol * 100, 1),
        "n_positions": int((optimized_weights > 0.001).sum()),
        "max_risk_contributor": available[int(np.argmax(risk_contribs))],
        "optimization_success": result.success,
        "regime_scalar": round(regime_scalar, 2),
        "vol_cap": round(effective_vol_cap * 100, 1),
        "max_rc": round(effective_max_rc * 100, 1),
        "fallback_used": fallback_used,
    }


# =========================
# VI. Position Sizing - VOL-TARGETING (2026-06-11)
#
# Signature unchanged from the Kelly-p version so get_trends and the
# optimizer need no plumbing changes. Internals replaced:
#
# WHY: per-asset ml_conf_slow had no OOS edge (transfer AUC 0.47 = noise
# on every non-SPY name). Kelly's discrete form f*=(bp-q)/b is unusable
# without an honest p. Continuous Kelly f*=mu/sigma^2 under the
# equal-Sharpe assumption (any instrument passing the trend filters has
# the same expected Sharpe) collapses to f* prop 1/sigma - vol-targeting.
# This IS Kelly, with the unestimable input (per-asset mu/p) removed.
#
# Parameter reinterpretation (names kept for signature stability):
#   ml_conf_slow        - IGNORED for sizing (accepted for compatibility)
#   kelly_fraction      - scales the per-position vol budget:
#                         vol_budget = 0.05 * (kelly_fraction / 0.25)
#                         default 0.25 -> 5% annualised vol contribution
#                         per full-strength position
#   divergence_discount - now carries the hurst + p_stop geometry
#                         discount from get_trends (ML divergence term
#                         removed at the call site)
#   delta_slope         - retained momentum adjust on the vol budget
#
# strength = tanh(slope*r2/8): saturating trend quality in [0,1).
#   slope*r2 ~ 4 -> 0.46, ~ 8 -> 0.76, ~ 16 -> 0.96 - no single hot name
#   dominates, replacing the role ml_conf played in conviction.
#
# exp_return: geometric expectation from the CORRECTED first-passage
# probabilities - reward*(1-p_stop) - risk*p_stop. Not an ML claim.
# =========================
def _compute_kelly_size(price, slope, atr, ml_conf_slow, r2,
                        portfolio_value=500000,
                        projection_days=63,
                        atr_stop_multiplier=2.5,
                        kelly_fraction=0.25,
                        max_allocation=0.15,
                        delta_slope=0.0,
                        divergence_discount=0.0):
    """
    Vol-targeted position sizing. Signature and return keys unchanged
    from the Kelly-p version; ml_conf_slow no longer affects sizing.
    """
    stop_price = price - (atr * atr_stop_multiplier)
    zero = {
        'dollar_amount': 0.0, 'shares': 0,
        'stop': round(stop_price, 2),
        'target': round(price, 2), 'rr_ratio': 0.0,
        'risk_dollar': 0.0, 'exp_return': 0.0
    }


    if slope <= 0 or r2 < 0.15:
        return zero

    if ml_conf_slow < 42:
        return zero

    if price <= 0 or atr <= 0 or stop_price >= price:
        return zero

    risk_per_share = price - stop_price

    # Vol budget from kelly_fraction knob (0.25 -> 5% per position)
    vol_budget = 0.05 * (kelly_fraction / 0.25)

    # Momentum adjust - retained behavior, now on the vol budget
    if delta_slope > 3:
        vol_budget = min(vol_budget * 1.25, 0.08)
    elif delta_slope < -3:
        vol_budget = vol_budget * 0.75

    # Target from slope projection, R2-damped (unchanged logic)
    # slope = 1000 * daily log return -> daily_return = slope / 1000
    daily_return_pct = slope / 1000

    projected_price = price * ((1 + daily_return_pct) ** projection_days)
    target_price = price + (projected_price - price) * r2
    target_price = max(target_price, price * 1.01)
    reward_per_share = target_price - price
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

    if rr_ratio < 1.0:
        return {**zero, 'stop': round(stop_price, 2),
                'target': round(target_price, 2), 'rr_ratio': round(rr_ratio, 2)}

    # Geometry - corrected first-passage probability
    p_stop = _stop_hit_probability(price, stop_price, target_price, slope, atr)

    # Trend-quality strength, saturating
    strength = float(np.tanh(slope * r2 / 8.0))

    # Apply combined discount (hurst + p_stop geometry, from get_trends)
    strength = strength * (1.0 - divergence_discount)
    strength = float(np.clip(strength, 0.0, 1.0))

    if strength <= 0.0:
        return {**zero, 'stop': round(stop_price, 2),
                'target': round(target_price, 2), 'rr_ratio': round(rr_ratio, 2)}

    # Vol-target weight: budget * strength / instrument vol
    instrument_vol = (atr / price) * np.sqrt(252.0)
    if instrument_vol <= 0:
        return zero
    raw_weight = (vol_budget * strength) / instrument_vol
    final_allocation = max(0.0, min(raw_weight, max_allocation))

    position_value = portfolio_value * final_allocation
    shares = int(position_value / price)
    actual_investment = shares * price
    risk_per_position = shares * risk_per_share
    reward_per_position = shares * reward_per_share

    # Geometric expectation from first-passage probabilities
    p_target = 1.0 - p_stop
    expected_return = reward_per_position * p_target - risk_per_position * p_stop

    return {
        'dollar_amount': round(actual_investment, 2),
        'shares': shares,
        'stop': round(stop_price, 2),
        'target': round(target_price, 2),
        'rr_ratio': round(rr_ratio, 2),
        'risk_dollar': round(risk_per_position, 2),
        'exp_return': round(expected_return, 2)
    }


# =========================
# VII. Trends (Main Entry Point)
# =========================
@ttl_cache(30)
def get_trends():
    global _portfolio_summary
    # 2026-06-11: per-asset ML is DISPLAY-ONLY telemetry. OOS validation
    # showed no edge on non-SPY names (transfer AUC 0.47), so ml_conf
    # values are still computed and returned for dashboard monitoring,
    # but no longer enter sizing, status logic, or conviction scoring.
    # 2026-06-18: that telemetry is now batched into ONE predict_proba per
    # model for the whole universe instead of one call per asset (~33x).
    # Outputs are identical (verified vs the per-asset path); this only
    # removes the per-call sklearn predict overhead that dominated load time.
    from macro.ml_engine import get_dual_ml_confidence_for_kelly_batch

    shared_data = _get_shared_market_data()
    results = []
    symbols = list(TREND_ASSETS.keys())

    # Pre-build each asset's frame once (reused in the loop below) and batch
    # the display-only dual-ML telemetry across the whole universe.
    _dfs = {}
    for _sym in symbols:
        try:
            if len(symbols) > 1:
                _d = shared_data.xs(_sym, level=1, axis=1).dropna()
            else:
                _d = shared_data.dropna()
            if not _d.empty:
                _dfs[_sym] = _d
        except Exception:
            continue
    try:
        _dual_map = get_dual_ml_confidence_for_kelly_batch(list(_dfs.items()))
    except Exception:
        _dual_map = {}

    for sym, name in TREND_ASSETS.items():
        try:
            df = _dfs.get(sym)
            if df is None or df.empty:
                continue

            c = df["Close"].squeeze()
            ma_50 = c.rolling(50, min_periods=1).mean()
            ma_200 = c.rolling(200, min_periods=1).mean()

            slope, r2 = _trend_stats(c, 20, 10)

            # Dual ML - TELEMETRY ONLY (2026-06-11). Real values returned
            # for dashboard monitoring; never consumed by sizing or status.
            # Sourced from the batched predict above; missing -> neutral.
            dual = _dual_map.get(sym)
            if dual is not None:
                ml_conf_slow = dual['slow']
                ml_conf_fast = dual['fast']
                ml_divergence = dual['divergence']
                ml_regime = dual['regime']
            else:
                ml_conf_slow = 50.0
                ml_conf_fast = 50.0
                ml_divergence = 0.0
                ml_regime = "N/A"

            atr = float(compute_ATR(df, 14).iloc[-1])
            last = float(c.iloc[-1])
            s50 = float(ma_50.iloc[-1])
            s200 = float(ma_200.iloc[-1])

            rsi14 = float(compute_RSI(c, 14).iloc[-1])

            # Hurst exponent - trend persistence
            hurst = _hurst_exponent(c, max_lag=40)

            # Z-Score calculation - vectorized rolling 20-day log-slope
            # (2026-06-29; window widened 10->20 with the headline trend).
            # For window=20, scale=10: slope = (centered_x . log_prices) /
            # 665 * 1000 - same closed-form OLS, now over a 20-bar window
            # so the z-score is built from 20-day slopes (consistent with
            # the 20-day headline slope above). 665 = sum((arange(20)-9.5)**2).
            # scale stays 10 -> slope remains "1000 x daily log return", the
            # convention kelly-size / p_stop / target projection all assume.
            # Scan window kept at 60 bars -> 41 historical slopes (was 51 at
            # window=10); widen the 60 to ~70 if you want the same sample depth.
            c_len = len(c)
            start_idx = max(0, c_len - 60)
            seg = np.log(c.values[start_idx:])
            if len(seg) >= 20:
                if sliding_window_view is not None:
                    _windows = sliding_window_view(seg, 20)
                else:  # numpy < 1.20 fallback
                    _windows = np.stack([seg[k:k + 20] for k in range(len(seg) - 19)])
                _xc = np.arange(20) - 9.5
                hist_slopes = np.round((_windows @ _xc) / 665.0 * 1000.0, 2)
            else:
                hist_slopes = np.empty(0)

            if hist_slopes.size:
                slope_mean = np.mean(hist_slopes)
                slope_std = np.std(hist_slopes)
                slope_z = (slope - slope_mean) / slope_std if slope_std > 0 else 0
            else:
                slope_z = 0

            delta_slope = _compute_delta_slope(c, window=20)

            # Stop hit probability - geometric assessment, CORRECTED formula.
            # Target floored at 1% above price to avoid the target<=price guard.
            _atr_stop = last - (atr * 2.5)
            _daily_return = slope / 1000
            _projected = last * ((1 + _daily_return) ** 63)
            _target_for_pstop = max(_projected, last * 1.01)

            p_stop = _stop_hit_probability(
                last, _atr_stop, _target_for_pstop, slope, atr
            )

            # Hurst discount - mean-reverting series defunded
            hurst_discount = 0.0
            if hurst < 0.45:
                hurst_discount = (0.45 - hurst) * 0.5   # up to +22.5% discount
            elif hurst > 0.55:
                hurst_discount = -(hurst - 0.55) * 0.2  # up to -10% (bonus)

            # Geometry discount - with the corrected p_stop this now binds
            p_stop_discount = max(0.0, (p_stop - 0.40) / 0.30 * 0.50) if p_stop > 0.40 else 0.0

            # Combined discount: hurst + geometry.
            # ML divergence term removed (2026-06-11) - fast model retired.
            combined_discount = float(np.clip(
                hurst_discount + p_stop_discount,
                -0.20,
                 0.80
            ))

            # Vol-targeted sizing (signature unchanged; ml arg neutral 50)
            position = _compute_kelly_size(
                last, slope, atr, ml_conf_slow, r2,
                delta_slope=delta_slope,
                divergence_discount=combined_discount
            )
            pos_size = position['dollar_amount']

            # strength for conviction scoring / dashboard
            strength = float(np.clip(
                np.tanh(slope * r2 / 8.0) * (1.0 - combined_discount), 0.0, 1.0
            )) if slope > 0 else 0.0

            # =============================================
            # Decision Logic - ML-free (2026-06-11)
            # Per-asset ML branches removed; price/MA/slope/RSI/geometry
            # conditions were what actually drove statuses for 32 of 33
            # tickers anyway. Status vocabulary unchanged so the optimizer
            # filter and dashboard need no changes.
            # =============================================
            # sell
            if last < position['stop']:
                status = "SELL (STOP)"

            elif last < s50 and slope < 0:
                # Price below MA50 AND slope negative -> confirmed downtrend
                status = "SELL (MA50)"

            elif slope_z > 2.0 and r2 > 0.7 and rsi14 < 70 and slope > 0:
                # Momentum breakout - slope extension with strong fit, not overbought
                status = "BUY (BREAKOUT)"

            elif slope_z > 2.0 and r2 > 0.8 and rsi14 > 70:
                # Slope very extended AND overbought -> trim
                status = "TRIM (EXTENDED)"

            elif slope_z > 1.5 and delta_slope < -3:

                status = "TRIM (FADING MOMENTUM)"

                # Mean-reversion swing: oversold dip inside an intact uptrend.
                # MUST sit above the geometry / negative-slope / position-size
                # trims below — each would otherwise shadow it. A down-leg sharp
                # enough to print rsi<30 also tends to push p_stop>0.55 and
                # slope<-2, and because _compute_kelly_size rejects slope<=0 it
                # forces pos_size==0. Naturally exclusive with SELL(MA50) via
                # last>s50 and with the slope_z>0 branches above.

            elif (hurst < 0.45 and last > s50 and rsi14 < 30
                and slope_z < -1.5 and slope < 0):
                status = "BUY (MR SWING)"
            elif p_stop > 0.55 and last > s50:
                status = "TRIM (GEOMETRY)"
            elif slope < -2:
                # Slope significantly negative
                status = "TRIM (NEGATIVE SLOPE)"

            elif pos_size == 0:
                status = "TRIM (POSITION SIZE)"

            # buy/hold zone
            elif (last > s200) and (last > s50) and (slope > 0) and (r2 > 0.6):
                # Strong uptrend - entry quality from momentum position
                if slope_z < 0 and rsi14 < 60:
                    # Slope below its own recent mean - pullback within uptrend
                    status = "BUY (PULLBACK)"
                elif slope_z > 1.0:
                    status = "BUY (BULL)"
                else:
                    status = "BUY"

            else:
                status = "HOLD"

            results.append({
                "sym": sym,
                "name": name,
                "price": round(last, 2),
                "status": status,
                "r2": round(r2, 2),
                # ml_conf fields are TELEMETRY - real model outputs for
                # dashboard monitoring. OOS-invalidated for decisions
                # (transfer AUC 0.47); nothing downstream consumes them.
                "ml_conf": round(strength*100,1),
                "ml_conf_slow": ml_conf_slow,
                "ml_conf_fast": ml_conf_fast,
                "divergence": ml_divergence,
                "regime": ml_regime,
                "strength": round(strength, 3),     # trend-quality multiplier [0,1]
                "rsi14": round(rsi14, 1),
                "slope": round(slope, 2),
                "slope_z": round(slope_z, 2),
                "delta_slope": round(delta_slope, 4),
                "hurst": hurst,                     # trend persistence [0,1]
                "p_stop": p_stop,                   # corrected stop-before-target prob
                "stop": position['stop'],
                "target": position['target'],
                "rr_ratio": position['rr_ratio'],
                "pos_size": f"${position['dollar_amount']:,.0f}",
                "dollar_amount": position['dollar_amount'],
                "shares": position['shares'],
                "risk_dollar": position['risk_dollar'],
                "exp_return": position['exp_return'],
                "weight_pct": 0.0,
                "risk_contribution": 0.0,
                "top_correlations": {},
            })

        except Exception as e:
            print(f"Error in trend loop for {sym}: {e}")
            continue

    # Sort by conviction: slope * r2 * strength, with p_stop as tiebreaker
    # for the slope<=0 cluster (where slope*r2*strength == 0 for all of
    # them) - lower p_stop ranks higher among ties, surfacing instruments
    # with the best stop geometry even when the system won't allocate to
    # them. reverse=True applies to both tuple elements; -p_stop ascending
    # under reverse=True yields p_stop ascending overall (verified).
    sorted_results = sorted(
        results,
        key=lambda x: (
            x["slope"] * x["r2"] * x.get("strength", x["r2"]),
            -x.get("p_stop", 0.5)
        ),
        reverse=True
    )

    # Get regime once - passed to optimizer for dynamic vol cap
    regime = get_risk_regime()
    scalar = get_regime_scalar(regime)

    optimized_results, summary = _kelly_covariance_optimizer(
        sorted_results, shared_data,
        portfolio_value=500000,
        max_single=0.25,
        max_risk_contribution=0.35,
        min_portfolio_vol=0.08,     # floor: 8%   (~1/9 Kelly at Sharpe ~0.7)
        max_portfolio_vol=0.135,     # ceiling: 13.5% (~1/5 Kelly) - see
                                    # note below; the Kelly fraction now
                                    # lives HERE, in one place
        regime_scalar=scalar,
        conviction_threshold=0.5,
        regime=regime,              # pass regime for dynamic vol cap
    )

    _portfolio_summary = summary
    return optimized_results

# NOTE on max_portfolio_vol=0.135 vs "~1/5 Kelly ~ 15%" comments elsewhere
# in this file (_effective_vol_cap docstring, empty_summary default):
# the live ceiling is 13.5%. Whether 13.5% or 15% is the intended final
# value is an open decision - not changed here. If 15% is intended,
# update this call site to 0.15; if 13.5% is intended, the "15%"
# comments in _effective_vol_cap's docstring and elsewhere should be
# updated to "13.5%" for consistency. Flagging only, not resolving.