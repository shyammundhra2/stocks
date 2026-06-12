import yfinance as yf
import numpy as np
import pandas as pd
import threading
import time
import math
from functools import wraps, lru_cache
from scipy.optimize import minimize

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
    raw = yf.download(all_tickers, period="1y", progress=False,
                      auto_adjust=False, threads=True, group_by='column')
    return raw


@ttl_cache(30)
def _get_extended_data():
    macro_proxies = ['HYG', 'IEF', '^TNX', '^IRX', 'JPY=X', 'HG=F', 'GC=F']
    risk_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys())))
    all_tickers = list(set(risk_tickers + macro_proxies + ['^VIX', '^MOVE']))
    raw = yf.download(all_tickers, period="300d", progress=False,
                      auto_adjust=False, threads=True)
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


def _trend_stats(series, window, scale):
    c = series.dropna().tail(window)
    if len(c) < 5:
        return 0.0, 0.0
    y = np.log(c.values)
    coeffs = np.polyfit(np.arange(len(y)), y, 1)
    slope = float(coeffs[0]) * scale * 100
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
# Hurst Exponent
#
#   H > 0.55  → trending     — momentum persists, trend system has edge
#   H 0.45–0.55 → random walk — no structural edge, reduce sizing
#   H < 0.45  → mean-reverting — trend system is fighting the series
#
# Method: Rescaled Range (R/S) analysis over log returns.
# =========================
def _hurst_exponent(series, max_lag=40):
    """
    Compute Hurst exponent via rescaled range (R/S) analysis.
    Returns float in [0, 1]. 0.5 = random walk default on error.
    """
    try:
        c = np.array(series.dropna()) if hasattr(series, 'dropna') else np.array(series)
        if len(c) < max_lag + 10:
            return 0.5

        log_returns = np.diff(np.log(c))
        if len(log_returns) < max_lag:
            return 0.5

        lags = range(10, max_lag)
        rs_values = []

        for lag in lags:
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

        if len(rs_values) < 5:
            return 0.5

        log_lags = np.log(list(lags)[:len(rs_values)])
        log_rs = np.log(rs_values)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        h = float(coeffs[0])

        return round(float(np.clip(h, 0.0, 1.0)), 3)

    except Exception:
        return 0.5


# =========================
# Stop Hit Probability — CORRECTED (2026-06-11)
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
# — both inverted. It passed the infinite-drift limit checks but was wrong
# at every finite drift, systematically UNDERSTATING p_stop (e.g. driftless
# 2.5xATR stop with rr=2: old said 0.33, truth is 0.67). Validated against
# Monte Carlo GBM 2026-06-11.
#
# Drift units also corrected: slope = 1000 * daily log return, so
# annual drift = slope/1000 * 252 = slope * 0.252 (old code used slope/100,
# ~25x understated).
#
# Interpretation (with honest values these gates now actually bind):
#   p_stop < 0.25   → strong geometry — full size
#   p_stop 0.25–0.40 → standard sizing
#   p_stop > 0.40   → reduce size
#   p_stop > 0.55   → unfavorable — do not enter / exit existing
# =========================
def _stop_hit_probability(price, stop, target, slope, atr, projection_days=63):
    """
    Probability that price hits stop before target under GBM with drift.
    Returns float in [0,1]; 0.5 on degenerate inputs.
    """
    try:
        if price <= 0 or stop >= price or target <= price:
            return 0.5
        if atr <= 0:
            return 0.5

        # slope = 1000 * daily log return → annualised = slope * 0.252
        annual_drift = slope * 0.252
        annual_vol = (atr / price) * np.sqrt(252)
        if annual_vol <= 0:
            return 0.5

        a = np.log(stop / price)    # < 0
        b = np.log(target / price)  # > 0

        mu = annual_drift - 0.5 * annual_vol ** 2   # Ito-corrected drift
        sigma2 = annual_vol ** 2

        if abs(mu) < 1e-10:
            # Driftless: optional stopping — nearer barrier hit more often
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
# Dynamic Vol Cap — REWEIGHTED (2026-06-11)
#
# Fast model retired from this calculation: holdout AUC 0.516 (coin flip).
# A vol ceiling modulated 50% by noise was adding randomness, not crash
# detection. New weighting:
#   regime_scalar 0.60 — six observable technical conditions (validated
#                        by construction: hard facts, no inference)
#   ml_slow       0.40 — SPY slow model, the one ML signal that survived
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
    ml_slow = regime.get("ml_slow", 50.0) / 100.0        # 0.0 – 1.0
    regime_scalar = get_regime_scalar(regime)             # 0.0 – 1.0

    combined = (
        regime_scalar * 0.60   # observable macro conditions — primary
        + ml_slow     * 0.40   # validated SPY structural signal
    )

    vol_cap = base_min + combined * (base_max - base_min)
    return round(max(base_min, min(vol_cap, base_max)), 4)


# =========================
# I. Risk Regime
# =========================
@ttl_cache(30)
def get_risk_regime():
    try:
        raw, risk_tickers = _get_extended_data()
        data = raw['Close'].ffill()

        # -----------------------------------------------
        # ML Risk Model — 20-day stride history (5 points)
        # RETAINED FOR DASHBOARD SPARKLINE ONLY (2026-06-11):
        # fast model holdout AUC 0.516 — no longer enters the
        # composite or any sizing/risk decision.
        # -----------------------------------------------
        history_points = []
        recent_dates = data.index[::-20][:5][::-1]

        for ts in recent_dates:
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

        ml_fast_conf = history_points[-1]

        # -----------------------------------------------
        # SPY Slow ML Prediction — the validated signal
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
            ml_slow_conf = 50.0   # neutral — do NOT fall back to fast (noise)
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
        # Composite RISK-ON / RISK-OFF Signal — REWEIGHTED (2026-06-11)
        #
        #   55% technical — 6 observable macro conditions, hard facts
        #   45% ML slow   — SPY structural trend, OOS-validated (0.625)
        #    0% ML fast   — RETIRED: holdout AUC 0.516 = coin flip.
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

        return {
            "status":         "RISK-ON" if is_risk_on else "RISK-OFF",
            "confidence":     round(composite_score * 100, 1),
            "ml_slow":        round(ml_slow_conf, 1),
            "ml_fast":        round(ml_fast_conf, 1),   # display only — not in composite
            "composite":      round(composite_score * 100, 1),
            "spy_regime":     spy_regime,
            "spy_divergence": round(spy_divergence, 1),
            "history":        history_points,
            "details":        details,
        }

    except Exception as e:
        print(f"Risk Regime Error: {e}")
        return {
            "status":         "ERROR",
            "confidence":     0,
            "ml_slow":        0,
            "ml_fast":        0,
            "composite":      0,
            "spy_regime":     "Error",
            "spy_divergence": 0,
            "history":        [],
            "details":        [],
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
            raise ValueError("RSI(2) returned NaN — insufficient data")

        price = float(c.iloc[-1])
        sma200 = float(c.rolling(200, min_periods=200).mean().iloc[-1])

        if pd.isna(sma200):
            raise ValueError("SMA200 returned NaN — insufficient data (need 200 bars)")

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

    The epsilon (1e-9) makes this safe at w≈0 for dashboard display, but
    non-homogeneous — do NOT use inside optimizer constraints; use the
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
        max_portfolio_vol=0.15,
        regime_scalar=1.0,
        conviction_threshold=0.5,
        regime=None,
):
    """
    Covariance portfolio optimizer with dynamic vol cap.
    (Name retains 'kelly' for signature stability; raw weights are now
    vol-targeted — continuous Kelly under the equal-Sharpe assumption.)

    Risk-contribution constraint notes (2026-06-10 fix):

    1. FEASIBILITY — Contributions sum to 1, so max contribution ≥ 1/n.
       effective_max_rc = max(max_risk_contribution, 1/n + 0.05).

    2. HOMOGENEITY — Constraint formulated as
       effective_max_rc * (wᵀΣw) - wᵢ(Σw)ᵢ ≥ 0 (degree-2 homogeneous),
       removing the epsilon-degenerate attractor near w=0.

    3. FALLBACK — On SLSQP failure: clip per-name at max_single, rescale
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

    def upper_bound(sym, kelly_w):
        if kelly_w > 0:
            return min(kelly_w, max_single)
        return 0.0

    bounds = [(0.0, upper_bound(sym, kelly_weights[i])) for i, sym in enumerate(available)]
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
# VI. Position Sizing — VOL-TARGETING (2026-06-11)
#
# Signature unchanged from the Kelly-p version so get_trends and the
# optimizer need no plumbing changes. Internals replaced:
#
# WHY: per-asset ml_conf_slow had no OOS edge (transfer AUC 0.47 = noise
# on every non-SPY name). Kelly's discrete form f*=(bp-q)/b is unusable
# without an honest p. Continuous Kelly f*=mu/sigma^2 under the
# equal-Sharpe assumption (any instrument passing the trend filters has
# the same expected Sharpe) collapses to f* ∝ 1/sigma — vol-targeting.
# This IS Kelly, with the unestimable input (per-asset mu/p) removed.
#
# Parameter reinterpretation (names kept for signature stability):
#   ml_conf_slow        — IGNORED for sizing (accepted for compatibility)
#   kelly_fraction      — scales the per-position vol budget:
#                         vol_budget = 0.05 * (kelly_fraction / 0.25)
#                         default 0.25 → 5% annualised vol contribution
#                         per full-strength position
#   divergence_discount — now carries the hurst + p_stop geometry
#                         discount from get_trends (ML divergence term
#                         removed at the call site)
#   delta_slope         — retained momentum adjust on the vol budget
#
# strength = tanh(slope*r2/8): saturating trend quality in [0,1).
#   slope*r2 ~ 4 → 0.46, ~ 8 → 0.76, ~ 16 → 0.96 — no single hot name
#   dominates, replacing the role ml_conf played in conviction.
#
# exp_return: geometric expectation from the CORRECTED first-passage
# probabilities — reward*(1-p_stop) - risk*p_stop. Not an ML claim.
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

    # Gate: positive trend with minimum fit quality.
    # ml_conf_slow gate removed (2026-06-11) — it was gating on noise.
    if slope <= 0 or r2 < 0.15:
        return zero

    if price <= 0 or atr <= 0 or stop_price >= price:
        return zero

    risk_per_share = price - stop_price

    # Vol budget from kelly_fraction knob (0.25 → 5% per position)
    vol_budget = 0.05 * (kelly_fraction / 0.25)

    # Momentum adjust — retained behavior, now on the vol budget
    if delta_slope > 3:
        vol_budget = min(vol_budget * 1.25, 0.08)
    elif delta_slope < -3:
        vol_budget = vol_budget * 0.75

    # Target from slope projection, R2-damped (unchanged logic)
    # slope = 1000 * daily log return → daily_return = slope / 1000
    daily_return_pct = slope / 1000

    projected_price = price * ((1 + daily_return_pct) ** projection_days)
    target_price = price + (projected_price - price) * r2
    target_price = max(target_price, price * 1.01)
    reward_per_share = target_price - price
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

    if rr_ratio < 1.0:
        return {**zero, 'stop': round(stop_price, 2),
                'target': round(target_price, 2), 'rr_ratio': round(rr_ratio, 2)}

    # Geometry — corrected first-passage probability
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
    from macro.ml_engine import get_dual_ml_confidence_for_kelly

    shared_data = _get_shared_market_data()
    results = []
    symbols = list(TREND_ASSETS.keys())

    for sym, name in TREND_ASSETS.items():
        try:
            if len(symbols) > 1:
                df = shared_data.xs(sym, level=1, axis=1).dropna()
            else:
                df = shared_data.dropna()

            if df.empty:
                continue

            c = df["Close"].squeeze()
            ma_50 = c.rolling(50, min_periods=1).mean()
            ma_200 = c.rolling(200, min_periods=1).mean()

            slope, r2 = _trend_stats(c, 10, 10)

            # Dual ML — TELEMETRY ONLY (2026-06-11). Real values returned
            # for dashboard monitoring; never consumed by sizing or status.
            # Failure degrades to neutral without affecting the loop.
            try:
                dual = get_dual_ml_confidence_for_kelly(df)
                ml_conf_slow = dual['slow']
                ml_conf_fast = dual['fast']
                ml_divergence = dual['divergence']
                ml_regime = dual['regime']
            except Exception:
                ml_conf_slow = 50.0
                ml_conf_fast = 50.0
                ml_divergence = 0.0
                ml_regime = "N/A"

            atr = float(compute_ATR(df, 14).iloc[-1])
            last = float(c.iloc[-1])
            s50 = float(ma_50.iloc[-1])
            s200 = float(ma_200.iloc[-1])

            rsi14 = float(compute_RSI(c, 14).iloc[-1])

            # Hurst exponent — trend persistence
            hurst = _hurst_exponent(c, max_lag=40)

            # Z-Score calculation
            c_len = len(c)
            start_idx = max(0, c_len - 60)
            hist_slopes = []
            for i in range(start_idx + 10, c_len + 1):
                window_slice = c.iloc[i - 10:i]
                if len(window_slice) >= 5:
                    s_val, _ = _trend_stats(window_slice, 10, 10)
                    hist_slopes.append(s_val)

            if hist_slopes:
                slope_mean = np.mean(hist_slopes)
                slope_std = np.std(hist_slopes)
                slope_z = (slope - slope_mean) / slope_std if slope_std > 0 else 0
            else:
                slope_z = 0

            delta_slope = _compute_delta_slope(c, window=20)

            # Stop hit probability — geometric assessment, CORRECTED formula.
            # Target floored at 1% above price to avoid the target<=price guard.
            _atr_stop = last - (atr * 2.5)
            _daily_return = slope / 1000
            _projected = last * ((1 + _daily_return) ** 63)
            _target_for_pstop = max(_projected, last * 1.01)

            p_stop = _stop_hit_probability(
                last, _atr_stop, _target_for_pstop, slope, atr
            )

            # Hurst discount — mean-reverting series defunded
            hurst_discount = 0.0
            if hurst < 0.45:
                hurst_discount = (0.45 - hurst) * 0.5   # up to +22.5% discount
            elif hurst > 0.55:
                hurst_discount = -(hurst - 0.55) * 0.2  # up to -10% (bonus)

            # Geometry discount — with the corrected p_stop this now binds
            p_stop_discount = max(0.0, (p_stop - 0.40) / 0.30 * 0.50) if p_stop > 0.40 else 0.0

            # Combined discount: hurst + geometry.
            # ML divergence term removed (2026-06-11) — fast model retired.
            combined_discount = float(np.clip(
                hurst_discount + p_stop_discount,
                -0.20,
                 0.80
            ))

            # Vol-targeted sizing (signature unchanged; ml arg neutral 50)
            position = _compute_kelly_size(
                last, slope, atr, 50.0, r2,
                delta_slope=delta_slope,
                divergence_discount=combined_discount
            )
            pos_size = position['dollar_amount']

            # strength for conviction scoring / dashboard
            strength = float(np.clip(
                np.tanh(slope * r2 / 8.0) * (1.0 - combined_discount), 0.0, 1.0
            )) if slope > 0 else 0.0

            # =============================================
            # Decision Logic — ML-free (2026-06-11)
            # Per-asset ML branches removed; price/MA/slope/RSI/geometry
            # conditions were what actually drove statuses for 32 of 33
            # tickers anyway. Status vocabulary unchanged so the optimizer
            # filter and dashboard need no changes.
            # =============================================
            # sell
            if last < position['stop']:
                status = "SELL (STOP)"

            elif last < s50 and slope < 0:
                # Price below MA50 AND slope negative → confirmed downtrend
                status = "SELL (MA50)"

            elif slope_z > 2.0 and r2 > 0.7 and rsi14 < 70 and slope > 0:
                # Momentum breakout — slope extension with strong fit, not overbought
                status = "BUY (BREAKOUT)"

            elif slope_z > 2.0 and r2 > 0.8 and rsi14 > 70:
                # Slope very extended AND overbought → trim
                status = "TRIM (EXTENDED)"

            elif slope_z > 1.5 and delta_slope < -3:
                # Momentum elevated but decelerating hard
                status = "TRIM (FADING MOMENTUM)"

            elif p_stop > 0.55 and last > s50:
                # Geometry unfavorable — stop more likely than target
                status = "TRIM (GEOMETRY)"

            elif slope < -2:
                # Slope significantly negative
                status = "TRIM (NEGATIVE SLOPE)"

            elif pos_size == 0:
                status = "TRIM (POSITION SIZE)"

            # buy/hold zone
            elif (last > s200) and (last > s50) and (slope > 0) and (r2 > 0.6):
                # Strong uptrend — entry quality from momentum position
                if slope_z < 0 and rsi14 < 60:
                    # Slope below its own recent mean — pullback within uptrend
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
                # ml_conf fields are TELEMETRY — real model outputs for
                # dashboard monitoring. OOS-invalidated for decisions
                # (transfer AUC 0.47); nothing downstream consumes them.
                "ml_conf": ml_conf_slow,
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

    # Sort by conviction: slope * r2 * strength
    sorted_results = sorted(
        results,
        key=lambda x: x["slope"] * x["r2"] * x.get("strength", x["r2"]),
        reverse=True
    )

    # Get regime once — passed to optimizer for dynamic vol cap
    regime = get_risk_regime()
    scalar = get_regime_scalar(regime)

    optimized_results, summary = _kelly_covariance_optimizer(
        sorted_results, shared_data,
        portfolio_value=500000,
        max_single=0.25,
        max_risk_contribution=0.35,
        min_portfolio_vol=0.08,     # floor: 8%   (~1/9 Kelly at Sharpe ~0.7)
        max_portfolio_vol=0.15,     # ceiling: 15% (~1/5 Kelly) — the Kelly
                                    # fraction now lives HERE, in one place
        regime_scalar=scalar,
        conviction_threshold=0.5,
        regime=regime,              # pass regime for dynamic vol cap
    )

    _portfolio_summary = summary
    return optimized_results