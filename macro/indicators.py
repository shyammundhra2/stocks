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
    """
    Production-hardened close series extraction.
    Handles MultiIndex, flat DataFrames, and single-ticker Series.
    """
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
@lru_cache(maxsize=128)
def _safe_r2_cached(y_tuple, coeffs_tuple):
    y = np.array(y_tuple)
    coeffs = np.array(coeffs_tuple)
    if len(y) < 2:
        return 0.0
    y_hat = np.polyval(coeffs, np.arange(len(y)))
    ss_res = np.sum(np.square(y - y_hat))
    y_mean = np.mean(y)
    ss_tot = np.sum(np.square(y - y_mean))
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def _safe_r2(y, coeffs):
    return _safe_r2_cached(tuple(y), tuple(coeffs))


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
# I. Risk Regime
# =========================
@ttl_cache(30)
def get_risk_regime():
    try:
        raw, risk_tickers = _get_extended_data()
        data = raw['Close'].ffill()

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

        current_conf = history_points[-1]
        is_risk_on_ml = current_conf > 60

        last_vals = data.iloc[-1]
        ma50 = data.rolling(50).mean().iloc[-1]

        credit_ratio = data['HYG'] / data['IEF']
        credit_pass = bool(last_vals['HYG'] / last_vals['IEF'] > credit_ratio.rolling(50).mean().iloc[-1])

        curve_spread = last_vals['^TNX'] - last_vals['^IRX']
        curve_pass = bool(curve_spread > 0)

        jpy_ret = data['JPY=X'].pct_change()
        jpy_vol = jpy_ret.rolling(20).std().iloc[-1] * np.sqrt(252)
        carry_pass = bool(last_vals['JPY=X'] > ma50['JPY=X'] and jpy_vol < 0.15)

        spy_ma200 = data['SPY'].rolling(200).mean().iloc[-1]
        spy_trend = bool(last_vals['SPY'] > spy_ma200)
        vix_low = bool(last_vals['^VIX'] < 20 and last_vals['^MOVE'] < 110)

        rsp_spy_ratio = data['RSP'] / data['SPY']
        breadth_pass = bool(rsp_spy_ratio.iloc[-1] > rsp_spy_ratio.rolling(50).mean().iloc[-1])

        details = [
            {"label": "Trend (SPY > 200MA)", "pass": spy_trend},
            {"label": "Fear (VIX/MOVE Low)", "pass": vix_low},
            {"label": "Breadth (RSP/SPY > 50MA)", "pass": breadth_pass},
            {"label": "Credit (HYG/IEF Ratio)", "pass": credit_pass},
            {"label": "Curve (10Y-3M Spread)", "pass": curve_pass},
            {"label": "Carry (JPY Weak/Stable)", "pass": carry_pass},
        ]

        return {
            "status": "RISK-ON" if is_risk_on_ml else "RISK-OFF",
            "confidence": current_conf,
            "history": history_points,
            "details": details
        }
    except Exception as e:
        print(f"Risk Regime Error: {e}")
        return {"status": "ERROR", "confidence": 0, "history": [], "details": []}


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
        return {
            "top": top, "bottom": bottom,
            "spread": round(top["confidence"] - bottom["confidence"], 1),
            "all": ranked
        }
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
        return {
            "top": top, "bottom": bottom,
            "spread": round(top["confidence"] - bottom["confidence"], 1),
            "all": ranked
        }
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
        return {
            "top": top, "bottom": bottom,
            "spread": round(top["confidence"] - bottom["confidence"], 1),
            "all": ranked
        }
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


@ttl_cache(30)
def get_mean_reversion():
    try:
        shared_data = _get_shared_market_data()
        c = _get_close(shared_data, 'QQQ')

        if c.empty:
            raise ValueError("QQQ data empty")

        rsi2 = float(compute_RSI(c, 2).iloc[-1])
        price = float(c.iloc[-1])
        s200 = float(c.rolling(200, min_periods=1).mean().iloc[-1])

        if rsi2 >= 70:
            signal = "EXIT"
        elif price < s200:
            signal = "RISK OFF"
        elif rsi2 <= 10:
            signal = "BUY"
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
                # Shield from infinity / division by zero anomalies
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
    port_var = _compute_portfolio_var(weights, cov_matrix)
    marginal = cov_matrix.values @ weights
    return weights * marginal / (port_var + 1e-9)


def _conviction_score(item):
    return max(item["slope"] * item["r2"] * item["ml_conf"] / 100, 0.0)


def _kelly_covariance_optimizer(
        trends,
        shared_data,
        portfolio_value=500000,
        max_single=0.15,
        max_risk_contribution=0.35,
        max_portfolio_vol=0.15,
        regime_scalar=1.0,
        conviction_threshold=0.5,
):
    empty_summary = {
        "total_allocated": 0.0, "portfolio_vol": 0.0, "n_positions": 0,
        "max_risk_contributor": "N/A", "optimization_success": False,
        "regime_scalar": round(regime_scalar, 2),
    }

    active_signals = [
        t for t in trends
        if any(s in t["status"] for s in ["BUY", "STRONG BUY", "BREAKOUT", "HOLD"])
    ]

    if not active_signals:
        return trends, empty_summary

    tickers = [t["sym"] for t in active_signals]
    available = [t for t in tickers if t in shared_data['Close'].columns]

    if not available:
        return trends, empty_summary

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
            "optimization_success": True, "regime_scalar": round(regime_scalar, 2),
        }

    cov_matrix, corr_matrix = _compute_covariance_matrix(shared_data, available)

    kelly_weights = np.array([
        next(t["dollar_amount"] for t in active_signals if t["sym"] == sym)
        / portfolio_value * regime_scalar
        for sym in available
    ])

    x0 = np.array([w if w > 0 else 0.05 for w in kelly_weights])

    def upper_bound(sym, kelly_w):
        if kelly_w > 0:
            return min(kelly_w, max_single)
        item = next(t for t in active_signals if t["sym"] == sym)
        cs = _conviction_score(item)
        return max_single if cs > conviction_threshold else 0.05

    bounds = [(0, upper_bound(sym, kelly_weights[i])) for i, sym in enumerate(available)]

    conviction_raw = np.array([
        _conviction_score(next(t for t in active_signals if t["sym"] == sym))
        for sym in available
    ])

    # Catch zero-conviction system edge case
    conviction_sum = conviction_raw.sum()
    if conviction_sum == 0:
        return trends, empty_summary

    conviction = conviction_raw / conviction_sum
    n = len(available)

    def neg_objective(weights):
        return -float(weights @ conviction)

    constraints = [
        {"type": "ineq", "fun": lambda w: 1.0 - w.sum()},
        {"type": "ineq", "fun": lambda w: max_portfolio_vol ** 2 - _compute_portfolio_var(w, cov_matrix)},
        *[
            {"type": "ineq",
             "fun": lambda w, i=i: max_risk_contribution - _compute_risk_contribution(w, cov_matrix)[i]}
            for i in range(n)
        ]
    ]

    result = minimize(
        neg_objective, x0=x0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 1000, "ftol": 1e-9}
    )

    optimized_weights = result.x if result.success else kelly_weights

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
    }


# =========================
# VI. Kelly Sizing
# =========================
def _compute_kelly_size(price, slope, atr, ml_conf, r2, portfolio_value=500000,
                        projection_days=63, atr_stop_multiplier=2.5,
                        kelly_fraction=0.25, max_allocation=0.15,
                        delta_slope=0.0):
    zero = {
        'dollar_amount': 0.0, 'shares': 0,
        'stop': round(price - (atr * atr_stop_multiplier), 2),
        'target': round(price, 2), 'rr_ratio': 0.0,
        'risk_dollar': 0.0, 'exp_return': 0.0
    }

    if slope <= 0 or ml_conf <= 50 or r2 < 0.15:
        return zero

    if delta_slope > 0:
        kelly_fraction = min(kelly_fraction * 1.25, 0.40)
    elif delta_slope < 0:
        kelly_fraction = kelly_fraction * 0.75

    stop_price = price - (atr * atr_stop_multiplier)
    risk_per_share = price - stop_price

    if stop_price >= price or risk_per_share <= 0:
        return {**zero, 'stop': round(stop_price, 2)}

    daily_return_pct = slope / price if price > 0 else 0
    projected_price = price * ((1 + daily_return_pct) ** projection_days)
    target_price = price + (projected_price - price) * r2
    reward_per_share = target_price - price
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

    if rr_ratio < 1.0:
        return {**zero, 'stop': round(stop_price, 2),
                'target': round(target_price, 2), 'rr_ratio': round(rr_ratio, 2)}

    p = ml_conf / 100.0
    q = 1.0 - p
    b = rr_ratio
    kelly = (b * p - q) / b
    fractional_kelly = kelly * kelly_fraction
    final_allocation = max(0, min(fractional_kelly, max_allocation))

    position_value = portfolio_value * final_allocation
    shares = int(position_value / price)
    actual_investment = shares * price
    risk_per_position = shares * risk_per_share
    reward_per_position = shares * reward_per_share
    expected_return = reward_per_position * p - risk_per_position * q

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
    from macro.ml_engine import get_ml_confidence

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
            ml_conf = get_ml_confidence(df)
            atr = float(compute_ATR(df, 14).iloc[-1])
            last = float(c.iloc[-1])
            s50 = float(ma_50.iloc[-1])
            s200 = float(ma_200.iloc[-1])

            # Vectorized Z-Score calculation to resolve slicing and tail overhead
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
            position = _compute_kelly_size(last, slope, atr, ml_conf, r2,
                                           delta_slope=delta_slope)
            pos_size = position['dollar_amount']

            # Decision Logic Execution
            if last < position['stop']:
                status = "SELL (STOP)"
            elif last < s50 and slope < 0:
                status = "SELL (MA50)"
            elif slope_z > 2.0 and ml_conf > 60 and r2 > 0.7:
                status = "BUY (BREAKOUT)"
            elif slope_z > 2.0 and r2 > 0.8:
                status = "TRIM (EXTENDED)"
            elif slope_z > 1.5 and ml_conf < 50:
                status = "TRIM (FADING MOMENTUM)"
            elif ml_conf < 45 and last > s50:
                status = "TRIM (ML FADE)"
            elif slope < -2:
                status = "TRIM (NEGATIVE SLOPE)"
            elif pos_size == 0:
                status = "TRIM (POSITION SIZE)"
            elif (last > s200) and (last > s50) and (slope > 0) and (r2 > 0.6):
                if slope_z < -1.0:
                    status = "BUY (SCALE IN)"
                elif ml_conf > 50:
                    status = "BUY"
                else:
                    status = "HOLD"
            else:
                status = "HOLD"

            rsi14 = float(compute_RSI(c, 14).iloc[-1])

            results.append({
                "sym": sym, "name": name,
                "price": round(last, 2),
                "status": status,
                "r2": round(r2, 2),
                "ml_conf": ml_conf,
                "rsi14": round(rsi14, 1),
                "slope": round(slope, 2),
                "slope_z": round(slope_z, 2),
                "delta_slope": round(delta_slope, 4),
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

    sorted_results = sorted(results, key=lambda x: x["slope"] * x["r2"] * x["ml_conf"] / 100, reverse=True)

    regime = get_risk_regime()
    scalar = get_regime_scalar(regime)

    optimized_results, summary = _kelly_covariance_optimizer(
        sorted_results, shared_data,
        portfolio_value=500000,
        max_single=0.15,
        max_risk_contribution=0.35,
        max_portfolio_vol=0.15,
        regime_scalar=scalar,
        conviction_threshold=0.5,
    )

    _portfolio_summary = summary
    return optimized_results
