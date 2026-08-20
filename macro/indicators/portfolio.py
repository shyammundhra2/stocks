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

from macro.indicators.cache import *
from macro.indicators.data import *
from macro.indicators.mathstats import *
from macro.indicators.regime import *


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

            # Trend vs. mean-reversion posture (~2-3mo). Hurst + variance
            # ratio (q=21) + weekly-return autocorr, blended to one score.
            vr21 = _variance_ratio(c, q=21)
            wk_autocorr = _return_autocorr(c, block=5, lag=1)
            persistence = _persistence_classify(hurst, vr21, wk_autocorr)

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
                "vr21": vr21,                       # Lo-MacKinlay VR (q=21): >1 trend, <1 revert
                "wk_autocorr": wk_autocorr,         # weekly-return lag-1 autocorr
                "persistence_score": persistence["persistence_score"],   # [-1,1]: +trend, -revert
                "persistence_label": persistence["persistence_label"],   # TREND/NEUTRAL/MEAN-REVERT
                "persistence_arrow": persistence["persistence_arrow"],   # ↑ / ↔ / ↻
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

    # Sort by conviction: slope * r2 * strength, with slope * r2 as tiebreaker
    # for the slope<=0 cluster (where strength == 0 for all of them, but
    # slope*r2 itself isn't) - surfaces instruments closest to a trend flip
    # even when the system won't allocate to them.
    sorted_results = sorted(
        results,
        key=lambda x: (
            x["slope"] * x["r2"] * x.get("strength", x["r2"]),
            x["slope"] * x["r2"]
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

__all__ = [
    "_portfolio_summary",
    "get_portfolio_summary",
    "_compute_covariance_matrix",
    "_compute_portfolio_var",
    "_compute_risk_contribution",
    "_conviction_score",
    "_kelly_covariance_optimizer",
    "_compute_kelly_size",
    "get_trends",
]
