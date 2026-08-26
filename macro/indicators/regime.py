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
# Per-date memo for the sparkline's strided slow-ML composite points. Past
# dates' values are immutable (data only up to that date), so each is computed
# once per process instead of on every 30s refresh. Display-only, output-identical.
_SLOW_HIST_CACHE = {}

_hmm_cache = {"model": None, "labels": None, "fitted_at": None}
_HMM_REFIT_INTERVAL = 7 * 24 * 3600  # 7 days


def _fit_regime_hmm(X, n_states=3, n_iter=100):
    from hmmlearn import hmm

    if len(X) < 60:
        return None, None

    # Sticky transition prior (Dirichlet pseudo-counts, diagonal-weighted):
    # without it, EM on this feature set converges to a degenerate solution
    # where two of the three states ping-pong daily (self-transition ~0.003
    # and ~0.024) instead of persisting - the "not converging" warnings seen
    # in practice are a symptom of this. A mild self-transition prior fixes
    # persistence AND finds a fit with materially higher data log-likelihood
    # (-893 vs -1683 on a 2016-2026 SPY/VIX backtest), so this isn't a
    # smoothness/accuracy tradeoff - the unregularized fit was a worse local
    # optimum, not a truer one.
    transmat_prior = np.ones((n_states, n_states))
    np.fill_diagonal(transmat_prior, 10.0)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42,
        transmat_prior=transmat_prior,
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

        ret_trend = np.log(spy / spy.shift(21))

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
        # (2026-08-25: a 4th VIX-term-structure feature was tried and reverted -
        # backtest_gss_hmm_eval / _regime_predictors showed it DOUBLED the state-
        # switch rate with no forward edge. Crises are not predictable from these
        # coincident features; the HMM is descriptive risk telemetry, and the
        # actionable term-structure signal lives in the deploy throttle instead.)

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
def _regime_details(data):
    """The 6 technical regime conditions evaluated at the last row of `data`.
    Extracted so the composite can also be computed at historical stride dates
    (pass data.loc[:ts]). Returns the list of {label, pass} dicts."""
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

    return [
        {"label": "Trend (SPY > 200MA)",      "pass": spy_trend},
        {"label": "Fear (VIX/MOVE Low)",       "pass": vix_low},
        {"label": "Breadth (RSP/SPY > 50MA)",  "pass": breadth_pass},
        {"label": "Credit (HYG/IEF Ratio)",    "pass": credit_pass},
        {"label": "Curve (10Y-3M Spread)",     "pass": curve_pass},
        {"label": "Carry (JPY Weak/Stable)",   "pass": carry_pass},
    ]


@ttl_cache(30)
def get_risk_regime():
    try:
        # Reuse the shared market data (already fetched by get_trends this
        # request) instead of a 2nd overlapping _get_extended_data download.
        # The 9 tickers _regime_details needs (SPY/RSP/HYG/IEF/^TNX/^IRX/^VIX/
        # ^MOVE/JPY=X) are all in the shared set; same group_by=column format,
        # 1y >= the 300d this used. risk_tickers is only a list for the sparkline
        # (predict_assets does its own fetch), so define it directly.
        raw = _get_shared_market_data()
        risk_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys())))
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
                model_path=model_path("risk_model.joblib"),
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
        # SPY slow-ML REMOVED 2026-08-26 (from get_trends + regime).
        # Honest walk-forward (backtest_ml_slow_walkforward) showed the "0.625"
        # was single-split luck; pooled AUC 0.51 = coin flip. It was never in
        # sizing (that uses the vol-target scalar) yet dominated the trading-tab
        # load: ~3.7s to load the model bundle + ~4s of predict_proba on EVERY
        # regime compute (and every strided sparkline point). Neutralized -
        # get_dual_ml_confidence_for_kelly is no longer called, so the model
        # bundle never loads. Composite is now the 6 macro conditions only;
        # ml_slow is shown neutral for display continuity.
        # -----------------------------------------------
        ml_slow_conf = 50.0     # neutral (dead signal removed)
        spy_regime = "N/A"
        spy_divergence = 0.0

        # -----------------------------------------------
        # Technical Conditions (6 independent checks)
        # -----------------------------------------------
        details = _regime_details(data)

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

        # -----------------------------------------------
        # Composite history for the sparkline (returned as "history"): the SAME
        # composite (0.55*technical + 0.45*ml_slow) at each 20-day stride date,
        # so the line's endpoint equals the current composite = the dot.
        # PERF 2026-08-25: the 4 past stride points use data only up to a fixed
        # past date, so their value is immutable - memoize per date so each is
        # computed once per process instead of re-running the RandomForest
        # predict on every 30s refresh. Output-identical. Falls back on error.
        # -----------------------------------------------
        # ml_slow NEUTRALIZED 2026-08-26 (dead signal removed) - each point is now
        # just the 6-condition technical score, still memoized per date.
        try:
            composite_history = []
            for _ts in recent_dates[:-1]:
                _key = pd.Timestamp(_ts).normalize()
                _cv = _SLOW_HIST_CACHE.get(_key)
                if _cv is None:
                    _passes = sum(1 for d in _regime_details(data.loc[:_ts]) if d["pass"])
                    _cv = round((_passes / 6.0 * 0.55 + 0.5 * 0.45) * 100, 1)   # neutral ml
                    _SLOW_HIST_CACHE[_key] = _cv
                composite_history.append(_cv)
            composite_history.append(float(round(composite_score * 100, 1)))   # endpoint = dot
        except Exception as e:
            print(f"Composite history error: {e}")
            composite_history = history_points

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
            "history":          composite_history,   # composite per stride date; endpoint = dot
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

__all__ = [
    "_hmm_cache",
    "_HMM_REFIT_INTERVAL",
    "_fit_regime_hmm",
    "get_regime_hmm_state",
    "_qualify_regime",
    "get_risk_regime",
]
