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
# IV. Rotation
# =========================
def _slope_r2_path(series, window=20, scale=20, days=21):
    """Daily [slope, r2] path over the last ~`days` sessions (oldest -> today),
    for the rotation-map hover ghost - mirrors get_trends' slope_r2_path. Each
    point is that day's trailing 20-bar trend stats. NaN-safe; falls back to a
    single current point when there isn't enough history."""
    s = series.dropna()
    nlen = len(s)
    path = [list(_trend_stats(s.iloc[:nlen - k], window, scale))
            for k in range(days, -1, -1) if nlen - k >= window + 5]
    if not path:
        path = [list(_trend_stats(s, window, scale))]
    return [[round(p[0], 3), round(p[1], 3)] for p in path]


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
                "gradient": gradient, "slope_change": slope_change,
                "slope_r2_path": _slope_r2_path(data[t]),
            })
        return {"all_ranked": sorted(out, key=lambda x: x["slope"], reverse=True)}
    except Exception as e:
        print(f"Sector Rotation Error: {e}")
        return {"all_ranked": []}


def _ret_126(series):
    """126d (~6mo) simple return, the validated ranking lookback. NaN-safe."""
    s = series.dropna()
    if len(s) < 127:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-127] - 1.0)


@ttl_cache(30)
def get_country_rotation():
    # Ranked by 6-month (126d) MOMENTUM. backtest_rank_predictors 2007-26:
    # mom126 is the only country signal with real edge (IC +0.044, top-pick
    # +1.5% excess/63d, 54% hit); it beats the 20d slope this used to sort by.
    # slope/r2/gradient kept for the rotation-map scatter.
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        results = []
        for s, n in COUNTRIES.items():
            slope, r2 = _trend_stats(data[s], 20, 20)
            gradient = _compute_gradient(data[s].tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(data[s])
            ret126 = _ret_126(data[s])
            results.append({"sym": s, "name": n, "slope": slope, "r2": r2,
                            "gradient": gradient, "slope_change": slope_change,
                            "ret_126": round(ret126 * 100, 1) if ret126 == ret126 else None,
                            "slope_r2_path": _slope_r2_path(data[s]),
                            "rank_score": ret126})            # + momentum: leaders first
        # Rank leaders first; NaN scores sink to the bottom.
        results.sort(key=lambda x: (x["rank_score"] if x["rank_score"] == x["rank_score"]
                                    else float("-inf")), reverse=True)
        return results
    except Exception as e:
        print(f"Country Rotation Error: {e}")
        return []


@ttl_cache(30)
def get_commodity_rotation():
    # Ranked by 6-month (126d) MEAN-REVERSION (buy the laggards). commodities
    # mean-revert at these horizons: mom126 IC is -0.113 (momentum backfires),
    # so the NEGATED 126d return is the edge (backtest_commodity_reversion
    # 2007-26: rev_126 IC +0.132, top-pick +4.7% excess/63d, 55% hit).
    # slope/r2/gradient kept for the rotation-map scatter.
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        results = []
        for s, n in COMMODITIES.items():
            slope, r2 = _trend_stats(data[s], 20, 20)
            gradient = _compute_gradient(data[s].tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(data[s])
            ret126 = _ret_126(data[s])
            results.append({"sym": s, "name": n, "slope": slope, "r2": r2,
                            "gradient": gradient, "slope_change": slope_change,
                            "ret_126": round(ret126 * 100, 1) if ret126 == ret126 else None,
                            "slope_r2_path": _slope_r2_path(data[s]),
                            "rank_score": (-ret126) if ret126 == ret126 else float("nan")})
        # Reversion: biggest 6-month LOSERS ranked first; NaN sinks.
        results.sort(key=lambda x: (x["rank_score"] if x["rank_score"] == x["rank_score"]
                                    else float("-inf")), reverse=True)
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
                            "gradient": gradient, "slope_change": slope_change,
                            "slope_r2_path": _slope_r2_path(c, window=60, scale=60)})
        return results
    except Exception as e:
        print(f"Currency Rotation Error: {e}")
        return []

__all__ = [
    "get_sector_rotation",
    "get_country_rotation",
    "get_commodity_rotation",
    "get_currency_rotation",
]
