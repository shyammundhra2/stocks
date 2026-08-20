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

__all__ = [
    "get_sector_rotation",
    "get_country_rotation",
    "get_commodity_rotation",
    "get_currency_rotation",
]
