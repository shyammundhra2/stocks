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

__all__ = [
    "get_vix_signal",
    "get_mean_reversion",
]
