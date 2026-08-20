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

__all__ = [
    "ttl_cache",
]
