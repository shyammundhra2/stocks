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
# Capitulation reserve sizing (2026-08-24): scale-in from $1k at the entry
# threshold up to the full reserve (20% of the $500k book, matching the
# backtested 20% reserve) at z >= 4.
_VIX_PORTFOLIO_VALUE = 500_000
_VIX_MAX_DEPLOY = int(0.20 * _VIX_PORTFOLIO_VALUE)   # $100k
_VIX_MIN_DEPLOY = 1_000


def capitulation_state(vix, spy):
    """PURE capitulation lifecycle on VIX and SPY close series (extracted 2026-11-26 so the live
    signal and the historical replay share one implementation; behaviour unchanged from the
    inline version it replaces). Returns the get_vix_signal dict. Raises ValueError on short data."""
    # Same gauge as before: VIX z vs its rolling 50d mean/std.
    zs = (vix - vix.rolling(50).mean()) / vix.rolling(50).std()
    rsi14 = compute_RSI(spy, 14)
    dd = spy / spy.rolling(252, min_periods=60).max() - 1.0
    # Exit-side comparison against the PRIOR day's trailing high, not
    # today's - rolling().max() includes the current row, so "today's
    # price > today's own high" is structurally impossible at a 0%/above
    # threshold (confirmed the hard way in backtest_capitulation_exit_sweep.py:
    # it silently degenerated to "never exits"). dd (inclusive) is kept
    # separately for the "from_high" display metric, where inclusive is
    # the semantically correct answer.
    dd_exit = spy / spy.rolling(252, min_periods=60).max().shift(1) - 1.0

    df = pd.concat({"z": zs, "rsi": rsi14, "dd": dd, "dd_exit": dd_exit}, axis=1).dropna()
    if df.empty:
        raise ValueError("insufficient history for VIX capitulation state")

    def ladder(zval):
        # Maintain-level from z: $1k at z=2, full $100k reserve at z>=4.
        depth = min(max((zval - 2.0) / 2.0, 0.0), 1.0)
        return int(round(_VIX_MIN_DEPLOY + depth * (_VIX_MAX_DEPLOY - _VIX_MIN_DEPLOY), -2))

    # Walk the episode state. Enter on capitulation (z>2 AND washout);
    # the MAINTAIN level ratchets UP with the episode's high-water z
    # (deeper panic -> bigger target; a cooling z does NOT sell you down
    # mid-episode - the backtested exit is price recovery, not z decay);
    # full exit (sell all) when SPY actually sets a new 252d high - best
    # Sharpe/CAGR with zero whipsaws in the full-period sweep (2026-11-23,
    # backtest_capitulation_exit_sweep.py); -10% is the better choice
    # specifically in the 2020-26 regime window (best Sharpe AND best
    # drawdown there) - flagged as a live, regime-dependent judgment call,
    # not a settled one, if this needs revisiting.
    deployed = False
    exited_today = False
    hw_z = 0.0
    for z_, r_, d_ in df[["z", "rsi", "dd_exit"]].itertuples(index=False):
        if deployed and d_ > 0.0:
            deployed = False
            exited_today = True
            hw_z = 0.0
        else:
            exited_today = False
            if not deployed and z_ > 2.0 and r_ < 30.0:
                deployed = True
                hw_z = z_
            elif deployed:
                hw_z = max(hw_z, z_)

    z = float(df["z"].iloc[-1])
    r_now = float(df["rsi"].iloc[-1])
    dd_now = float(df["dd"].iloc[-1])
    v_last = float(vix.iloc[-1])

    target = ladder(hw_z) if deployed else 0
    if deployed and z > 2.0 and r_now < 30.0:
        signal = "BUY (CAPITULATION)"       # trigger live: add up to target
    elif exited_today:
        signal = "EXIT (RECOVERED)"         # sell all - reserve back to cash
    elif deployed:
        signal = "DEPLOYED_HOLD"            # maintain target until recovery
    else:
        signal = "HOLD"

    return {
        "vix": round(v_last, 2), "z": round(z, 2), "signal": signal,
        "spy_rsi14": round(r_now, 1), "from_high": round(dd_now * 100, 1),
        "target": target,                    # $ to maintain (0 = flat)
    }


@ttl_cache(30)
def get_vix_signal():
    """VIX fear-gauge SPY buyer - capitulation lifecycle (2026-08-24).

    Buy/sell rewired to the validated crisis-only tactical-buyer form
    (backtest_gss_capitulation, 2007-26: +0.06 Sharpe / +1.1%/yr on a 20%
    reserve). Two rules the old z-only version lacked, both of which the
    backtest showed matter:
      - Entry needs the AND: vol spike (z > 2) AND price washout
        (SPY RSI14 < 30). Pure z-triggers deployed too early in 2008;
        the AND was the only entry that did not backfire.
      - Defined exit: SPY back within 10% of its 252d high -> reserve
        returns to cash. (The old z-based TRIMs were never validated and
        are removed - this recovery exit is the sell side now.) Moved from
        5% to 10% (2026-11-23, backtest_capitulation_exit_sweep.py): the
        5% threshold was never itself validated against alternatives - a
        sweep found 10% gives the smallest maxDD in the 2020/2022-style
        correction windows (the more relevant comparison than the 2008 GFC
        tail case, where all thresholds converge), at the cost of a lower
        Sharpe/CAGR than tighter exits. Chosen for drawdown control over
        return-maximization, matching this book's stated objective.

    Scale-in: deploy grows linearly from $1k at z=2 to the full $100k
    reserve (20% of the $500k book) at z >= 4.
    Deployment state is walked deterministically over history, so the
    signal needs no persisted state.
    """
    try:
        shared_data = _get_shared_market_data()
        vix = _get_close(shared_data, '^VIX')
        spy = _get_close(shared_data, 'SPY')
        return capitulation_state(vix, spy)
    except Exception as e:
        print(f"VIX Signal Error: {e}")
        return {"vix": 0, "z": 0, "signal": "ERROR",
                "spy_rsi14": 0, "from_high": 0, "target": 0}


EMPTY_BOOK_MAX_FRAC = 0.20   # cap on tactical deploy when the book is all cash (= capitulation reserve size)


def _vol_headroom_deploy(shared_data, tactical_sym="QQQ",
                         vol_cap=0.15, portfolio_value=500_000):
    """Tactical deploy sized by the vol budget: the largest w such that
    book + w*tactical stays at/below the 15% vol cap,
        sig_p^2 + w^2 sig_q^2 + 2 w sig_p sig_q rho = cap^2.
    Correlation is what makes this honest - a rho~0.75 tactical dollar costs
    more vol than a diversifying one, and the formula charges for it.
    Countercyclical for free: when the risk gate has de-risked the book
    (low sig_p), headroom is large - exactly when RSI2/capitulation fire.
    Uses live book weights from the portfolio summary (get_trends runs
    first in app.py); returns 0 if unavailable or already at cap.
    """
    try:
        from macro.indicators.portfolio import get_portfolio_summary
        summary = get_portfolio_summary() or {}
        # Distinguish "summary not computed yet" (no 'weights' key: stay at 0)
        # from "computed, and the book is all cash" ({}): that is exactly when
        # vol headroom is largest, so it must NOT return 0 (fixed 2026-11-26).
        if summary.get("weights") is None:
            return 0
        weights = summary["weights"]
        rets = shared_data["Close"].pct_change().tail(63)
        if tactical_sym not in rets.columns:
            return 0
        q = rets[tactical_sym].fillna(0)
        sig_q = float(q.std() * np.sqrt(252))
        if not weights:                                 # all-cash book: sig_p = 0
            if sig_q <= 0 or not np.isfinite(sig_q):
                return 0
            w_max = min(vol_cap / sig_q, EMPTY_BOOK_MAX_FRAC)
            return int(round(w_max * portfolio_value, -3))
        book = None
        for s, wt in weights.items():
            if s in rets.columns:
                leg = rets[s].fillna(0) * wt
                book = leg if book is None else book + leg
        if book is None:
            return 0
        sig_p = float(book.std() * np.sqrt(252))
        rho = float(np.corrcoef(book, q)[0, 1])
        if not np.isfinite(rho):
            rho = 1.0                                   # conservative
        if sig_p >= vol_cap or sig_q <= 0:
            return 0
        a = sig_q ** 2
        b = 2.0 * sig_p * sig_q * rho
        c = sig_p ** 2 - vol_cap ** 2
        w_max = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        return int(round(max(w_max, 0.0) * portfolio_value, -3))
    except Exception as e:
        print(f"Vol headroom error: {e}")
        return 0


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
        # BUY tightened <=10 -> <=5 (2026-08-24): per-trade study 2005-26,
        # 5d max hold: RSI2<=5 carries the whole edge (+0.36%/trade, n=410);
        # the 5-10 band is negative (-0.12%, n=43).
        if price < sma200:
            signal = "RISK OFF"
        elif rsi2 <= 5:
            signal = "BUY"
        elif rsi2 >= 70:
            signal = "EXIT"
        else:
            signal = "HOLD"

        # Deploy sized by vol headroom (15% cap - current book vol), only
        # when the signal actually fires.
        deploy = _vol_headroom_deploy(shared_data) if signal == "BUY" else 0

        return {"price": round(price, 2), "rsi2": round(rsi2, 1),
                "signal": signal, "deploy": deploy}

    except Exception as e:
        print(f"Mean Reversion Error: {e}")
        return {"price": 0, "rsi2": 0, "signal": "ERROR", "deploy": 0}

__all__ = [
    "capitulation_state",
    "get_vix_signal",
    "get_mean_reversion",
]
