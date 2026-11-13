import yfinance as yf
import numpy as np
import pandas as pd
import threading
import time
import math
import os
import io
import ssl
import json
import hashlib
import tempfile
import urllib.request
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


def _splice_splits(stored, fresh, start, tol=0.02):
    """Rescale stored history when a ticker has split since the last full sync.

    THE BUG THIS FIXES (2026-11-09). The incremental refresh only re-downloads a
    trailing _TRAILING_DAYS window, but a split retroactively rewrites a ticker's
    ENTIRE price history at the source. So after a split the store holds
    pre-split prices before the window and post-split prices inside it, leaving a
    fake cliff mid-series. Observed: SOXX and ITA both split 3:1 on 2026-11-05,
    and the store showed a -67% one-day "move" on 2026-10-23 with pre-split
    prices (~$500) sitting next to post-split ones (~$159). _trend_stats read
    slope -90.65 for SOXX against -11.35 on clean data - the router was sizing
    off a corporate action, not a price move. Nothing downstream could catch it:
    the artifact filters in the stock sleeve (JUMP_CAP) do not apply here.

    Detection uses the OVERLAP, which is unambiguous: for dates present in both
    stored and fresh, a split shows up as a constant ratio across every
    overlapping bar. A real price revision would not be constant. When the ratio
    is consistent (within `tol`) and materially different from 1, the stored
    history before `start` is multiplied by it. Anything less clean is left
    alone and caught by the >25% tripwire in _core_history_ok's sibling checks.
    """
    try:
        ov = stored.index.intersection(fresh.index)
        if len(ov) < 3:
            return stored
        cols = stored.columns.intersection(fresh.columns)
        scale = {}
        for c in cols:
            if isinstance(c, tuple) and c[0] == "Volume":
                continue                      # volume scales inversely; not used downstream
            a = pd.to_numeric(stored.loc[ov, c], errors="coerce")
            b = pd.to_numeric(fresh.loc[ov, c], errors="coerce")
            m = a.notna() & b.notna() & (a != 0)
            if m.sum() < 3:
                continue
            r = (b[m] / a[m]).astype(float)
            med = float(r.median())
            # constant ratio across the whole overlap, and not ~1
            if abs(med - 1.0) > tol and float((r / med - 1.0).abs().max()) < tol:
                scale[c] = med
        if not scale:
            return stored
        out = stored.copy()
        pre = out.index < start
        for c, k in scale.items():
            out.loc[pre, c] = pd.to_numeric(out.loc[pre, c], errors="coerce") * k
        syms = sorted({c[1] for c in scale if isinstance(c, tuple) and len(c) > 1})
        print(f"split-adjust: rescaled stored history for {syms or list(scale)[:4]}")
        return out
    except Exception as e:
        print(f"split-adjust skipped: {e}")
        return stored


def _merge_trailing(stored, fresh, start):
    """Keep stored rows before `start`; overwrite [start..] with fresh bars."""
    fresh = fresh.reindex(columns=stored.columns)
    stored = _splice_splits(stored, fresh, start)
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


def _period_min_bars(period, frac=0.6):
    """Rough floor on how many sessions a period= request should return.

    Used to recognise a truncated full fetch BEFORE it is stamped as the 30-day
    full sync. Deliberately loose (60% of expected sessions) - it only needs to
    separate "a year of data" from "six rows", not to be precise.
    """
    try:
        p = str(period).strip().lower()
        if p.endswith("y"):
            days = float(p[:-1]) * 365.0
        elif p.endswith("mo"):
            days = float(p[:-2]) * 30.0
        elif p.endswith("d"):
            days = float(p[:-1])
        else:
            return 0
    except Exception:
        return 0
    return int(days * (252.0 / 365.0) * frac)


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
            # A short full-period fetch is a truncated response, not a real sync.
            # Stamping full_ts on it froze a 6-row store for the whole 30d resync
            # window (2026-09-01). Prefer better stored data, and leave full_ts
            # unset so the next call retries the full fetch instead of drifting
            # forward on 7-day incremental refreshes.
            floor_bars = _period_min_bars(period)
            if floor_bars and len(raw) < floor_bars:
                if stored is not None and len(stored) > len(raw):
                    _store_write(base, stored, ts=now,
                                 full_ts=float((spec or {}).get("full_ts", 0.0)))
                    return _slice_period(stored, period)
                _store_write(base, raw, ts=now, full_ts=0.0)
                return raw
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
# Tickers that DRIVE DASHBOARD TILES - a sparse column in any of these must trip
# the self-heal. QQQ added 2026-11-12: yfinance dropped the QQQ column in a full
# resync (7 bars against 252 for SPY in the same frame), the whole-frame row
# floor passed because the BATCH was fine, and QQQ was never on this list - so
# the Tactical RSI2 tile errored while the guard reported healthy. Per-column
# dropout is the third distinct poisoning mode of this store, after the
# truncated-batch mode (2026-09-01) and the stale-split mode (2026-11-09).
_CORE_CHECK = ("^VIX", "SPY", "^MOVE", "QQQ")


def _core_history_ok(data, min_frac=0.5, min_bars=60):
    """False if a CORE ticker came back suspiciously sparse - i.e. a bad batch
    fetch (yfinance intermittently truncates a column, or drops a ^-index ticker
    entirely) that the incremental cache would otherwise FREEZE forever (e.g.
    ^VIX at 7 points, breaking the VIX 50d z-score).

    The reference length is the FULLEST column in the batch, NOT SPY - because
    SPY itself can be the poisoned one. (Old bug: SPY froze at 6 points -> the
    `ref<60` short-history escape hatch fired on SPY's own count -> the guard
    passed the poison and the whole book fell back.) A core ticker MISSING
    entirely, or under min_frac of the fullest column, trips the self-heal.

    2026-09-01: the escape hatch had to go the rest of the way. Referencing the
    fullest column still passed a WHOLLY truncated batch, because when every
    column is poisoned the fullest one is poisoned too - a 6-row store returned
    ref=6, fired the `ref<60` hatch, and froze for the full 30d resync window
    (VIX + QQQ-RSI2 tiles read ERROR because the 50d z-score and SMA200 had no
    bars). A batch whose FULLEST column is under min_bars is not a short batch,
    it IS the poison, so it now trips the self-heal instead of bypassing it.
    Callers requesting a genuinely short period must pass a smaller min_bars."""
    try:
        cols = getattr(data, "columns", None)
        close = data["Close"] if (cols is not None and hasattr(cols, "get_level_values")
                                   and "Close" in cols.get_level_values(0)) else data
        counts = close.notna().sum()
        ref = int(counts.max()) if len(counts) else 0
        if ref < min_bars:
            return False                      # whole batch truncated -> self-heal, don't wave it through
        need = max(min_bars, min_frac * ref)
        for t in _CORE_CHECK:                  # includes SPY, ^VIX, ^MOVE
            if t not in close.columns:
                return False                   # core ticker dropped from the batch entirely
            if int(close[t].notna().sum()) < need:
                return False                   # core ticker sparse vs the fullest column
        return True
    except Exception:
        return True


def _invalidate_store(label, tickers):
    try:
        base = _store_base(label, tickers)
        for p in (base + ".parquet", base + ".meta"):
            if os.path.exists(p):
                os.remove(p)
    except Exception:
        pass


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
    data = _persistent_fetch("shared", all_tickers, "1y", group_by="column")
    # Self-heal: if a core ticker is sparse (a transient bad fetch the incremental
    # cache would otherwise freeze), drop the store and re-fetch once.
    if not _core_history_ok(data):
        print("shared-data self-heal: core ticker sparse -> invalidating cache + re-fetch")
        _invalidate_store("shared", all_tickers)
        data = _persistent_fetch("shared", all_tickers, "1y", group_by="column")
    return data


@ttl_cache(30)
def _get_extended_data():
    macro_proxies = ['HYG', 'IEF', '^TNX', '^IRX', 'JPY=X', 'HG=F', 'GC=F']
    risk_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys())))
    all_tickers = list(set(risk_tickers + macro_proxies + ['^VIX', '^MOVE']))
    raw = _persistent_fetch("extended", all_tickers, "300d", group_by="column")
    return raw, risk_tickers


def split_artifact_scan(shared_data, tickers, thresh=0.25):
    """Tickers whose stored series contains a split-shaped discontinuity.

    Backstop to _splice_splits: if an unadjusted split ever slips through, this
    names it instead of letting the router size off a corporate action. A 2:1 /
    3:1 / 4:1 split leaves a near-exact ratio jump, which is what is matched;
    genuine moves above `thresh` are reported separately as 'large'.
    """
    out = {}
    for t in tickers:
        s = _get_close(shared_data, t)
        if len(s) < 50:
            continue
        r = s.pct_change().dropna()
        if r.empty:
            continue
        i = r.abs().idxmax()
        mv = float(r.loc[i])
        if abs(mv) < thresh:
            continue
        ratio = 1.0 + mv
        kind = "large"
        for k in (2, 3, 4, 5, 10):
            if abs(ratio - k) < 0.05 or abs(ratio - 1.0 / k) < 0.05:
                kind = f"split-shaped x{ratio:.2f}"
        out[t] = {"date": str(i.date()), "move": mv, "kind": kind}
    return out


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
# VIX term structure (VIX / VIX3M)  -- added 2026-08-25
#
# The forward-looking stress signal: backwardation (VIX > VIX3M, ratio > 1)
# flags stress BEFORE spot VIX level rises; contango (< 1) = calm. yfinance
# serves ^VIX3M as a SINGLE latest row only (no history), so the series comes
# from CBOE's public daily-history CSV. Disk-cached 6h; serves stale on network
# failure; yfinance ^VIX3M latest as a last-resort single-point fallback so the
# live throttle survives a CBOE outage. Empty series only if every source fails
# (throttle then fails OPEN -> no throttle -> current behavior).
#
# Drives the deploy throttle (validated backtest_gss_vix_termstructure: 2020-26
# maxDD -10.7 -> -5.7% for ~0.5% CAGR give-up, 4x cheaper than the spot-VIX
# throttle) and the regime HMM's 4th feature.
# =========================
_VIX_TS_CACHE = os.path.join(_STORE_DIR, "vix_termstructure.parquet")
_VIX_TS_DISK_TTL = 6 * 3600


def _cboe_index_series(sym):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, context=ctx, timeout=20).read()
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().upper() for c in df.columns]
    dcol = next(c for c in df.columns if "DATE" in c)
    ccol = next(c for c in df.columns if "CLOSE" in c)
    df[dcol] = pd.to_datetime(df[dcol])
    return df.set_index(dcol)[ccol].sort_index()


@ttl_cache(30)
def get_vix_term_structure():
    """VIX/VIX3M term-structure ratio series (CBOE primary, 6h disk cache,
    yfinance latest fallback). Returns an empty Series if all sources fail."""
    try:
        if os.path.exists(_VIX_TS_CACHE) and (time.time() - os.path.getmtime(_VIX_TS_CACHE)) < _VIX_TS_DISK_TTL:
            return pd.read_parquet(_VIX_TS_CACHE)["ts"]
    except Exception:
        pass
    try:
        vix = _cboe_index_series("VIX")
        vix3m = _cboe_index_series("VIX3M")
        ts = (vix / vix3m).dropna()
        if len(ts):
            try:
                ts.to_frame("ts").to_parquet(_VIX_TS_CACHE)
            except Exception:
                pass
            return ts
    except Exception:
        pass
    try:                                   # network failed -> serve stale disk
        return pd.read_parquet(_VIX_TS_CACHE)["ts"]
    except Exception:
        pass
    try:                                   # last resort: yfinance latest point
        v = yf.download("^VIX", period="5d", progress=False, auto_adjust=False)["Close"].dropna()
        v3 = yf.download("^VIX3M", period="5d", progress=False, auto_adjust=False)["Close"].dropna()
        if len(v) and len(v3):
            return pd.Series([float(v.iloc[-1]) / float(v3.iloc[-1])], index=[v.index[-1]])
    except Exception:
        pass
    return pd.Series(dtype=float)


def vix_ts_deploy_throttle(floor=0.30):
    """Continuous deploy throttle from the latest VIX term structure:
    f = clip(1.0 - 2.5*(ratio - 0.90), floor, 1.0). Contango (ratio <= 0.9) ->
    1.0 (full deploy); backwardation (ratio > 1) -> throttled toward floor.
    Fails OPEN (returns 1.0) when data is unavailable, so a data outage can
    never wrongly cut the book - it just reverts to un-throttled behavior."""
    ts = get_vix_term_structure()
    if ts is None or len(ts) == 0:
        return 1.0
    last = float(ts.iloc[-1])
    if not np.isfinite(last):
        return 1.0
    return float(np.clip(1.0 - 2.5 * (last - 0.90), floor, 1.0))


__all__ = [
    "_STORE_DIR",
    "_STORE_TTL",
    "_TRAILING_DAYS",
    "_FULL_RESYNC_SECS",
    "_STORE_MAX_ROWS",
    "_INCREMENTAL",
    "_STORE_WARNED",
    "_store_base",
    "_store_read",
    "_store_write",
    "_slice_period",
    "_merge_trailing",
    "_yf_dl",
    "_persistent_fetch",
    "_risk_history_base",
    "_risk_history_load",
    "_risk_history_save",
    "_get_shared_market_data",
    "_get_extended_data",
    "_get_close",
    "split_artifact_scan",
    "get_vix_term_structure",
    "vix_ts_deploy_throttle",
]
