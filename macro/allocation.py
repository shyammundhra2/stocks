"""Live $-allocation of the $500k book across the six sleeves of research variant "F"
(research/kelly/: stock sieve 100k, momentum-trend optimizer book 100k, trend ETFs 100k, VIX/SPY capitulation 100k,
QQQ RSI2 25k, top-5 countries 10k; the rest, 65k, is cash).

Cadence (user's choice): QQQ RSI2 and VIX/SPY are acted on DAILY; the other four sleeves MONTHLY.
Every number is the live signal ("what the rules say to hold at today's close"); for the monthly sleeves
`next_rebalance` is the month-end to act on. Each sleeve is
isolated: one failing data source degrades that sleeve to cash with a note, never breaks the page.

Display-only. Nothing here places or sizes real orders beyond the budget split.
"""
import os
import sys
import json
import calendar
from datetime import date

import numpy as np
import pandas as pd

from macro.helpers import compute_RSI

BOOK = 500_000
BUDGETS = {                       # variant F
    "stock_sieve": 100_000,
    "mom_trend": 100_000,
    "trend_etf": 100_000,
    "vix_spy": 100_000,
    "qqq_rsi2": 25_000,
    "country": 10_000,
}
LABELS = {
    "stock_sieve": "Stock sieve (12-1, top 10, <=3/sector, vol-managed)",
    "mom_trend": "Momentum-trend ETFs (live optimizer book)",
    "trend_etf": "Trend ETFs (equal-weight, above 200DMA)",
    "vix_spy": "VIX capitulation -> SPY",
    "qqq_rsi2": "QQQ RSI2",
    "country": "Top-5 countries (6-month momentum, gated)",
}
CADENCE = {"stock_sieve": "monthly", "mom_trend": "monthly", "trend_etf": "monthly",
           "vix_spy": "daily", "qqq_rsi2": "daily", "country": "monthly"}
COUNTRY_TOP_K = 5
COUNTRY_LOOKBACK = 126   # 6-month momentum
ETF_TOP_K = 8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ETF_CSV = os.path.join(_ROOT, "trendselection", "trend_etf_ranking.csv")
_STOCK_CSV = os.path.join(_ROOT, "stockselection", "macro_2027_ranking_enhanced.csv")
HOLDINGS_PATH = os.path.join(_ROOT, "data", "allocation_holdings.json")   # what you actually hold, per sleeve
MIN_TRADE = 250        # ignore drift smaller than this ($)


def _sleeve(key, positions=None, note="", signal=""):
    """positions: {ticker: dollars}. Deployed = sum; the rest of the budget is the sleeve's cash."""
    positions = {t: round(float(v)) for t, v in (positions or {}).items() if v and v > 0}
    deployed = sum(positions.values())
    budget = BUDGETS[key]
    return {"key": key, "label": LABELS[key], "cadence": CADENCE[key], "budget": budget, "deployed": deployed,
            "cash": budget - deployed, "positions": positions, "signal": signal, "note": note}


def _close(shared, ticker):
    from macro.indicators.data import _get_close
    return _get_close(shared, ticker)


def _above_200(series):
    s = series.dropna()
    if len(s) < 200:
        return None
    return bool(s.iloc[-1] > s.rolling(200).mean().iloc[-1])


def next_rebalance(today=None):
    """Last weekday of the current month (holidays ignored - shift to the prior trading day if closed)."""
    t = today or date.today()
    last = calendar.monthrange(t.year, t.month)[1]
    d = date(t.year, t.month, last)
    while d.weekday() >= 5:
        d = date.fromordinal(d.toordinal() - 1)
    return d


def _sieve():
    try:
        df = pd.read_csv(_STOCK_CSV)
        gate = str(df["Gate"].iloc[0]) if len(df) else ""
        w = min(float(df["Deploy_W"].iloc[0]), 1.0) if "Deploy_W" in df else 1.0
        buys = df[df["Signal"] == "BUY"]
        if buys.empty:
            return _sleeve("stock_sieve", signal="GATE: CASH", note=gate or "No BUY names (gate risk-off).")
        per = BUDGETS["stock_sieve"] * w / len(buys)
        pos = {r["Ticker"]: per for _, r in buys.iterrows()}
        return _sleeve("stock_sieve", pos, signal=f"{len(buys)} names", note=f"vol-managed deploy {w:.0%} of sleeve")
    except Exception as e:
        return _sleeve("stock_sieve", signal="ERROR", note=f"stock ranking unavailable: {e}")


def _etf_universe_and_mom():
    t = pd.read_csv(_ETF_CSV)
    return t["Ticker"].tolist(), t.set_index("Ticker")["Mom_12_1"].astype(float)


def _mom_trend():
    """The main momentum-trend sleeve = the live get_trends() optimizer book (inverse-vol/covariance optimizer,
    SPY-200DMA risk gate, defensives). Its weights are fractions of ITS OWN book, so position $ = weight x sleeve
    budget - the optimizer's own deployment (typically 40-60%) and cash are preserved, as in the replay backtest.
    Needs get_trends() to have run first (app.py does this); otherwise the sleeve reports ERROR/cash."""
    try:
        from macro.indicators import get_portfolio_summary
        summ = get_portfolio_summary() or {}
        if summ.get("weights") is None:
            return _sleeve("mom_trend", signal="ERROR", note="optimizer book not computed yet (get_trends must run first)")
        w = {t: float(v) for t, v in summ["weights"].items() if float(v) > 0}
        gross = sum(w.values())
        pos = {t: v * BUDGETS["mom_trend"] for t, v in w.items()}
        return _sleeve("mom_trend", pos, signal=f"{len(pos)} names, {gross:.0%} deployed",
                       note="optimizer weights x sleeve budget; backtest standalone Sharpe ~0.04 (low-vol diversifier)")
    except Exception as e:
        return _sleeve("mom_trend", signal="ERROR", note=f"{e}")


def _trend_etf(shared):
    try:
        tickers, _ = _etf_universe_and_mom()
        have = {t: _close(shared, t) for t in tickers}
        have = {t: s for t, s in have.items() if len(s.dropna()) >= 200}
        if not have:
            return _sleeve("trend_etf", signal="ERROR", note="no ETF history")
        slot = BUDGETS["trend_etf"] / len(have)
        pos = {t: slot for t, s in have.items() if _above_200(s)}
        return _sleeve("trend_etf", pos, signal=f"{len(pos)} of {len(have)} above 200DMA")
    except Exception as e:
        return _sleeve("trend_etf", signal="ERROR", note=f"{e}")


def _vix_spy(vix_signal):
    try:
        v = vix_signal or {}
        if v.get("signal") == "ERROR":
            return _sleeve("vix_spy", signal="ERROR", note="VIX signal unavailable")
        tgt = min(max(float(v.get("target", 0) or 0), 0), BUDGETS["vix_spy"])
        return _sleeve("vix_spy", {"SPY": tgt}, signal=str(v.get("signal", "")),
                       note=f"VIX z {v.get('z')}, SPY RSI14 {v.get('spy_rsi14')}")
    except Exception as e:
        return _sleeve("vix_spy", signal="ERROR", note=f"{e}")


def _qqq(shared):
    try:
        c = _close(shared, "QQQ").dropna()
        if len(c) < 205:
            return _sleeve("qqq_rsi2", signal="ERROR", note="QQQ history too short")
        rsi, sma = compute_RSI(c, 2), c.rolling(200, min_periods=200).mean()
        cur = 0
        for p, s, r in zip(c, sma, rsi):          # stateful: enter <=5, exit >=70, flat <200DMA
            if np.isnan(s) or p < s:
                cur = 0
            elif r <= 5:
                cur = 1
            elif r >= 70:
                cur = 0
        note = f"RSI2 {rsi.iloc[-1]:.0f} (enter <=5, exit >=70, flat <200DMA)"
        return _sleeve("qqq_rsi2", {"QQQ": BUDGETS["qqq_rsi2"]} if cur else {}, signal="IN" if cur else "FLAT", note=note)
    except Exception as e:
        return _sleeve("qqq_rsi2", signal="ERROR", note=f"{e}")


def _country(shared):
    """Top-5 countries by 6-month (126d) momentum, equal-weight, each must be above its 200DMA with positive
    6m momentum (research/kelly country tests: Sharpe 0.37, just past the random-pick 95th pct 0.36; empty
    slots stay cash). Universe = macro.constants.COUNTRIES (same 16 ETFs as the test, incl. SPY)."""
    try:
        from macro.constants import COUNTRIES
        mom, up = {}, {}
        for t in [t for t in COUNTRIES if not t.startswith("^")]:
            c = _close(shared, t).dropna()
            if len(c) < 200:
                continue
            mom[t] = float(c.iloc[-1] / c.iloc[-1 - COUNTRY_LOOKBACK] - 1.0) if len(c) > COUNTRY_LOOKBACK else np.nan
            up[t] = bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1])
        if not mom:
            return _sleeve("country", signal="ERROR", note="no country price history")
        ranked = sorted(((t, m) for t, m in mom.items() if np.isfinite(m) and m > 0 and up[t]), key=lambda x: -x[1])
        top = ranked[:COUNTRY_TOP_K]
        slot = BUDGETS["country"] / COUNTRY_TOP_K
        top6 = sorted(((t, m) for t, m in mom.items() if np.isfinite(m)), key=lambda x: -x[1])[:5]
        note = (f"{len(ranked)} of {len(mom)} pass gate (6m>0 and >200DMA); empty slots stay cash. "
                "6m leaders: " + ", ".join(f"{t} {m:+.0%}{'' if up[t] else ' (<200DMA)'}" for t, m in top6))
        return _sleeve("country", {t: slot for t, _ in top}, signal=f"{len(top)} of {COUNTRY_TOP_K} slots", note=note)
    except Exception as e:
        return _sleeve("country", signal="ERROR", note=f"{e}")


def get_allocation(vix_signal=None):
    """Return the full live allocation. `vix_signal` = get_vix_signal() output (avoids recomputing)."""
    from macro.indicators import _get_shared_market_data
    try:
        shared = _get_shared_market_data()
    except Exception:
        shared = pd.DataFrame()
    sleeves = [_sieve(), _mom_trend(), _trend_etf(shared), _vix_spy(vix_signal), _qqq(shared), _country(shared)]

    by_ticker = {}
    for s in sleeves:
        for t, v in s["positions"].items():
            by_ticker[t] = by_ticker.get(t, 0) + v
    for t in ("SPY", "QQQ"):                      # always show the index lines, even when flat
        by_ticker.setdefault(t, 0)
    invested = sum(by_ticker.values())
    cash = BOOK - invested
    rows = [{"ticker": t, "dollars": v, "pct": v / BOOK,
             "sleeves": [s["key"] for s in sleeves if t in s["positions"]]}
            for t, v in sorted(by_ticker.items(), key=lambda kv: -kv[1])]
    out_orders = build_orders(sleeves, load_held(), shared)
    return {"orders": out_orders, "book": BOOK, "sleeves": sleeves, "by_ticker": rows, "invested": invested, "cash": cash,
            "cash_pct": cash / BOOK, "budget_cash": BOOK - sum(BUDGETS.values()),
            "next_rebalance": next_rebalance().isoformat(), "errors": [s["key"] for s in sleeves if s["signal"] == "ERROR"]}


# ---------------------------------------------------------------- holdings ledger + orders
def load_held():
    """{sleeve: {"positions": {ticker: $}, "as_of": "YYYY-MM-DD"}} - what you last recorded as held."""
    try:
        with open(HOLDINGS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_held(held):
    os.makedirs(os.path.dirname(HOLDINGS_PATH), exist_ok=True)
    with open(HOLDINGS_PATH, "w") as f:
        json.dump(held, f, indent=1, sort_keys=True)


def _prices(shared):
    px = {}
    for path in (_ETF_CSV, _STOCK_CSV):
        try:
            d = pd.read_csv(path)
            px.update({r["Ticker"]: float(r["Current_Price"]) for _, r in d.iterrows() if pd.notna(r.get("Current_Price"))})
        except Exception:
            pass
    for t in ("SPY", "QQQ"):
        try:
            c = _close(shared, t)
            if len(c):
                px[t] = float(c.iloc[-1])
        except Exception:
            pass
    return px


def monthly_due(today=None):
    """Monthly sleeves are acted on at the month-end rebalance day (last weekday of the month)."""
    t = today or date.today()
    return t == next_rebalance(t)


def build_orders(sleeves, held, shared=None, today=None):
    """Diff each sleeve's live target against what is recorded as held.
    DAILY sleeves (vix_spy, qqq_rsi2): orders every day. MONTHLY sleeves: `due` only on the rebalance day;
    otherwise the same diff is shown as a PREVIEW. Netted by ticker across the sleeves that are due.
    No holdings recorded for a sleeve => the order is the full initial build."""
    px = _prices(shared if shared is not None else pd.DataFrame())
    t = today or date.today()
    month_due = monthly_due(t)
    per_sleeve, net = [], {}
    for s in sleeves:
        key = s["key"]
        due = (s["cadence"] == "daily") or month_due
        h = (held.get(key) or {}).get("positions", {})
        tg = s["positions"]
        for tick in sorted(set(h) | set(tg)):
            diff = float(tg.get(tick, 0)) - float(h.get(tick, 0))
            if abs(diff) < MIN_TRADE and not (tick not in tg and h.get(tick, 0) > 0):
                continue
            p = px.get(tick)
            o = {"sleeve": key, "ticker": tick, "action": "BUY" if diff > 0 else "SELL", "dollars": round(abs(diff)),
                 "shares": int(round(abs(diff) / p)) if p else None, "price": round(p, 2) if p else None,
                 "due": due, "held": round(h.get(tick, 0)), "target": round(tg.get(tick, 0))}
            per_sleeve.append(o)
            if due:
                net[tick] = net.get(tick, 0) + diff
    netted = [{"ticker": k, "action": "BUY" if v > 0 else "SELL", "dollars": round(abs(v)),
               "shares": int(round(abs(v) / px[k])) if px.get(k) else None}
              for k, v in sorted(net.items(), key=lambda kv: -abs(kv[1])) if abs(v) >= 1]
    return {"per_sleeve": per_sleeve, "net_due": netted, "monthly_due": month_due,
            "first_run": not held, "n_due": sum(1 for o in per_sleeve if o["due"]),
            "held_asof": {k: v.get("as_of") for k, v in held.items()}}


def mark_executed(alloc, which="due", today=None):
    """Record targets as held AFTER you have placed the orders. which: 'due' (daily sleeves, plus monthly
    sleeves if it is rebalance day) | 'all'. Returns the keys updated."""
    t = today or date.today()
    held = load_held()
    done = []
    for s in alloc["sleeves"]:
        if which == "all" or s["cadence"] == "daily" or monthly_due(t):
            held[s["key"]] = {"positions": s["positions"], "as_of": t.isoformat()}
            done.append(s["key"])
    save_held(held)
    return done


if __name__ == "__main__":
    # python -m macro.allocation orders   -> print today's orders
    # python -m macro.allocation mark     -> after trading: record due sleeves as held ('mark all' = initial build)
    from macro.indicators import get_vix_signal, get_trends
    get_trends()                      # runs the optimizer so the momentum-trend sleeve has its weights
    a = get_allocation(get_vix_signal())
    cmd = sys.argv[1] if len(sys.argv) > 1 else "orders"
    if cmd == "mark":
        print("recorded as held:", mark_executed(a, sys.argv[2] if len(sys.argv) > 2 else "due"))
    else:
        o = a["orders"]
        print(f"{'FIRST RUN - no holdings recorded (initial build)' if o['first_run'] else 'vs recorded holdings'}; "
              f"monthly sleeves {'DUE' if o['monthly_due'] else 'not due (preview) until ' + a['next_rebalance']}")
        for x in o["per_sleeve"]:
            print(f"  {'DUE ' if x['due'] else 'prev'} {x['sleeve']:12s} {x['action']:4s} {x['ticker']:6s} ${x['dollars']:>8,}  ~{x['shares']} sh @ {x['price']}")
        print("NET (due only):", [(n['action'], n['ticker'], n['dollars']) for n in o["net_due"]])
