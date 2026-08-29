"""
Rental-property BUY scorecard for the Macro Cockpit.

Turns the validated real-asset signals into one decision table per metro:

    net cap rate  vs  30yr mortgage rate   -> the GO/NO-GO gate
    (income yield)    (cost of leverage)

RULE (user): only buy a metro if NET CAP RATE > MORTGAGE RATE. Above that line
leverage is POSITIVE (the property out-yields the debt, so it cash-flows and
leverage amplifies returns); below it you're negatively levered = a pure
appreciation bet. Among the metros that pass, rank by an expected-levered-return
proxy: cap-rate spread (cash flow) + forecast rent growth + price momentum.

Everything is computed from free, keyless sources:
  - rent   : Zillow ZORI SFR (monthly, ~3BR proxy)          [via rents.py]
  - price  : Zillow ZHVI SFR/condo tiered                    [via rents.py]
  - rent fwd: national CPI-rent + Case-Shiller model         [rents._fit_national]
  - mortgage: FRED MORTGAGE30US (30yr fixed)
Cap rate uses an SFR pro-forma: NOI = gross rent - opex(~30% of rent) - property
tax(state-specific % of price); net cap = NOI / price. State property tax is
first-order (TX/IL/NJ high erode cap rate; CA Prop-13 low), so it's modeled.

Cached 30 DAYS (rents/prices are monthly and slow-moving) - stale-while-revalidate.
"""
import os
import json
import time
import threading

import numpy as np
import pandas_datareader.data as web

from macrocockpit.macro_board import _STORE_DIR
from macrocockpit.rents import _read_csv, _metro_series, _fit_national, _ZORI, _ZHVI

_CACHE = os.path.join(_STORE_DIR, "buy_scorecard.json")
_TTL = 30 * 24 * 3600          # 30 days - real-estate data moves monthly at most
_lock = threading.Lock()

# Curated investable SFR-rental metros (ZORI RegionName format). Spans coastal
# appreciation plays (low cap) and Midwest/South cash-flow plays (high cap) so the
# cap-vs-mortgage gate is meaningful. Includes the two the user is weighing.
_METROS = [
    "San Jose, CA", "San Francisco, CA", "Raleigh, NC", "Charlotte, NC",
    "Austin, TX", "Dallas, TX", "Houston, TX", "San Antonio, TX",
    "Atlanta, GA", "Phoenix, AZ", "Tampa, FL", "Orlando, FL", "Jacksonville, FL",
    "Nashville, TN", "Memphis, TN", "Denver, CO", "Chicago, IL",
    "Indianapolis, IN", "Columbus, OH", "Cleveland, OH", "Kansas City, MO",
    "Oklahoma City, OK", "Birmingham, AL", "Pittsburgh, PA", "Las Vegas, NV",
]

# Effective property-tax rate by state (% of home value/yr, approx). First-order
# driver of net cap rate. Default 1.1% for anything unlisted.
_STATE_TAX = {
    "CA": 0.0075, "TX": 0.0160, "NC": 0.0080, "GA": 0.0090, "AZ": 0.0060,
    "FL": 0.0090, "TN": 0.0065, "CO": 0.0050, "IL": 0.0210, "IN": 0.0085,
    "OH": 0.0150, "MO": 0.0100, "OK": 0.0090, "AL": 0.0040, "PA": 0.0150,
    "NV": 0.0060, "NY": 0.0140, "NJ": 0.0220, "WA": 0.0090, "OR": 0.0090,
}
_OPEX_OF_RENT = 0.30           # maintenance + mgmt + vacancy + insurance, ~30% of gross rent


def _mortgage_rate():
    try:
        s = web.DataReader("MORTGAGE30US", "fred", "2020-01-01").iloc[:, 0].dropna()
        return float(s.iloc[-1])
    except Exception:
        return 6.5             # sane fallback if FRED hiccups


def compute_scorecard():
    a, b1, b2 = _fit_national()
    mort = _mortgage_rate()
    zori = _read_csv(_ZORI)
    zhvi = _read_csv(_ZHVI)
    rows = []
    for region in _METROS:
        rent = _metro_series(zori, region)
        price = _metro_series(zhvi, region)
        if rent is None or price is None or len(rent) < 13 or len(price) < 13:
            continue
        cur_rent = float(rent.iloc[-1])
        cur_price = float(price.iloc[-1])
        state = region.split(", ")[-1].strip()
        tax = _STATE_TAX.get(state, 0.011)
        # SFR pro-forma net cap rate
        net_cap = (cur_rent * 12.0 * (1.0 - _OPEX_OF_RENT) - cur_price * tax) / cur_price * 100.0
        spread = net_cap - mort                     # >0 = positive leverage (BUY gate)
        rent_yoy = float(rent.iloc[-1] / rent.iloc[-13] - 1) * 100.0
        hp_yoy = float(price.iloc[-1] / price.iloc[-13] - 1) * 100.0
        rent_fwd = a + b1 * rent_yoy + b2 * hp_yoy  # 12m rent-growth forecast
        # expected levered-return proxy: cash-flow spread + appreciation + rent growth
        score = spread + hp_yoy * 0.5 + rent_fwd * 0.3
        rows.append({
            "metro": region, "rent": round(cur_rent), "price": round(cur_price),
            "net_cap": round(net_cap, 2), "spread": round(spread, 2),
            "hp_yoy": round(hp_yoy, 1), "rent_fwd": round(rent_fwd, 1),
            "tax_rate": round(tax * 100, 2),
            "verdict": "BUY" if spread > 0 else "PASS",
            "score": round(score, 2),
        })
    # BUY-eligible first, then by expected-return score
    rows.sort(key=lambda r: (r["verdict"] == "BUY", r["score"]), reverse=True)
    return {"asof": str(rent.index[-1].date()), "mortgage_rate": round(mort, 2),
            "coefs": [round(a, 2), round(b1, 2), round(b2, 2)],
            "opex_of_rent": _OPEX_OF_RENT, "metros": rows}


def _fresh(c):
    return c and (time.time() - c.get("_ts", 0) < _TTL)


def _refresh_async():
    if _lock.locked():
        return
    def work():
        with _lock:
            try:
                d = compute_scorecard()
                d["_ts"] = time.time()
                with open(_CACHE, "w") as f:
                    json.dump(d, f)
            except Exception as e:
                print(f"buy_scorecard refresh error: {e}")
    threading.Thread(target=work, daemon=True).start()


def get_buy_scorecard():
    """Stale-while-revalidate: return cached scorecard instantly, refresh in the
    background only when >30 days old. Blocks once on a cold cache."""
    cache = None
    if os.path.exists(_CACHE):
        try:
            with open(_CACHE) as f:
                cache = json.load(f)
        except Exception:
            cache = None
    if _fresh(cache):
        cache["refreshing"], cache["stale"] = False, False
        return cache
    if cache is None:
        d = compute_scorecard()          # cold cache: compute once, synchronously
        d["_ts"] = time.time()
        try:
            with open(_CACHE, "w") as f:
                json.dump(d, f)
        except Exception:
            pass
        d["refreshing"], d["stale"] = False, False
        return d
    _refresh_async()                     # stale: serve old, refresh behind
    cache["refreshing"], cache["stale"] = True, True
    return cache


if __name__ == "__main__":
    d = compute_scorecard()
    print(f"30yr mortgage: {d['mortgage_rate']}%   (BUY gate: net cap rate > this)   asof {d['asof']}\n")
    h = f"{'metro':20s}{'rent/mo':>8s}{'price':>10s}{'tax%':>6s}{'netCap%':>9s}{'spread':>8s}{'HP YoY':>8s}{'rentFwd':>9s}  verdict"
    print(h); print("-" * len(h))
    for m in d["metros"]:
        print(f"{m['metro']:20s}{m['rent']:>7,}{m['price']:>10,}{m['tax_rate']:>6.1f}"
              f"{m['net_cap']:>9.2f}{m['spread']:>+8.2f}{m['hp_yoy']:>+7.1f}%{m['rent_fwd']:>+8.1f}%  {m['verdict']}")
