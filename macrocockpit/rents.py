"""
Rent tracker + 1-year-forward rent forecast for the Macro Cockpit.

Rents are one of the MOST forecastable series (validated: persistence rank IC
+0.55, home-price-leads-rent +0.56, market-rent-leads-CPI +0.58). This surfaces
current single-family rent (Zillow ZORI SFR ~ 3BR proxy) for the US, Raleigh, and
San Jose (Milpitas) plus a 1yr-forward growth forecast from a model fit nationally
on CPI-rent + Case-Shiller:

    fwd_12m_rent_growth = a + b1 * trailing_rent_yoy + b2 * home_price_yoy
    (persistence)                                     (the ~12mo housing lead)

Zillow SFR ZORI (rent) + ZHVI (home price) per metro drive the per-metro forecast.
Stale-while-revalidate cached (daily) - rents are monthly, so daily is plenty.
"""
import os
import json
import time
import ssl
import threading
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pandas_datareader.data as web

from macrocockpit.macro_board import _STORE_DIR

_CACHE = os.path.join(_STORE_DIR, "rents.json")
_TTL = 24 * 3600
_lock = threading.Lock()

_ZORI = "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfr_sm_sa_month.csv"
_ZHVI = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
_METROS = {"United States": "National", "Raleigh, NC": "Raleigh", "San Jose, CA": "Milpitas (San Jose)"}


def _read_csv(url):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE          # Zillow public CSVs fail strict verify
    return pd.read_csv(urlopen(url, context=ctx, timeout=60))


def _metro_series(df, region):
    r = df[df["RegionName"] == region]
    if not len(r):
        return None
    dc = [c for c in df.columns if c[:2] == "20" and "-" in c]
    s = pd.Series(r[dc].values.flatten(), index=pd.to_datetime(dc)).dropna()
    return s


def _fit_national():
    """fwd-12m rent growth ~ a + b1*trailing rent YoY + b2*home-price YoY (CPI + CS)."""
    cpi = web.DataReader("CUSR0000SEHA", "fred", "1990-01-01").iloc[:, 0].resample("ME").last()
    cs = web.DataReader("CSUSHPINSA", "fred", "1990-01-01").iloc[:, 0].resample("ME").last()
    d = pd.DataFrame({"rent": cpi.pct_change(12) * 100, "hp": cs.pct_change(12) * 100})
    d["fwd"] = d["rent"].shift(-12)
    d = d.dropna()
    X = np.c_[np.ones(len(d)), d["rent"], d["hp"]]
    beta, *_ = np.linalg.lstsq(X, d["fwd"].values, rcond=None)
    return [float(b) for b in beta]     # a, b1, b2


def compute_rents():
    a, b1, b2 = _fit_national()
    zori = _read_csv(_ZORI)
    zhvi = _read_csv(_ZHVI)
    out = []
    for region, label in _METROS.items():
        rent = _metro_series(zori, region)
        hv = _metro_series(zhvi, region)
        if rent is None or len(rent) < 13:
            continue
        cur = float(rent.iloc[-1])
        rent_yoy = float(rent.iloc[-1] / rent.iloc[-13] - 1) * 100
        hp_yoy = float(hv.iloc[-1] / hv.iloc[-13] - 1) * 100 if hv is not None and len(hv) > 13 else 0.0
        fwd = a + b1 * rent_yoy + b2 * hp_yoy
        out.append({"metro": label, "rent": round(cur), "rent_yoy": round(rent_yoy, 1),
                    "hp_yoy": round(hp_yoy, 1), "fwd": round(fwd, 1),
                    "rent_1yr": round(cur * (1 + fwd / 100))})
    return {"asof": str(rent.index[-1].date()), "coefs": [round(a, 2), round(b1, 2), round(b2, 2)],
            "metros": out}


def _refresh():
    if not _lock.acquire(blocking=False):
        return
    def run():
        try:
            d = compute_rents()
            with open(_CACHE + ".tmp", "w") as f:
                json.dump({"ts": time.time(), **d}, f)
            os.replace(_CACHE + ".tmp", _CACHE)
        except Exception as e:
            print(f"rents refresh error: {e}")
        finally:
            _lock.release()
    threading.Thread(target=run, daemon=True).start()


def get_rents():
    out = {"metros": [], "asof": "?", "refreshing": _lock.locked(), "stale": True}
    try:
        with open(_CACHE) as f:
            c = json.load(f)
        out.update({k: c.get(k) for k in ["metros", "asof", "coefs"]})
        out["stale"] = (time.time() - c.get("ts", 0)) > _TTL
    except Exception:
        out["stale"] = True
    if out["stale"]:
        _refresh()
        out["refreshing"] = True
    return out


if __name__ == "__main__":
    d = compute_rents()
    a, b1, b2 = d["coefs"]
    print(f"forecast model: fwd_rent = {a} + {b1}*rent_yoy + {b2}*hp_yoy   (as of {d['asof']})\n")
    print(f"{'metro':22s}{'rent/mo':>9s}{'trail YoY':>10s}{'HP YoY':>8s}{'1yr fwd':>9s}{'rent in 1yr':>12s}")
    for m in d["metros"]:
        print(f"{m['metro']:22s}{m['rent']:>8,}{m['rent_yoy']:>+9.1f}%{m['hp_yoy']:>+7.1f}%{m['fwd']:>+8.1f}%{m['rent_1yr']:>11,}")
