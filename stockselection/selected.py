"""
Stale-while-revalidate accessor for the top-10 stock sleeve, for the trading tab.
Reads the already-produced ranking CSV INSTANTLY (never blocks the page), and if
that cache is older than a day, kicks off a BACKGROUND refresh (re-runs the
ranker as a subprocess) - so the page always shows the last-known 10 immediately
and the fresh set appears on the next load. Never increases trading-tab load time.
"""
import os
import sys
import time
import threading
import subprocess

import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.join(_DIR, "macro_2027_ranking_enhanced.csv")
_TTL = 24 * 3600
_lock = threading.Lock()


def _bg_refresh():
    """Re-run the ranker in a daemon thread (downloads S&P 500 - slow), guarded
    by a non-blocking lock so only one refresh runs at a time."""
    if not _lock.acquire(blocking=False):
        return
    def run():
        try:
            subprocess.run([sys.executable, "predict_top_stocks.py"], cwd=_DIR,
                           timeout=900, capture_output=True)
        except Exception:
            pass
        finally:
            _lock.release()
    threading.Thread(target=run, daemon=True).start()


def get_selected_stocks():
    out = {"stocks": [], "gate": "", "asof": "?", "stale": True, "refreshing": _lock.locked()}
    try:
        mtime = os.path.getmtime(_CSV)
        df = pd.read_csv(_CSV)
        b = df[df["Signal"] == "BUY"].sort_values("Mom_12_1", ascending=False)
        out["stocks"] = [{
            "ticker": r["Ticker"],
            "sector": (r.get("Sector", "") or "")[:14],
            "mom": int(round(float(r["Mom_12_1"]) * 100)),
            "alloc": float(r.get("Size_Allocation", 0) or 0),
        } for _, r in b.iterrows()]
        out["gate"] = str(df["Gate"].iloc[0]) if "Gate" in df.columns and len(df) else ""
        out["asof"] = time.strftime("%b %d", time.localtime(mtime))
        out["stale"] = (time.time() - mtime) > _TTL
    except Exception:
        out["stale"] = True
    if out["stale"]:
        _bg_refresh()               # non-blocking; refreshes for the NEXT load
        out["refreshing"] = True
    return out
