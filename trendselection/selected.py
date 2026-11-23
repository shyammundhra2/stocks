"""
Stale-while-revalidate accessor for the trend-asset sleeve, for the trading tab.
Same pattern as stockselection/selected.py: reads the already-produced ranking
CSV INSTANTLY (never blocks the page), and if that cache is older than a day,
kicks off a BACKGROUND refresh (re-runs the ranker as a subprocess) - so the
page always shows the last-known holdings immediately and the fresh set
appears on the next load.
"""
import os
import sys
import time
import threading
import subprocess

import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.join(_DIR, "trend_etf_ranking.csv")
_TTL = 24 * 3600
_lock = threading.Lock()


def _bg_refresh():
    if not _lock.acquire(blocking=False):
        return
    def run():
        try:
            subprocess.run([sys.executable, "predict_top_trends.py"], cwd=_DIR,
                           timeout=300, capture_output=True)
        except Exception as e:
            print(f"trendselection/selected.py background refresh error (serving stale sleeve): {e}")
        finally:
            _lock.release()
    threading.Thread(target=run, daemon=True).start()


def get_selected_trends():
    out = {"holdings": [], "asof": "?", "stale": True, "refreshing": _lock.locked()}
    try:
        mtime = os.path.getmtime(_CSV)
        df = pd.read_csv(_CSV)
        h = df[df["Weight"] > 0].sort_values("Weight", ascending=False)
        out["holdings"] = [{
            "ticker": r["Ticker"],
            "name": (r.get("Name", "") or "")[:18],
            "mom": int(round(float(r["Mom_12_1"]) * 100)),
            "dd": round(float(r["DD_From_High"]) * 100, 1) if pd.notna(r.get("DD_From_High")) else None,
            "tilted": bool(r.get("Tilt_Applied", False)),
            "weight": round(float(r["Weight"]) * 100, 1),
        } for _, r in h.iterrows()]
        out["asof"] = time.strftime("%b %d", time.localtime(mtime))
        out["stale"] = (time.time() - mtime) > _TTL
    except Exception:
        out["stale"] = True
    if out["stale"]:
        _bg_refresh()
        out["refreshing"] = True
    return out
