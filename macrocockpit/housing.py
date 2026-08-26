"""
Stale-while-revalidate accessor for the validated metro housing ranking, for the
Macro Cockpit. The model (compute_metro_ranking) fetches 19 Case-Shiller metros +
macro and trains a GBM (~15s), so it must NEVER block the tab: read the cached
ranking instantly, and if it's older than a day, refresh in a BACKGROUND thread.
Housing data is monthly, so a daily refresh is plenty.
"""
import os
import json
import time
import threading

from macrocockpit.macro_board import _STORE_DIR

_CACHE = os.path.join(_STORE_DIR, "housing_ranking.json")
_TTL = 24 * 3600
_lock = threading.Lock()


def _refresh():
    if not _lock.acquire(blocking=False):
        return
    def run():
        try:
            from housing.predict_housing import compute_metro_ranking
            d = compute_metro_ranking()
            tmp = _CACHE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"ts": time.time(), **d}, f)
            os.replace(tmp, _CACHE)
        except Exception:
            pass
        finally:
            _lock.release()
    threading.Thread(target=run, daemon=True).start()


def get_metro_ranking():
    out = {"ranking": [], "skill": {}, "asof": "?", "refreshing": _lock.locked(), "stale": True}
    try:
        with open(_CACHE) as f:
            c = json.load(f)
        out["ranking"] = c.get("ranking", [])
        out["skill"] = c.get("skill", {})
        out["asof"] = c.get("asof", "?")
        out["stale"] = (time.time() - c.get("ts", 0)) > _TTL
    except Exception:
        out["stale"] = True
    if out["stale"]:
        _refresh()               # non-blocking; populates for the next load
        out["refreshing"] = True
    return out
