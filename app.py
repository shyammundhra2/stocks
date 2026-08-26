# app.py
from flask import Flask, render_template, make_response
from datetime import datetime
import json

app = Flask(__name__)


@app.route("/")
def index():
    from macro.indicators import (
        get_risk_regime,
        get_vix_signal,
        get_mean_reversion,
        get_sector_rotation,
        get_country_rotation,
        get_commodity_rotation,
        get_currency_rotation,
        get_trends,
        get_portfolio_summary,
        get_ml_sector_prediction,
        get_ml_country_prediction,
        get_ml_commodity_prediction
    )
    from macro.indicators import _get_shared_market_data

    # get_trends() must be called before get_portfolio_summary()
    # so the optimizer runs and populates the module-level cache
    trends = get_trends()

    # Freshness stamp so a stale (cached) tab is obvious at a glance.
    try:
        _idx = _get_shared_market_data().index
        data_asof = str(_idx[-1].date()) if len(_idx) else "?"
    except Exception:
        data_asof = "?"
    rendered_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_data = {
        "regime": get_risk_regime(),
        "vix_mr": get_vix_signal(),
        "mr": get_mean_reversion(),
        "sr": get_sector_rotation(),
        "countries": get_country_rotation(),
        "commodities": get_commodity_rotation(),
        "currencies": get_currency_rotation(),
        "trends": trends,
        "portfolio_summary": get_portfolio_summary(),
        "ml_sector": get_ml_sector_prediction(),
        "ml_country": get_ml_country_prediction(),
        "ml_commodity": get_ml_commodity_prediction()
    }

    def convert_to_serializable(obj):
        """Recursively convert non-JSON-serializable objects"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        else:
            return str(obj)

    serializable_data = convert_to_serializable(all_data)

    # Top-10 stock sleeve: stale-while-revalidate - reads the cached CSV instantly
    # (never blocks) and refreshes in a background thread only when >1 day old.
    from stockselection.selected import get_selected_stocks
    selected_stocks = get_selected_stocks()

    resp = make_response(render_template(
        "dashboard.html",
        selected_stocks=selected_stocks,
        regime=all_data["regime"],
        vix_mr=all_data["vix_mr"],
        mr=all_data["mr"],
        sr=all_data["sr"],
        countries=all_data["countries"],
        commodities=all_data["commodities"],
        currencies=all_data["currencies"],
        trends=all_data["trends"],
        portfolio_summary=all_data["portfolio_summary"],
        ml_sector=all_data["ml_sector"],
        ml_country=all_data["ml_country"],
        ml_commodity=all_data["ml_commodity"],
        all_data_json=json.dumps(serializable_data),
        data_asof=data_asof,
        rendered_at=rendered_at,
        active_tab="trading",
    ))
    # Never let the browser serve a stale cached dashboard.
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/macro")
def macro():
    # Lazy: the macro data (FRED + markets) is fetched ONLY here, so the trading
    # tab's initial load is never delayed by it. Disk-cached + incremental FRED.
    # macrocockpit is a standalone package, separate from the trading engine.
    from macrocockpit.macro_board import get_macro_dashboard
    from macrocockpit.cycles import get_cycles
    md = get_macro_dashboard()
    cy = get_cycles()
    charts = dict(md.get("charts", {}))
    charts["cycles"] = cy      # cycle chart series bundled for the client
    resp = make_response(render_template(
        "macro_dashboard.html",
        md=md,
        cy=cy,
        charts_json=json.dumps(charts),
        rendered_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        active_tab="macro",
    ))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


if __name__ == "__main__":
    app.run(debug=True)
