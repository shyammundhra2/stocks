# app.py
from flask import Flask, render_template

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
        get_ml_sector_prediction,    # NEW
        get_ml_country_prediction,   # NEW
        get_ml_commodity_prediction  # NEW
    )

    return render_template(
        "dashboard.html",
        regime=get_risk_regime(),
        vix_mr=get_vix_signal(),
        mr=get_mean_reversion(),
        sr=get_sector_rotation(),
        countries=get_country_rotation(),
        commodities=get_commodity_rotation(),
        currencies=get_currency_rotation(),
        trends=get_trends(),
        ml_sector=get_ml_sector_prediction(),        # NEW
        ml_country=get_ml_country_prediction(),      # NEW
        ml_commodity=get_ml_commodity_prediction()   # NEW
    )

if __name__ == "__main__":
    app.run(debug=True)

