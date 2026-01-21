# app.py
from flask import Flask, render_template
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
        get_ml_sector_prediction,
        get_ml_country_prediction,
        get_ml_commodity_prediction
    )

    # Collect all data
    all_data = {
        "regime": get_risk_regime(),
        "vix_mr": get_vix_signal(),
        "mr": get_mean_reversion(),
        "sr": get_sector_rotation(),
        "countries": get_country_rotation(),
        "commodities": get_commodity_rotation(),
        "currencies": get_currency_rotation(),
        "trends": get_trends(),
        "ml_sector": get_ml_sector_prediction(),
        "ml_country": get_ml_country_prediction(),
        "ml_commodity": get_ml_commodity_prediction()
    }

    # Custom JSON encoder to handle non-serializable types
    def convert_to_serializable(obj):
        """Recursively convert non-JSON-serializable objects"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        else:
            # Convert other types to string
            return str(obj)

    # Convert data to JSON-serializable format
    serializable_data = convert_to_serializable(all_data)

    return render_template(
        "dashboard.html",
        regime=all_data["regime"],
        vix_mr=all_data["vix_mr"],
        mr=all_data["mr"],
        sr=all_data["sr"],
        countries=all_data["countries"],
        commodities=all_data["commodities"],
        currencies=all_data["currencies"],
        trends=all_data["trends"],
        ml_sector=all_data["ml_sector"],
        ml_country=all_data["ml_country"],
        ml_commodity=all_data["ml_commodity"],
        all_data_json=json.dumps(serializable_data)  # Pass serialized data to template
    )


if __name__ == "__main__":
    app.run(debug=True)