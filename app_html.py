from __future__ import annotations

from flask import Flask, render_template, request

from config import MODEL_PATH
from src.explain_prediction import load_model_ranking, simple_price_explanation
from src.input_validation import validate_prediction_input
from src.predict import predict_price
from src.train import train_all


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


@app.after_request
def set_utf8_headers(response):
    if response.mimetype == "text/html":
        response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


DEFAULT_INPUT = {
    "DatasetSource": "king_county_house_sales",
    "PriceUnit": "USD",
    "LivingArea": 1710,
    "LotArea": 8450,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "QualityScore": 7,
    "ConditionScore": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "Neighborhood": "CollgCr",
    "HouseType": "1Fam",
    "SaleCondition": "Normal",
}

NUMERIC_FIELDS = {
    "LivingArea",
    "LotArea",
    "Bedrooms",
    "Bathrooms",
    "QualityScore",
    "ConditionScore",
    "YearBuilt",
    "YearRemodAdd",
}


def parse_form(form) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, default in DEFAULT_INPUT.items():
        raw_value = form.get(key, default)
        if key in NUMERIC_FIELDS:
            values[key] = float(raw_value)
        else:
            values[key] = str(raw_value)
    return values


@app.route("/", methods=["GET", "POST"])
def index():
    input_values = DEFAULT_INPUT.copy()
    prediction = None
    notes = []
    errors = []
    warnings = []
    ranking = load_model_ranking().to_dict(orient="records")

    if request.method == "POST":
        input_values = parse_form(request.form)
        validation = validate_prediction_input(input_values)
        errors = validation["errors"]
        warnings = validation["warnings"]
        if validation["is_valid"]:
            if not MODEL_PATH.exists():
                train_all()
            prediction = float(predict_price(input_values).iloc[0])
            notes = simple_price_explanation(input_values)
        ranking = load_model_ranking().to_dict(orient="records")

    return render_template(
        "index.html",
        input_values=input_values,
        prediction=prediction,
        notes=notes,
        errors=errors,
        warnings=warnings,
        ranking=ranking,
    )


if __name__ == "__main__":
    app.run(debug=False)
