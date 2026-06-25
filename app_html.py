from __future__ import annotations

from flask import Flask, render_template, request

from config import MODEL_PATH
from src.explain_prediction import load_model_ranking, simple_price_explanation
from src.predict import predict_price
from src.train import train_all


app = Flask(__name__)


DEFAULT_INPUT = {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotArea": 8450,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "GrLivArea": 1710,
    "FullBath": 2,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "GarageCars": 2,
    "GarageArea": 548,
    "Neighborhood": "CollgCr",
    "HouseStyle": "2Story",
    "SaleCondition": "Normal",
}

NUMERIC_FIELDS = {
    "MSSubClass",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "FullBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "GarageCars",
    "GarageArea",
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
    ranking = load_model_ranking().to_dict(orient="records")

    if request.method == "POST":
        input_values = parse_form(request.form)
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
        ranking=ranking,
    )


if __name__ == "__main__":
    app.run(debug=False)
