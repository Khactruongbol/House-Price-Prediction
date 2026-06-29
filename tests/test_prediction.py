from config import METRICS_PATH, MODEL_PATH
from src.data_loader import load_dataset, split_features_target
from src.predict import predict_price
from src.train import train_all


def test_train_creates_model_and_prediction_is_positive():
    if not MODEL_PATH.exists():
        train_all(save_artifacts=True)
    assert MODEL_PATH.exists()

    df = load_dataset()
    X, _ = split_features_target(df)
    prediction = predict_price(X.head(1))
    assert prediction.iloc[0] > 0
    assert METRICS_PATH.exists()
    assert (METRICS_PATH.parent / "model_tuning_report.csv").exists()


def test_html_app_renders_form():
    from app_html import app

    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"House Price Prediction" in response.data
    assert "charset=utf-8" in response.content_type.lower()


def test_html_app_preserves_vietnamese_text():
    from app_html import DEFAULT_INPUT, app

    client = app.test_client()
    response = client.post("/", data=DEFAULT_INPUT)
    html = response.data.decode("utf-8")
    assert response.status_code == 200
    assert "Thông tin căn nhà" in html
    assert "Dự đoán giá nhà" in html
    assert "Giải thích nhanh" in html
    assert "Kết quả dự đoán trả về" in html



def test_prediction_input_validation_rejects_invalid_rows():
    from src.input_validation import validate_prediction_input

    result = validate_prediction_input({
        "DatasetSource": "unknown_source",
        "PriceUnit": "USD",
        "LivingArea": 0,
        "LotArea": 1000,
        "Bedrooms": -1,
        "Bathrooms": 1,
        "QualityScore": 5,
        "ConditionScore": 5,
        "YearBuilt": 1700,
        "YearRemodAdd": 1600,
        "Neighborhood": "",
        "HouseType": "House",
        "SaleCondition": "Normal",
    })

    assert result["is_valid"] is False
    assert result["errors"]
    assert result["warnings"]
    assert result["return_shape"]["PredictedSalePrice"] == "numeric"


def test_prediction_input_validation_accepts_default_input():
    from app_html import DEFAULT_INPUT
    from src.input_validation import validate_prediction_input

    result = validate_prediction_input(DEFAULT_INPUT)

    assert result["is_valid"] is True
    assert result["return_shape"]["DatasetSource"] == DEFAULT_INPUT["DatasetSource"]
    assert result["return_shape"]["PriceUnit"]
