from config import METRICS_PATH, MODEL_PATH
from src.data_loader import load_dataset, split_features_target
from src.predict import predict_price
from src.train import train_all


def test_train_creates_model_and_prediction_is_positive():
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
