from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH
from src.feature_engineering import add_house_features


def load_model(model_path: str | Path = MODEL_PATH) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_price(input_data: dict | pd.DataFrame, model_path: str | Path = MODEL_PATH) -> pd.Series:
    artifact = load_model(model_path)
    pipeline = artifact["pipeline"]
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    df = add_house_features(df)
    feature_columns = artifact["feature_columns"]
    for column in feature_columns:
        if column not in df.columns:
            df[column] = np.nan
    predictions = pipeline.predict(df[feature_columns])
    return pd.Series(predictions, name="PredictedSalePrice")
