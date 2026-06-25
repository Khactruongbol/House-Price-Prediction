from __future__ import annotations

import pandas as pd

from config import METRICS_PATH


def load_model_ranking(metrics_path=METRICS_PATH) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame(columns=["model", "RMSE", "MAE", "R2"])
    return pd.read_csv(metrics_path)


def simple_price_explanation(input_row: dict) -> list[str]:
    notes: list[str] = []
    if input_row.get("OverallQual") is not None:
        notes.append("OverallQual cao thường làm giá dự đoán tăng.")
    if input_row.get("GrLivArea") is not None:
        notes.append("GrLivArea lớn thường có tương quan dương với SalePrice.")
    if input_row.get("GarageCars") is not None:
        notes.append("Số chỗ đậu xe/garage là một tín hiệu tiện ích quan trọng.")
    if input_row.get("YearBuilt") is not None:
        notes.append("Nhà xây mới hơn thường có lợi thế về chất lượng và giá.")
    return notes or ["Dự đoán dựa trên toàn bộ feature đã nhập và pipeline đã train."]
