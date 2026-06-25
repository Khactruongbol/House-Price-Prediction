from __future__ import annotations

import pandas as pd

from config import METRICS_PATH


def load_model_ranking(metrics_path=METRICS_PATH) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame(columns=["model", "RMSE", "MAE", "R2"])
    return pd.read_csv(metrics_path)


def simple_price_explanation(input_row: dict) -> list[str]:
    notes: list[str] = []
    if input_row.get("DatasetSource") is not None:
        notes.append("DatasetSource xác định ngữ cảnh thị trường và đơn vị giá cần diễn giải.")
    if input_row.get("PriceUnit") is not None:
        notes.append(f"Kết quả dự đoán trả về theo đơn vị: {input_row.get('PriceUnit')}.")
    if input_row.get("QualityScore") is not None:
        notes.append("QualityScore cao thường làm giá dự đoán tăng.")
    if input_row.get("LivingArea") is not None:
        notes.append("LivingArea lớn thường có tương quan dương với SalePrice.")
    if input_row.get("YearBuilt") is not None:
        notes.append("Nhà xây mới hơn thường có lợi thế về chất lượng và giá.")
    return notes or ["Dự đoán dựa trên toàn bộ feature đã nhập và pipeline đã train."]
