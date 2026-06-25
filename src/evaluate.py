from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def metrics_to_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(metrics, orient="index")
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values(["RMSE", "MAE"], ascending=True)
    )
