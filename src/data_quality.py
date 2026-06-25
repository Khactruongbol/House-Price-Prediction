from __future__ import annotations

import pandas as pd

from config import TARGET_COLUMN


def summarize_quality(df: pd.DataFrame) -> dict[str, object]:
    missing = df.isna().sum().sort_values(ascending=False)
    duplicate_rows = int(df.duplicated().sum())
    target_missing = int(df[TARGET_COLUMN].isna().sum()) if TARGET_COLUMN in df else None
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "target_missing": target_missing,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "top_missing": missing[missing > 0].head(15).to_dict(),
    }


def remove_target_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Remove extreme target outliers using IQR; useful for optional experiments."""
    q1 = df[TARGET_COLUMN].quantile(0.25)
    q3 = df[TARGET_COLUMN].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return df[(df[TARGET_COLUMN] >= lower) & (df[TARGET_COLUMN] <= upper)].copy()
