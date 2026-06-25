from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DEFAULT_RAW_DATASET, DERIVED_LABEL_COLUMNS, OPENML_RAW_DATASET, PROCESSED_DATASET, TARGET_COLUMN


def resolve_dataset_path(dataset_path: str | Path | None = None) -> Path:
    """Return a real dataset path. This function never falls back to sample data."""
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    if PROCESSED_DATASET.exists():
        return PROCESSED_DATASET

    if DEFAULT_RAW_DATASET.exists():
        return DEFAULT_RAW_DATASET

    if OPENML_RAW_DATASET.exists():
        return OPENML_RAW_DATASET

    raise FileNotFoundError(
        "No real dataset found. Run `python -m src.data_pipeline` first, "
        "or place Kaggle train.csv at data/raw/train.csv."
    )


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    path = resolve_dataset_path(dataset_path)
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET_COLUMN}'.")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}'.")
    drop_columns = [TARGET_COLUMN] + [col for col in DERIVED_LABEL_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_columns)
    y = df[TARGET_COLUMN]
    return X, y
