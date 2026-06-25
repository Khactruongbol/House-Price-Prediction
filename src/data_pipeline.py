from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml

from config import (
    DATA_COLLECTION_REPORT,
    DATA_QUALITY_REPORT,
    DEFAULT_RAW_DATASET,
    FIGURES_DIR,
    LABEL_REPORT,
    LABELS_DIR,
    METRICS_DIR,
    OPENML_RAW_DATASET,
    PROCESSED_DATASET,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)


def collect_real_data(force_openml: bool = False) -> pd.DataFrame:
    """Collect real Ames/House Prices data from Kaggle CSV or public OpenML."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if DEFAULT_RAW_DATASET.exists() and not force_openml:
        df = pd.read_csv(DEFAULT_RAW_DATASET)
        source = "kaggle_train_csv"
        source_path = str(DEFAULT_RAW_DATASET)
    else:
        bunch = fetch_openml(name="house_prices", as_frame=True, version=1)
        df = bunch.frame.copy()
        if TARGET_COLUMN not in df.columns and getattr(bunch, "target", None) is not None:
            df[TARGET_COLUMN] = bunch.target
        df.to_csv(OPENML_RAW_DATASET, index=False)
        source = "openml_house_prices"
        source_path = str(OPENML_RAW_DATASET)

    DATA_COLLECTION_REPORT.write_text(
        json.dumps(
            {
                "source": source,
                "source_path": source_path,
                "rows_collected": int(df.shape[0]),
                "columns_collected": int(df.shape[1]),
                "target_column": TARGET_COLUMN,
                "sample_data_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return df


def filter_noise_columns(df: pd.DataFrame, missing_threshold: float = 0.8) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned = df.copy()
    before_shape = cleaned.shape
    cleaned = cleaned.drop_duplicates()

    high_missing_cols = [
        col for col in cleaned.columns if col != TARGET_COLUMN and cleaned[col].isna().mean() >= missing_threshold
    ]
    cleaned = cleaned.drop(columns=high_missing_cols)

    constant_cols = [col for col in cleaned.columns if col != TARGET_COLUMN and cleaned[col].nunique(dropna=True) <= 1]
    cleaned = cleaned.drop(columns=constant_cols)

    report = {
        "shape_before_noise_filter": before_shape,
        "shape_after_noise_filter": cleaned.shape,
        "removed_duplicate_rows": int(before_shape[0] - df.drop_duplicates().shape[0]),
        "removed_high_missing_columns": high_missing_cols,
        "removed_constant_columns": constant_cols,
    }
    return cleaned, report


def filter_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target label column: {TARGET_COLUMN}")

    cleaned = df.copy()
    before_rows = len(cleaned)
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned = cleaned[cleaned[TARGET_COLUMN] > 0]

    # Ames Housing has a known suspicious pattern: very large living area sold cheaply.
    removed_known_outliers = 0
    if {"GrLivArea", TARGET_COLUMN}.issubset(cleaned.columns):
        mask = (cleaned["GrLivArea"] > 4000) & (cleaned[TARGET_COLUMN] < 300000)
        removed_known_outliers = int(mask.sum())
        cleaned = cleaned[~mask]

    report = {
        "rows_before_label_filter": int(before_rows),
        "rows_after_label_filter": int(len(cleaned)),
        "removed_missing_or_invalid_labels": int(before_rows - len(cleaned) - removed_known_outliers),
        "removed_known_target_feature_outliers": removed_known_outliers,
        "target_min": float(cleaned[TARGET_COLUMN].min()),
        "target_max": float(cleaned[TARGET_COLUMN].max()),
        "target_median": float(cleaned[TARGET_COLUMN].median()),
    }
    return cleaned, report


def assign_data_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["PriceSegment"] = pd.qcut(
        labeled[TARGET_COLUMN],
        q=4,
        labels=["budget", "standard", "premium", "luxury"],
        duplicates="drop",
    )
    if "OverallQual" in labeled.columns:
        labeled["QualityLabel"] = pd.cut(
            pd.to_numeric(labeled["OverallQual"], errors="coerce"),
            bins=[0, 4, 6, 8, 10],
            labels=["low_quality", "average_quality", "high_quality", "excellent_quality"],
            include_lowest=True,
        )
    if "GrLivArea" in labeled.columns:
        labeled["AreaSegment"] = pd.qcut(
            pd.to_numeric(labeled["GrLivArea"], errors="coerce"),
            q=4,
            labels=["small", "medium", "large", "very_large"],
            duplicates="drop",
        )
    return labeled


def save_dataset_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="PriceSegment", order=["budget", "standard", "premium", "luxury"])
    plt.title("Data Labels - Price Segment Counts")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_price_segment_counts.png", dpi=150)
    plt.close()

    if "QualityLabel" in df.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df, x="QualityLabel", y=TARGET_COLUMN)
        plt.xticks(rotation=20, ha="right")
        plt.title("SalePrice by Quality Label")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "saleprice_by_quality_label.png", dpi=150)
        plt.close()

    if {"Neighborhood", TARGET_COLUMN}.issubset(df.columns):
        top = df.groupby("Neighborhood", observed=False)[TARGET_COLUMN].median().sort_values(ascending=False).head(12)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top.index, y=top.values)
        plt.xticks(rotation=35, ha="right")
        plt.title("Top Neighborhoods by Median SalePrice")
        plt.ylabel("Median SalePrice")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "top_neighborhoods_median_saleprice.png", dpi=150)
        plt.close()

    if {"PriceSegment", "OverallQual"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="PriceSegment", y="OverallQual", order=["budget", "standard", "premium", "luxury"])
        plt.title("Overall Quality by Price Segment")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "overall_quality_by_price_segment.png", dpi=150)
        plt.close()


def prepare_real_dataset(force_openml: bool = False) -> pd.DataFrame:
    df = collect_real_data(force_openml=force_openml)
    cleaned, noise_report = filter_noise_columns(df)
    cleaned, label_filter_report = filter_labels(cleaned)
    labeled = assign_data_labels(cleaned)

    PROCESSED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(PROCESSED_DATASET, index=False)
    labeled[[TARGET_COLUMN, "PriceSegment"]].to_csv(LABEL_REPORT, index=False)
    save_dataset_figures(labeled)

    DATA_QUALITY_REPORT.write_text(
        json.dumps(
            {
                **noise_report,
                **label_filter_report,
                "processed_path": str(PROCESSED_DATASET),
                "rows_final": int(labeled.shape[0]),
                "columns_final": int(labeled.shape[1]),
                "label_columns_added": [col for col in ["PriceSegment", "QualityLabel", "AreaSegment"] if col in labeled],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return labeled


if __name__ == "__main__":
    prepared = prepare_real_dataset()
    print(f"Prepared real dataset: {PROCESSED_DATASET.name}")
    print(f"Rows: {prepared.shape[0]}; Columns: {prepared.shape[1]}")
