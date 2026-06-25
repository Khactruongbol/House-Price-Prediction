from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    DATA_COLLECTION_REPORT,
    DATA_QUALITY_REPORT,
    EDGE_CASE_REPORT,
    FIGURES_DIR,
    KAGGLE_RAW_DIR,
    LABEL_REPORT,
    LABELS_DIR,
    METRICS_DIR,
    PROCESSED_DATASET,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class KaggleSource:
    name: str
    slug: str
    price_unit: str
    expected_files: tuple[str, ...]


KAGGLE_SOURCES = [
    KaggleSource(
        name="ames_housing",
        slug="shashanknecrothapa/ames-housing-dataset",
        price_unit="USD",
        expected_files=("AmesHousing.csv",),
    ),
    KaggleSource(
        name="simple_housing_prices",
        slug="yasserh/housing-prices-dataset",
        price_unit="dataset_original_unit",
        expected_files=("Housing.csv",),
    ),
    KaggleSource(
        name="bengaluru_house_prices",
        slug="amitabhajoy/bengaluru-house-price-data",
        price_unit="lakh_INR",
        expected_files=("Bengaluru_House_Data.csv",),
    ),
    KaggleSource(
        name="king_county_house_sales",
        slug="harlfoxem/housesalesprediction",
        price_unit="USD",
        expected_files=("kc_house_data.csv",),
    ),
]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()


def _copy_csv_files(downloaded_path: str | Path, source: KaggleSource) -> list[Path]:
    source_dir = KAGGLE_RAW_DIR / source.name
    source_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for csv_path in Path(downloaded_path).rglob("*.csv"):
        target = source_dir / csv_path.name
        shutil.copy2(csv_path, target)
        copied.append(target)
    return copied


def collect_kaggle_raw_data() -> tuple[list[Path], list[dict[str, object]]]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_files: list[Path] = []
    collection_rows: list[dict[str, object]] = []
    for source in KAGGLE_SOURCES:
        try:
            downloaded_path = kagglehub.dataset_download(source.slug)
            copied_files = _copy_csv_files(downloaded_path, source)
            raw_files.extend(copied_files)
            collection_rows.append(
                {
                    "source": source.name,
                    "slug": source.slug,
                    "status": "downloaded",
                    "price_unit": source.price_unit,
                    "raw_files": [str(path) for path in copied_files],
                }
            )
        except Exception as exc:
            collection_rows.append(
                {
                    "source": source.name,
                    "slug": source.slug,
                    "status": "failed",
                    "price_unit": source.price_unit,
                    "error": str(exc),
                }
            )

    DATA_COLLECTION_REPORT.write_text(json.dumps(collection_rows, indent=2), encoding="utf-8")
    return raw_files, collection_rows


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [_safe_name(col) for col in cleaned.columns]
    return cleaned


def _parse_sqft(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan
    if "-" in text and len(numbers) >= 2:
        return float(np.mean(numbers[:2]))
    return numbers[0]


def _parse_bhk(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return float(match.group(0)) if match else np.nan


def _pick(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate]
    return pd.Series([np.nan] * len(df), index=df.index)


def _canonicalize_ames(df: pd.DataFrame) -> pd.DataFrame:
    raw = _clean_columns(df)
    out = pd.DataFrame(index=raw.index)
    out[TARGET_COLUMN] = raw["saleprice"]
    out["LivingArea"] = raw.get("gr_liv_area")
    out["LotArea"] = raw.get("lot_area")
    out["Bedrooms"] = raw.get("bedroom_abvgr")
    out["Bathrooms"] = raw.get("full_bath")
    out["QualityScore"] = raw.get("overall_qual")
    out["ConditionScore"] = raw.get("overall_cond")
    out["YearBuilt"] = raw.get("year_built")
    out["YearRemodAdd"] = raw.get("year_remod_add")
    out["GarageCars"] = raw.get("garage_cars")
    out["GarageArea"] = raw.get("garage_area")
    out["Neighborhood"] = raw.get("neighborhood")
    out["HouseType"] = raw.get("house_style")
    out["SaleCondition"] = raw.get("sale_condition")
    return out


def _canonicalize_simple_housing(df: pd.DataFrame) -> pd.DataFrame:
    raw = _clean_columns(df)
    out = pd.DataFrame(index=raw.index)
    out[TARGET_COLUMN] = raw["price"]
    out["LivingArea"] = raw.get("area")
    out["LotArea"] = raw.get("area")
    out["Bedrooms"] = raw.get("bedrooms")
    out["Bathrooms"] = raw.get("bathrooms")
    out["QualityScore"] = raw.get("stories")
    out["ConditionScore"] = np.nan
    out["YearBuilt"] = np.nan
    out["YearRemodAdd"] = np.nan
    out["GarageCars"] = raw.get("parking")
    out["GarageArea"] = np.nan
    out["Neighborhood"] = raw.get("prefarea")
    out["HouseType"] = raw.get("furnishingstatus")
    out["SaleCondition"] = raw.get("mainroad")
    return out


def _canonicalize_bengaluru(df: pd.DataFrame) -> pd.DataFrame:
    raw = _clean_columns(df)
    out = pd.DataFrame(index=raw.index)
    out[TARGET_COLUMN] = raw["price"]
    out["LivingArea"] = raw["total_sqft"].map(_parse_sqft)
    out["LotArea"] = out["LivingArea"]
    out["Bedrooms"] = raw["size"].map(_parse_bhk)
    out["Bathrooms"] = raw.get("bath")
    out["QualityScore"] = np.nan
    out["ConditionScore"] = np.nan
    out["YearBuilt"] = np.nan
    out["YearRemodAdd"] = np.nan
    out["GarageCars"] = np.nan
    out["GarageArea"] = np.nan
    out["Neighborhood"] = raw.get("location")
    out["HouseType"] = raw.get("area_type")
    out["SaleCondition"] = raw.get("availability")
    return out


def _canonicalize_king_county(df: pd.DataFrame) -> pd.DataFrame:
    raw = _clean_columns(df)
    out = pd.DataFrame(index=raw.index)
    out[TARGET_COLUMN] = raw["price"]
    out["LivingArea"] = raw.get("sqft_living")
    out["LotArea"] = raw.get("sqft_lot")
    out["Bedrooms"] = raw.get("bedrooms")
    out["Bathrooms"] = raw.get("bathrooms")
    out["QualityScore"] = raw.get("grade")
    out["ConditionScore"] = raw.get("condition")
    out["YearBuilt"] = raw.get("yr_built")
    out["YearRemodAdd"] = raw.get("yr_renovated").replace(0, np.nan) if "yr_renovated" in raw else np.nan
    out["GarageCars"] = np.nan
    out["GarageArea"] = np.nan
    out["Neighborhood"] = raw.get("zipcode").astype(str) if "zipcode" in raw else np.nan
    out["HouseType"] = raw.get("waterfront").astype(str) if "waterfront" in raw else np.nan
    out["SaleCondition"] = raw.get("view").astype(str) if "view" in raw else np.nan
    return out


def canonicalize_source(raw_file: Path, source_name: str, price_unit: str) -> pd.DataFrame:
    df = pd.read_csv(raw_file)
    if source_name == "ames_housing":
        canonical = _canonicalize_ames(df)
    elif source_name == "simple_housing_prices":
        canonical = _canonicalize_simple_housing(df)
    elif source_name == "bengaluru_house_prices":
        canonical = _canonicalize_bengaluru(df)
    elif source_name == "king_county_house_sales":
        canonical = _canonicalize_king_county(df)
    else:
        raise ValueError(f"Unsupported source: {source_name}")
    canonical["DatasetSource"] = source_name
    canonical["PriceUnit"] = price_unit
    canonical["RawFile"] = raw_file.name
    return canonical


def collect_and_canonicalize_kaggle_data() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    raw_files, collection_rows = collect_kaggle_raw_data()
    by_source = {source.name: source for source in KAGGLE_SOURCES}
    frames: list[pd.DataFrame] = []
    for row in collection_rows:
        if row.get("status") != "downloaded":
            continue
        source = by_source[str(row["source"])]
        for raw_file in row["raw_files"]:
            path = Path(str(raw_file))
            frames.append(canonicalize_source(path, source.name, source.price_unit))

    if not frames:
        raise RuntimeError("No Kaggle raw CSV files were downloaded successfully.")

    return pd.concat(frames, ignore_index=True), collection_rows


def filter_noise_columns(df: pd.DataFrame, missing_threshold: float = 0.9) -> tuple[pd.DataFrame, dict[str, object]]:
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

    impossible_area = cleaned["LivingArea"].notna() & (pd.to_numeric(cleaned["LivingArea"], errors="coerce") <= 0)
    impossible_rooms = cleaned["Bedrooms"].notna() & (pd.to_numeric(cleaned["Bedrooms"], errors="coerce") < 0)
    cleaned = cleaned[~(impossible_area | impossible_rooms)]

    report = {
        "rows_before_label_filter": int(before_rows),
        "rows_after_label_filter": int(len(cleaned)),
        "removed_missing_or_invalid_labels": int(before_rows - len(cleaned)),
        "target_min": float(cleaned[TARGET_COLUMN].min()),
        "target_max": float(cleaned[TARGET_COLUMN].max()),
        "target_median": float(cleaned[TARGET_COLUMN].median()),
    }
    return cleaned, report


def assign_data_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    categorical_context = ["DatasetSource", "PriceUnit", "RawFile", "Neighborhood", "HouseType", "SaleCondition"]
    for column in categorical_context:
        if column in labeled.columns:
            labeled[column] = labeled[column].astype("string")
    labeled["PriceSegment"] = labeled.groupby("DatasetSource", group_keys=False)[TARGET_COLUMN].transform(
        lambda s: pd.qcut(s, q=4, labels=["budget", "standard", "premium", "luxury"], duplicates="drop")
    )
    if "QualityScore" in labeled.columns:
        labeled["QualityLabel"] = pd.cut(
            pd.to_numeric(labeled["QualityScore"], errors="coerce"),
            bins=[-np.inf, 4, 6, 8, np.inf],
            labels=["low_quality", "average_quality", "high_quality", "excellent_quality"],
        )
    if "LivingArea" in labeled.columns:
        labeled["AreaSegment"] = labeled.groupby("DatasetSource", group_keys=False)["LivingArea"].transform(
            lambda s: pd.qcut(pd.to_numeric(s, errors="coerce"), q=4, labels=["small", "medium", "large", "very_large"], duplicates="drop")
        )
    return labeled


def build_edge_case_report(df: pd.DataFrame) -> dict[str, object]:
    report = {
        "prediction_return_shape": {
            "PredictedSalePrice": "numeric prediction in the selected DatasetSource price unit",
            "DatasetSource": "market/source context required for interpretation",
            "PriceUnit": "USD, lakh_INR, or dataset_original_unit depending on source",
        },
        "special_cases_checked": {
            "missing_numeric_values": "handled by median imputation in preprocessing",
            "missing_categorical_values": "handled by most-frequent imputation in preprocessing",
            "unknown_categories_at_inference": "handled by OneHotEncoder(handle_unknown='ignore')",
            "mixed_price_units": "DatasetSource and PriceUnit are explicit features/metadata; do not compare raw prices across markets without conversion",
            "invalid_labels": "removed when SalePrice is missing, nonnumeric, or <= 0",
            "impossible_area_or_rooms": "removed when living area <= 0 or bedrooms < 0",
        },
        "source_price_units": df.groupby("DatasetSource")["PriceUnit"].first().to_dict(),
        "rows_by_source": df["DatasetSource"].value_counts().to_dict(),
        "price_range_by_source": df.groupby("DatasetSource")[TARGET_COLUMN].agg(["min", "median", "max"]).to_dict("index"),
    }
    EDGE_CASE_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def save_dataset_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x="DatasetSource", order=df["DatasetSource"].value_counts().index)
    plt.xticks(rotation=25, ha="right")
    plt.title("Kaggle Raw Rows by Source After Filtering")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kaggle_rows_by_source.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x="PriceSegment", hue="DatasetSource")
    plt.title("Price Segment Labels by Kaggle Source")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_price_segment_counts.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="DatasetSource", y=TARGET_COLUMN)
    plt.yscale("log")
    plt.xticks(rotation=25, ha="right")
    plt.title("SalePrice Distribution by Kaggle Source (log scale)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "saleprice_by_kaggle_source.png", dpi=150)
    plt.close()

    if "QualityLabel" in df.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df, x="QualityLabel", y=TARGET_COLUMN)
        plt.yscale("log")
        plt.xticks(rotation=20, ha="right")
        plt.title("SalePrice by Quality Label")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "saleprice_by_quality_label.png", dpi=150)
        plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(len(df), 3000), random_state=42), x="LivingArea", y=TARGET_COLUMN, hue="DatasetSource", alpha=0.65)
    plt.yscale("log")
    plt.title("Living Area vs SalePrice by Kaggle Source")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "living_area_vs_saleprice_by_source.png", dpi=150)
    plt.close()


def prepare_real_dataset() -> pd.DataFrame:
    df, collection_rows = collect_and_canonicalize_kaggle_data()
    cleaned, noise_report = filter_noise_columns(df)
    cleaned, label_filter_report = filter_labels(cleaned)
    labeled = assign_data_labels(cleaned)

    PROCESSED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(PROCESSED_DATASET, index=False)
    labeled[[TARGET_COLUMN, "DatasetSource", "PriceUnit", "PriceSegment"]].to_csv(LABEL_REPORT, index=False)
    save_dataset_figures(labeled)
    edge_report = build_edge_case_report(labeled)

    DATA_QUALITY_REPORT.write_text(
        json.dumps(
            {
                "collection": collection_rows,
                **noise_report,
                **label_filter_report,
                "processed_path": str(PROCESSED_DATASET),
                "rows_final": int(labeled.shape[0]),
                "columns_final": int(labeled.shape[1]),
                "rows_by_source": labeled["DatasetSource"].value_counts().to_dict(),
                "label_columns_added": [col for col in ["PriceSegment", "QualityLabel", "AreaSegment"] if col in labeled],
                "edge_case_report_path": str(EDGE_CASE_REPORT),
                "prediction_return_shape": edge_report["prediction_return_shape"],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return labeled


if __name__ == "__main__":
    prepared = prepare_real_dataset()
    print(f"Prepared Kaggle-only dataset: {PROCESSED_DATASET.name}")
    print(f"Rows: {prepared.shape[0]}; Columns: {prepared.shape[1]}")
    print(prepared["DatasetSource"].value_counts().to_string())
