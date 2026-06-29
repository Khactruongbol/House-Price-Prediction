from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import EDGE_CASE_REPORT


DEFAULT_SOURCE_UNITS = {
    "ames_housing": "USD",
    "bengaluru_house_prices": "lakh_INR",
    "king_county_house_sales": "USD",
    "simple_housing_prices": "dataset_original_unit",
}


def load_source_units(edge_case_report: Path = EDGE_CASE_REPORT) -> dict[str, str]:
    if not edge_case_report.exists():
        return DEFAULT_SOURCE_UNITS.copy()
    try:
        report = json.loads(edge_case_report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_SOURCE_UNITS.copy()
    source_units = report.get("source_price_units", {})
    return source_units or DEFAULT_SOURCE_UNITS.copy()


def validate_prediction_input(input_row: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    source_units = load_source_units()

    dataset_source = str(input_row.get("DatasetSource", "")).strip()
    price_unit = str(input_row.get("PriceUnit", "")).strip()
    if dataset_source not in source_units:
        errors.append(f"DatasetSource '{dataset_source}' chưa có trong dữ liệu train.")
    elif price_unit and price_unit != source_units[dataset_source]:
        warnings.append(f"PriceUnit nên là '{source_units[dataset_source]}' cho nguồn {dataset_source}.")

    numeric_rules = {
        "LivingArea": (0, None, "LivingArea phải lớn hơn 0."),
        "LotArea": (0, None, "LotArea phải lớn hơn 0."),
        "Bedrooms": (0, None, "Bedrooms không được âm."),
        "Bathrooms": (0, None, "Bathrooms không được âm."),
        "QualityScore": (0, None, "QualityScore phải lớn hơn 0."),
        "ConditionScore": (0, None, "ConditionScore phải lớn hơn 0."),
    }
    for field, (minimum, maximum, message) in numeric_rules.items():
        value = input_row.get(field)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{field} phải là số.")
            continue
        if minimum is not None and numeric_value < minimum:
            errors.append(message)
        if maximum is not None and numeric_value > maximum:
            errors.append(message)

    year_built = _as_number(input_row.get("YearBuilt"))
    year_remod = _as_number(input_row.get("YearRemodAdd"))
    for field, year in {"YearBuilt": year_built, "YearRemodAdd": year_remod}.items():
        if year is None:
            errors.append(f"{field} phải là năm hợp lệ.")
        elif year < 1800 or year > 2035:
            warnings.append(f"{field} nằm ngoài khoảng thông thường 1800-2035.")
    if year_built is not None and year_remod is not None and year_remod < year_built:
        warnings.append("YearRemodAdd nhỏ hơn YearBuilt; model vẫn dự đoán nhưng dữ liệu cần được kiểm tra lại.")

    for field in ["Neighborhood", "HouseType", "SaleCondition"]:
        if not str(input_row.get(field, "")).strip():
            warnings.append(f"{field} đang trống; pipeline sẽ xử lý nhưng dự đoán có thể kém ổn định.")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "source_units": source_units,
        "return_shape": {
            "PredictedSalePrice": "numeric",
            "DatasetSource": dataset_source,
            "PriceUnit": source_units.get(dataset_source, price_unit),
        },
    }


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
