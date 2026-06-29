from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import EDGE_CASE_REPORT, FIGURES_DIR, LABEL_REPORT, METRICS_PATH, MODEL_PATH, PROCESSED_DATASET
from src.explain_prediction import load_model_ranking, simple_price_explanation
from src.input_validation import load_source_units, validate_prediction_input
from src.predict import predict_price
from src.train import train_all


st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("House Price Prediction")
st.caption("Python UI bằng Streamlit là giao diện chính; Flask HTML vẫn được giữ làm fallback.")


@st.cache_data(show_spinner=False)
def load_processed_preview() -> pd.DataFrame:
    if not PROCESSED_DATASET.exists():
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_DATASET, low_memory=False)


@st.cache_data(show_spinner=False)
def load_json_report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


DEFAULT_INPUT = {
    "DatasetSource": "king_county_house_sales",
    "PriceUnit": "USD",
    "LivingArea": 1710.0,
    "LotArea": 8450.0,
    "Bedrooms": 3.0,
    "Bathrooms": 2.0,
    "QualityScore": 7.0,
    "ConditionScore": 5.0,
    "YearBuilt": 2003.0,
    "YearRemodAdd": 2003.0,
    "Neighborhood": "CollgCr",
    "HouseType": "1Fam",
    "SaleCondition": "Normal",
}


def prediction_form() -> dict[str, object]:
    source_units = load_source_units()
    sources = list(source_units.keys())
    dataset_source = st.selectbox("DatasetSource", sources, index=sources.index(DEFAULT_INPUT["DatasetSource"]))
    expected_unit = source_units.get(dataset_source, DEFAULT_INPUT["PriceUnit"])
    price_unit = st.text_input("PriceUnit", value=expected_unit)

    left, right = st.columns(2)
    with left:
        living_area = st.number_input("LivingArea", min_value=1.0, value=DEFAULT_INPUT["LivingArea"], step=50.0)
        lot_area = st.number_input("LotArea", min_value=1.0, value=DEFAULT_INPUT["LotArea"], step=100.0)
        bedrooms = st.number_input("Bedrooms", min_value=0.0, value=DEFAULT_INPUT["Bedrooms"], step=1.0)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, value=DEFAULT_INPUT["Bathrooms"], step=0.5)
    with right:
        quality = st.number_input("QualityScore", min_value=1.0, value=DEFAULT_INPUT["QualityScore"], step=1.0)
        condition = st.number_input("ConditionScore", min_value=1.0, value=DEFAULT_INPUT["ConditionScore"], step=1.0)
        year_built = st.number_input("YearBuilt", min_value=1800.0, max_value=2035.0, value=DEFAULT_INPUT["YearBuilt"], step=1.0)
        year_remod = st.number_input("YearRemodAdd", min_value=1800.0, max_value=2035.0, value=DEFAULT_INPUT["YearRemodAdd"], step=1.0)

    neighborhood = st.text_input("Neighborhood", value=DEFAULT_INPUT["Neighborhood"])
    house_type = st.text_input("HouseType", value=DEFAULT_INPUT["HouseType"])
    sale_condition = st.text_input("SaleCondition", value=DEFAULT_INPUT["SaleCondition"])

    return {
        "DatasetSource": dataset_source,
        "PriceUnit": price_unit,
        "LivingArea": living_area,
        "LotArea": lot_area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "QualityScore": quality,
        "ConditionScore": condition,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "Neighborhood": neighborhood,
        "HouseType": house_type,
        "SaleCondition": sale_condition,
    }


with st.sidebar:
    st.header("Model")
    st.write("Model artifact:", MODEL_PATH.name if MODEL_PATH.exists() else "chưa có")
    if st.button("Train / refresh model"):
        with st.spinner("Training models from Kaggle processed data..."):
            result = train_all()
        st.success(f"Best model: {result['best_model_name']}")

predict_tab, dataset_tab, metrics_tab, figures_tab, edge_tab = st.tabs(
    ["Predict", "Dataset", "Model Metrics", "Figures", "Edge Cases"]
)

with predict_tab:
    st.subheader("Dự đoán giá nhà")
    input_row = prediction_form()
    validation = validate_prediction_input(input_row)
    for warning in validation["warnings"]:
        st.warning(warning)
    for error in validation["errors"]:
        st.error(error)

    if st.button("Predict SalePrice", type="primary"):
        if not validation["is_valid"]:
            st.error("Vui lòng sửa lỗi dữ liệu đầu vào trước khi dự đoán.")
        else:
            if not MODEL_PATH.exists():
                with st.spinner("Model chưa có, đang train..."):
                    train_all()
            prediction = float(predict_price(input_row).iloc[0])
            unit = validation["return_shape"]["PriceUnit"]
            st.metric("Predicted SalePrice", f"{prediction:,.2f} {unit}")
            st.dataframe(pd.DataFrame([input_row]), use_container_width=True)
            for note in simple_price_explanation(input_row):
                st.write(f"- {note}")

with dataset_tab:
    st.subheader("Kaggle-only processed dataset")
    df = load_processed_preview()
    if df.empty:
        st.info("Chưa có processed dataset. Chạy `python -m src.data_pipeline` trước.")
    else:
        st.write(f"Rows: {len(df):,}; Columns: {len(df.columns):,}")
        if "DatasetSource" in df.columns:
            st.bar_chart(df["DatasetSource"].value_counts())
        st.dataframe(df.head(100), use_container_width=True)
    if LABEL_REPORT.exists():
        st.download_button(
            "Download price segment labels",
            LABEL_REPORT.read_bytes(),
            file_name=LABEL_REPORT.name,
            mime="text/csv",
        )

with metrics_tab:
    st.subheader("Model metrics")
    ranking = load_model_ranking()
    if ranking.empty:
        st.info("Chưa có metrics. Chạy `python -m src.train` trước.")
    else:
        st.dataframe(ranking, use_container_width=True)
        st.bar_chart(ranking.set_index("model")["RMSE"])

with figures_tab:
    st.subheader("Generated figures")
    figure_files = sorted(FIGURES_DIR.glob("*.png"))
    if not figure_files:
        st.info("Chưa có figure trong reports/figures.")
    for figure_path in figure_files:
        st.markdown(f"**{figure_path.name}**")
        st.image(str(figure_path), use_container_width=True)

with edge_tab:
    st.subheader("Edge cases và khả năng trả về")
    edge_report = load_json_report(EDGE_CASE_REPORT)
    if not edge_report:
        st.info("Chưa có edge case report.")
    else:
        st.json(edge_report)
    st.markdown(
        """
        - Missing numeric values được xử lý bằng median imputation.
        - Missing categorical values được xử lý bằng most-frequent imputation.
        - Unknown categories được xử lý bằng `OneHotEncoder(handle_unknown='ignore')`.
        - Kết quả dự đoán cần đọc cùng `DatasetSource` và `PriceUnit`.
        """
    )
