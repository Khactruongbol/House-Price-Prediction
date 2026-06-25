import pandas as pd
import streamlit as st

from config import KAGGLE_COMPETITION_URL, MODEL_PATH
from src.explain_prediction import load_model_ranking, simple_price_explanation
from src.predict import predict_price
from src.train import train_all


st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("House Price Prediction")

with st.sidebar:
    st.header("Model")
    st.caption(f"Dataset chính: {KAGGLE_COMPETITION_URL}")
    if st.button("Train model"):
        with st.spinner("Training models..."):
            result = train_all()
        st.success(f"Best model: {result['best_model_name']}")

default_values = {
    "DatasetSource": "king_county_house_sales",
    "PriceUnit": "USD",
    "LivingArea": 1710,
    "LotArea": 8450,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "QualityScore": 7,
    "ConditionScore": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "Neighborhood": "CollgCr",
    "HouseType": "1Fam",
    "SaleCondition": "Normal",
}

left, right = st.columns([1, 1])
with left:
    st.subheader("Input")
    input_row = {
        "DatasetSource": st.selectbox("DatasetSource", ["king_county_house_sales", "ames_housing", "bengaluru_house_prices", "simple_housing_prices"], index=0),
        "PriceUnit": st.text_input("PriceUnit", value=default_values["PriceUnit"]),
        "LivingArea": st.number_input("LivingArea", min_value=100, max_value=20000, value=default_values["LivingArea"], step=50),
        "LotArea": st.number_input("LotArea", min_value=500, max_value=100000, value=default_values["LotArea"], step=100),
        "Bedrooms": st.number_input("Bedrooms", min_value=0, max_value=20, value=default_values["Bedrooms"], step=1),
        "Bathrooms": st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=float(default_values["Bathrooms"]), step=0.5),
        "QualityScore": st.slider("QualityScore", 1, 13, default_values["QualityScore"]),
        "ConditionScore": st.slider("ConditionScore", 1, 10, default_values["ConditionScore"]),
        "YearBuilt": st.number_input("YearBuilt", min_value=1870, max_value=2026, value=default_values["YearBuilt"], step=1),
        "YearRemodAdd": st.number_input("YearRemodAdd", min_value=1870, max_value=2026, value=default_values["YearRemodAdd"], step=1),
        "Neighborhood": st.text_input("Neighborhood", value=default_values["Neighborhood"]),
        "HouseType": st.text_input("HouseType", value=default_values["HouseType"]),
        "SaleCondition": st.text_input("SaleCondition", value=default_values["SaleCondition"]),
    }

with right:
    st.subheader("Prediction")
    if not MODEL_PATH.exists():
        st.info("Chưa có model. Bấm Train model ở sidebar hoặc chạy `python -m src.train`.")
    if st.button("Predict SalePrice", type="primary"):
        if not MODEL_PATH.exists():
            with st.spinner("Training sample model first..."):
                train_all()
        prediction = float(predict_price(input_row).iloc[0])
        st.metric("Predicted SalePrice", f"${prediction:,.0f}")
        st.write(pd.DataFrame([input_row]))
        for note in simple_price_explanation(input_row):
            st.write(f"- {note}")

    st.subheader("Model Ranking")
    ranking = load_model_ranking()
    if ranking.empty:
        st.caption("Train model để xem bảng metric.")
    else:
        st.dataframe(ranking, use_container_width=True)
