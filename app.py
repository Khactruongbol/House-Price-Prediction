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
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotArea": 8450,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "GrLivArea": 1710,
    "FullBath": 2,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "GarageCars": 2,
    "GarageArea": 548,
    "Neighborhood": "CollgCr",
    "HouseStyle": "2Story",
    "SaleCondition": "Normal",
}

left, right = st.columns([1, 1])
with left:
    st.subheader("Input")
    input_row = {
        "MSSubClass": st.number_input("MSSubClass", min_value=20, max_value=190, value=default_values["MSSubClass"], step=5),
        "MSZoning": st.selectbox("MSZoning", ["RL", "RM", "FV", "RH", "C (all)"], index=0),
        "LotArea": st.number_input("LotArea", min_value=500, max_value=100000, value=default_values["LotArea"], step=100),
        "OverallQual": st.slider("OverallQual", 1, 10, default_values["OverallQual"]),
        "OverallCond": st.slider("OverallCond", 1, 10, default_values["OverallCond"]),
        "YearBuilt": st.number_input("YearBuilt", min_value=1870, max_value=2026, value=default_values["YearBuilt"], step=1),
        "YearRemodAdd": st.number_input("YearRemodAdd", min_value=1870, max_value=2026, value=default_values["YearRemodAdd"], step=1),
        "GrLivArea": st.number_input("GrLivArea", min_value=300, max_value=10000, value=default_values["GrLivArea"], step=50),
        "FullBath": st.number_input("FullBath", min_value=0, max_value=5, value=default_values["FullBath"], step=1),
        "BedroomAbvGr": st.number_input("BedroomAbvGr", min_value=0, max_value=10, value=default_values["BedroomAbvGr"], step=1),
        "KitchenAbvGr": st.number_input("KitchenAbvGr", min_value=0, max_value=4, value=default_values["KitchenAbvGr"], step=1),
        "GarageCars": st.number_input("GarageCars", min_value=0, max_value=5, value=default_values["GarageCars"], step=1),
        "GarageArea": st.number_input("GarageArea", min_value=0, max_value=2000, value=default_values["GarageArea"], step=20),
        "Neighborhood": st.text_input("Neighborhood", value=default_values["Neighborhood"]),
        "HouseStyle": st.text_input("HouseStyle", value=default_values["HouseStyle"]),
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
