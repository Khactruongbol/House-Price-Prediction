# House Price Prediction

Python project for predicting house prices with a complete machine learning workflow in one notebook, model tuning, report figures, and an HTML interface.

## Dataset Links

- Main dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Alternative datasets:
  - [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
  - [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)
  - [Real Estate Price Prediction](https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction)
  - [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)

## How To Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
python app_html.py
```

Put Kaggle `train.csv` in `data/raw/train.csv`. If Kaggle credentials/data are not available, run the real OpenML Ames Housing collection pipeline. The main training workflow does not use sample data.

## Google Colab Training

Open `notebooks/99_house_price_workflow.ipynb` in Google Colab, run the Colab setup cell, upload `kaggle.json`, download the Kaggle competition data, and run all cells. If Kaggle is unavailable, the notebook collects real OpenML house price data. The same notebook saves:

- `models/house_price_pipeline.joblib`
- `reports/metrics/model_metrics.csv`
- `reports/metrics/model_tuning_report.csv`
- EDA and model figures under `reports/figures/`

## HTML Interface

The HTML UI uses Flask:

```powershell
python app_html.py
```

Then open `http://127.0.0.1:5000`.

The older Streamlit app is still available:

```powershell
streamlit run app.py
```

## Main Workflow

Open `notebooks/99_house_price_workflow.ipynb` and run cells from top to bottom. It covers dataset sources, Colab setup, data quality, EDA, preprocessing, feature engineering, model training, hyperparameter tuning, optimization checks, residual analysis, learning curve, artifact saving, inference demo, and HTML UI instructions.

## Real Data Pipeline

```powershell
python -m src.data_pipeline
python -m src.train
```

This collects real data, removes noisy rows/columns, filters invalid labels, adds analysis labels such as `PriceSegment`, creates figures, and trains the model from `data/processed/house_prices_clean_labeled.csv`.
