# House Price Prediction

Python project for predicting house prices with a complete machine learning workflow in one notebook, model tuning, report figures, and an HTML interface.

## Dataset Links

- Main dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Kaggle-only raw sources used by the pipeline:
  - [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)
  - [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
  - [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
  - [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

## How To Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

The main workflow collects raw data only from Kaggle using `kagglehub`. It does not use OpenML or sample data.

## Google Colab Training

Open `notebooks/99_house_price_workflow.ipynb` in Google Colab, run the Colab setup cell, install `kagglehub`, and run all cells. The same notebook saves:

- `models/house_price_pipeline.joblib`
- `reports/metrics/model_metrics.csv`
- `reports/metrics/model_tuning_report.csv`
- EDA and model figures under `reports/figures/`

## Python UI - Streamlit

The recommended local UI is Streamlit because it keeps the interface in Python, displays charts and metrics easily, and avoids most HTML/font issues:

```powershell
streamlit run app.py
```

Then open `http://localhost:8501`.

The Streamlit UI includes:

- `Predict`: input validation and price prediction
- `Dataset`: processed Kaggle dataset preview
- `Model Metrics`: model ranking and RMSE chart
- `Figures`: generated EDA/model figures
- `Edge Cases`: prediction return shape and special-case handling

## HTML Interface Fallback

The HTML fallback uses Flask:

```powershell
python app_html.py
```

Then open `http://127.0.0.1:5000`.

## Main Workflow

Open `notebooks/99_house_price_workflow.ipynb` and run cells from top to bottom. It covers dataset sources, Colab setup, data quality, EDA, preprocessing, feature engineering, model training, hyperparameter tuning, optimization checks, residual analysis, learning curve, artifact saving, inference demo, and HTML UI instructions.

## Real Data Pipeline

```powershell
python -m src.data_pipeline
python -m src.train
```

This collects Kaggle raw data, removes noisy rows/columns, filters invalid labels, adds analysis labels such as `PriceSegment`, creates figures, records edge cases, and trains the model from `data/processed/kaggle_house_prices_clean_labeled.csv`.

Current best model after adding Kaggle sources and extra models: `hist_gradient_boosting_tuned`.
