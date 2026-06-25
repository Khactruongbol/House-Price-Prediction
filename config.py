from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
LABELS_DIR = REPORTS_DIR / "labels"

TARGET_COLUMN = "SalePrice"
DERIVED_LABEL_COLUMNS = ["PriceSegment", "QualityLabel", "AreaSegment"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

KAGGLE_COMPETITION_URL = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques"
DEFAULT_RAW_DATASET = RAW_DATA_DIR / "train.csv"
OPENML_RAW_DATASET = RAW_DATA_DIR / "ames_house_prices_openml.csv"
PROCESSED_DATASET = PROCESSED_DATA_DIR / "house_prices_clean_labeled.csv"
DATA_COLLECTION_REPORT = METRICS_DIR / "data_collection_report.json"
DATA_QUALITY_REPORT = METRICS_DIR / "data_quality_report.json"
LABEL_REPORT = LABELS_DIR / "price_segment_labels.csv"
MODEL_PATH = MODELS_DIR / "house_price_pipeline.joblib"
METRICS_PATH = METRICS_DIR / "model_metrics.csv"
