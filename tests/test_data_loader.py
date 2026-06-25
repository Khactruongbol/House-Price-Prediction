from config import TARGET_COLUMN
from src.data_loader import load_dataset, split_features_target


def test_load_dataset_contains_target():
    df = load_dataset()
    assert TARGET_COLUMN in df.columns
    assert len(df) > 0


def test_split_features_target_removes_target_from_features():
    df = load_dataset()
    X, y = split_features_target(df)
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y)
