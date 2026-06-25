from src.data_loader import load_dataset, split_features_target
from src.feature_engineering import add_house_features
from src.preprocessing import build_preprocessor, infer_feature_types


def test_preprocessor_transforms_rows():
    df = add_house_features(load_dataset())
    X, _ = split_features_target(df)
    numeric_features, categorical_features = infer_feature_types(X)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] >= len(numeric_features)
