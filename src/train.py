from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, learning_curve, train_test_split
from sklearn.pipeline import Pipeline

from config import DEFAULT_RAW_DATASET, FIGURES_DIR, METRICS_PATH, MODEL_PATH, PROCESSED_DATASET, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.data_loader import load_dataset, split_features_target
from src.data_pipeline import prepare_real_dataset
from src.evaluate import metrics_to_frame, regression_metrics
from src.feature_engineering import add_house_features
from src.preprocessing import build_preprocessor, infer_feature_types


MAX_TUNING_ROWS = 5000
MAX_SLOW_MODEL_ROWS = 8000
MAX_LEARNING_CURVE_ROWS = 1200


def build_models(preprocessor) -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())]),
        "ridge": Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE))]),
        "lasso": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", Lasso(alpha=100.0, max_iter=100000, tol=1e-3, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=50,
                        random_state=RANDOM_STATE,
                        min_samples_leaf=1,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(n_estimators=80, random_state=RANDOM_STATE)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=60,
                        random_state=RANDOM_STATE,
                        min_samples_leaf=1,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }


def build_tuning_grids(preprocessor) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "gradient_boosting_tuned": (
            Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
            {
                "model__n_estimators": [120],
                "model__learning_rate": [0.08],
                "model__max_depth": [3],
            },
        ),
        "hist_gradient_boosting_tuned": (
            Pipeline([("preprocessor", preprocessor), ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))]),
            {
                "model__learning_rate": [0.08],
                "model__max_iter": [120],
                "model__max_leaf_nodes": [31],
            },
        ),
    }


def tune_models(preprocessor, X_train, y_train, cv_splits: int) -> dict[str, object]:
    tuned_models: dict[str, Pipeline] = {}
    tuning_rows: list[dict[str, object]] = []
    if cv_splits < 2:
        return {"models": tuned_models, "report": pd.DataFrame(tuning_rows)}

    if len(X_train) > MAX_TUNING_ROWS:
        X_tune = X_train.sample(MAX_TUNING_ROWS, random_state=RANDOM_STATE)
        y_tune = y_train.loc[X_tune.index]
    else:
        X_tune = X_train
        y_tune = y_train

    cv = KFold(n_splits=min(3, cv_splits), shuffle=True, random_state=RANDOM_STATE)
    for name, (pipeline, param_grid) in build_tuning_grids(preprocessor).items():
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_tune, y_tune)
        best_estimator = search.best_estimator_
        best_estimator.fit(X_train, y_train)
        tuned_models[name] = best_estimator
        tuning_rows.append(
            {
                "model": name,
                "best_cv_rmse": float(-search.best_score_),
                "best_params": search.best_params_,
                "tuning_rows": int(len(X_tune)),
            }
        )

    return {"models": tuned_models, "report": pd.DataFrame(tuning_rows)}


def _fit_data_for_model(name: str, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    slow_models = {"random_forest", "gradient_boosting", "extra_trees"}
    if name in slow_models and len(X_train) > MAX_SLOW_MODEL_ROWS:
        sampled_X = X_train.sample(MAX_SLOW_MODEL_ROWS, random_state=RANDOM_STATE)
        return sampled_X, y_train.loc[sampled_X.index]
    return X_train, y_train


def save_training_figures(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    y_test=None,
    best_predictions=None,
    best_model: Pipeline | None = None,
    X_train=None,
    y_train=None,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=min(30, max(5, len(df) // 2)))
    plt.title("SalePrice Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "saleprice_distribution.png", dpi=150)
    plt.close()

    if "GrLivArea" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="GrLivArea", y=TARGET_COLUMN)
        plt.title("GrLivArea vs SalePrice")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "grlivarea_vs_saleprice.png", dpi=150)
        plt.close()

    numeric_df = df.select_dtypes(include="number")
    if TARGET_COLUMN in numeric_df.columns and numeric_df.shape[1] > 1:
        corr = numeric_df.corr(numeric_only=True)[[TARGET_COLUMN]].sort_values(TARGET_COLUMN, ascending=False).head(12)
        plt.figure(figsize=(6, 7))
        sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Top Correlations With SalePrice")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "saleprice_correlation_heatmap.png", dpi=150)
        plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="model", y="RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.title("Model RMSE Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_rmse_comparison.png", dpi=150)
    plt.close()

    if y_test is not None and best_predictions is not None:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=best_predictions)
        min_value = min(float(min(y_test)), float(min(best_predictions)))
        max_value = max(float(max(y_test)), float(max(best_predictions)))
        plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
        plt.xlabel("Actual SalePrice")
        plt.ylabel("Predicted SalePrice")
        plt.title("Actual vs Predicted SalePrice")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "actual_vs_predicted_saleprice.png", dpi=150)
        plt.close()

        residuals = pd.Series(y_test).to_numpy() - np.asarray(best_predictions)
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, kde=True)
        plt.title("Residual Distribution")
        plt.xlabel("Actual - Predicted")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "residual_distribution.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=best_predictions, y=residuals)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted SalePrice")
        plt.ylabel("Residual")
        plt.title("Residuals vs Predicted")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "residuals_vs_predicted.png", dpi=150)
        plt.close()

    if best_model is not None and X_train is not None and y_train is not None and len(X_train) >= 8:
        if len(X_train) > MAX_LEARNING_CURVE_ROWS:
            X_curve = X_train.sample(MAX_LEARNING_CURVE_ROWS, random_state=RANDOM_STATE)
            y_curve = y_train.loc[X_curve.index]
        else:
            X_curve = X_train
            y_curve = y_train
        cv_splits = min(3, len(X_train))
        train_sizes, train_scores, validation_scores = learning_curve(
            best_model,
            X_curve,
            y_curve,
            cv=cv_splits,
            scoring="neg_root_mean_squared_error",
            train_sizes=np.linspace(0.35, 1.0, 4),
            n_jobs=-1,
        )
        train_rmse = -train_scores.mean(axis=1)
        validation_rmse = -validation_scores.mean(axis=1)
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_rmse, marker="o", label="Train RMSE")
        plt.plot(train_sizes, validation_rmse, marker="o", label="Validation RMSE")
        plt.title("Learning Curve - Best Model")
        plt.xlabel("Training examples")
        plt.ylabel("RMSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "learning_curve_best_model.png", dpi=150)
        plt.close()


def train_all(dataset_path: str | Path | None = None, save_artifacts: bool = True) -> dict[str, object]:
    if dataset_path is None and not PROCESSED_DATASET.exists() and not DEFAULT_RAW_DATASET.exists():
        prepare_real_dataset()
    df = add_house_features(load_dataset(dataset_path))
    X, y = split_features_target(df)
    numeric_features, categorical_features = infer_feature_types(X)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    split_kwargs = {"test_size": TEST_SIZE, "random_state": RANDOM_STATE}
    if len(df) < 10:
        split_kwargs["test_size"] = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}
    model_fit_rows: dict[str, int] = {}
    for name, model in models.items():
        fit_X, fit_y = _fit_data_for_model(name, X_train, y_train)
        model_fit_rows[name] = int(len(fit_X))
        model.fit(fit_X, fit_y)
        predictions = model.predict(X_test)
        metrics[name] = regression_metrics(y_test, predictions)
        fitted_models[name] = model

    cv_splits = min(5, len(X_train))
    tuning_result = tune_models(preprocessor, X_train, y_train, cv_splits=cv_splits)
    tuning_report = tuning_result["report"]
    for name, model in tuning_result["models"].items():
        model_fit_rows[name] = int(len(X_train))
        predictions = model.predict(X_test)
        metrics[name] = regression_metrics(y_test, predictions)
        fitted_models[name] = model

    metrics_df = metrics_to_frame(metrics)
    best_model_name = str(metrics_df.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_predictions = best_model.predict(X_test)
    baseline_best_rmse = float(metrics_to_frame({k: v for k, v in metrics.items() if not k.endswith("_tuned")}).iloc[0]["RMSE"])
    best_rmse = float(metrics_df.iloc[0]["RMSE"])
    optimization_status = "optimized" if best_rmse <= baseline_best_rmse else "baseline_preferred"

    if save_artifacts:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": best_model_name,
                "pipeline": best_model,
                "feature_columns": X.columns.tolist(),
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "metrics": metrics_df.to_dict(orient="records"),
                "model_fit_rows": model_fit_rows,
            },
            MODEL_PATH,
        )
        metrics_df.to_csv(METRICS_PATH, index=False)
        if not tuning_report.empty:
            tuning_report.to_csv(METRICS_PATH.parent / "model_tuning_report.csv", index=False)
        (METRICS_PATH.parent / "training_summary.json").write_text(
            json.dumps(
                {
                    "best_model": best_model_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "baseline_best_rmse": baseline_best_rmse,
                    "best_rmse_after_tuning": best_rmse,
                    "optimization_status": optimization_status,
                    "model_fit_rows": model_fit_rows,
                    "note": "All rows come from filtered Kaggle data. Slow comparison models can use representative filtered-data samples for local runtime; tuned final candidates use the full training split.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        save_training_figures(df, metrics_df, y_test, best_predictions, best_model, X_train, y_train)

    return {
        "data": df,
        "metrics": metrics_df,
        "tuning_report": tuning_report,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "model_path": MODEL_PATH,
        "optimization_status": optimization_status,
    }


if __name__ == "__main__":
    result = train_all()
    print(result["metrics"].to_string(index=False))
    if not result["tuning_report"].empty:
        print(result["tuning_report"].to_string(index=False))
    print(f"Best model: {result['best_model_name']}")
    print(f"Optimization status: {result['optimization_status']}")
    print(f"Saved model: {Path(result['model_path']).name}")
