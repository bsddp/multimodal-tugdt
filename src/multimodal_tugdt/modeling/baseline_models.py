"""Interpretable baseline estimators with fold-local preprocessing."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_pipeline(
    model_name: str,
    task_type: str,
    *,
    random_seed: int,
) -> Pipeline:
    """Create an sklearn Pipeline so imputation/scaling fit only on training folds."""
    if task_type == "classification":
        if model_name == "logistic_regression":
            estimator = LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=random_seed,
            )
            steps = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        elif model_name == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=1,
            )
            steps = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", estimator),
            ]
        else:
            raise ValueError(f"Unsupported classification baseline: {model_name}")
    elif task_type == "regression":
        if model_name == "linear_regression":
            estimator = LinearRegression()
        elif model_name == "ridge":
            estimator = Ridge(alpha=1.0)
        elif model_name == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=200,
                random_state=random_seed,
                n_jobs=1,
            )
        else:
            raise ValueError(f"Unsupported regression baseline: {model_name}")
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if model_name in {"linear_regression", "ridge"}:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    return Pipeline(steps)
