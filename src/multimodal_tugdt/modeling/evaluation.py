"""Leakage-aware evaluation of single- and multimodal baseline feature sets."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from multimodal_tugdt.config import FusionConfig, ModelingConfig
from multimodal_tugdt.fusion.feature_level import TRIAL_KEYS
from multimodal_tugdt.modeling.baseline_models import build_baseline_pipeline


@dataclass(frozen=True)
class BaselineEvaluation:
    """All tabular artifacts produced by grouped cross-validation."""

    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    predictions: pd.DataFrame
    split_audit: pd.DataFrame
    skipped: pd.DataFrame
    successful_evaluations: int


def _target_column(config: ModelingConfig) -> str:
    return (
        config.target_column
        if "__" in config.target_column
        else f"clinical__{config.target_column}"
    )


def _selected_feature_columns(
    frame: pd.DataFrame,
    modalities: tuple[str, ...],
    target: str,
) -> tuple[list[str], list[str]]:
    raw = [
        column
        for column in frame.columns
        if any(column.startswith(f"{modality}__") for modality in modalities)
        and column != target
        and is_numeric_dtype(frame[column])
    ]
    availability = [
        f"availability__{modality}"
        for modality in modalities
        if f"availability__{modality}" in frame
    ]
    return raw + availability, raw


def _cohort(
    frame: pd.DataFrame,
    modalities: tuple[str, ...],
    target: str,
    mode: str,
) -> pd.DataFrame:
    selected = frame.loc[frame[target].notna()].copy()
    if mode == "complete_modalities":
        for modality in modalities:
            availability = f"availability__{modality}"
            if availability not in selected:
                return selected.iloc[0:0]
            selected = selected.loc[selected[availability] == 1]
    return selected.reset_index(drop=True)


def _splitter(
    data: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: ModelingConfig,
):
    group_count = int(groups.nunique())
    if config.task_type == "classification":
        classes = pd.unique(y)
        if len(classes) != 2:
            raise ValueError(
                "Classification baselines currently require exactly two target classes."
            )
        class_group_counts = data.groupby(y.name, dropna=False)[config.group_column].nunique()
        folds = min(config.folds, group_count, int(class_group_counts.min()))
        if folds < 2:
            raise ValueError(
                "Classification requires at least two participant groups containing each class."
            )
        return StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=config.random_seed,
        )
    folds = min(config.folds, group_count)
    if folds < 2:
        raise ValueError("Regression requires at least two participant groups.")
    return GroupKFold(n_splits=folds)


def _classification_context(y: pd.Series, configured_positive: object | None):
    classes = sorted(pd.unique(y), key=lambda value: str(value))
    if len(classes) != 2:
        raise ValueError("Classification baselines currently require exactly two target classes.")
    positive = classes[-1] if configured_positive is None else configured_positive
    if positive not in classes:
        raise ValueError(f"Configured positive_label {positive!r} is not present in the target.")
    negative = next(value for value in classes if value != positive)
    return positive, negative


def _classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    positive: object,
    negative: object,
) -> dict[str, float]:
    binary_true = np.asarray(y_true == positive, dtype=int)
    roc_auc = float(roc_auc_score(binary_true, y_score)) if len(set(binary_true)) == 2 else math.nan
    sensitivity = float(recall_score(y_true, y_pred, pos_label=positive, zero_division=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "precision": float(precision_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        "recall": sensitivity,
        "f1": float(f1_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": float(recall_score(y_true, y_pred, pos_label=negative, zero_division=0)),
    }


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) > 1:
        r_squared = float(r2_score(y_true, y_pred))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            correlation = float(spearmanr(y_true, y_pred).statistic)
    else:
        r_squared = math.nan
        correlation = math.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r_squared,
        "spearman": correlation,
    }


def _safe_summary(values: pd.Series) -> tuple[float, float, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan, math.nan, 0
    return float(numeric.mean()), float(numeric.std(ddof=1)), len(numeric)


def _summary_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "task_type",
        "target_column",
        "feature_set",
        "modalities",
        "cohort",
        "model",
        "n_samples",
        "n_groups",
        "n_features",
        "n_folds",
    ]
    metric_columns = [column for column in fold_metrics if column not in {*identity, "fold"}]
    rows: list[dict[str, object]] = []
    for keys, group in fold_metrics.groupby(identity, dropna=False, sort=False):
        base = dict(zip(identity, keys, strict=True))
        for metric in metric_columns:
            mean, standard_deviation, valid_folds = _safe_summary(group[metric])
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                    "valid_fold_count": valid_folds,
                }
            )
    return pd.DataFrame(rows)


def evaluate_baselines(
    frame: pd.DataFrame,
    fusion: FusionConfig,
    config: ModelingConfig,
) -> BaselineEvaluation:
    """Evaluate configured feature sets without row-level or preprocessing leakage."""
    target = _target_column(config)
    if target not in frame:
        raise ValueError(f"Configured modeling target is absent from fused features: {target}")
    if config.group_column not in frame:
        raise ValueError(f"Grouping column is absent from fused features: {config.group_column}")

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    successful = 0
    for feature_set, modalities in fusion.feature_sets.items():
        feature_columns, raw_feature_columns = _selected_feature_columns(frame, modalities, target)
        for cohort_mode in config.cohort_modes:
            data = _cohort(frame, modalities, target, cohort_mode)
            context = {
                "task_type": config.task_type,
                "target_column": target,
                "feature_set": feature_set,
                "modalities": "+".join(modalities),
                "cohort": cohort_mode,
            }
            if data.empty:
                skipped_rows.append({**context, "model": "all", "reason": "cohort is empty"})
                continue
            if not feature_columns or not raw_feature_columns:
                skipped_rows.append(
                    {**context, "model": "all", "reason": "no numeric feature columns"}
                )
                continue
            if not data[raw_feature_columns].notna().any(axis=None):
                skipped_rows.append(
                    {**context, "model": "all", "reason": "all selected features are missing"}
                )
                continue
            y = data[target]
            groups = data[config.group_column]
            try:
                splitter = _splitter(data, y, groups, config)
                splits = list(splitter.split(data[feature_columns], y, groups))
                classification_context = (
                    _classification_context(y, config.positive_label)
                    if config.task_type == "classification"
                    else None
                )
            except ValueError as exc:
                skipped_rows.append({**context, "model": "all", "reason": str(exc)})
                continue

            for fold, (train_index, test_index) in enumerate(splits, start=1):
                train_groups = sorted(set(groups.iloc[train_index].astype(str)))
                test_groups = sorted(set(groups.iloc[test_index].astype(str)))
                overlap = sorted(set(train_groups) & set(test_groups))
                audit_rows.append(
                    {
                        **context,
                        "fold": fold,
                        "train_group_count": len(train_groups),
                        "test_group_count": len(test_groups),
                        "group_overlap_count": len(overlap),
                        "train_groups": json.dumps(train_groups),
                        "test_groups": json.dumps(test_groups),
                    }
                )
                if overlap:
                    raise RuntimeError("Participant leakage detected in grouped split.")

            for model_name in config.models:
                model_failed = False
                for fold, (train_index, test_index) in enumerate(splits, start=1):
                    pipeline = build_baseline_pipeline(
                        model_name,
                        config.task_type,
                        random_seed=config.random_seed,
                    )
                    x_train = data.iloc[train_index][feature_columns]
                    x_test = data.iloc[test_index][feature_columns]
                    y_train = y.iloc[train_index]
                    y_test = y.iloc[test_index]
                    try:
                        pipeline.fit(x_train, y_train)
                        prediction = pipeline.predict(x_test)
                        score = np.full(len(test_index), np.nan)
                        if config.task_type == "classification":
                            assert classification_context is not None
                            positive, negative = classification_context
                            classes = list(pipeline.named_steps["model"].classes_)
                            positive_index = classes.index(positive)
                            score = pipeline.predict_proba(x_test)[:, positive_index]
                            metrics = _classification_metrics(
                                y_test,
                                prediction,
                                score,
                                positive=positive,
                                negative=negative,
                            )
                        else:
                            metrics = _regression_metrics(y_test, prediction)
                    except ValueError as exc:
                        skipped_rows.append(
                            {**context, "model": model_name, "reason": f"fold {fold}: {exc}"}
                        )
                        model_failed = True
                        break

                    fold_rows.append(
                        {
                            **context,
                            "model": model_name,
                            "fold": fold,
                            "n_samples": len(data),
                            "n_groups": int(groups.nunique()),
                            "n_features": len(feature_columns),
                            "n_folds": len(splits),
                            **metrics,
                        }
                    )
                    for position, row_index in enumerate(test_index):
                        row = data.iloc[row_index]
                        prediction_rows.append(
                            {
                                **{key: row[key] for key in TRIAL_KEYS},
                                **context,
                                "model": model_name,
                                "fold": fold,
                                "y_true": y_test.iloc[position],
                                "y_pred": prediction[position],
                                "y_score": score[position],
                            }
                        )
                if model_failed:
                    fold_rows = [
                        row
                        for row in fold_rows
                        if not (
                            row["feature_set"] == feature_set
                            and row["cohort"] == cohort_mode
                            and row["model"] == model_name
                        )
                    ]
                    prediction_rows = [
                        row
                        for row in prediction_rows
                        if not (
                            row["feature_set"] == feature_set
                            and row["cohort"] == cohort_mode
                            and row["model"] == model_name
                        )
                    ]
                else:
                    successful += 1

    fold_metrics = pd.DataFrame(fold_rows)
    return BaselineEvaluation(
        fold_metrics=fold_metrics,
        summary_metrics=_summary_table(fold_metrics) if not fold_metrics.empty else pd.DataFrame(),
        predictions=pd.DataFrame(prediction_rows),
        split_audit=pd.DataFrame(audit_rows),
        skipped=pd.DataFrame(skipped_rows),
        successful_evaluations=successful,
    )
