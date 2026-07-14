import numpy as np
import pandas as pd

from multimodal_tugdt.config import FusionConfig, ModelingConfig
from multimodal_tugdt.modeling.baseline_models import build_baseline_pipeline
from multimodal_tugdt.modeling.evaluation import evaluate_baselines


def _frame(*, classification: bool = False) -> pd.DataFrame:
    rows = []
    for participant_index in range(10):
        participant = f"P{participant_index:03d}"
        target = participant_index % 2 if classification else 30 - participant_index
        for trial_index, condition in enumerate(("single_task", "dual_task"), start=1):
            rows.append(
                {
                    "participant_id": participant,
                    "session_id": "S01",
                    "trial_id": f"T{trial_index:02d}",
                    "condition": condition,
                    "imu__cadence": 80 + participant_index + trial_index,
                    "audio__speech_ratio": (
                        np.nan if participant_index in {2, 7} else 0.3 + participant_index / 100
                    ),
                    "availability__imu": 1,
                    "availability__audio": 0 if participant_index in {2, 7} else 1,
                    "clinical__target": target,
                }
            )
    return pd.DataFrame(rows)


def _fusion() -> FusionConfig:
    return FusionConfig(
        modalities=("imu", "audio"),
        feature_sets={"imu": ("imu",), "imu_audio": ("imu", "audio")},
    )


def test_imputation_statistics_are_fit_from_training_data_only() -> None:
    pipeline = build_baseline_pipeline("ridge", "regression", random_seed=42)
    x_train = pd.DataFrame({"feature": [1.0, 3.0, np.nan]})
    y_train = [1.0, 3.0, 2.0]

    pipeline.fit(x_train, y_train)

    assert pipeline.named_steps["imputer"].statistics_[0] == 2.0


def test_regression_baselines_use_grouped_splits_and_report_comparisons() -> None:
    config = ModelingConfig(
        enabled=True,
        target_column="clinical__target",
        task_type="regression",
        group_column="participant_id",
        folds=5,
        random_seed=42,
        models=("ridge",),
        cohort_modes=("all_samples", "complete_modalities"),
        positive_label=None,
    )

    result = evaluate_baselines(_frame(), _fusion(), config)

    assert result.successful_evaluations == 4
    assert result.split_audit["group_overlap_count"].eq(0).all()
    assert set(result.summary_metrics["metric"]) == {"mae", "rmse", "r2", "spearman"}
    assert set(result.summary_metrics["feature_set"]) == {"imu", "imu_audio"}
    assert set(result.summary_metrics["cohort"]) == {"all_samples", "complete_modalities"}
    prediction_counts = result.predictions.groupby(
        ["feature_set", "cohort", "model", "participant_id", "trial_id"]
    ).size()
    assert prediction_counts.eq(1).all()


def test_classification_baselines_report_binary_metrics_without_group_leakage() -> None:
    config = ModelingConfig(
        enabled=True,
        target_column="clinical__target",
        task_type="classification",
        group_column="participant_id",
        folds=5,
        random_seed=42,
        models=("logistic_regression",),
        cohort_modes=("all_samples",),
        positive_label=1,
    )

    result = evaluate_baselines(_frame(classification=True), _fusion(), config)

    assert result.successful_evaluations == 2
    assert result.split_audit["group_overlap_count"].eq(0).all()
    assert set(result.summary_metrics["metric"]) == {
        "balanced_accuracy",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "sensitivity",
        "specificity",
    }
    assert result.predictions["y_score"].between(0, 1).all()
