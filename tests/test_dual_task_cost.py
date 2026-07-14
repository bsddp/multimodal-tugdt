from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal_tugdt.config import load_config
from multimodal_tugdt.features.dual_task_cost import calculate_dual_task_costs


def _config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
paths:
  manifest: participants.csv
dual_task_cost:
  enabled: true
  group_columns: [participant_id, session_id]
  single_condition: single_task
  dual_condition: dual_task
  metrics:
    imu__cadence_steps_min: higher_is_better
    imu__turn_duration_s: higher_is_worse
    audio__response_accuracy: higher_is_better
""",
        encoding="utf-8",
    )
    return load_config(path).dual_task_cost


def test_dual_task_cost_preserves_values_and_normalizes_deterioration(tmp_path: Path) -> None:
    fused = pd.DataFrame(
        {
            "participant_id": ["P001", "P001"],
            "session_id": ["S01", "S01"],
            "trial_id": ["T01", "T02"],
            "condition": ["single_task", "dual_task"],
            "imu__cadence_steps_min": [100.0, 80.0],
            "imu__turn_duration_s": [2.0, 2.5],
            "audio__response_accuracy": [0.0, 0.0],
        }
    )

    result = calculate_dual_task_costs(fused, _config(tmp_path))
    row = result.frame.iloc[0]

    assert result.paired_group_count == 1
    assert result.skipped_group_count == 0
    assert row["single__imu_cadence_steps_min"] == 100.0
    assert row["dual__imu_cadence_steps_min"] == 80.0
    assert row["dtc__imu_cadence_steps_min_pct"] == pytest.approx(20.0)
    assert row["dtc__imu_turn_duration_s_pct"] == pytest.approx(25.0)
    assert np.isnan(row["dtc__audio_response_accuracy_pct"])
    assert row["dtc_valid_metric_count"] == 2


def test_dual_task_cost_skips_incomplete_pairs_and_rejects_duplicates(tmp_path: Path) -> None:
    base = pd.DataFrame(
        {
            "participant_id": ["P001", "P001", "P002"],
            "session_id": ["S01", "S01", "S01"],
            "trial_id": ["T01", "T02", "T01"],
            "condition": ["single_task", "dual_task", "single_task"],
            "imu__cadence_steps_min": [100.0, 90.0, 80.0],
            "imu__turn_duration_s": [2.0, 2.2, 2.5],
            "audio__response_accuracy": [0.9, 0.8, 0.7],
        }
    )

    result = calculate_dual_task_costs(base, _config(tmp_path))
    assert result.paired_group_count == 1
    assert result.skipped_group_count == 1

    duplicated = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one trial per group"):
        calculate_dual_task_costs(duplicated, _config(tmp_path))
