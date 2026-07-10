from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal_tugdt.config import load_config
from multimodal_tugdt.features.imu_features import extract_trial_and_phase_features
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.segmentation.manual import Segment


def _imu_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
project:
  root: "."
paths:
  manifest: participants.csv
imu:
  step_min_interval_s: 0.35
  step_prominence: 0.2
""",
        encoding="utf-8",
    )
    return load_config(path).imu


def test_trial_and_phase_features_use_straight_walk_annotations(tmp_path: Path) -> None:
    timestamps = np.arange(0.0, 20.0 + 0.01, 0.01)
    walking = ((timestamps >= 3) & (timestamps < 8)) | (
        (timestamps >= 10) & (timestamps < 15)
    )
    turning = (timestamps >= 8) & (timestamps < 10)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acc_ap": 0.2 * np.sin(2 * np.pi * 2 * timestamps) * walking,
            "acc_ml": 0.1 * np.cos(2 * np.pi * 2 * timestamps) * walking,
            "acc_vertical": np.sin(2 * np.pi * 2 * timestamps) * walking,
            "gyro_yaw": np.sin(np.pi * (timestamps - 8) / 2) * turning,
        }
    )
    segments = [
        Segment("baseline_sitting", 0, 2),
        Segment("sit_to_stand", 2, 3),
        Segment("outbound_walk", 3, 8),
        Segment("turn_1", 8, 10),
        Segment("return_walk", 10, 15),
        Segment("turn_to_sit", 15, 17),
        Segment("final_sitting", 17, 20),
    ]
    record = TrialRecord(
        participant_id="P001",
        session_id="S01",
        condition="dual_task",
        trial_id="T01",
        paths={},
    )

    rows = extract_trial_and_phase_features(record, frame, segments, _imu_config(tmp_path))
    features = pd.DataFrame(rows)
    trial = features.loc[features["feature_level"] == "trial"].iloc[0]
    outbound = features.loc[features["segment_name"] == "outbound_walk"].iloc[0]
    turn = features.loc[features["segment_name"] == "turn_1"].iloc[0]

    assert len(features) == 8
    assert trial["imu__cadence_steps_min"] == pytest.approx(120, rel=0.1)
    assert outbound["imu__step_count"] == pytest.approx(10, abs=1)
    assert np.isnan(turn["imu__step_count"])
    assert turn["imu__turn_duration_s"] == pytest.approx(2.0)
    assert turn["imu__peak_yaw_velocity_rad_s"] == pytest.approx(1.0, rel=0.01)

