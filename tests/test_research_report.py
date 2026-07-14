from pathlib import Path

import pandas as pd

from multimodal_tugdt.config import load_config
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.reporting.research_report import generate_research_report


def test_research_report_is_aggregate_deterministic_and_privacy_conscious(tmp_path: Path) -> None:
    (tmp_path / "participants.csv").write_text("placeholder\n", encoding="utf-8")
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        """
project:
  root: "."
paths:
  manifest: participants.csv
  output_dir: outputs
""",
        encoding="utf-8",
    )
    qc_dir = tmp_path / "outputs/qc"
    feature_dir = tmp_path / "outputs/features"
    qc_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    pd.DataFrame({"qc_status": ["pass", "warning"]}).to_csv(
        qc_dir / "imu_preprocessing.csv", index=False
    )
    pd.DataFrame(
        {
            "feature_level": ["trial", "phase", "phase"],
            "imu__duration_s": [20.0, 2.0, 5.0],
        }
    ).to_csv(feature_dir / "imu_features.csv", index=False)
    records = [
        TrialRecord(
            "P001",
            "S01",
            "single_task",
            "T01",
            {"imu_path": "private/P001.csv", "clinical_path": "private/clinical.csv"},
        ),
        TrialRecord(
            "P002",
            "S01",
            "dual_task",
            "T01",
            {"imu_path": "private/P002.csv", "clinical_path": "private/clinical.csv"},
        ),
    ]

    first = generate_research_report(load_config(config_path), records)
    first_text = first.path.read_text(encoding="utf-8")
    second = generate_research_report(load_config(config_path), records)

    assert first.participant_count == 2
    assert first.trial_count == 2
    assert first_text == second.path.read_text(encoding="utf-8")
    assert "| Participants | 2 |" in first_text
    assert "| IMU preprocessing | available | 2 | 1 | 1 | 0 |" in first_text
    assert "Baseline modeling is disabled" in first_text
    assert "P001" not in first_text
    assert "P002" not in first_text
    assert "private/" not in first_text
