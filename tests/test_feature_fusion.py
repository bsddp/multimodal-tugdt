from pathlib import Path

import pandas as pd

from multimodal_tugdt.config import load_config
from multimodal_tugdt.fusion.feature_level import build_trial_feature_table
from multimodal_tugdt.io.manifest import TrialRecord


def _records() -> list[TrialRecord]:
    return [
        TrialRecord(
            participant_id="P001",
            session_id="S01",
            condition="single_task",
            trial_id="T01",
            paths={"clinical_path": "clinical.csv"},
        ),
        TrialRecord(
            participant_id="P002",
            session_id="S01",
            condition="dual_task",
            trial_id="T01",
            paths={"clinical_path": "clinical.csv"},
        ),
    ]


def _write_feature_table(path: Path, modality: str, participant_ids: list[str]) -> None:
    rows = [
        {
            "participant_id": participant_id,
            "session_id": "S01",
            "trial_id": "T01",
            "condition": "single_task" if participant_id == "P001" else "dual_task",
            "feature_level": "trial",
            "segment_name": "trial",
            f"{modality}__value": index + 1.0,
        }
        for index, participant_id in enumerate(participant_ids)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_feature_fusion_preserves_missing_trials_and_adds_availability(tmp_path: Path) -> None:
    feature_dir = tmp_path / "outputs/features"
    feature_dir.mkdir(parents=True)
    _write_feature_table(feature_dir / "imu_features.csv", "imu", ["P001", "P002"])
    _write_feature_table(feature_dir / "audio_features.csv", "audio", ["P001"])
    _write_feature_table(feature_dir / "footswitch_features.csv", "footswitch", ["P002"])
    pd.DataFrame(
        columns=[
            "participant_id",
            "session_id",
            "trial_id",
            "condition",
            "feature_level",
            "video__duration_s",
        ]
    ).to_csv(feature_dir / "video_features.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": ["P001", "P002"],
            "age": [71, 72],
            "moca": [25, 23],
            "diagnosis": ["control", "MCI"],
        }
    ).to_csv(tmp_path / "clinical.csv", index=False)
    (tmp_path / "participants.csv").write_text("placeholder\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  root: "."
paths:
  manifest: participants.csv
  output_dir: outputs
fusion:
  modalities: [imu, audio, video, footswitch, clinical]
  feature_sets:
    all: [imu, audio, video, footswitch, clinical]
""",
        encoding="utf-8",
    )

    result = build_trial_feature_table(load_config(config_path), _records())

    assert len(result.frame) == 2
    p1 = result.frame.loc[result.frame["participant_id"] == "P001"].iloc[0]
    p2 = result.frame.loc[result.frame["participant_id"] == "P002"].iloc[0]
    assert p1["availability__audio"] == 1
    assert p2["availability__audio"] == 0
    assert pd.isna(p2["audio__value"])
    assert result.frame["availability__video"].eq(0).all()
    assert set(result.frame["clinical__diagnosis"]) == {"control", "MCI"}
    assert "clinical__moca" in set(result.inventory["column"])
