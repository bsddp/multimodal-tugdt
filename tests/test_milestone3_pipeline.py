import json
from pathlib import Path

import pandas as pd

from multimodal_tugdt.cli import main
from multimodal_tugdt.synthetic import generate_synthetic_dataset


def test_synchronization_pipeline_writes_metadata_qc_shifted_csv_and_plots(
    tmp_path: Path,
) -> None:
    generate_synthetic_dataset(
        tmp_path / "data/synthetic",
        project_root=tmp_path,
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "example.yaml"
    config_path.write_text(
        """
project:
  root: ".."
paths:
  manifest: data/synthetic/participants.csv
  processed_dir: data/processed
  output_dir: outputs
study:
  allowed_conditions: [single_task, dual_task]
imu:
  columns:
    timestamp: timestamp
    acc_ap: pelvis_acc_ap
    acc_ml: pelvis_acc_ml
    acc_vertical: pelvis_acc_vertical
    gyro_yaw: pelvis_gyro_yaw
    quat_w: quat_w
    quat_x: quat_x
    quat_y: quat_y
    quat_z: quat_z
  gravity_removal: constant
  generate_plots: false
synchronization:
  reference_modality: imu
  method: manual_offset
  offsets_seconds:
    audio: 0.25
    footswitch: -0.10
  uncertainty_seconds:
    audio: 0.05
    footswitch: 0.02
  timestamp_columns:
    footswitch: timestamp
  operator: test_operator
  notes: known synthetic offsets
  maximum_duration_difference_s: 0.5
  minimum_overlap_ratio: 0.9
  generate_plots: true
""",
        encoding="utf-8",
    )

    assert main(["preprocess", "--config", str(config_path)]) == 0
    assert main(["synchronize", "--config", str(config_path)]) == 0

    trial_dir = tmp_path / "data/processed/P001/S01/T01"
    metadata = json.loads((trial_dir / "sync_metadata.json").read_text())
    assert metadata["clock_mapping"] == "reference_time = native_time + offset_seconds"
    alignments = {item["target_modality"]: item for item in metadata["alignments"]}
    assert alignments["audio"]["offset_seconds"] == 0.25
    assert alignments["footswitch"]["offset_seconds"] == -0.10
    assert alignments["audio"]["operator"] == "test_operator"

    synchronized = pd.read_csv(trial_dir / "footswitch_synced.csv")
    assert synchronized["native_timestamp"].iloc[0] == 0.0
    assert synchronized["timestamp"].iloc[0] == -0.10
    qc = pd.read_csv(tmp_path / "outputs/qc/synchronization.csv")
    assert len(qc) == 4
    assert set(qc["qc_status"]) == {"pass"}
    assert len(list((tmp_path / "outputs/plots").glob("*_synchronization.png"))) == 2


def test_synchronization_fails_when_available_modality_offset_is_undeclared(
    tmp_path: Path,
) -> None:
    generate_synthetic_dataset(
        tmp_path / "data/synthetic",
        project_root=tmp_path,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  root: "."
paths:
  manifest: data/synthetic/participants.csv
  processed_dir: data/processed
  output_dir: outputs
imu:
  columns:
    timestamp: timestamp
    acc_ap: pelvis_acc_ap
  generate_plots: false
synchronization:
  offsets_seconds:
    audio: 0.0
  uncertainty_seconds:
    audio: 0.0
  generate_plots: false
""",
        encoding="utf-8",
    )

    assert main(["preprocess", "--config", str(config_path)]) == 0
    assert main(["synchronize", "--config", str(config_path)]) == 1
    qc = pd.read_csv(tmp_path / "outputs/qc/synchronization.csv")
    footswitch = qc.loc[qc["target_modality"] == "footswitch"]
    assert set(footswitch["qc_status"]) == {"fail"}
    assert footswitch["qc_notes"].str.contains("no explicit").all()
