from pathlib import Path

import pandas as pd

from multimodal_tugdt.cli import main
from multimodal_tugdt.synthetic import generate_synthetic_dataset


def test_run_all_creates_processed_data_qc_features_and_plots(tmp_path: Path) -> None:
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
  format: wide_csv
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
  target_sampling_rate_hz: 100
  lowpass_cutoff_hz: 6
  filter_order: 4
  gravity_removal: constant
  generate_plots: true
synchronization:
  offsets_seconds:
    audio: 0.0
    footswitch: 0.0
  uncertainty_seconds:
    audio: 0.0
    footswitch: 0.0
  generate_plots: false
""",
        encoding="utf-8",
    )

    result = main(["run-all", "--config", str(config_path)])

    assert result == 0
    assert (tmp_path / "data/processed/P001/S01/T01/imu.csv").is_file()
    assert (tmp_path / "data/processed/P001/S01/T02/imu_qc.json").is_file()
    assert (tmp_path / "outputs/qc/imu_preprocessing.csv").is_file()
    feature_path = tmp_path / "outputs/features/imu_features.csv"
    assert feature_path.is_file()
    features = pd.read_csv(feature_path)
    assert len(features) == 16
    assert set(features["feature_level"]) == {"trial", "phase"}
    assert len(list((tmp_path / "outputs/plots").glob("*.png"))) == 2
