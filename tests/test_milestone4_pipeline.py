from pathlib import Path

import pandas as pd

from multimodal_tugdt.cli import main
from multimodal_tugdt.synthetic import generate_synthetic_dataset


def test_milestone4_run_all_writes_audio_footswitch_features_and_qc(tmp_path: Path) -> None:
    generate_synthetic_dataset(tmp_path / "data/synthetic", project_root=tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  root: "."
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
  offsets_seconds:
    audio: 0.0
    footswitch: 0.0
  uncertainty_seconds:
    audio: 0.0
    footswitch: 0.0
  generate_plots: false
audio:
  energy_threshold_dbfs: -35
  task_start_seconds: 0.0
footswitch:
  event_matching_tolerance_s: 0.15
""",
        encoding="utf-8",
    )

    assert main(["run-all", "--config", str(config_path)]) == 0

    trial_dir = tmp_path / "data/processed/P001/S01/T01"
    assert (trial_dir / "audio_frames.csv").is_file()
    assert (trial_dir / "audio_activity.csv").is_file()
    assert (trial_dir / "audio_qc.json").is_file()
    assert (trial_dir / "footswitch_processed.csv").is_file()
    assert (trial_dir / "footswitch_events.csv").is_file()
    audio = pd.read_csv(tmp_path / "outputs/features/audio_features.csv")
    footswitch = pd.read_csv(tmp_path / "outputs/features/footswitch_features.csv")
    assert len(audio) == 16
    assert len(footswitch) == 16
    audio_trials = audio.loc[audio["feature_level"] == "trial"]
    assert set(audio_trials["audio__speech_segment_count"]) == {4}
    assert audio_trials["audio__response_accuracy"].isna().all()
    foot_trials = footswitch.loc[footswitch["feature_level"] == "trial"]
    assert (foot_trials["footswitch__imu_event_agreement_f1"] > 0.8).all()
    assert (tmp_path / "outputs/qc/audio_processing.csv").is_file()
    assert (tmp_path / "outputs/qc/footswitch_processing.csv").is_file()
