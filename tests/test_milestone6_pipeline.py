from pathlib import Path

import pandas as pd

from multimodal_tugdt.cli import main
from multimodal_tugdt.synthetic import generate_synthetic_dataset


def test_milestone6_run_all_fuses_features_and_skips_disabled_modeling(tmp_path: Path) -> None:
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
fusion:
  modalities: [imu, audio, video, footswitch, clinical]
  feature_sets:
    imu: [imu]
    all: [imu, audio, video, footswitch, clinical]
modeling:
  enabled: false
  target_column: clinical__moca
  task_type: regression
  models: [ridge]
""",
        encoding="utf-8",
    )

    assert main(["run-all", "--config", str(config_path)]) == 0
    fused = pd.read_csv(tmp_path / "outputs/features/multimodal_features.csv")
    inventory = pd.read_csv(tmp_path / "outputs/features/feature_inventory.csv")
    assert len(fused) == 2
    assert fused["availability__imu"].eq(1).all()
    assert fused["availability__video"].eq(0).all()
    assert "clinical__moca" in fused
    assert not inventory.empty
    assert (tmp_path / "outputs/reports/research_summary.md").is_file()
    assert not (tmp_path / "outputs/modeling/summary_metrics.csv").exists()

    public_report = tmp_path / "public/report.md"
    assert (
        main(
            [
                "generate-report",
                "--config",
                str(config_path),
                "--output",
                str(public_report),
            ]
        )
        == 0
    )
    assert "Baseline modeling is disabled" in public_report.read_text(encoding="utf-8")

    assert main(["run-baselines", "--config", str(config_path)]) == 1
    skipped = pd.read_csv(tmp_path / "outputs/modeling/skipped_evaluations.csv")
    assert skipped["reason"].str.contains("participant groups|cohort is empty").any()


def test_milestone6_cli_writes_successful_grouped_modeling_artifacts(tmp_path: Path) -> None:
    feature_dir = tmp_path / "outputs/features"
    feature_dir.mkdir(parents=True)
    (tmp_path / "dummy_imu.csv").write_text("timestamp,value\n0,0\n", encoding="utf-8")
    manifest_rows = []
    clinical_rows = []
    imu_rows = []
    audio_rows = []
    for participant_index in range(8):
        participant = f"P{participant_index:03d}"
        clinical_rows.append(
            {
                "participant_id": participant,
                "moca": 30 - participant_index,
                "age": 65 + participant_index,
            }
        )
        for trial_index, condition in enumerate(("single_task", "dual_task"), start=1):
            trial = f"T{trial_index:02d}"
            manifest_rows.append(
                {
                    "participant_id": participant,
                    "session_id": "S01",
                    "condition": condition,
                    "trial_id": trial,
                    "imu_path": "dummy_imu.csv",
                    "video_path": "",
                    "audio_path": "",
                    "footswitch_path": "",
                    "annotation_path": "",
                    "clinical_path": "clinical.csv",
                }
            )
            identity = {
                "participant_id": participant,
                "session_id": "S01",
                "trial_id": trial,
                "condition": condition,
                "feature_level": "trial",
                "segment_name": "trial",
            }
            imu_rows.append(
                {**identity, "imu__cadence_steps_min": 90 - participant_index - trial_index}
            )
            if participant_index != 3:
                audio_rows.append(
                    {**identity, "audio__speech_ratio": 0.3 + participant_index / 100}
                )
    pd.DataFrame(manifest_rows).to_csv(tmp_path / "participants.csv", index=False)
    pd.DataFrame(clinical_rows).to_csv(tmp_path / "clinical.csv", index=False)
    pd.DataFrame(imu_rows).to_csv(feature_dir / "imu_features.csv", index=False)
    pd.DataFrame(audio_rows).to_csv(feature_dir / "audio_features.csv", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  root: "."
paths:
  manifest: participants.csv
  output_dir: outputs
study:
  allowed_conditions: [single_task, dual_task]
fusion:
  modalities: [imu, audio, clinical]
  feature_sets:
    imu: [imu]
    imu_audio: [imu, audio]
    all: [imu, audio, clinical]
modeling:
  enabled: true
  target_column: clinical__moca
  task_type: regression
  group_column: participant_id
  folds: 4
  models: [ridge]
  cohort_modes: [all_samples, complete_modalities]
""",
        encoding="utf-8",
    )

    assert main(["run-baselines", "--config", str(config_path)]) == 0
    summary = pd.read_csv(tmp_path / "outputs/modeling/summary_metrics.csv")
    predictions = pd.read_csv(tmp_path / "outputs/modeling/predictions.csv")
    audit = pd.read_csv(tmp_path / "outputs/modeling/split_audit.csv")
    metadata_path = tmp_path / "outputs/modeling/modeling_metadata.json"
    assert not summary.empty
    assert set(summary["metric"]) == {"mae", "rmse", "r2", "spearman"}
    assert not predictions.empty
    assert audit["group_overlap_count"].eq(0).all()
    assert metadata_path.is_file()
