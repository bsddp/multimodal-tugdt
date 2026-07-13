import json
from pathlib import Path

import pandas as pd

from multimodal_tugdt.cli import main
from multimodal_tugdt.io.video_loader import VideoMetadata
from multimodal_tugdt.synthetic import generate_synthetic_dataset


def test_milestone5_metadata_only_video_pipeline(tmp_path: Path, monkeypatch) -> None:
    generate_synthetic_dataset(tmp_path / "data/synthetic", project_root=tmp_path)
    manifest_path = tmp_path / "data/synthetic/participants.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["video_path"] = ""
    for row_index, row in manifest.iterrows():
        video = tmp_path / f"data/synthetic/{row.participant_id}/S01/{row.trial_id}/video.mp4"
        video.write_bytes(b"synthetic placeholder")
        manifest.loc[row_index, "video_path"] = str(video.relative_to(tmp_path))
    manifest.to_csv(manifest_path, index=False)

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
video:
  enable_pose_estimation: false
""",
        encoding="utf-8",
    )
    assert main(["preprocess", "--config", str(config_path)]) == 0
    for row in manifest.itertuples():
        trial_dir = tmp_path / f"data/processed/{row.participant_id}/S01/{row.trial_id}"
        (trial_dir / "sync_metadata.json").write_text(
            json.dumps(
                {
                    "alignments": [
                        {
                            "target_modality": "video",
                            "offset_seconds": 0.25,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "multimodal_tugdt.pipeline.inspect_video",
        lambda _: VideoMetadata(20.0, 30.0, 600, 1280, 720, "h264", False),
    )
    assert main(["process-video", "--config", str(config_path)]) == 0

    trial_dir = tmp_path / "data/processed/P001/S01/T01"
    metadata = json.loads((trial_dir / "video_metadata.json").read_text())
    assert metadata["offset_seconds"] == 0.25
    assert metadata["pose_status"] == "not_requested"
    features = pd.read_csv(tmp_path / "outputs/features/video_features.csv")
    assert len(features) == 16
    assert features["video__pose_estimation_enabled"].eq(False).all()  # noqa: E712
    assert features["video__pose_detection_rate"].isna().all()
    qc = pd.read_csv(tmp_path / "outputs/qc/video_processing.csv")
    assert set(qc["qc_status"]) == {"pass"}


def test_milestone5_run_all_without_video_writes_parseable_empty_tables(tmp_path: Path) -> None:
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
""",
        encoding="utf-8",
    )

    assert main(["run-all", "--config", str(config_path)]) == 0
    features = pd.read_csv(tmp_path / "outputs/features/video_features.csv")
    qc = pd.read_csv(tmp_path / "outputs/qc/video_processing.csv")
    assert features.empty
    assert qc.empty
    assert "video__pose_detection_rate" in features.columns
    assert "qc_status" in qc.columns
