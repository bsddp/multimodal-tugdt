import csv
from pathlib import Path

from multimodal_tugdt.config import load_config
from multimodal_tugdt.io.manifest import REQUIRED_COLUMNS, validate_manifest
from multimodal_tugdt.synthetic import generate_synthetic_dataset


def _write_config(root: Path, manifest: str = "data/synthetic/participants.csv") -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / "test.yaml"
    path.write_text(
        f"""
project:
  root: ".."
paths:
  manifest: {manifest}
study:
  allowed_conditions: [single_task, dual_task]
""",
        encoding="utf-8",
    )
    return path


def test_synthetic_manifest_validates_and_warns_for_missing_video(tmp_path: Path) -> None:
    generate_synthetic_dataset(
        tmp_path / "data/synthetic",
        project_root=tmp_path,
    )
    report = validate_manifest(load_config(_write_config(tmp_path)))

    assert report.is_valid
    assert len(report.records) == 2
    assert len(report.warnings) == 2
    assert all("video" in warning for warning in report.warnings)


def test_manifest_reports_missing_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "participants.csv"
    manifest.write_text("participant_id,trial_id\nP001,T01\n", encoding="utf-8")
    report = validate_manifest(
        load_config(_write_config(tmp_path, "participants.csv")),
        check_files=False,
    )

    assert not report.is_valid
    assert "Missing required columns" in report.errors[0]


def test_manifest_rejects_duplicate_trial_key_and_unknown_condition(tmp_path: Path) -> None:
    data_file = tmp_path / "imu.csv"
    data_file.write_text("timestamp,x\n0,0\n", encoding="utf-8")
    manifest = tmp_path / "participants.csv"
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update(
        participant_id="P001",
        session_id="S01",
        trial_id="T01",
        condition="unsupported",
        imu_path="imu.csv",
    )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows([row, row])

    report = validate_manifest(load_config(_write_config(tmp_path, "participants.csv")))

    assert not report.is_valid
    assert sum("unsupported condition" in error for error in report.errors) == 2
    assert sum("duplicate trial key" in error for error in report.errors) == 1


def test_manifest_reports_missing_referenced_file(tmp_path: Path) -> None:
    manifest = tmp_path / "participants.csv"
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update(
        participant_id="P001",
        session_id="S01",
        trial_id="T01",
        condition="single_task",
        imu_path="missing.csv",
    )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    report = validate_manifest(load_config(_write_config(tmp_path, "participants.csv")))

    assert not report.is_valid
    assert any("does not exist" in error for error in report.errors)

