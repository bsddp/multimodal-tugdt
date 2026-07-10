"""Project-level orchestration for the implemented research milestones."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from multimodal_tugdt.config import ProjectConfig
from multimodal_tugdt.features.imu_features import extract_trial_and_phase_features
from multimodal_tugdt.io.imu_loader import create_imu_loader
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.preprocessing.imu import preprocess_imu
from multimodal_tugdt.segmentation.manual import Segment, load_segments
from multimodal_tugdt.visualization.imu_plots import plot_imu_overview

LOGGER = logging.getLogger(__name__)
IDENTIFIER_COLUMNS = ("participant_id", "session_id", "trial_id", "condition")


@dataclass(frozen=True)
class WorkflowResult:
    """Summary returned to the CLI without hiding per-trial failures."""

    succeeded: int
    failed: int
    skipped: int
    output_path: Path


def _trial_directory(config: ProjectConfig, record: TrialRecord) -> Path:
    return config.processed_dir / record.participant_id / record.session_id / record.trial_id


def _plot_path(config: ProjectConfig, record: TrialRecord) -> Path:
    filename = f"{record.participant_id}_{record.session_id}_{record.trial_id}_imu.png"
    return config.output_dir / "plots" / filename


def _segments_for_record(
    config: ProjectConfig,
    record: TrialRecord,
    frame: pd.DataFrame,
) -> list[Segment]:
    annotation_value = record.paths.get("annotation_path", "")
    if not annotation_value:
        LOGGER.warning("Trial %s has no phase annotations; trial-level features only.", record.key)
        return []
    return load_segments(
        config.resolve_path(annotation_value),
        trial_start=float(frame["timestamp"].iloc[0]),
        trial_end=float(frame["timestamp"].iloc[-1]),
    )


def preprocess_project(config: ProjectConfig, records: list[TrialRecord]) -> WorkflowResult:
    """Preprocess every available IMU trial and write QC evidence."""
    loader = create_imu_loader(config.imu)
    qc_rows: list[dict[str, object]] = []
    succeeded = failed = skipped = 0
    for record in records:
        imu_value = record.paths.get("imu_path", "")
        identity = {
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "condition": record.condition,
        }
        if not imu_value:
            skipped += 1
            qc_rows.append({**identity, "qc_status": "warning", "qc_notes": "IMU absent"})
            continue
        try:
            raw = loader.load(config.resolve_path(imu_value))
            result = preprocess_imu(raw, config.imu)
            trial_dir = _trial_directory(config, record)
            trial_dir.mkdir(parents=True, exist_ok=True)
            output_frame = result.frame.copy()
            for index, column in enumerate(IDENTIFIER_COLUMNS):
                output_frame.insert(index, column, identity[column])
            output_frame.to_csv(trial_dir / "imu.csv", index=False)
            (trial_dir / "imu_qc.json").write_text(
                json.dumps(result.quality.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            segments = _segments_for_record(config, record, result.frame)
            if config.imu.generate_plots:
                plot_imu_overview(
                    result.frame,
                    segments,
                    _plot_path(config, record),
                    title=f"{record.participant_id} · {record.condition} · {record.trial_id}",
                )
            quality = result.quality.to_dict()
            qc_rows.append(
                {
                    **identity,
                    "qc_status": "warning" if quality["warnings"] else "pass",
                    "input_sample_count": quality["input_sample_count"],
                    "output_sample_count": quality["output_sample_count"],
                    "estimated_input_rate_hz": quality["estimated_input_rate_hz"],
                    "output_rate_hz": quality["output_rate_hz"],
                    "duplicate_timestamp_count": quality["duplicate_timestamp_count"],
                    "invalid_timestamp_count": quality["invalid_timestamp_count"],
                    "timestamps_were_nonmonotonic": quality["timestamps_were_nonmonotonic"],
                    "sampling_interval_cv_pct": quality["sampling_interval_cv_pct"],
                    "missing_ratio_by_column": json.dumps(
                        quality["missing_ratio_by_column"], sort_keys=True
                    ),
                    "anomaly_ratio_by_column": json.dumps(
                        quality["anomaly_ratio_by_column"], sort_keys=True
                    ),
                    "qc_notes": " | ".join(quality["warnings"]),
                }
            )
            succeeded += 1
        except (OSError, ValueError) as exc:
            failed += 1
            LOGGER.error("Failed to preprocess trial %s: %s", record.key, exc)
            qc_rows.append({**identity, "qc_status": "fail", "qc_notes": str(exc)})

    output_path = config.output_dir / "qc" / "imu_preprocessing.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(output_path, index=False)
    return WorkflowResult(succeeded, failed, skipped, output_path)


def _load_processed_trial(config: ProjectConfig, record: TrialRecord) -> pd.DataFrame:
    path = _trial_directory(config, record) / "imu.csv"
    if not path.is_file():
        raise ValueError(f"Processed IMU not found; run 'tugdt preprocess' first: {path}")
    frame = pd.read_csv(path)
    return frame.drop(columns=[column for column in IDENTIFIER_COLUMNS if column in frame])


def extract_project_features(config: ProjectConfig, records: list[TrialRecord]) -> WorkflowResult:
    """Extract trial- and phase-level IMU features from processed signals."""
    feature_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, str]] = []
    succeeded = failed = skipped = 0
    for record in records:
        if not record.paths.get("imu_path", ""):
            skipped += 1
            continue
        try:
            frame = _load_processed_trial(config, record)
            segments = _segments_for_record(config, record, frame)
            feature_rows.extend(
                extract_trial_and_phase_features(record, frame, segments, config.imu)
            )
            succeeded += 1
        except (OSError, ValueError) as exc:
            failed += 1
            LOGGER.error("Failed to extract IMU features for trial %s: %s", record.key, exc)
            failure_rows.append(
                {
                    "participant_id": record.participant_id,
                    "session_id": record.session_id,
                    "trial_id": record.trial_id,
                    "error": str(exc),
                }
            )

    output_path = config.output_dir / "features" / "imu_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_rows).to_csv(output_path, index=False)
    failure_path = config.output_dir / "features" / "imu_feature_failures.csv"
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(failure_path, index=False)
    elif failure_path.exists():
        failure_path.unlink()
    return WorkflowResult(succeeded, failed, skipped, output_path)

