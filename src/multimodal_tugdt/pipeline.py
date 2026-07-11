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
from multimodal_tugdt.synchronization.timeline import (
    AlignmentResult,
    ManualOffsetSynchronizer,
    SynchronizationError,
    Timeline,
    apply_offset_to_timestamps,
    read_csv_timeline,
    read_media_timeline,
)
from multimodal_tugdt.visualization.imu_plots import plot_imu_overview
from multimodal_tugdt.visualization.sync_plots import plot_synchronization_overview

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


def _synchronization_plot_path(config: ProjectConfig, record: TrialRecord) -> Path:
    filename = f"{record.participant_id}_{record.session_id}_{record.trial_id}_synchronization.png"
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


def _reference_timeline(config: ProjectConfig, record: TrialRecord) -> Timeline:
    path = _trial_directory(config, record) / "imu.csv"
    timeline, _ = read_csv_timeline(path, "imu", "timestamp")
    return timeline


def _target_timeline(
    config: ProjectConfig,
    record: TrialRecord,
    modality: str,
) -> tuple[Timeline, pd.DataFrame | None]:
    path_value = record.paths[f"{modality}_path"]
    path = config.resolve_path(path_value)
    if modality == "footswitch":
        timestamp_column = config.synchronization.timestamp_columns.get(modality)
        if not timestamp_column:
            raise SynchronizationError(
                "Available footswitch data requires "
                "synchronization.timestamp_columns.footswitch."
            )
        return read_csv_timeline(path, modality, timestamp_column)
    return read_media_timeline(path, modality), None


def _save_synchronized_footswitch(
    config: ProjectConfig,
    record: TrialRecord,
    frame: pd.DataFrame,
    alignment: AlignmentResult,
) -> None:
    timestamp_column = config.synchronization.timestamp_columns["footswitch"]
    native = pd.to_numeric(frame[timestamp_column], errors="raise").to_numpy(dtype=float)
    signal_columns = frame.drop(columns=[timestamp_column]).copy()
    reference_timestamps = apply_offset_to_timestamps(native, alignment.offset_seconds)
    signal_columns.insert(0, "timestamp", reference_timestamps)
    signal_columns.insert(0, "native_timestamp", native)
    identity = {
        "participant_id": record.participant_id,
        "session_id": record.session_id,
        "trial_id": record.trial_id,
        "condition": record.condition,
    }
    for index, column in enumerate(IDENTIFIER_COLUMNS):
        signal_columns.insert(index, column, identity[column])
    signal_columns.to_csv(_trial_directory(config, record) / "footswitch_synced.csv", index=False)


def _save_reference_segments(
    config: ProjectConfig,
    record: TrialRecord,
    reference: Timeline,
) -> None:
    annotation_value = record.paths.get("annotation_path", "")
    if not annotation_value:
        return
    segments = load_segments(
        config.resolve_path(annotation_value),
        trial_start=reference.native_start_seconds,
        trial_end=reference.native_end_seconds,
    )
    rows = [
        {
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "condition": record.condition,
            "segment_name": segment.name,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "source": segment.source,
            "confidence": segment.confidence,
            "notes": segment.notes,
            "reference_modality": "imu",
        }
        for segment in segments
    ]
    pd.DataFrame(rows).to_csv(_trial_directory(config, record) / "segments.csv", index=False)


def synchronize_project(config: ProjectConfig, records: list[TrialRecord]) -> WorkflowResult:
    """Map available target modalities to the IMU clock and save auditable evidence."""
    synchronizer = ManualOffsetSynchronizer(config.synchronization)
    qc_rows: list[dict[str, object]] = []
    succeeded = failed = skipped = 0
    target_modalities = ("video", "audio", "footswitch")
    for record in records:
        identity = {
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "condition": record.condition,
        }
        available = [
            modality
            for modality in target_modalities
            if record.paths.get(f"{modality}_path", "")
        ]
        if not available:
            skipped += 1
            continue
        alignments: list[AlignmentResult] = []
        errors: list[dict[str, str]] = []
        try:
            reference = _reference_timeline(config, record)
            _save_reference_segments(config, record, reference)
        except (OSError, ValueError) as exc:
            failed += 1
            message = str(exc)
            LOGGER.error("Failed to load reference timeline for trial %s: %s", record.key, message)
            for modality in available:
                qc_rows.append(
                    {
                        **identity,
                        "reference_modality": "imu",
                        "target_modality": modality,
                        "qc_status": "fail",
                        "qc_notes": message,
                    }
                )
            continue

        for modality in available:
            try:
                target, frame = _target_timeline(config, record, modality)
                alignment = synchronizer.align(reference, target)
                alignments.append(alignment)
                if modality == "footswitch" and frame is not None:
                    _save_synchronized_footswitch(config, record, frame, alignment)
                row = {**identity, **alignment.to_dict()}
                row["qc_notes"] = " | ".join(alignment.qc_notes)
                qc_rows.append(row)
            except (OSError, ValueError) as exc:
                message = str(exc)
                LOGGER.error(
                    "Failed to synchronize %s for trial %s: %s",
                    modality,
                    record.key,
                    message,
                )
                errors.append({"target_modality": modality, "error": message})
                qc_rows.append(
                    {
                        **identity,
                        "reference_modality": "imu",
                        "target_modality": modality,
                        "synchronization_method": config.synchronization.method,
                        "qc_status": "fail",
                        "qc_notes": message,
                    }
                )

        metadata = {
            "schema_version": 1,
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "condition": record.condition,
            "clock_mapping": "reference_time = native_time + offset_seconds",
            "synchronization_method": config.synchronization.method,
            "qc_thresholds": {
                "maximum_duration_difference_s": (
                    config.synchronization.maximum_duration_difference_s
                ),
                "minimum_overlap_ratio": config.synchronization.minimum_overlap_ratio,
            },
            "reference": reference.to_dict(),
            "alignments": [alignment.to_dict() for alignment in alignments],
            "errors": errors,
        }
        trial_dir = _trial_directory(config, record)
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "sync_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if config.synchronization.generate_plots:
            plot_synchronization_overview(
                reference,
                alignments,
                _synchronization_plot_path(config, record),
                title=f"{record.participant_id} · {record.condition} · {record.trial_id}",
            )
        if errors:
            failed += 1
        else:
            succeeded += 1

    output_path = config.output_dir / "qc" / "synchronization.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(output_path, index=False)
    return WorkflowResult(succeeded, failed, skipped, output_path)


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
