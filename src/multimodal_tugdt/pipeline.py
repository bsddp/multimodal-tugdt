"""Project-level orchestration for the integrated research pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn import __version__ as sklearn_version

from multimodal_tugdt.config import ProjectConfig
from multimodal_tugdt.features.audio_features import extract_trial_and_phase_audio_features
from multimodal_tugdt.features.dual_task_cost import calculate_dual_task_costs
from multimodal_tugdt.features.footswitch_features import (
    extract_trial_and_phase_footswitch_features,
)
from multimodal_tugdt.features.imu_features import extract_trial_and_phase_features
from multimodal_tugdt.features.video_features import VIDEO_FEATURE_COLUMNS, extract_video_features
from multimodal_tugdt.fusion.feature_level import build_trial_feature_table
from multimodal_tugdt.io.audio_loader import load_audio
from multimodal_tugdt.io.imu_loader import create_imu_loader
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.io.video_loader import (
    PoseExtractionResult,
    extract_mediapipe_pose,
    inspect_video,
)
from multimodal_tugdt.modeling.evaluation import evaluate_baselines
from multimodal_tugdt.preprocessing.audio import run_energy_vad
from multimodal_tugdt.preprocessing.footswitch import process_footswitch
from multimodal_tugdt.preprocessing.imu import preprocess_imu
from multimodal_tugdt.reporting.research_report import generate_research_report
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
                "Available footswitch data requires synchronization.timestamp_columns.footswitch."
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
            modality for modality in target_modalities if record.paths.get(f"{modality}_path", "")
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


def _synchronization_offset(
    config: ProjectConfig,
    record: TrialRecord,
    modality: str,
) -> float:
    path = _trial_directory(config, record) / "sync_metadata.json"
    if not path.is_file():
        raise ValueError(f"Synchronization metadata not found; run 'tugdt synchronize': {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    for alignment in metadata.get("alignments", []):
        if alignment.get("target_modality") == modality:
            return float(alignment["offset_seconds"])
    raise ValueError(f"No successful {modality} alignment in synchronization metadata: {path}")


def process_audio_project(config: ProjectConfig, records: list[TrialRecord]) -> WorkflowResult:
    """Run audio loading, VAD, QC, and trial/phase feature extraction."""
    feature_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    succeeded = failed = skipped = 0
    for record in records:
        audio_value = record.paths.get("audio_path", "")
        identity = {
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "condition": record.condition,
        }
        if not audio_value:
            skipped += 1
            continue
        try:
            offset = _synchronization_offset(config, record, "audio")
            audio = load_audio(config.resolve_path(audio_value), config.audio)
            vad = run_energy_vad(audio, config.audio)
            activity_rows = [
                {
                    **identity,
                    "activity": interval.activity,
                    "native_start_time": interval.start_seconds,
                    "native_end_time": interval.end_seconds,
                    "start_time": interval.start_seconds + offset,
                    "end_time": interval.end_seconds + offset,
                    "duration_s": interval.duration_seconds,
                }
                for interval in vad.intervals
            ]
            activity = pd.DataFrame(activity_rows)
            trial_dir = _trial_directory(config, record)
            activity.to_csv(trial_dir / "audio_activity.csv", index=False)
            audio_frames = vad.frames.copy()
            audio_frames["start_time"] = audio_frames["native_start_time"] + offset
            audio_frames["end_time"] = audio_frames["native_end_time"] + offset
            for index, column in enumerate(IDENTIFIER_COLUMNS):
                audio_frames.insert(index, column, identity[column])
            audio_frames.to_csv(trial_dir / "audio_frames.csv", index=False)
            (trial_dir / "audio_qc.json").write_text(
                json.dumps(vad.quality.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            imu_frame = _load_processed_trial(config, record)
            segments = _segments_for_record(config, record, imu_frame)
            feature_rows.extend(
                extract_trial_and_phase_audio_features(
                    record,
                    activity,
                    segments,
                    config.audio,
                    trial_start=float(imu_frame["timestamp"].iloc[0]),
                    trial_end=float(imu_frame["timestamp"].iloc[-1]),
                )
            )
            quality = vad.quality.to_dict()
            qc_status = (
                "warning"
                if quality["speech_segment_count"] == 0 or quality["clipping_ratio"] > 0.01
                else "pass"
            )
            qc_rows.append({**identity, "qc_status": qc_status, **quality})
            succeeded += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failed += 1
            LOGGER.error("Failed to process audio for trial %s: %s", record.key, exc)
            qc_rows.append({**identity, "qc_status": "fail", "qc_notes": str(exc)})
    feature_path = config.output_dir / "features" / "audio_features.csv"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    qc_path = config.output_dir / "qc" / "audio_processing.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    return WorkflowResult(succeeded, failed, skipped, feature_path)


def process_footswitch_project(
    config: ProjectConfig,
    records: list[TrialRecord],
) -> WorkflowResult:
    """Debounce synchronized contacts and extract timing/agreement features."""
    feature_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    succeeded = failed = skipped = 0
    for record in records:
        identity = {
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "condition": record.condition,
        }
        if not record.paths.get("footswitch_path", ""):
            skipped += 1
            continue
        try:
            trial_dir = _trial_directory(config, record)
            synced_path = trial_dir / "footswitch_synced.csv"
            if not synced_path.is_file():
                raise ValueError(
                    f"Synchronized footswitch not found; run 'tugdt synchronize': {synced_path}"
                )
            result = process_footswitch(pd.read_csv(synced_path), config.footswitch)
            result.frame.to_csv(trial_dir / "footswitch_processed.csv", index=False)
            events = result.events.copy()
            for index, column in enumerate(IDENTIFIER_COLUMNS):
                events.insert(index, column, identity[column])
            events.to_csv(trial_dir / "footswitch_events.csv", index=False)
            (trial_dir / "footswitch_qc.json").write_text(
                json.dumps(result.quality.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            imu_frame = _load_processed_trial(config, record)
            segments = _segments_for_record(config, record, imu_frame)
            feature_rows.extend(
                extract_trial_and_phase_footswitch_features(
                    record,
                    result.events,
                    imu_frame,
                    segments,
                    config.imu,
                    config.footswitch,
                )
            )
            quality = result.quality.to_dict()
            contact_count = quality["left_contact_count"] + quality["right_contact_count"]
            qc_rows.append(
                {
                    **identity,
                    "qc_status": "pass" if contact_count > 0 else "warning",
                    **quality,
                }
            )
            succeeded += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failed += 1
            LOGGER.error("Failed to process footswitch for trial %s: %s", record.key, exc)
            qc_rows.append({**identity, "qc_status": "fail", "qc_notes": str(exc)})
    feature_path = config.output_dir / "features" / "footswitch_features.csv"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    qc_path = config.output_dir / "qc" / "footswitch_processing.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    return WorkflowResult(succeeded, failed, skipped, feature_path)


def process_video_project(config: ProjectConfig, records: list[TrialRecord]) -> WorkflowResult:
    """Inspect video files and optionally extract aligned MediaPipe pose landmarks."""
    feature_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    succeeded = failed = skipped = 0
    for record in records:
        video_value = record.paths.get("video_path", "")
        identity = {
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "condition": record.condition,
        }
        if not video_value:
            skipped += 1
            continue
        try:
            source = config.resolve_path(video_value)
            offset = _synchronization_offset(config, record, "video")
            metadata = inspect_video(source)
            pose: PoseExtractionResult | None = None
            pose_status = "not_requested"
            if config.video.enable_pose_estimation:
                assert config.video.pose_model_path is not None
                raw_pose = extract_mediapipe_pose(
                    source,
                    config.resolve_path(config.video.pose_model_path),
                    config.video,
                )
                landmarks = raw_pose.landmarks.copy()
                landmarks["timestamp"] = landmarks["native_timestamp"] + offset
                frames = raw_pose.frames.copy()
                frames["timestamp"] = frames["native_timestamp"] + offset
                pose = PoseExtractionResult(
                    landmarks=landmarks,
                    frames=frames,
                    processed_frame_count=raw_pose.processed_frame_count,
                    detected_frame_count=raw_pose.detected_frame_count,
                    backend=raw_pose.backend,
                )
                pose_status = "complete"

            trial_dir = _trial_directory(config, record)
            trial_dir.mkdir(parents=True, exist_ok=True)
            if pose is not None:
                pose.landmarks.to_csv(trial_dir / "video_pose_landmarks.csv", index=False)
                pose.frames.to_csv(trial_dir / "video_pose_frames.csv", index=False)
            video_metadata = {
                **metadata.to_dict(),
                "source_path": str(source),
                "reference_modality": "imu",
                "offset_seconds": offset,
                "pose_status": pose_status,
                "pose_backend": pose.backend if pose is not None else config.video.pose_backend,
                "pose_model_path": (
                    str(config.resolve_path(config.video.pose_model_path))
                    if config.video.pose_model_path is not None
                    else None
                ),
                "frame_step": config.video.frame_step,
                "processed_frame_count": pose.processed_frame_count if pose is not None else 0,
                "detected_frame_count": pose.detected_frame_count if pose is not None else 0,
                "pose_detection_rate": pose.detection_rate if pose is not None else None,
            }
            (trial_dir / "video_metadata.json").write_text(
                json.dumps(video_metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            imu_frame = _load_processed_trial(config, record)
            segments = _segments_for_record(config, record, imu_frame)
            feature_rows.extend(
                extract_video_features(
                    record,
                    metadata,
                    pose,
                    segments,
                    config.video,
                    trial_start=float(imu_frame["timestamp"].iloc[0]),
                    trial_end=float(imu_frame["timestamp"].iloc[-1]),
                )
            )
            mean_visibility = None
            if pose is not None and not pose.landmarks.empty:
                mean_visibility = float(pose.landmarks["visibility"].mean())
            qc_rows.append(
                {
                    **identity,
                    "qc_status": "pass",
                    "duration_seconds": metadata.duration_seconds,
                    "frame_rate_hz": metadata.frame_rate_hz,
                    "total_frames": metadata.total_frames,
                    "frame_count_is_estimated": metadata.frame_count_is_estimated,
                    "width_pixels": metadata.width_pixels,
                    "height_pixels": metadata.height_pixels,
                    "pose_status": pose_status,
                    "pose_backend": pose.backend if pose is not None else config.video.pose_backend,
                    "processed_frame_count": (
                        pose.processed_frame_count if pose is not None else 0
                    ),
                    "detected_frame_count": pose.detected_frame_count if pose is not None else 0,
                    "pose_detection_rate": pose.detection_rate if pose is not None else None,
                    "mean_landmark_confidence": mean_visibility,
                    "qc_notes": (
                        "Pose estimation disabled; metadata only." if pose is None else ""
                    ),
                }
            )
            succeeded += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failed += 1
            LOGGER.error("Failed to process video for trial %s: %s", record.key, exc)
            qc_rows.append({**identity, "qc_status": "fail", "qc_notes": str(exc)})

    feature_path = config.output_dir / "features" / "video_features.csv"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_rows, columns=VIDEO_FEATURE_COLUMNS).to_csv(feature_path, index=False)
    qc_path = config.output_dir / "qc" / "video_processing.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_columns = (
        *IDENTIFIER_COLUMNS,
        "qc_status",
        "duration_seconds",
        "frame_rate_hz",
        "total_frames",
        "frame_count_is_estimated",
        "width_pixels",
        "height_pixels",
        "pose_status",
        "pose_backend",
        "processed_frame_count",
        "detected_frame_count",
        "pose_detection_rate",
        "mean_landmark_confidence",
        "qc_notes",
    )
    pd.DataFrame(qc_rows, columns=qc_columns).to_csv(qc_path, index=False)
    return WorkflowResult(succeeded, failed, skipped, feature_path)


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


def fuse_project_features(config: ProjectConfig, records: list[TrialRecord]) -> WorkflowResult:
    """Build a trial-level multimodal matrix and feature inventory."""
    result = build_trial_feature_table(config, records)
    output_path = _write_fusion_artifacts(config, result.frame, result.inventory)
    return WorkflowResult(len(result.frame), 0, 0, output_path)


def _write_fusion_artifacts(
    config: ProjectConfig,
    fused: pd.DataFrame,
    inventory: pd.DataFrame,
) -> Path:
    """Write trial fusion and the optional pair-level dual-task cost artifact."""
    feature_dir = config.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / "multimodal_features.csv"
    fused.to_csv(output_path, index=False)
    inventory.to_csv(feature_dir / "feature_inventory.csv", index=False)
    cost_path = feature_dir / "dual_task_costs.csv"
    if config.dual_task_cost.enabled:
        costs = calculate_dual_task_costs(fused, config.dual_task_cost)
        costs.frame.to_csv(cost_path, index=False)
        LOGGER.info(
            "Dual-task cost complete: %d paired group(s), %d incomplete group(s) skipped. "
            "Output: %s",
            costs.paired_group_count,
            costs.skipped_group_count,
            cost_path,
        )
    elif cost_path.exists():
        cost_path.unlink()
    return output_path


def _write_modeling_artifact(
    frame: pd.DataFrame,
    path: Path,
    empty_columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        pd.DataFrame(columns=empty_columns).to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def run_baselines_project(config: ProjectConfig, records: list[TrialRecord]) -> WorkflowResult:
    """Fuse current features and evaluate participant-grouped baseline models."""
    fused = build_trial_feature_table(config, records)
    _write_fusion_artifacts(config, fused.frame, fused.inventory)

    evaluation = evaluate_baselines(fused.frame, config.fusion, config.modeling)
    output_dir = config.output_dir / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)
    modeling_metadata = {
        "schema_version": 1,
        "task_type": config.modeling.task_type,
        "target_column": config.modeling.target_column,
        "group_column": config.modeling.group_column,
        "requested_folds": config.modeling.folds,
        "random_seed": config.modeling.random_seed,
        "models": list(config.modeling.models),
        "cohort_modes": list(config.modeling.cohort_modes),
        "positive_label": config.modeling.positive_label,
        "feature_sets": {
            name: list(modalities) for name, modalities in config.fusion.feature_sets.items()
        },
        "preprocessing_scope": "fit within each training fold",
        "scikit_learn_version": sklearn_version,
        "successful_evaluations": evaluation.successful_evaluations,
        "skipped_evaluation_count": len(evaluation.skipped),
    }
    (output_dir / "modeling_metadata.json").write_text(
        json.dumps(modeling_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    common = (
        "task_type",
        "target_column",
        "feature_set",
        "modalities",
        "cohort",
        "model",
    )
    _write_modeling_artifact(
        evaluation.fold_metrics,
        output_dir / "fold_metrics.csv",
        (*common, "fold", "n_samples", "n_groups", "n_features", "n_folds"),
    )
    summary_path = output_dir / "summary_metrics.csv"
    _write_modeling_artifact(
        evaluation.summary_metrics,
        summary_path,
        (
            *common,
            "n_samples",
            "n_groups",
            "n_features",
            "n_folds",
            "metric",
            "mean",
            "standard_deviation",
            "valid_fold_count",
        ),
    )
    _write_modeling_artifact(
        evaluation.predictions,
        output_dir / "predictions.csv",
        (*IDENTIFIER_COLUMNS, *common, "fold", "y_true", "y_pred", "y_score"),
    )
    _write_modeling_artifact(
        evaluation.split_audit,
        output_dir / "split_audit.csv",
        (
            *common[:-1],
            "fold",
            "train_group_count",
            "test_group_count",
            "group_overlap_count",
            "train_groups",
            "test_groups",
        ),
    )
    _write_modeling_artifact(
        evaluation.skipped,
        output_dir / "skipped_evaluations.csv",
        (*common, "reason"),
    )
    return WorkflowResult(
        evaluation.successful_evaluations,
        0 if evaluation.successful_evaluations else 1,
        len(evaluation.skipped),
        summary_path,
    )


def generate_report_project(
    config: ProjectConfig,
    records: list[TrialRecord],
    output_path: str | Path | None = None,
) -> WorkflowResult:
    """Generate the aggregate Markdown research summary."""
    report = generate_research_report(config, records, output_path)
    return WorkflowResult(report.trial_count, 0, 0, report.path)
