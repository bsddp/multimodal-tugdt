"""Interpretable two-dimensional proxy features from standardized pose landmarks."""

from __future__ import annotations

import math

import pandas as pd

from multimodal_tugdt.config import VideoConfig
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.io.video_loader import PoseExtractionResult, VideoMetadata
from multimodal_tugdt.segmentation.manual import Segment

VIDEO_FEATURE_COLUMNS = (
    "participant_id",
    "session_id",
    "trial_id",
    "condition",
    "feature_level",
    "segment_name",
    "segment_start_time",
    "segment_end_time",
    "video__duration_s",
    "video__frame_rate_hz",
    "video__total_frames",
    "video__width_pixels",
    "video__height_pixels",
    "video__pose_estimation_enabled",
    "video__processed_frame_count",
    "video__detected_frame_count",
    "video__pose_detection_rate",
    "video__mean_landmark_confidence",
    "video__trunk_lean_mean_degrees",
    "video__trunk_lean_range_degrees",
    "video__pelvis_vertical_displacement_proxy",
    "video__sit_to_stand_trunk_flexion_degrees",
    "video__left_right_step_length_proxy",
    "video__lower_limb_symmetry_proxy",
)


def _visible_point(
    landmarks: pd.DataFrame,
    name: str,
    minimum_visibility: float,
) -> tuple[float, float] | None:
    match = landmarks.loc[landmarks["landmark_name"] == name]
    if match.empty:
        return None
    row = match.iloc[0]
    if float(row["visibility"]) < minimum_visibility:
        return None
    return float(row["x"]), float(row["y"])


def compute_pose_frame_metrics(
    landmarks: pd.DataFrame,
    minimum_visibility: float,
) -> pd.DataFrame:
    """Reduce long-form landmarks to transparent frame-wise 2D proxy measures."""
    columns = [
        "frame_index",
        "timestamp",
        "trunk_lean_degrees",
        "pelvis_center_y",
        "left_right_step_length_proxy",
        "lower_limb_symmetry_proxy",
    ]
    if landmarks.empty:
        return pd.DataFrame(columns=columns)
    required = {"frame_index", "timestamp", "landmark_name", "x", "y", "visibility"}
    missing = sorted(required - set(landmarks.columns))
    if missing:
        raise ValueError(f"Pose landmarks are missing required columns: {', '.join(missing)}")

    rows: list[dict[str, float | int]] = []
    for (frame_index, timestamp), group in landmarks.groupby(
        ["frame_index", "timestamp"], sort=True
    ):
        points = {
            name: _visible_point(group, name, minimum_visibility)
            for name in (
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_ankle",
                "right_ankle",
            )
        }
        trunk_lean = pelvis_y = step_proxy = symmetry_proxy = math.nan
        shoulders = (points["left_shoulder"], points["right_shoulder"])
        hips = (points["left_hip"], points["right_hip"])
        ankles = (points["left_ankle"], points["right_ankle"])
        if all(point is not None for point in shoulders + hips):
            left_shoulder, right_shoulder = shoulders
            left_hip, right_hip = hips
            assert left_shoulder and right_shoulder and left_hip and right_hip
            shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_x = (left_hip[0] + right_hip[0]) / 2
            pelvis_y = (left_hip[1] + right_hip[1]) / 2
            trunk_lean = math.degrees(math.atan2(shoulder_x - hip_x, pelvis_y - shoulder_y))
        if all(point is not None for point in ankles):
            left_ankle, right_ankle = ankles
            assert left_ankle and right_ankle
            step_proxy = abs(left_ankle[0] - right_ankle[0])
        if all(point is not None for point in hips + ankles):
            left_hip, right_hip = hips
            left_ankle, right_ankle = ankles
            assert left_hip and right_hip and left_ankle and right_ankle
            left_length = math.dist(left_hip, left_ankle)
            right_length = math.dist(right_hip, right_ankle)
            denominator = (left_length + right_length) / 2
            if denominator > 0:
                symmetry_proxy = abs(left_length - right_length) / denominator
        rows.append(
            {
                "frame_index": int(frame_index),
                "timestamp": float(timestamp),
                "trunk_lean_degrees": trunk_lean,
                "pelvis_center_y": pelvis_y,
                "left_right_step_length_proxy": step_proxy,
                "lower_limb_symmetry_proxy": symmetry_proxy,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _slice_half_open(frame: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]


def _nan_stat(series: pd.Series, operation: str) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return math.nan
    if operation == "mean":
        return float(values.mean())
    if operation == "range":
        return float(values.max() - values.min())
    if operation == "max_abs":
        return float(values.abs().max())
    raise ValueError(f"Unsupported operation: {operation}")


def _summarize_interval(
    landmarks: pd.DataFrame,
    frames: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    start: float,
    end: float,
    sit_to_stand: tuple[float, float] | None,
) -> dict[str, float | int]:
    landmark_slice = _slice_half_open(landmarks, start, end)
    frame_slice = _slice_half_open(frames, start, end)
    metric_slice = _slice_half_open(metrics, start, end)
    processed = len(frame_slice)
    detected = int(frame_slice["pose_detected"].astype(bool).sum()) if processed else 0
    confidence = pd.to_numeric(landmark_slice.get("visibility"), errors="coerce").dropna()
    flexion = math.nan
    if sit_to_stand is not None:
        flexion_slice = _slice_half_open(metrics, *sit_to_stand)
        flexion = _nan_stat(flexion_slice["trunk_lean_degrees"], "max_abs")
    return {
        "video__processed_frame_count": processed,
        "video__detected_frame_count": detected,
        "video__pose_detection_rate": detected / processed if processed else math.nan,
        "video__mean_landmark_confidence": (
            float(confidence.mean()) if not confidence.empty else math.nan
        ),
        "video__trunk_lean_mean_degrees": _nan_stat(metric_slice["trunk_lean_degrees"], "mean"),
        "video__trunk_lean_range_degrees": _nan_stat(metric_slice["trunk_lean_degrees"], "range"),
        "video__pelvis_vertical_displacement_proxy": _nan_stat(
            metric_slice["pelvis_center_y"], "range"
        ),
        "video__sit_to_stand_trunk_flexion_degrees": flexion,
        "video__left_right_step_length_proxy": _nan_stat(
            metric_slice["left_right_step_length_proxy"], "mean"
        ),
        "video__lower_limb_symmetry_proxy": _nan_stat(
            metric_slice["lower_limb_symmetry_proxy"], "mean"
        ),
    }


def extract_video_features(
    record: TrialRecord,
    metadata: VideoMetadata,
    pose: PoseExtractionResult | None,
    segments: list[Segment],
    config: VideoConfig,
    *,
    trial_start: float,
    trial_end: float,
) -> list[dict[str, object]]:
    """Extract trial- and phase-level metadata and optional 2D pose proxies."""
    identity: dict[str, object] = {
        "participant_id": record.participant_id,
        "session_id": record.session_id,
        "trial_id": record.trial_id,
        "condition": record.condition,
    }
    sit_segment = next((segment for segment in segments if segment.name == "sit_to_stand"), None)
    sit_bounds = (sit_segment.start_time, sit_segment.end_time) if sit_segment is not None else None
    empty_landmarks = pd.DataFrame(columns=["timestamp", "visibility"])
    empty_frames = pd.DataFrame(columns=["timestamp", "pose_detected"])
    empty_metrics = pd.DataFrame(
        columns=[
            "timestamp",
            "trunk_lean_degrees",
            "pelvis_center_y",
            "left_right_step_length_proxy",
            "lower_limb_symmetry_proxy",
        ]
    )
    landmarks = pose.landmarks if pose is not None else empty_landmarks
    frames = pose.frames if pose is not None else empty_frames
    metrics = (
        compute_pose_frame_metrics(landmarks, config.minimum_visibility)
        if pose is not None
        else empty_metrics
    )

    trial_features: dict[str, object] = {
        **identity,
        "feature_level": "trial",
        "segment_name": "trial",
        "segment_start_time": trial_start,
        "segment_end_time": trial_end,
        "video__duration_s": metadata.duration_seconds,
        "video__frame_rate_hz": metadata.frame_rate_hz,
        "video__total_frames": metadata.total_frames,
        "video__width_pixels": metadata.width_pixels,
        "video__height_pixels": metadata.height_pixels,
        "video__pose_estimation_enabled": pose is not None,
        **_summarize_interval(
            landmarks,
            frames,
            metrics,
            start=trial_start,
            end=trial_end,
            sit_to_stand=sit_bounds,
        ),
    }
    rows = [trial_features]
    for segment in segments:
        rows.append(
            {
                **identity,
                "feature_level": "phase",
                "segment_name": segment.name,
                "segment_start_time": segment.start_time,
                "segment_end_time": segment.end_time,
                "video__duration_s": segment.duration,
                "video__frame_rate_hz": metadata.frame_rate_hz,
                "video__total_frames": math.nan,
                "video__width_pixels": metadata.width_pixels,
                "video__height_pixels": metadata.height_pixels,
                "video__pose_estimation_enabled": pose is not None,
                **_summarize_interval(
                    landmarks,
                    frames,
                    metrics,
                    start=segment.start_time,
                    end=segment.end_time,
                    sit_to_stand=(
                        (segment.start_time, segment.end_time)
                        if segment.name == "sit_to_stand"
                        else None
                    ),
                ),
            }
        )
    return rows
