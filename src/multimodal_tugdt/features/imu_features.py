"""Interpretable trial-level and phase-level IMU features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from multimodal_tugdt.config import IMUConfig
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.preprocessing.imu import estimate_sampling_rate
from multimodal_tugdt.segmentation.manual import Segment, slice_segment

WALKING_PHASES = {"outbound_walk", "return_walk", "combined_straight_walk"}


def _rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def _base_signal_features(frame: pd.DataFrame, duration: float) -> dict[str, float | int]:
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    features: dict[str, float | int] = {
        "imu__sample_count": int(len(frame)),
        "imu__duration_s": float(duration),
        "imu__sampling_rate_hz": estimate_sampling_rate(timestamps),
    }
    axis_names = {
        "acc_ap": "imu__pelvis_acc_rms_ap_m_s2",
        "acc_ml": "imu__pelvis_acc_rms_ml_m_s2",
        "acc_vertical": "imu__pelvis_acc_rms_vertical_m_s2",
    }
    for column, feature_name in axis_names.items():
        features[feature_name] = (
            _rms(frame[column].to_numpy(dtype=float)) if column in frame else float("nan")
        )
    if "acc_ml" in frame:
        values = frame["acc_ml"].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        features["imu__pelvis_acc_range_ml_m_s2"] = (
            float(np.max(finite_values) - np.min(finite_values))
            if finite_values.size
            else float("nan")
        )
    else:
        features["imu__pelvis_acc_range_ml_m_s2"] = float("nan")

    acceleration_columns = [
        column for column in ("acc_ap", "acc_ml", "acc_vertical") if column in frame
    ]
    if acceleration_columns and len(frame) >= 2:
        acceleration = frame[acceleration_columns].to_numpy(dtype=float)
        jerk = np.gradient(acceleration, timestamps, axis=0)
        features["imu__pelvis_jerk_rms_m_s3"] = _rms(np.linalg.norm(jerk, axis=1))
    else:
        features["imu__pelvis_jerk_rms_m_s3"] = float("nan")
    features["imu__angular_velocity_rms_rad_s"] = (
        _rms(frame["gyro_yaw"].to_numpy(dtype=float))
        if "gyro_yaw" in frame
        else float("nan")
    )
    return features


def _step_events(frame: pd.DataFrame, config: IMUConfig) -> tuple[np.ndarray, np.ndarray]:
    if "acc_vertical" not in frame or len(frame) < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    values = frame["acc_vertical"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    sampling_rate = estimate_sampling_rate(timestamps)
    minimum_distance = max(1, int(round(config.step_min_interval_s * sampling_rate)))
    peaks, _ = find_peaks(
        np.nan_to_num(values, nan=float(np.nanmedian(values))),
        distance=minimum_distance,
        prominence=config.step_prominence,
    )
    peak_times = timestamps[peaks]
    return peak_times, np.diff(peak_times)


def detect_step_events(frame: pd.DataFrame, config: IMUConfig) -> np.ndarray:
    """Return IMU-derived step-event times for validation against a reference sensor."""
    event_times, _ = _step_events(frame, config)
    return event_times


def _step_statistics(
    event_times: Iterable[np.ndarray],
    intervals: Iterable[np.ndarray],
    walking_duration: float,
) -> dict[str, float | int]:
    event_arrays = list(event_times)
    interval_arrays = [array for array in intervals if array.size]
    step_count = int(sum(array.size for array in event_arrays))
    step_intervals = (
        np.concatenate(interval_arrays) if interval_arrays else np.array([], dtype=float)
    )
    mean_step_time = float(np.mean(step_intervals)) if step_intervals.size else float("nan")
    step_time_sd = (
        float(np.std(step_intervals, ddof=1)) if step_intervals.size >= 2 else float("nan")
    )
    step_time_cv = (
        float(step_time_sd / mean_step_time * 100)
        if np.isfinite(step_time_sd) and mean_step_time > 0
        else float("nan")
    )
    return {
        "imu__step_count": step_count,
        "imu__cadence_steps_min": (
            float(step_count / walking_duration * 60) if walking_duration > 0 else float("nan")
        ),
        "imu__mean_step_time_s": mean_step_time,
        "imu__step_time_sd_s": step_time_sd,
        "imu__step_time_cv_pct": step_time_cv,
    }


def _empty_step_statistics() -> dict[str, float]:
    return {
        "imu__step_count": float("nan"),
        "imu__cadence_steps_min": float("nan"),
        "imu__mean_step_time_s": float("nan"),
        "imu__step_time_sd_s": float("nan"),
        "imu__step_time_cv_pct": float("nan"),
    }


def _turn_statistics(frame: pd.DataFrame, duration: float) -> dict[str, float]:
    if "gyro_yaw" not in frame:
        return {
            "imu__turn_duration_s": float(duration),
            "imu__peak_yaw_velocity_rad_s": float("nan"),
            "imu__mean_abs_yaw_velocity_rad_s": float("nan"),
            "imu__turn_smoothness_rad_s2": float("nan"),
        }
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    yaw = frame["gyro_yaw"].to_numpy(dtype=float)
    finite_yaw = yaw[np.isfinite(yaw)]
    if finite_yaw.size == 0:
        return {
            "imu__turn_duration_s": float(duration),
            "imu__peak_yaw_velocity_rad_s": float("nan"),
            "imu__mean_abs_yaw_velocity_rad_s": float("nan"),
            "imu__turn_smoothness_rad_s2": float("nan"),
        }
    yaw_acceleration = np.gradient(yaw, timestamps) if len(frame) >= 2 else np.array([])
    return {
        "imu__turn_duration_s": float(duration),
        "imu__peak_yaw_velocity_rad_s": float(np.max(np.abs(finite_yaw))),
        "imu__mean_abs_yaw_velocity_rad_s": float(np.mean(np.abs(finite_yaw))),
        "imu__turn_smoothness_rad_s2": _rms(yaw_acceleration),
    }


def _empty_turn_statistics() -> dict[str, float]:
    return {
        "imu__turn_duration_s": float("nan"),
        "imu__peak_yaw_velocity_rad_s": float("nan"),
        "imu__mean_abs_yaw_velocity_rad_s": float("nan"),
        "imu__turn_smoothness_rad_s2": float("nan"),
    }


def _identity(record: TrialRecord, level: str, segment_name: str) -> dict[str, str]:
    return {
        "participant_id": record.participant_id,
        "session_id": record.session_id,
        "trial_id": record.trial_id,
        "condition": record.condition,
        "feature_level": level,
        "segment_name": segment_name,
    }


def extract_trial_and_phase_features(
    record: TrialRecord,
    frame: pd.DataFrame,
    segments: list[Segment],
    config: IMUConfig,
) -> list[dict[str, object]]:
    """Extract one trial row plus one row per externally annotated TUG phase."""
    trial_duration = float(frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0])
    trial_row: dict[str, object] = _identity(record, "trial", "full_trial")
    trial_row.update(_base_signal_features(frame, trial_duration))

    walking_data: list[tuple[pd.DataFrame, float]] = []
    turning_data: list[tuple[pd.DataFrame, float]] = []
    phase_rows: list[dict[str, object]] = []
    for segment in segments:
        phase_frame = slice_segment(frame, segment)
        phase_row: dict[str, object] = _identity(record, "phase", segment.name)
        phase_row.update(_base_signal_features(phase_frame, segment.duration))
        if segment.name in WALKING_PHASES:
            peak_times, intervals = _step_events(phase_frame, config)
            phase_row.update(_step_statistics([peak_times], [intervals], segment.duration))
            walking_data.append((phase_frame, segment.duration))
        else:
            phase_row.update(_empty_step_statistics())
        if segment.name.startswith("turn"):
            phase_row.update(_turn_statistics(phase_frame, segment.duration))
            turning_data.append((phase_frame, segment.duration))
        else:
            phase_row.update(_empty_turn_statistics())
        phase_rows.append(phase_row)

    if walking_data:
        walking_events: list[np.ndarray] = []
        walking_intervals: list[np.ndarray] = []
        for walking_frame, _ in walking_data:
            peak_times, intervals = _step_events(walking_frame, config)
            walking_events.append(peak_times)
            walking_intervals.append(intervals)
        trial_row.update(
            _step_statistics(
                walking_events,
                walking_intervals,
                sum(duration for _, duration in walking_data),
            )
        )
    else:
        peak_times, intervals = _step_events(frame, config)
        trial_row.update(_step_statistics([peak_times], [intervals], trial_duration))

    if turning_data:
        turn_frame = pd.concat([item[0] for item in turning_data], ignore_index=True)
        trial_row.update(
            _turn_statistics(turn_frame, sum(duration for _, duration in turning_data))
        )
    else:
        trial_row.update(_empty_turn_statistics())
    return [trial_row, *phase_rows]
