"""Foot-contact timing and agreement with IMU-derived step events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multimodal_tugdt.config import FootswitchConfig, IMUConfig
from multimodal_tugdt.features.imu_features import detect_step_events
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.segmentation.manual import Segment, slice_segment

WALKING_PHASES = {"outbound_walk", "return_walk", "combined_straight_walk"}


@dataclass(frozen=True)
class EventMeasurements:
    left_contacts: np.ndarray
    right_contacts: np.ndarray
    stance_left: np.ndarray
    stance_right: np.ndarray
    swing_left: np.ndarray
    swing_right: np.ndarray
    step_intervals: np.ndarray


def _identity(record: TrialRecord, level: str, segment_name: str) -> dict[str, str]:
    return {
        "participant_id": record.participant_id,
        "session_id": record.session_id,
        "trial_id": record.trial_id,
        "condition": record.condition,
        "feature_level": level,
        "segment_name": segment_name,
    }


def _paired_durations(events: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray]:
    selected = events.loc[events["side"] == side].sort_values("timestamp")
    stance: list[float] = []
    swing: list[float] = []
    previous_event: str | None = None
    previous_time: float | None = None
    for row in selected.itertuples(index=False):
        timestamp = float(row.timestamp)
        if previous_time is not None:
            if previous_event == "contact" and row.event == "toe_off":
                stance.append(timestamp - previous_time)
            elif previous_event == "toe_off" and row.event == "contact":
                swing.append(timestamp - previous_time)
        previous_event = str(row.event)
        previous_time = timestamp
    return np.asarray(stance), np.asarray(swing)


def _measure_window(events: pd.DataFrame, start: float, end: float) -> EventMeasurements:
    selected = events.loc[(events["timestamp"] >= start) & (events["timestamp"] < end)].copy()
    contacts = selected.loc[selected["event"] == "contact"].sort_values("timestamp")
    left = contacts.loc[contacts["side"] == "left", "timestamp"].to_numpy(dtype=float)
    right = contacts.loc[contacts["side"] == "right", "timestamp"].to_numpy(dtype=float)
    stance_left, swing_left = _paired_durations(selected, "left")
    stance_right, swing_right = _paired_durations(selected, "right")
    all_contacts = contacts["timestamp"].to_numpy(dtype=float)
    return EventMeasurements(
        left,
        right,
        stance_left,
        stance_right,
        swing_left,
        swing_right,
        np.diff(all_contacts),
    )


def _combine(measurements: list[EventMeasurements]) -> EventMeasurements:
    def concatenate(attribute: str) -> np.ndarray:
        arrays = [
            getattr(item, attribute) for item in measurements if getattr(item, attribute).size
        ]
        return np.concatenate(arrays) if arrays else np.array([], dtype=float)

    return EventMeasurements(
        left_contacts=concatenate("left_contacts"),
        right_contacts=concatenate("right_contacts"),
        stance_left=concatenate("stance_left"),
        stance_right=concatenate("stance_right"),
        swing_left=concatenate("swing_left"),
        swing_right=concatenate("swing_right"),
        step_intervals=concatenate("step_intervals"),
    )


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _measurement_features(values: EventMeasurements) -> dict[str, float | int]:
    step_times = values.step_intervals
    mean_left_stance = _mean(values.stance_left)
    mean_right_stance = _mean(values.stance_right)
    denominator = (mean_left_stance + mean_right_stance) / 2
    asymmetry = (
        abs(mean_left_stance - mean_right_stance) / denominator * 100
        if np.isfinite(denominator) and denominator > 0
        else float("nan")
    )
    step_sd = float(np.std(step_times, ddof=1)) if step_times.size >= 2 else float("nan")
    mean_step = _mean(step_times)
    return {
        "footswitch__left_contact_count": int(values.left_contacts.size),
        "footswitch__right_contact_count": int(values.right_contacts.size),
        "footswitch__step_count": int(values.left_contacts.size + values.right_contacts.size),
        "footswitch__mean_left_stance_time_s": mean_left_stance,
        "footswitch__mean_right_stance_time_s": mean_right_stance,
        "footswitch__mean_left_swing_time_s": _mean(values.swing_left),
        "footswitch__mean_right_swing_time_s": _mean(values.swing_right),
        "footswitch__stance_time_asymmetry_pct": asymmetry,
        "footswitch__mean_step_time_s": mean_step,
        "footswitch__step_time_sd_s": step_sd,
        "footswitch__step_time_cv_pct": (
            step_sd / mean_step * 100 if np.isfinite(step_sd) and mean_step > 0 else float("nan")
        ),
    }


def _match_events(
    imu_events: np.ndarray,
    reference_events: np.ndarray,
    tolerance: float,
) -> dict[str, float | int]:
    candidates = sorted(
        (
            (abs(imu_time - reference_time), imu_index, reference_index)
            for imu_index, imu_time in enumerate(imu_events)
            for reference_index, reference_time in enumerate(reference_events)
            if abs(imu_time - reference_time) <= tolerance
        ),
        key=lambda item: item[0],
    )
    used_imu: set[int] = set()
    used_reference: set[int] = set()
    errors: list[float] = []
    for error, imu_index, reference_index in candidates:
        if imu_index not in used_imu and reference_index not in used_reference:
            used_imu.add(imu_index)
            used_reference.add(reference_index)
            errors.append(error)
    matched = len(errors)
    precision = matched / len(imu_events) if len(imu_events) else float("nan")
    recall = matched / len(reference_events) if len(reference_events) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and precision + recall > 0
        else float("nan")
    )
    return {
        "footswitch__imu_matched_event_count": matched,
        "footswitch__imu_event_precision": precision,
        "footswitch__imu_event_recall": recall,
        "footswitch__imu_event_agreement_f1": f1,
        "footswitch__imu_event_mean_abs_error_s": (
            float(np.mean(errors)) if errors else float("nan")
        ),
    }


def _empty_agreement() -> dict[str, float]:
    return {
        "footswitch__imu_matched_event_count": float("nan"),
        "footswitch__imu_event_precision": float("nan"),
        "footswitch__imu_event_recall": float("nan"),
        "footswitch__imu_event_agreement_f1": float("nan"),
        "footswitch__imu_event_mean_abs_error_s": float("nan"),
    }


def extract_trial_and_phase_footswitch_features(
    record: TrialRecord,
    events: pd.DataFrame,
    imu_frame: pd.DataFrame,
    segments: list[Segment],
    imu_config: IMUConfig,
    footswitch_config: FootswitchConfig,
) -> list[dict[str, object]]:
    """Extract contact timing and one-to-one event agreement within straight walking."""
    walking_segments = [segment for segment in segments if segment.name in WALKING_PHASES]
    if walking_segments:
        measurements = [
            _measure_window(events, segment.start_time, segment.end_time)
            for segment in walking_segments
        ]
        trial_measurements = _combine(measurements)
        imu_events = np.concatenate(
            [
                detect_step_events(slice_segment(imu_frame, segment), imu_config)
                for segment in walking_segments
            ]
        )
    else:
        start = float(imu_frame["timestamp"].iloc[0])
        end = float(imu_frame["timestamp"].iloc[-1])
        trial_measurements = _measure_window(events, start, end)
        imu_events = detect_step_events(imu_frame, imu_config)
    reference_events = np.sort(
        np.concatenate([trial_measurements.left_contacts, trial_measurements.right_contacts])
    )
    trial_row: dict[str, object] = _identity(record, "trial", "full_trial")
    trial_row.update(_measurement_features(trial_measurements))
    trial_row.update(
        _match_events(
            imu_events,
            reference_events,
            footswitch_config.event_matching_tolerance_s,
        )
    )
    rows = [trial_row]
    for segment in segments:
        phase_values = _measure_window(events, segment.start_time, segment.end_time)
        row: dict[str, object] = _identity(record, "phase", segment.name)
        row.update(_measurement_features(phase_values))
        if segment.name in WALKING_PHASES:
            phase_imu = detect_step_events(slice_segment(imu_frame, segment), imu_config)
            phase_reference = np.sort(
                np.concatenate([phase_values.left_contacts, phase_values.right_contacts])
            )
            row.update(
                _match_events(
                    phase_imu,
                    phase_reference,
                    footswitch_config.event_matching_tolerance_s,
                )
            )
        else:
            row.update(_empty_agreement())
        rows.append(row)
    return rows
