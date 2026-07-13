"""Trial- and phase-level behavior features from synchronized VAD intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from multimodal_tugdt.config import AudioConfig
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.segmentation.manual import Segment


def _identity(record: TrialRecord, level: str, segment_name: str) -> dict[str, str]:
    return {
        "participant_id": record.participant_id,
        "session_id": record.session_id,
        "trial_id": record.trial_id,
        "condition": record.condition,
        "feature_level": level,
        "segment_name": segment_name,
    }


def _clipped_speech(
    activity: pd.DataFrame,
    start_time: float,
    end_time: float,
) -> list[tuple[float, float]]:
    speech = activity.loc[activity["activity"] == "speech"]
    intervals: list[tuple[float, float]] = []
    for row in speech.itertuples(index=False):
        start = max(float(row.start_time), start_time)
        end = min(float(row.end_time), end_time)
        if end > start:
            intervals.append((start, end))
    return intervals


def _window_features(
    activity: pd.DataFrame,
    start_time: float,
    end_time: float,
    config: AudioConfig,
    *,
    first_response_reference: float | None,
) -> dict[str, float | int]:
    duration = end_time - start_time
    speech = _clipped_speech(activity, start_time, end_time)
    speech_duration = sum(end - start for start, end in speech)
    pauses = [
        speech[index + 1][0] - speech[index][1]
        for index in range(len(speech) - 1)
        if speech[index + 1][0] - speech[index][1] >= config.minimum_pause_duration_s
    ]
    eligible_response_starts = (
        [start for start, _ in speech if start >= first_response_reference]
        if first_response_reference is not None
        else []
    )
    return {
        "audio__speech_segment_count": len(speech),
        "audio__speech_duration_s": speech_duration,
        "audio__silence_duration_s": max(0.0, duration - speech_duration),
        "audio__speech_ratio": speech_duration / duration if duration > 0 else float("nan"),
        "audio__pause_count": len(pauses),
        "audio__mean_pause_duration_s": (
            float(np.mean(pauses)) if pauses else float("nan")
        ),
        "audio__max_pause_duration_s": (
            float(np.max(pauses)) if pauses else float("nan")
        ),
        "audio__first_response_latency_s": (
            eligible_response_starts[0] - first_response_reference
            if first_response_reference is not None and eligible_response_starts
            else float("nan")
        ),
        "audio__response_count": float("nan"),
        "audio__correct_response_count": float("nan"),
        "audio__response_accuracy": float("nan"),
    }


def extract_trial_and_phase_audio_features(
    record: TrialRecord,
    activity: pd.DataFrame,
    segments: list[Segment],
    config: AudioConfig,
    *,
    trial_start: float,
    trial_end: float,
) -> list[dict[str, object]]:
    """Extract VAD-only features without inventing transcript-derived outcomes."""
    trial_row: dict[str, object] = _identity(record, "trial", "full_trial")
    trial_row.update(
        _window_features(
            activity,
            trial_start,
            trial_end,
            config,
            first_response_reference=config.task_start_seconds,
        )
    )
    rows = [trial_row]
    for segment in segments:
        phase_row: dict[str, object] = _identity(record, "phase", segment.name)
        phase_row.update(
            _window_features(
                activity,
                segment.start_time,
                segment.end_time,
                config,
                first_response_reference=None,
            )
        )
        rows.append(phase_row)
    return rows

