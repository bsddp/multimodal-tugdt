"""Interpretable frame-energy voice activity detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from multimodal_tugdt.config import AudioConfig
from multimodal_tugdt.io.audio_loader import AudioData


@dataclass(frozen=True)
class AudioInterval:
    """Half-open native-clock audio activity interval."""

    activity: str
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class AudioQualityReport:
    """Audio loading and VAD evidence."""

    original_sampling_rate_hz: int
    output_sampling_rate_hz: int
    sample_count: int
    duration_seconds: float
    clipping_ratio: float
    frame_count: int
    energy_threshold_dbfs: float
    speech_segment_count: int
    speech_duration_seconds: float
    speech_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioVADResult:
    intervals: list[AudioInterval]
    frames: pd.DataFrame
    quality: AudioQualityReport


def _runs(labels: np.ndarray) -> list[tuple[int, int, bool]]:
    if labels.size == 0:
        return []
    changes = np.flatnonzero(np.diff(labels.astype(np.int8)) != 0) + 1
    boundaries = np.concatenate(([0], changes, [labels.size]))
    return [
        (int(start), int(end), bool(labels[start]))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def _stabilize_labels(
    labels: np.ndarray,
    frame_seconds: float,
    config: AudioConfig,
) -> np.ndarray:
    stable = labels.copy()
    for start, end, is_speech in _runs(stable):
        if is_speech and (end - start) * frame_seconds < config.minimum_speech_duration_s:
            stable[start:end] = False
    for start, end, is_speech in _runs(stable):
        internal = start > 0 and end < stable.size
        if (
            not is_speech
            and internal
            and (end - start) * frame_seconds < config.minimum_pause_duration_s
        ):
            stable[start:end] = True
    return stable


def run_energy_vad(audio: AudioData, config: AudioConfig) -> AudioVADResult:
    """Classify fixed frames by absolute RMS threshold and stabilize short runs."""
    frame_size = max(1, round(config.frame_duration_ms / 1000 * audio.sampling_rate_hz))
    frame_seconds = frame_size / audio.sampling_rate_hz
    frame_count = int(np.ceil(len(audio.samples) / frame_size))
    padded = np.pad(audio.samples, (0, frame_count * frame_size - len(audio.samples)))
    matrix = padded.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(matrix), axis=1))
    threshold = 10 ** (config.energy_threshold_dbfs / 20)
    labels = _stabilize_labels(rms >= threshold, frame_seconds, config)
    frame_start = np.arange(frame_count) * frame_seconds
    frame_end = np.minimum(frame_start + frame_seconds, audio.duration_seconds)
    frames = pd.DataFrame(
        {
            "native_start_time": frame_start,
            "native_end_time": frame_end,
            "rms": rms,
            "rms_dbfs": 20 * np.log10(np.maximum(rms, np.finfo(float).tiny)),
            "is_speech": labels,
        }
    )
    intervals: list[AudioInterval] = []
    for start, end, is_speech in _runs(labels):
        interval_start = float(frame_start[start])
        interval_end = float(min(end * frame_seconds, audio.duration_seconds))
        if interval_end > interval_start:
            intervals.append(
                AudioInterval(
                    "speech" if is_speech else "silence",
                    interval_start,
                    interval_end,
                )
            )
    speech = [interval for interval in intervals if interval.activity == "speech"]
    speech_duration = sum(interval.duration_seconds for interval in speech)
    quality = AudioQualityReport(
        original_sampling_rate_hz=audio.original_sampling_rate_hz,
        output_sampling_rate_hz=audio.sampling_rate_hz,
        sample_count=len(audio.samples),
        duration_seconds=audio.duration_seconds,
        clipping_ratio=float((np.abs(audio.samples) >= config.clipping_threshold).mean()),
        frame_count=frame_count,
        energy_threshold_dbfs=config.energy_threshold_dbfs,
        speech_segment_count=len(speech),
        speech_duration_seconds=speech_duration,
        speech_ratio=(speech_duration / audio.duration_seconds),
    )
    return AudioVADResult(intervals, frames, quality)
