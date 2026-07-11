"""Timeline readers and auditable manual-offset alignment."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multimodal_tugdt.config import SynchronizationConfig


class SynchronizationError(ValueError):
    """Raised when clocks cannot be aligned without an unsafe assumption."""


@dataclass(frozen=True)
class Timeline:
    """Native-clock extent of one modality file."""

    modality: str
    native_start_seconds: float
    native_end_seconds: float
    source_path: str

    @property
    def duration_seconds(self) -> float:
        return self.native_end_seconds - self.native_start_seconds

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "duration_seconds": self.duration_seconds}


@dataclass
class AlignmentResult:
    """Complete evidence for mapping a target modality to the reference clock."""

    reference_modality: str
    target_modality: str
    synchronization_method: str
    offset_seconds: float
    alignment_event: str
    estimated_uncertainty_seconds: float | None
    operator: str
    notes: str
    target_source_path: str
    native_start_seconds: float
    native_end_seconds: float
    reference_start_seconds: float
    reference_end_seconds: float
    duration_seconds: float
    reference_duration_seconds: float
    duration_difference_seconds: float
    overlap_duration_seconds: float
    overlap_ratio: float
    qc_status: str
    qc_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_offset_to_timestamps(
    timestamps: pd.Series | np.ndarray,
    offset_seconds: float,
) -> np.ndarray:
    """Map native timestamps to reference time using reference = native + offset."""
    values = np.asarray(timestamps, dtype=float)
    if not np.isfinite(values).all():
        raise SynchronizationError("Timestamps must be finite before applying an offset.")
    if not math.isfinite(offset_seconds):
        raise SynchronizationError("Synchronization offset must be finite.")
    return values + offset_seconds


class ManualOffsetSynchronizer:
    """Apply a declared offset and evaluate temporal coverage against a reference."""

    def __init__(self, config: SynchronizationConfig) -> None:
        self.config = config

    def align(self, reference: Timeline, target: Timeline) -> AlignmentResult:
        if target.modality not in self.config.offsets_seconds:
            raise SynchronizationError(
                f"Available modality '{target.modality}' has no explicit "
                "synchronization.offsets_seconds entry."
            )
        offset = self.config.offsets_seconds[target.modality]
        target_start = target.native_start_seconds + offset
        target_end = target.native_end_seconds + offset
        overlap_start = max(reference.native_start_seconds, target_start)
        overlap_end = min(reference.native_end_seconds, target_end)
        overlap = max(0.0, overlap_end - overlap_start)
        denominator = min(reference.duration_seconds, target.duration_seconds)
        overlap_ratio = overlap / denominator if denominator > 0 else 0.0
        duration_difference = abs(target.duration_seconds - reference.duration_seconds)

        qc_notes: list[str] = []
        if overlap <= 0:
            qc_status = "fail"
            qc_notes.append("Target has no temporal overlap with the reference timeline.")
        else:
            qc_status = "pass"
            if overlap_ratio < self.config.minimum_overlap_ratio:
                qc_status = "warning"
                qc_notes.append(
                    f"Overlap ratio {overlap_ratio:.3f} is below configured minimum "
                    f"{self.config.minimum_overlap_ratio:.3f}."
                )
            if duration_difference > self.config.maximum_duration_difference_s:
                qc_status = "warning"
                qc_notes.append(
                    f"Duration difference {duration_difference:.3f} s exceeds configured "
                    f"maximum {self.config.maximum_duration_difference_s:.3f} s."
                )
        uncertainty = self.config.uncertainty_seconds.get(target.modality)
        if uncertainty is None:
            if qc_status == "pass":
                qc_status = "warning"
            qc_notes.append("Estimated synchronization uncertainty was not provided.")

        return AlignmentResult(
            reference_modality=reference.modality,
            target_modality=target.modality,
            synchronization_method=self.config.method,
            offset_seconds=offset,
            alignment_event="manual_configuration",
            estimated_uncertainty_seconds=uncertainty,
            operator=self.config.operator,
            notes=self.config.notes,
            target_source_path=target.source_path,
            native_start_seconds=target.native_start_seconds,
            native_end_seconds=target.native_end_seconds,
            reference_start_seconds=target_start,
            reference_end_seconds=target_end,
            duration_seconds=target.duration_seconds,
            reference_duration_seconds=reference.duration_seconds,
            duration_difference_seconds=duration_difference,
            overlap_duration_seconds=overlap,
            overlap_ratio=overlap_ratio,
            qc_status=qc_status,
            qc_notes=qc_notes,
        )


def _timeline_from_timestamps(
    path: str | Path,
    modality: str,
    timestamps: pd.Series,
) -> Timeline:
    source = Path(path).expanduser().resolve()
    numeric = pd.to_numeric(timestamps, errors="coerce")
    if numeric.isna().any():
        raise SynchronizationError(f"{modality} contains invalid timestamps: {source}")
    values = numeric.to_numpy(dtype=float)
    if values.size < 2:
        raise SynchronizationError(f"{modality} needs at least two timestamps: {source}")
    if not np.isfinite(values).all():
        raise SynchronizationError(f"{modality} timestamps must be finite: {source}")
    if np.any(np.diff(values) < 0):
        raise SynchronizationError(f"{modality} timestamps are nonmonotonic: {source}")
    start = float(values[0])
    end = float(values[-1])
    if end <= start:
        raise SynchronizationError(f"{modality} timeline duration must be positive: {source}")
    return Timeline(modality, start, end, source.as_posix())


def read_csv_timeline(
    path: str | Path,
    modality: str,
    timestamp_column: str,
) -> tuple[Timeline, pd.DataFrame]:
    """Read a timestamped CSV strictly, retaining the frame for synchronized export."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SynchronizationError(f"{modality} file does not exist: {source}")
    frame = pd.read_csv(source)
    if timestamp_column not in frame:
        raise SynchronizationError(
            f"Missing {modality} timestamp column '{timestamp_column}': {source}"
        )
    timeline = _timeline_from_timestamps(source, modality, frame[timestamp_column])
    return timeline, frame


def read_wav_timeline(path: str | Path, modality: str = "audio") -> Timeline:
    """Read exact duration metadata from a PCM WAV container."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SynchronizationError(f"{modality} file does not exist: {source}")
    try:
        with wave.open(str(source), "rb") as handle:
            rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (OSError, wave.Error) as exc:
        raise SynchronizationError(f"Could not read WAV metadata {source}: {exc}") from exc
    if rate <= 0 or frame_count <= 0:
        raise SynchronizationError(f"WAV has invalid frame count or sampling rate: {source}")
    duration = frame_count / rate
    return Timeline(modality, 0.0, float(duration), source.as_posix())


def read_media_timeline(path: str | Path, modality: str) -> Timeline:
    """Read WAV directly or use ffprobe for other audio/video containers."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SynchronizationError(f"{modality} file does not exist: {source}")
    if source.suffix.lower() == ".wav":
        return read_wav_timeline(source, modality)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise SynchronizationError(
            f"ffprobe is required to read {source.suffix or 'media'} duration for {modality}."
        )
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        raise SynchronizationError(f"Could not read media duration {source}: {exc}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise SynchronizationError(f"Media duration must be positive and finite: {source}")
    return Timeline(modality, 0.0, duration, source.as_posix())
