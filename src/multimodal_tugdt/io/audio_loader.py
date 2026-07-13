"""Waveform loading with WAV-native and optional FFmpeg decoding."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from multimodal_tugdt.config import AudioConfig


class AudioLoadError(ValueError):
    """Raised when an audio file cannot be decoded safely."""


@dataclass(frozen=True)
class AudioData:
    """Mono floating-point waveform normalized to approximately [-1, 1]."""

    samples: np.ndarray
    sampling_rate_hz: int
    original_sampling_rate_hz: int
    source_path: Path

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sampling_rate_hz


def _normalize_pcm(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.integer):
        information = np.iinfo(samples.dtype)
        values = samples.astype(np.float64)
        if information.min == 0:
            midpoint = (information.max + 1) / 2
            values = (values - midpoint) / midpoint
        else:
            values = values / max(abs(information.min), information.max)
    elif np.issubdtype(samples.dtype, np.floating):
        values = samples.astype(np.float64)
    else:
        raise AudioLoadError(f"Unsupported waveform dtype: {samples.dtype}")
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1:
        raise AudioLoadError("Audio must be mono or a two-dimensional channel array.")
    if not np.isfinite(values).all():
        raise AudioLoadError("Audio contains non-finite sample values.")
    return values


def _decode_wav(path: Path) -> tuple[int, np.ndarray]:
    try:
        rate, samples = wavfile.read(path)
    except (OSError, ValueError) as exc:
        raise AudioLoadError(f"Could not decode WAV {path}: {exc}") from exc
    return int(rate), _normalize_pcm(samples)


def _decode_with_ffmpeg(path: Path, target_rate_hz: int) -> tuple[int, np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise AudioLoadError(
            f"ffmpeg and ffprobe are required to decode audio container: {path.suffix}"
        )
    try:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        original_rate = int(probe.stdout.strip())
        completed = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(target_rate_hz),
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (ValueError, subprocess.SubprocessError) as exc:
        raise AudioLoadError(f"Could not decode audio with ffmpeg {path}: {exc}") from exc
    samples = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64)
    return original_rate, samples


def load_audio(path: str | Path, config: AudioConfig) -> AudioData:
    """Decode, downmix, normalize, and resample an audio file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise AudioLoadError(f"Audio file does not exist: {source}")
    if source.suffix.lower() == ".wav":
        original_rate, samples = _decode_wav(source)
        decoded_rate = original_rate
    else:
        original_rate, samples = _decode_with_ffmpeg(
            source, config.target_sampling_rate_hz
        )
        decoded_rate = config.target_sampling_rate_hz
    if original_rate <= 0 or samples.size == 0:
        raise AudioLoadError(f"Audio has no samples or an invalid sampling rate: {source}")
    if decoded_rate != config.target_sampling_rate_hz:
        ratio = Fraction(config.target_sampling_rate_hz, decoded_rate).limit_denominator(1000)
        samples = resample_poly(samples, ratio.numerator, ratio.denominator)
    return AudioData(
        samples=samples,
        sampling_rate_hz=config.target_sampling_rate_hz,
        original_sampling_rate_hz=original_rate,
        source_path=source,
    )
