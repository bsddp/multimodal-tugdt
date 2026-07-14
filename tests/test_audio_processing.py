from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from multimodal_tugdt.config import AudioConfig
from multimodal_tugdt.features.audio_features import (
    extract_trial_and_phase_audio_features,
)
from multimodal_tugdt.io.audio_loader import load_audio
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.preprocessing.audio import run_energy_vad
from multimodal_tugdt.segmentation.manual import Segment


def _config() -> AudioConfig:
    return AudioConfig(
        target_sampling_rate_hz=16_000,
        frame_duration_ms=20,
        energy_threshold_dbfs=-35,
        minimum_speech_duration_s=0.2,
        minimum_pause_duration_s=0.2,
        task_start_seconds=0.0,
        clipping_threshold=0.999,
    )


def _write_test_wav(path: Path) -> None:
    rate = 8_000
    time = np.arange(3 * rate) / rate
    active = ((time >= 0.5) & (time < 1.0)) | ((time >= 2.0) & (time < 2.5))
    samples = np.where(active, 0.2 * np.sin(2 * np.pi * 220 * time), 0.0)
    wavfile.write(path, rate, np.asarray(samples * 32767, dtype=np.int16))


def test_audio_loader_resamples_and_energy_vad_finds_speech(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    _write_test_wav(path)

    audio = load_audio(path, _config())
    vad = run_energy_vad(audio, _config())

    assert audio.original_sampling_rate_hz == 8_000
    assert audio.sampling_rate_hz == 16_000
    assert audio.duration_seconds == pytest.approx(3.0, rel=1e-3)
    assert vad.quality.speech_segment_count == 2
    assert vad.quality.speech_duration_seconds == pytest.approx(1.0, abs=0.05)
    assert vad.quality.clipping_ratio == 0


def test_audio_features_leave_transcript_outcomes_blank(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    _write_test_wav(path)
    vad = run_energy_vad(load_audio(path, _config()), _config())
    activity = pd.DataFrame(
        {
            "activity": [item.activity for item in vad.intervals],
            "start_time": [item.start_seconds for item in vad.intervals],
            "end_time": [item.end_seconds for item in vad.intervals],
        }
    )
    record = TrialRecord("P001", "S01", "dual_task", "T01", {})
    rows = extract_trial_and_phase_audio_features(
        record,
        activity,
        [Segment("outbound_walk", 0.0, 1.5)],
        _config(),
        trial_start=0.0,
        trial_end=3.0,
    )
    trial = rows[0]

    assert trial["audio__speech_segment_count"] == 2
    assert trial["audio__pause_count"] == 1
    assert trial["audio__mean_pause_duration_s"] == pytest.approx(1.0, abs=0.05)
    assert trial["audio__first_response_latency_s"] == pytest.approx(0.5, abs=0.05)
    assert np.isnan(trial["audio__response_accuracy"])
    assert np.isnan(trial["audio__correct_response_count"])
