from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal_tugdt.config import SynchronizationConfig
from multimodal_tugdt.synchronization.timeline import (
    ManualOffsetSynchronizer,
    SynchronizationError,
    Timeline,
    apply_offset_to_timestamps,
    read_csv_timeline,
)


def _config(
    *,
    offsets: dict[str, float],
    uncertainties: dict[str, float] | None = None,
    maximum_duration_difference_s: float = 0.5,
    minimum_overlap_ratio: float = 0.9,
) -> SynchronizationConfig:
    return SynchronizationConfig(
        reference_modality="imu",
        method="manual_offset",
        offsets_seconds=offsets,
        uncertainty_seconds=uncertainties or {},
        timestamp_columns={"footswitch": "timestamp"},
        operator="tester",
        notes="known test offset",
        maximum_duration_difference_s=maximum_duration_difference_s,
        minimum_overlap_ratio=minimum_overlap_ratio,
        generate_plots=False,
    )


def test_manual_offset_maps_target_to_reference_clock() -> None:
    reference = Timeline("imu", 0.0, 20.0, "imu.csv")
    target = Timeline("audio", 0.0, 20.0, "audio.wav")

    result = ManualOffsetSynchronizer(
        _config(offsets={"audio": 1.5}, uncertainties={"audio": 0.05})
    ).align(reference, target)

    assert result.reference_start_seconds == pytest.approx(1.5)
    assert result.reference_end_seconds == pytest.approx(21.5)
    assert result.overlap_duration_seconds == pytest.approx(18.5)
    assert result.overlap_ratio == pytest.approx(0.925)
    assert result.qc_status == "pass"
    assert result.estimated_uncertainty_seconds == pytest.approx(0.05)


def test_offset_application_supports_negative_values() -> None:
    mapped = apply_offset_to_timestamps(pd.Series([0.0, 0.5, 1.0]), -0.2)
    assert np.allclose(mapped, [-0.2, 0.3, 0.8])


def test_available_modality_requires_explicit_offset() -> None:
    reference = Timeline("imu", 0.0, 10.0, "imu.csv")
    target = Timeline("footswitch", 0.0, 10.0, "footswitch.csv")
    with pytest.raises(SynchronizationError, match="no explicit"):
        ManualOffsetSynchronizer(_config(offsets={})).align(reference, target)


def test_low_overlap_and_duration_difference_produce_warning() -> None:
    reference = Timeline("imu", 0.0, 20.0, "imu.csv")
    target = Timeline("audio", 0.0, 18.0, "audio.wav")
    result = ManualOffsetSynchronizer(
        _config(
            offsets={"audio": 3.0},
            uncertainties={"audio": 0.1},
            maximum_duration_difference_s=0.5,
            minimum_overlap_ratio=0.95,
        )
    ).align(reference, target)

    assert result.qc_status == "warning"
    assert len(result.qc_notes) == 2


def test_csv_timeline_rejects_nonmonotonic_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "footswitch.csv"
    pd.DataFrame({"timestamp": [0.0, 0.2, 0.1], "left": [0, 1, 0]}).to_csv(
        path, index=False
    )
    with pytest.raises(SynchronizationError, match="nonmonotonic"):
        read_csv_timeline(path, "footswitch", "timestamp")

