from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal_tugdt.config import load_config
from multimodal_tugdt.preprocessing.imu import estimate_sampling_rate, preprocess_imu


def _imu_config(tmp_path: Path, overrides: str = ""):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
project:
  root: "."
paths:
  manifest: participants.csv
imu:
  target_sampling_rate_hz: 50
  lowpass_cutoff_hz: 6
  filter_order: 4
  gravity_removal: none
  generate_plots: false
{overrides}
""",
        encoding="utf-8",
    )
    return load_config(path).imu


def test_preprocess_sorts_deduplicates_interpolates_and_normalizes(tmp_path: Path) -> None:
    timestamps = np.arange(0, 2.0, 0.01)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acc_ap": np.sin(2 * np.pi * timestamps),
            "acc_vertical": np.cos(2 * np.pi * timestamps),
            "quat_w": 2.0,
            "quat_x": 0.0,
            "quat_y": 0.0,
            "quat_z": 0.0,
        }
    )
    frame.loc[10, "acc_ap"] = np.nan
    frame = pd.concat([frame.iloc[50:], frame.iloc[:51]], ignore_index=True)

    result = preprocess_imu(frame, _imu_config(tmp_path))

    assert result.quality.timestamps_were_nonmonotonic
    assert result.quality.duplicate_timestamp_count == 1
    assert result.quality.missing_ratio_by_column["acc_ap"] > 0
    assert result.frame["timestamp"].is_monotonic_increasing
    assert estimate_sampling_rate(result.frame["timestamp"]) == pytest.approx(50, rel=1e-3)
    quaternion_norms = np.linalg.norm(
        result.frame[["quat_w", "quat_x", "quat_y", "quat_z"]], axis=1
    )
    assert np.allclose(quaternion_norms, 1.0)


def test_short_signal_does_not_crash_filtering(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [0.00, 0.01, 0.02, 0.03, 0.04],
            "acc_ap": [0.0, 1.0, 0.0, -1.0, 0.0],
        }
    )

    result = preprocess_imu(frame, _imu_config(tmp_path))

    assert len(result.frame) >= 2
    assert any("too short" in warning for warning in result.quality.warnings)


def test_cutoff_must_be_below_target_nyquist(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"timestamp": np.arange(0, 1, 0.01), "acc_ap": np.zeros(100)}
    )
    config = _imu_config(
        tmp_path,
        "  target_sampling_rate_hz: 10\n  lowpass_cutoff_hz: 6",
    )

    with pytest.raises(ValueError, match="target Nyquist"):
        preprocess_imu(frame, config)


def test_unit_conversion_produces_si_values(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(0, 1, 0.01),
            "acc_ap": np.ones(100),
            "gyro_yaw": np.full(100, 180.0),
        }
    )
    config = _imu_config(
        tmp_path,
        "  input_acceleration_unit: g\n  input_angular_velocity_unit: deg/s",
    )

    result = preprocess_imu(frame, config)

    assert result.frame["acc_ap"].mean() == pytest.approx(9.80665, rel=1e-3)
    assert result.frame["gyro_yaw"].mean() == pytest.approx(np.pi, rel=1e-3)

