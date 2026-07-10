"""Safe, configurable preprocessing for canonical IMU signals."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from multimodal_tugdt.config import IMUConfig

LOGGER = logging.getLogger(__name__)
ACCELERATION_COLUMNS = ("acc_ap", "acc_ml", "acc_vertical")
ANGULAR_VELOCITY_COLUMNS = ("gyro_x", "gyro_y", "gyro_z", "gyro_yaw")
QUATERNION_COLUMNS = ("quat_w", "quat_x", "quat_y", "quat_z")


@dataclass
class IMUQualityReport:
    """Machine-readable evidence about an input and its preprocessing decisions."""

    input_sample_count: int
    output_sample_count: int = 0
    invalid_timestamp_count: int = 0
    duplicate_timestamp_count: int = 0
    timestamps_were_nonmonotonic: bool = False
    estimated_input_rate_hz: float | None = None
    output_rate_hz: float | None = None
    sampling_interval_cv_pct: float | None = None
    missing_ratio_by_column: dict[str, float] = field(default_factory=dict)
    anomaly_ratio_by_column: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable QC metadata."""
        return asdict(self)


@dataclass(frozen=True)
class IMUPreprocessResult:
    """Uniform, SI-unit IMU signals paired with their QC evidence."""

    frame: pd.DataFrame
    quality: IMUQualityReport


def estimate_sampling_rate(timestamps: pd.Series | np.ndarray) -> float:
    """Estimate sample rate from the median positive timestamp interval."""
    values = np.asarray(timestamps, dtype=float)
    differences = np.diff(values)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if positive.size == 0:
        raise ValueError("At least two increasing timestamps are required.")
    return float(1.0 / np.median(positive))


def _sampling_interval_cv(timestamps: np.ndarray) -> float | None:
    differences = np.diff(timestamps)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if positive.size < 2 or np.mean(positive) == 0:
        return None
    return float(np.std(positive, ddof=1) / np.mean(positive) * 100)


def _interpolate_missing(frame: pd.DataFrame, quality: IMUQualityReport) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column == "timestamp":
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].notna().any():
            result[column] = result[column].interpolate(limit_direction="both")
        else:
            message = f"Signal '{column}' is entirely missing and was retained as NaN."
            quality.warnings.append(message)
            LOGGER.warning(message)
    return result


def _standardize_units(frame: pd.DataFrame, config: IMUConfig) -> pd.DataFrame:
    result = frame.copy()
    if config.input_acceleration_unit == "g":
        for column in ACCELERATION_COLUMNS:
            if column in result:
                result[column] = result[column] * config.gravity_value_m_s2
    if config.input_angular_velocity_unit == "deg/s":
        for column in ANGULAR_VELOCITY_COLUMNS:
            if column in result:
                result[column] = np.deg2rad(result[column])
    if config.gravity_removal == "constant" and "acc_vertical" in result:
        result["acc_vertical"] = result["acc_vertical"] - config.gravity_value_m_s2
    return result


def _anomaly_ratios(frame: pd.DataFrame, config: IMUConfig) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for column in ACCELERATION_COLUMNS:
        if column in frame:
            ratios[column] = float(
                (frame[column].abs() > config.maximum_abs_acceleration_m_s2).mean()
            )
    for column in ANGULAR_VELOCITY_COLUMNS:
        if column in frame:
            ratios[column] = float(
                (frame[column].abs() > config.maximum_abs_angular_velocity_rad_s).mean()
            )
    return ratios


def _lowpass_series(
    values: pd.Series,
    *,
    sampling_rate_hz: float,
    cutoff_hz: float,
    order: int,
    column: str,
    quality: IMUQualityReport,
) -> pd.Series:
    if cutoff_hz >= sampling_rate_hz / 2:
        raise ValueError(
            f"Low-pass cutoff {cutoff_hz:g} Hz must be below the input Nyquist frequency "
            f"({sampling_rate_hz / 2:g} Hz) for '{column}'."
        )
    if values.isna().all():
        return values
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sampling_rate_hz, output="sos")
    try:
        filtered = sosfiltfilt(sos, values.to_numpy(dtype=float))
    except ValueError:
        message = (
            f"Signal '{column}' is too short for zero-phase filtering; "
            "the interpolated unfiltered values were retained."
        )
        quality.warnings.append(message)
        LOGGER.warning(message)
        return values.copy()
    return pd.Series(filtered, index=values.index, name=values.name)


def _resample_uniform(frame: pd.DataFrame, target_rate_hz: float) -> pd.DataFrame:
    start = float(frame["timestamp"].iloc[0])
    end = float(frame["timestamp"].iloc[-1])
    step = 1.0 / target_rate_hz
    target_time = np.arange(start, end + step * 0.5, step)
    target_time = target_time[target_time <= end + 1e-9]
    result = {"timestamp": target_time}
    source_time = frame["timestamp"].to_numpy(dtype=float)
    for column in frame.columns:
        if column == "timestamp":
            continue
        values = frame[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            result[column] = np.full(target_time.shape, np.nan)
        elif finite.sum() == 1:
            result[column] = np.full(target_time.shape, values[finite][0])
        else:
            result[column] = np.interp(target_time, source_time[finite], values[finite])
    return pd.DataFrame(result)


def _normalize_quaternions(frame: pd.DataFrame, quality: IMUQualityReport) -> pd.DataFrame:
    present = [column for column in QUATERNION_COLUMNS if column in frame]
    if not present:
        return frame
    if len(present) != len(QUATERNION_COLUMNS):
        message = "Incomplete quaternion columns were not normalized."
        quality.warnings.append(message)
        LOGGER.warning(message)
        return frame
    result = frame.copy()
    values = result[list(QUATERNION_COLUMNS)].to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1)
    invalid = ~np.isfinite(norms) | (norms <= np.finfo(float).eps)
    norms[invalid] = np.nan
    result.loc[:, list(QUATERNION_COLUMNS)] = values / norms[:, None]
    if invalid.any():
        message = f"{int(invalid.sum())} quaternion row(s) had zero or invalid norm."
        quality.warnings.append(message)
        LOGGER.warning(message)
    return result


def preprocess_imu(frame: pd.DataFrame, config: IMUConfig) -> IMUPreprocessResult:
    """Validate, clean, convert, filter, and resample canonical IMU signals."""
    if "timestamp" not in frame:
        raise ValueError("Canonical IMU data must contain a 'timestamp' column.")
    quality = IMUQualityReport(input_sample_count=len(frame))
    working = frame.copy()
    timestamp_numeric = pd.to_numeric(working["timestamp"], errors="coerce")
    quality.invalid_timestamp_count = int(timestamp_numeric.isna().sum())
    working["timestamp"] = timestamp_numeric
    working = working.dropna(subset=["timestamp"])
    if len(working) < 2:
        raise ValueError("IMU data must contain at least two valid timestamps.")

    original_timestamps = working["timestamp"].to_numpy(dtype=float)
    quality.timestamps_were_nonmonotonic = bool(np.any(np.diff(original_timestamps) < 0))
    quality.duplicate_timestamp_count = int(working["timestamp"].duplicated().sum())
    if quality.timestamps_were_nonmonotonic:
        quality.warnings.append("Input timestamps were nonmonotonic and were sorted.")
    if quality.duplicate_timestamp_count:
        quality.warnings.append(
            f"Dropped {quality.duplicate_timestamp_count} duplicate timestamp row(s)."
        )
    working = working.sort_values("timestamp", kind="stable").drop_duplicates(
        "timestamp", keep="first"
    )
    if len(working) < 2:
        raise ValueError("IMU data must contain at least two unique timestamps.")

    for column in working.columns:
        if column != "timestamp":
            working[column] = pd.to_numeric(working[column], errors="coerce")
    quality.missing_ratio_by_column = {
        column: float(working[column].isna().mean())
        for column in working.columns
        if column != "timestamp"
    }
    working = _interpolate_missing(working, quality)
    working = _standardize_units(working, config)
    quality.anomaly_ratio_by_column = _anomaly_ratios(working, config)

    timestamps = working["timestamp"].to_numpy(dtype=float)
    input_rate = estimate_sampling_rate(timestamps)
    quality.estimated_input_rate_hz = input_rate
    quality.sampling_interval_cv_pct = _sampling_interval_cv(timestamps)
    if config.lowpass_cutoff_hz >= config.target_sampling_rate_hz / 2:
        raise ValueError(
            f"Low-pass cutoff {config.lowpass_cutoff_hz:g} Hz must be below the target "
            f"Nyquist frequency ({config.target_sampling_rate_hz / 2:g} Hz)."
        )

    filter_columns = [
        column
        for column in (*ACCELERATION_COLUMNS, *ANGULAR_VELOCITY_COLUMNS)
        if column in working
    ]
    for column in filter_columns:
        working[column] = _lowpass_series(
            working[column],
            sampling_rate_hz=input_rate,
            cutoff_hz=config.lowpass_cutoff_hz,
            order=config.filter_order,
            column=column,
            quality=quality,
        )

    resampled = _resample_uniform(working, config.target_sampling_rate_hz)
    resampled = _normalize_quaternions(resampled, quality)
    if not math.isclose(input_rate, config.target_sampling_rate_hz, rel_tol=0.01):
        quality.warnings.append(
            f"Resampled from approximately {input_rate:.3f} Hz "
            f"to {config.target_sampling_rate_hz:.3f} Hz."
        )
    quality.output_sample_count = len(resampled)
    quality.output_rate_hz = estimate_sampling_rate(resampled["timestamp"])
    return IMUPreprocessResult(frame=resampled, quality=quality)

