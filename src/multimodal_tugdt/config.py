"""Typed project configuration loading and path resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a project configuration is missing or malformed."""


DEFAULT_IMU_COLUMNS = {
    "timestamp": "timestamp",
    "acc_ap": "pelvis_acc_ap",
    "acc_ml": "pelvis_acc_ml",
    "acc_vertical": "pelvis_acc_vertical",
    "gyro_yaw": "pelvis_gyro_yaw",
    "quat_w": "quat_w",
    "quat_x": "quat_x",
    "quat_y": "quat_y",
    "quat_z": "quat_z",
}


@dataclass(frozen=True)
class IMUConfig:
    """Validated settings for IMU ingestion, preprocessing, and basic features."""

    format: str
    target_sensor: str
    columns: dict[str, str]
    target_sampling_rate_hz: float
    lowpass_cutoff_hz: float
    filter_order: int
    input_acceleration_unit: str
    input_angular_velocity_unit: str
    gravity_removal: str
    gravity_value_m_s2: float
    maximum_abs_acceleration_m_s2: float
    maximum_abs_angular_velocity_rad_s: float
    step_min_interval_s: float
    step_prominence: float
    generate_plots: bool


@dataclass(frozen=True)
class SynchronizationConfig:
    """Explicit alignment settings for mapping target clocks to a reference clock."""

    reference_modality: str
    method: str
    offsets_seconds: dict[str, float]
    uncertainty_seconds: dict[str, float]
    timestamp_columns: dict[str, str]
    operator: str
    notes: str
    maximum_duration_difference_s: float
    minimum_overlap_ratio: float
    generate_plots: bool


@dataclass(frozen=True)
class ProjectConfig:
    """Validated configuration plus reproducible path-resolution rules."""

    source: Path
    project_root: Path
    manifest_path: Path
    interim_dir: Path
    processed_dir: Path
    output_dir: Path
    allowed_conditions: tuple[str, ...]
    imu: IMUConfig
    synchronization: SynchronizationConfig
    values: dict[str, Any]

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a manifest or config path against the configured project root."""
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section '{section}' must be a mapping.")
    return value


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ConfigurationError(f"'{field}' must be a positive number.")
    return float(value)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"'{field}' must be a positive integer.")
    return value


def _nonnegative_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ConfigurationError(f"'{field}' must be a nonnegative number.")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"'{field}' must be finite.")
    return number


def _load_imu_config(root: dict[str, Any]) -> IMUConfig:
    values = _mapping(root.get("imu", {}), "imu")
    columns = _mapping(values.get("columns", DEFAULT_IMU_COLUMNS), "imu.columns")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in columns.items()):
        raise ConfigurationError("'imu.columns' keys and values must be strings.")
    if "timestamp" not in columns:
        raise ConfigurationError("'imu.columns' must define a timestamp mapping.")

    data_format = values.get("format", "wide_csv")
    if data_format not in {"wide_csv", "long_csv", "mvnx"}:
        raise ConfigurationError("'imu.format' must be wide_csv, long_csv, or mvnx.")
    acceleration_unit = values.get("input_acceleration_unit", "m/s^2")
    if acceleration_unit not in {"m/s^2", "g"}:
        raise ConfigurationError("'imu.input_acceleration_unit' must be 'm/s^2' or 'g'.")
    angular_unit = values.get("input_angular_velocity_unit", "rad/s")
    if angular_unit not in {"rad/s", "deg/s"}:
        raise ConfigurationError(
            "'imu.input_angular_velocity_unit' must be 'rad/s' or 'deg/s'."
        )
    gravity_removal = values.get("gravity_removal", "none")
    if gravity_removal not in {"none", "constant"}:
        raise ConfigurationError("'imu.gravity_removal' must be 'none' or 'constant'.")

    generate_plots = values.get("generate_plots", True)
    if not isinstance(generate_plots, bool):
        raise ConfigurationError("'imu.generate_plots' must be true or false.")

    target_sensor = values.get("target_sensor", "pelvis")
    if not isinstance(target_sensor, str) or not target_sensor.strip():
        raise ConfigurationError("'imu.target_sensor' must be a non-empty string.")

    return IMUConfig(
        format=data_format,
        target_sensor=target_sensor,
        columns=dict(columns),
        target_sampling_rate_hz=_positive_float(
            values.get("target_sampling_rate_hz", 100),
            "imu.target_sampling_rate_hz",
        ),
        lowpass_cutoff_hz=_positive_float(
            values.get("lowpass_cutoff_hz", 6),
            "imu.lowpass_cutoff_hz",
        ),
        filter_order=_positive_int(values.get("filter_order", 4), "imu.filter_order"),
        input_acceleration_unit=acceleration_unit,
        input_angular_velocity_unit=angular_unit,
        gravity_removal=gravity_removal,
        gravity_value_m_s2=_positive_float(
            values.get("gravity_value_m_s2", 9.80665),
            "imu.gravity_value_m_s2",
        ),
        maximum_abs_acceleration_m_s2=_positive_float(
            values.get("maximum_abs_acceleration_m_s2", 50),
            "imu.maximum_abs_acceleration_m_s2",
        ),
        maximum_abs_angular_velocity_rad_s=_positive_float(
            values.get("maximum_abs_angular_velocity_rad_s", 20),
            "imu.maximum_abs_angular_velocity_rad_s",
        ),
        step_min_interval_s=_positive_float(
            values.get("step_min_interval_s", 0.35),
            "imu.step_min_interval_s",
        ),
        step_prominence=_positive_float(
            values.get("step_prominence", 0.15),
            "imu.step_prominence",
        ),
        generate_plots=generate_plots,
    )


def _numeric_mapping(value: Any, field: str, *, nonnegative: bool) -> dict[str, float]:
    mapping = _mapping(value, field)
    result: dict[str, float] = {}
    for key, raw_value in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigurationError(f"'{field}' keys must be non-empty strings.")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ConfigurationError(f"'{field}.{key}' must be numeric.")
        number = float(raw_value)
        if not math.isfinite(number) or (nonnegative and number < 0):
            qualifier = "finite and nonnegative" if nonnegative else "finite"
            raise ConfigurationError(f"'{field}.{key}' must be {qualifier}.")
        result[key] = number
    return result


def _load_synchronization_config(root: dict[str, Any]) -> SynchronizationConfig:
    values = _mapping(root.get("synchronization", {}), "synchronization")
    reference = values.get("reference_modality", "imu")
    if reference != "imu":
        raise ConfigurationError(
            "Milestone 3 currently requires 'synchronization.reference_modality: imu'."
        )
    method = values.get("method", "manual_offset")
    if method != "manual_offset":
        raise ConfigurationError(
            "Milestone 3 currently supports only 'synchronization.method: manual_offset'."
        )
    timestamp_columns = _mapping(
        values.get("timestamp_columns", {"footswitch": "timestamp"}),
        "synchronization.timestamp_columns",
    )
    if not all(
        isinstance(key, str) and isinstance(column, str) and column.strip()
        for key, column in timestamp_columns.items()
    ):
        raise ConfigurationError(
            "'synchronization.timestamp_columns' keys and values must be non-empty strings."
        )
    operator = values.get("operator", "not_specified")
    notes = values.get("notes", "")
    if not isinstance(operator, str) or not operator.strip():
        raise ConfigurationError("'synchronization.operator' must be a non-empty string.")
    if not isinstance(notes, str):
        raise ConfigurationError("'synchronization.notes' must be a string.")
    minimum_overlap = values.get("minimum_overlap_ratio", 0.9)
    if (
        isinstance(minimum_overlap, bool)
        or not isinstance(minimum_overlap, int | float)
        or not 0 <= minimum_overlap <= 1
    ):
        raise ConfigurationError(
            "'synchronization.minimum_overlap_ratio' must be between 0 and 1."
        )
    generate_plots = values.get("generate_plots", True)
    if not isinstance(generate_plots, bool):
        raise ConfigurationError("'synchronization.generate_plots' must be true or false.")
    return SynchronizationConfig(
        reference_modality=reference,
        method=method,
        offsets_seconds=_numeric_mapping(
            values.get("offsets_seconds", {}),
            "synchronization.offsets_seconds",
            nonnegative=False,
        ),
        uncertainty_seconds=_numeric_mapping(
            values.get("uncertainty_seconds", {}),
            "synchronization.uncertainty_seconds",
            nonnegative=True,
        ),
        timestamp_columns=dict(timestamp_columns),
        operator=operator,
        notes=notes,
        maximum_duration_difference_s=_nonnegative_float(
            values.get("maximum_duration_difference_s", 0.5),
            "synchronization.maximum_duration_difference_s",
        ),
        minimum_overlap_ratio=float(minimum_overlap),
        generate_plots=generate_plots,
    )


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate project, path, study, and IMU settings."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {source}")

    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {source}: {exc}") from exc

    root = _mapping(raw, "root")
    project = _mapping(root.get("project", {}), "project")
    paths = _mapping(root.get("paths"), "paths")
    study = _mapping(root.get("study", {}), "study")

    manifest_value = paths.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ConfigurationError("'paths.manifest' must be a non-empty path string.")

    root_value = project.get("root", "..")
    if not isinstance(root_value, str):
        raise ConfigurationError("'project.root' must be a path string.")
    project_root = (source.parent / root_value).resolve()

    def project_path(key: str, default: str) -> Path:
        value = paths.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"'paths.{key}' must be a non-empty path string.")
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (project_root / candidate).resolve()

    allowed = study.get(
        "allowed_conditions",
        ["single_task", "dual_task", "motor_only", "cognitive_only"],
    )
    if not isinstance(allowed, list) or not allowed or not all(isinstance(x, str) for x in allowed):
        raise ConfigurationError("'study.allowed_conditions' must be a non-empty string list.")

    return ProjectConfig(
        source=source,
        project_root=project_root,
        manifest_path=project_path("manifest", "data/metadata/participants.csv"),
        interim_dir=project_path("interim_dir", "data/interim"),
        processed_dir=project_path("processed_dir", "data/processed"),
        output_dir=project_path("output_dir", "outputs"),
        allowed_conditions=tuple(allowed),
        imu=_load_imu_config(root),
        synchronization=_load_synchronization_config(root),
        values=root,
    )
