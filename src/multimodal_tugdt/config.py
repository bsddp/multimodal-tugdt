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
class AudioConfig:
    """Settings for waveform normalization and interpretable energy VAD."""

    target_sampling_rate_hz: int
    frame_duration_ms: int
    energy_threshold_dbfs: float
    minimum_speech_duration_s: float
    minimum_pause_duration_s: float
    task_start_seconds: float | None
    clipping_threshold: float


@dataclass(frozen=True)
class FootswitchConfig:
    """Settings for binary contact stabilization and gait-event comparison."""

    timestamp_column: str
    left_contact_column: str
    right_contact_column: str
    threshold: float
    minimum_contact_duration_s: float
    minimum_swing_duration_s: float
    event_matching_tolerance_s: float


@dataclass(frozen=True)
class VideoConfig:
    """Settings for video inspection and optional MediaPipe pose extraction."""

    enable_pose_estimation: bool
    pose_backend: str
    pose_model_path: str | None
    frame_step: int
    minimum_visibility: float
    minimum_pose_detection_confidence: float
    minimum_pose_presence_confidence: float
    minimum_tracking_confidence: float


@dataclass(frozen=True)
class FusionConfig:
    """Settings for trial-level feature-table fusion and modality comparisons."""

    modalities: tuple[str, ...]
    feature_sets: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ModelingConfig:
    """Leakage-aware participant-grouped baseline evaluation settings."""

    enabled: bool
    target_column: str
    task_type: str
    group_column: str
    folds: int
    random_seed: int
    models: tuple[str, ...]
    cohort_modes: tuple[str, ...]
    positive_label: Any | None


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
    audio: AudioConfig
    footswitch: FootswitchConfig
    video: VideoConfig
    fusion: FusionConfig
    modeling: ModelingConfig
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
        raise ConfigurationError("'imu.input_angular_velocity_unit' must be 'rad/s' or 'deg/s'.")
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
        raise ConfigurationError("'synchronization.minimum_overlap_ratio' must be between 0 and 1.")
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


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"'{field}' must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"'{field}' must be finite.")
    return number


def _load_audio_config(root: dict[str, Any]) -> AudioConfig:
    values = _mapping(root.get("audio", {}), "audio")
    threshold_dbfs = _finite_float(
        values.get("energy_threshold_dbfs", -35),
        "audio.energy_threshold_dbfs",
    )
    if threshold_dbfs > 0:
        raise ConfigurationError("'audio.energy_threshold_dbfs' must be at most 0 dBFS.")
    task_start_value = values.get("task_start_seconds")
    task_start = (
        None
        if task_start_value is None
        else _finite_float(task_start_value, "audio.task_start_seconds")
    )
    clipping_threshold = _finite_float(
        values.get("clipping_threshold", 0.999),
        "audio.clipping_threshold",
    )
    if not 0 < clipping_threshold <= 1:
        raise ConfigurationError("'audio.clipping_threshold' must be in (0, 1].")
    return AudioConfig(
        target_sampling_rate_hz=_positive_int(
            values.get("target_sampling_rate_hz", 16_000),
            "audio.target_sampling_rate_hz",
        ),
        frame_duration_ms=_positive_int(
            values.get("frame_duration_ms", 30),
            "audio.frame_duration_ms",
        ),
        energy_threshold_dbfs=threshold_dbfs,
        minimum_speech_duration_s=_positive_float(
            values.get("minimum_speech_duration_s", 0.2),
            "audio.minimum_speech_duration_s",
        ),
        minimum_pause_duration_s=_positive_float(
            values.get("minimum_pause_duration_s", 0.2),
            "audio.minimum_pause_duration_s",
        ),
        task_start_seconds=task_start,
        clipping_threshold=clipping_threshold,
    )


def _load_footswitch_config(root: dict[str, Any]) -> FootswitchConfig:
    values = _mapping(root.get("footswitch", {}), "footswitch")
    column_values = {
        "timestamp_column": values.get("timestamp_column", "timestamp"),
        "left_contact_column": values.get("left_contact_column", "left_contact"),
        "right_contact_column": values.get("right_contact_column", "right_contact"),
    }
    if not all(isinstance(value, str) and value.strip() for value in column_values.values()):
        raise ConfigurationError("Footswitch column names must be non-empty strings.")
    return FootswitchConfig(
        **column_values,
        threshold=_finite_float(values.get("threshold", 0.5), "footswitch.threshold"),
        minimum_contact_duration_s=_positive_float(
            values.get("minimum_contact_duration_s", 0.08),
            "footswitch.minimum_contact_duration_s",
        ),
        minimum_swing_duration_s=_positive_float(
            values.get("minimum_swing_duration_s", 0.08),
            "footswitch.minimum_swing_duration_s",
        ),
        event_matching_tolerance_s=_positive_float(
            values.get("event_matching_tolerance_s", 0.15),
            "footswitch.event_matching_tolerance_s",
        ),
    )


def _probability(value: Any, field: str) -> float:
    number = _finite_float(value, field)
    if not 0 <= number <= 1:
        raise ConfigurationError(f"'{field}' must be between 0 and 1.")
    return number


def _load_video_config(root: dict[str, Any]) -> VideoConfig:
    values = _mapping(root.get("video", {}), "video")
    enabled = values.get("enable_pose_estimation", False)
    if not isinstance(enabled, bool):
        raise ConfigurationError("'video.enable_pose_estimation' must be true or false.")
    backend = values.get("pose_backend", "mediapipe")
    if backend != "mediapipe":
        raise ConfigurationError("'video.pose_backend' currently supports only 'mediapipe'.")
    model_path = values.get("pose_model_path")
    if model_path is not None and (not isinstance(model_path, str) or not model_path.strip()):
        raise ConfigurationError("'video.pose_model_path' must be a non-empty path or null.")
    if enabled and model_path is None:
        raise ConfigurationError(
            "'video.pose_model_path' is required when pose estimation is enabled."
        )
    return VideoConfig(
        enable_pose_estimation=enabled,
        pose_backend=backend,
        pose_model_path=model_path,
        frame_step=_positive_int(values.get("frame_step", 1), "video.frame_step"),
        minimum_visibility=_probability(
            values.get("minimum_visibility", 0.5),
            "video.minimum_visibility",
        ),
        minimum_pose_detection_confidence=_probability(
            values.get("minimum_pose_detection_confidence", 0.5),
            "video.minimum_pose_detection_confidence",
        ),
        minimum_pose_presence_confidence=_probability(
            values.get("minimum_pose_presence_confidence", 0.5),
            "video.minimum_pose_presence_confidence",
        ),
        minimum_tracking_confidence=_probability(
            values.get("minimum_tracking_confidence", 0.5),
            "video.minimum_tracking_confidence",
        ),
    )


SUPPORTED_FUSION_MODALITIES = ("imu", "audio", "video", "footswitch", "clinical")


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"'{field}' must be a non-empty list.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigurationError(f"'{field}' entries must be non-empty strings.")
    if len(set(value)) != len(value):
        raise ConfigurationError(f"'{field}' must not contain duplicates.")
    return tuple(value)


def _load_fusion_config(root: dict[str, Any]) -> FusionConfig:
    values = _mapping(root.get("fusion", {}), "fusion")
    modalities = _string_list(
        values.get("modalities", list(SUPPORTED_FUSION_MODALITIES)),
        "fusion.modalities",
    )
    unsupported = sorted(set(modalities) - set(SUPPORTED_FUSION_MODALITIES))
    if unsupported:
        raise ConfigurationError(
            "'fusion.modalities' contains unsupported values: " + ", ".join(unsupported)
        )
    default_sets = {modality: [modality] for modality in modalities}
    default_sets.update(
        {
            "imu_clinical": ["imu", "clinical"],
            "imu_audio": ["imu", "audio"],
            "imu_video": ["imu", "video"],
            "imu_audio_video": ["imu", "audio", "video"],
            "all_available": list(modalities),
        }
    )
    raw_sets = _mapping(values.get("feature_sets", default_sets), "fusion.feature_sets")
    feature_sets: dict[str, tuple[str, ...]] = {}
    for name, raw_modalities in raw_sets.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("'fusion.feature_sets' names must be non-empty strings.")
        selected = _string_list(raw_modalities, f"fusion.feature_sets.{name}")
        unknown = sorted(set(selected) - set(modalities))
        if unknown:
            raise ConfigurationError(
                f"'fusion.feature_sets.{name}' references disabled modalities: "
                + ", ".join(unknown)
            )
        feature_sets[name] = selected
    return FusionConfig(modalities=modalities, feature_sets=feature_sets)


def _load_modeling_config(root: dict[str, Any]) -> ModelingConfig:
    values = _mapping(root.get("modeling", {}), "modeling")
    enabled = values.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigurationError("'modeling.enabled' must be true or false.")
    target = values.get("target_column", "clinical__moca")
    if not isinstance(target, str) or not target.strip():
        raise ConfigurationError("'modeling.target_column' must be a non-empty string.")
    task_type = values.get("task_type", "regression")
    if task_type not in {"classification", "regression"}:
        raise ConfigurationError("'modeling.task_type' must be classification or regression.")
    group_column = values.get("group_column", "participant_id")
    if group_column != "participant_id":
        raise ConfigurationError(
            "Milestone 6 requires 'modeling.group_column: participant_id' to prevent leakage."
        )
    default_models = (
        ["logistic_regression", "random_forest"]
        if task_type == "classification"
        else ["ridge", "random_forest"]
    )
    models = _string_list(values.get("models", default_models), "modeling.models")
    supported_models = (
        {"logistic_regression", "random_forest"}
        if task_type == "classification"
        else {"linear_regression", "ridge", "random_forest"}
    )
    unsupported = sorted(set(models) - supported_models)
    if unsupported:
        raise ConfigurationError(f"Unsupported {task_type} models: " + ", ".join(unsupported))
    cohort_modes = _string_list(
        values.get("cohort_modes", ["all_samples", "complete_modalities"]),
        "modeling.cohort_modes",
    )
    invalid_cohorts = sorted(set(cohort_modes) - {"all_samples", "complete_modalities"})
    if invalid_cohorts:
        raise ConfigurationError("Unsupported cohort modes: " + ", ".join(invalid_cohorts))
    seed = values.get("random_seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("'modeling.random_seed' must be an integer.")
    positive_label = values.get("positive_label")
    if task_type == "classification" and positive_label is None:
        raise ConfigurationError("'modeling.positive_label' must be declared for classification.")
    return ModelingConfig(
        enabled=enabled,
        target_column=target,
        task_type=task_type,
        group_column=group_column,
        folds=_positive_int(values.get("folds", 5), "modeling.folds"),
        random_seed=seed,
        models=models,
        cohort_modes=cohort_modes,
        positive_label=positive_label,
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
        audio=_load_audio_config(root),
        footswitch=_load_footswitch_config(root),
        video=_load_video_config(root),
        fusion=_load_fusion_config(root),
        modeling=_load_modeling_config(root),
        values=root,
    )
