"""Configurable adapters for IMU CSV and future MVNX inputs."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from multimodal_tugdt.config import IMUConfig

LOGGER = logging.getLogger(__name__)


class IMULoadError(ValueError):
    """Raised when an IMU source cannot satisfy the canonical data contract."""


class BaseIMULoader(ABC):
    """Abstract adapter returning canonical, unprocessed IMU columns."""

    def __init__(self, config: IMUConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self, path: str | Path) -> pd.DataFrame:
        """Load a source into a canonical DataFrame."""

    @staticmethod
    def _read_csv(path: str | Path) -> tuple[Path, pd.DataFrame]:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise IMULoadError(f"IMU file does not exist: {source}")
        try:
            return source, pd.read_csv(source)
        except (OSError, pd.errors.ParserError) as exc:
            raise IMULoadError(f"Could not read IMU CSV {source}: {exc}") from exc


class WideCSVLoader(BaseIMULoader):
    """Map one-column-per-signal CSV files to canonical signal names."""

    def load(self, path: str | Path) -> pd.DataFrame:
        source, frame = self._read_csv(path)
        timestamp_source = self.config.columns["timestamp"]
        if timestamp_source not in frame.columns:
            raise IMULoadError(
                f"Missing timestamp column '{timestamp_source}' in IMU file: {source}"
            )

        available = {
            canonical: source_column
            for canonical, source_column in self.config.columns.items()
            if source_column in frame.columns
        }
        missing_optional = sorted(set(self.config.columns) - set(available) - {"timestamp"})
        if missing_optional:
            LOGGER.warning(
                "IMU file %s is missing configured optional signals: %s",
                source,
                ", ".join(missing_optional),
            )
        signal_columns = [name for name in available if name != "timestamp"]
        if not signal_columns:
            raise IMULoadError(f"No configured IMU signal columns were found in: {source}")

        canonical = frame[[available[name] for name in available]].rename(
            columns={source_column: name for name, source_column in available.items()}
        )
        return canonical


class LongCSVLoader(BaseIMULoader):
    """Select one configured sensor from a long-format IMU CSV."""

    def load(self, path: str | Path) -> pd.DataFrame:
        source, frame = self._read_csv(path)
        timestamp_source = self.config.columns["timestamp"]
        sensor_source = self.config.columns.get("sensor_name", "sensor_name")
        required = {timestamp_source, sensor_source}
        missing_required = sorted(required - set(frame.columns))
        if missing_required:
            raise IMULoadError(
                f"Missing long-format IMU columns in {source}: {', '.join(missing_required)}"
            )

        selected = frame.loc[frame[sensor_source].astype(str) == self.config.target_sensor].copy()
        if selected.empty:
            raise IMULoadError(
                f"Target sensor '{self.config.target_sensor}' was not found in: {source}"
            )
        available = {
            canonical: source_column
            for canonical, source_column in self.config.columns.items()
            if canonical != "sensor_name" and source_column in selected.columns
        }
        signal_columns = [name for name in available if name != "timestamp"]
        if not signal_columns:
            raise IMULoadError(f"No configured signals were found for target sensor in: {source}")
        return selected[[available[name] for name in available]].rename(
            columns={source_column: name for name, source_column in available.items()}
        )


class MVNXLoader(BaseIMULoader):
    """Explicit boundary for direct MVNX support planned after the CSV workflow."""

    def load(self, path: str | Path) -> pd.DataFrame:
        source = Path(path).expanduser().resolve()
        raise IMULoadError(
            "Direct MVNX parsing is not implemented. "
            f"Export a CSV and configure imu.format as wide_csv or long_csv: {source}"
        )


def create_imu_loader(config: IMUConfig) -> BaseIMULoader:
    """Create the adapter declared by the project configuration."""
    loaders: dict[str, type[BaseIMULoader]] = {
        "wide_csv": WideCSVLoader,
        "long_csv": LongCSVLoader,
        "mvnx": MVNXLoader,
    }
    return loaders[config.format](config)
