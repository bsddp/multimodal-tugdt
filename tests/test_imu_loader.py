from pathlib import Path

import pandas as pd
import pytest

from multimodal_tugdt.config import load_config
from multimodal_tugdt.io.imu_loader import IMULoadError, create_imu_loader


def _config(tmp_path: Path, imu_yaml: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
project:
  root: "."
paths:
  manifest: participants.csv
imu:
{imu_yaml}
""",
        encoding="utf-8",
    )
    return path


def test_wide_csv_loader_maps_configured_columns(tmp_path: Path) -> None:
    source = tmp_path / "imu.csv"
    pd.DataFrame(
        {"time_s": [0.0, 0.01], "pelvis_ap": [1.0, 2.0], "unused": [9, 9]}
    ).to_csv(source, index=False)
    config = load_config(
        _config(
            tmp_path,
            """  format: wide_csv
  columns:
    timestamp: time_s
    acc_ap: pelvis_ap
""",
        )
    )

    frame = create_imu_loader(config.imu).load(source)

    assert list(frame.columns) == ["timestamp", "acc_ap"]
    assert frame["acc_ap"].tolist() == [1.0, 2.0]


def test_wide_csv_loader_rejects_missing_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "imu.csv"
    pd.DataFrame({"acc": [1.0, 2.0]}).to_csv(source, index=False)
    config = load_config(
        _config(
            tmp_path,
            """  columns:
    timestamp: time_s
    acc_ap: acc
""",
        )
    )

    with pytest.raises(IMULoadError, match="Missing timestamp"):
        create_imu_loader(config.imu).load(source)


def test_long_csv_loader_selects_target_sensor(tmp_path: Path) -> None:
    source = tmp_path / "imu_long.csv"
    pd.DataFrame(
        {
            "time": [0.0, 0.0, 0.01, 0.01],
            "sensor": ["pelvis", "foot", "pelvis", "foot"],
            "ax": [1.0, 8.0, 2.0, 9.0],
        }
    ).to_csv(source, index=False)
    config = load_config(
        _config(
            tmp_path,
            """  format: long_csv
  target_sensor: pelvis
  columns:
    timestamp: time
    sensor_name: sensor
    acc_ap: ax
""",
        )
    )

    frame = create_imu_loader(config.imu).load(source)

    assert frame["acc_ap"].tolist() == [1.0, 2.0]


def test_mvnx_adapter_has_explicit_export_guidance(tmp_path: Path) -> None:
    config = load_config(
        _config(
            tmp_path,
            """  format: mvnx
  columns:
    timestamp: timestamp
""",
        )
    )
    with pytest.raises(IMULoadError, match="Export a CSV"):
        create_imu_loader(config.imu).load(tmp_path / "trial.mvnx")

