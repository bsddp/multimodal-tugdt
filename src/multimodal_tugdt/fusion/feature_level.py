"""Feature-level fusion with explicit modality availability indicators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from multimodal_tugdt.config import ProjectConfig
from multimodal_tugdt.io.manifest import TrialRecord

TRIAL_KEYS = ("participant_id", "session_id", "trial_id", "condition")
SENSOR_MODALITIES = ("imu", "audio", "video", "footswitch")


@dataclass(frozen=True)
class FusedDataset:
    """A fused trial table and a machine-readable feature inventory."""

    frame: pd.DataFrame
    inventory: pd.DataFrame


def _base_trials(records: list[TrialRecord]) -> pd.DataFrame:
    rows = [
        {
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "trial_id": record.trial_id,
            "condition": record.condition,
        }
        for record in records
    ]
    frame = pd.DataFrame(rows, columns=TRIAL_KEYS)
    if frame.empty:
        raise ValueError("Cannot fuse features because the manifest contains no trials.")
    duplicated = frame.duplicated(list(TRIAL_KEYS), keep=False)
    if duplicated.any():
        raise ValueError("Manifest trial identifiers are not unique during feature fusion.")
    return frame


def _load_trial_feature_table(path: Path, modality: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(
            f"{modality} feature table does not exist; run feature extraction first: {path}"
        )
    frame = pd.read_csv(path)
    required = {*TRIAL_KEYS, "feature_level"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{modality} feature table is missing required columns: {', '.join(missing)}"
        )
    trial = frame.loc[frame["feature_level"] == "trial"].copy()
    feature_columns = [column for column in trial if column.startswith(f"{modality}__")]
    if not feature_columns and not trial.empty:
        raise ValueError(f"{modality} trial table has no '{modality}__' feature columns.")
    selected = trial.loc[:, [*TRIAL_KEYS, *feature_columns]]
    if selected.duplicated(list(TRIAL_KEYS), keep=False).any():
        raise ValueError(f"{modality} feature table contains duplicate trial rows.")
    selected[f"availability__{modality}"] = 1
    return selected


def _load_clinical_table(config: ProjectConfig, records: list[TrialRecord]) -> pd.DataFrame:
    paths = sorted(
        {
            config.resolve_path(value)
            for record in records
            if (value := record.paths.get("clinical_path", ""))
        }
    )
    if not paths:
        return pd.DataFrame(columns=["participant_id", "availability__clinical"])
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Clinical table does not exist: {path}")
        frame = pd.read_csv(path)
        if "participant_id" not in frame:
            raise ValueError(f"Clinical table is missing participant_id: {path}")
        frames.append(frame)
    clinical = pd.concat(frames, ignore_index=True).drop_duplicates()
    if clinical["participant_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Clinical table contains a blank participant_id.")
    if clinical.duplicated("participant_id", keep=False).any():
        duplicated = sorted(
            clinical.loc[
                clinical.duplicated("participant_id", keep=False), "participant_id"
            ].astype(str)
        )
        raise ValueError("Conflicting clinical rows for participant(s): " + ", ".join(duplicated))
    renamed = {
        column: f"clinical__{column}" for column in clinical.columns if column != "participant_id"
    }
    clinical = clinical.rename(columns=renamed)
    clinical["availability__clinical"] = 1
    return clinical


def _feature_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if "__" not in column:
            continue
        prefix = column.split("__", maxsplit=1)[0]
        modality = column.removeprefix("availability__") if prefix == "availability" else prefix
        rows.append(
            {
                "column": column,
                "modality": modality,
                "role": "availability" if prefix == "availability" else "feature_or_target",
                "dtype": str(frame[column].dtype),
                "observed_count": int(frame[column].notna().sum()),
                "missing_count": int(frame[column].isna().sum()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["column", "modality", "role", "dtype", "observed_count", "missing_count"],
    )


def build_trial_feature_table(
    config: ProjectConfig,
    records: list[TrialRecord],
) -> FusedDataset:
    """Outer-join trial features and clinical metadata without dropping missing modalities."""
    fused = _base_trials(records)
    for modality in SENSOR_MODALITIES:
        if modality not in config.fusion.modalities:
            continue
        feature_path = config.output_dir / "features" / f"{modality}_features.csv"
        table = _load_trial_feature_table(feature_path, modality)
        fused = fused.merge(table, on=list(TRIAL_KEYS), how="left", validate="one_to_one")
        availability = f"availability__{modality}"
        fused[availability] = fused[availability].fillna(0).astype(int)

    if "clinical" in config.fusion.modalities:
        clinical = _load_clinical_table(config, records)
        fused = fused.merge(clinical, on="participant_id", how="left", validate="many_to_one")
        fused["availability__clinical"] = fused["availability__clinical"].fillna(0).astype(int)

    return FusedDataset(frame=fused, inventory=_feature_inventory(fused))
