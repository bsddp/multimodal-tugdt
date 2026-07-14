"""Participant manifest loading and validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from multimodal_tugdt.config import ProjectConfig

IDENTIFIER_COLUMNS = ("participant_id", "session_id", "condition", "trial_id")
PATH_COLUMNS = (
    "imu_path",
    "video_path",
    "audio_path",
    "footswitch_path",
    "annotation_path",
    "clinical_path",
)
REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + PATH_COLUMNS
TRIAL_MODALITY_COLUMNS = PATH_COLUMNS[:-1]


@dataclass(frozen=True)
class TrialRecord:
    """A single trial row from the study manifest."""

    participant_id: str
    session_id: str
    condition: str
    trial_id: str
    paths: dict[str, str]

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the participant/session/trial identity used for duplicate detection."""
        return (self.participant_id, self.session_id, self.trial_id)


@dataclass
class ManifestReport:
    """Structured validation result suitable for both CLI and future QC reports."""

    manifest_path: Path
    records: list[TrialRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _read_rows(path: Path, report: ManifestReport) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        report.errors.append(f"Manifest file does not exist: {path}")
        return [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            report.errors.append("Manifest is empty or has no header row.")
            return [], []
        return list(reader.fieldnames), list(reader)


def validate_manifest(config: ProjectConfig, *, check_files: bool = True) -> ManifestReport:
    """Validate schema, identifiers, study conditions, duplicates, and referenced files."""
    report = ManifestReport(manifest_path=config.manifest_path)
    columns, rows = _read_rows(config.manifest_path, report)
    if report.errors:
        return report

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        report.errors.append("Missing required columns: " + ", ".join(missing_columns))
        return report
    if not rows:
        report.errors.append("Manifest contains no trial rows.")
        return report

    seen: dict[tuple[str, str, str], int] = {}
    for row_number, row in enumerate(rows, start=2):
        identifiers = {name: (row.get(name) or "").strip() for name in IDENTIFIER_COLUMNS}
        blank_identifiers = [name for name, value in identifiers.items() if not value]
        if blank_identifiers:
            report.errors.append(
                f"Row {row_number}: blank required identifiers: {', '.join(blank_identifiers)}"
            )
            continue

        if identifiers["condition"] not in config.allowed_conditions:
            report.errors.append(
                f"Row {row_number}: unsupported condition '{identifiers['condition']}'. "
                f"Allowed: {', '.join(config.allowed_conditions)}"
            )

        paths = {name: (row.get(name) or "").strip() for name in PATH_COLUMNS}
        if not any(paths[name] for name in TRIAL_MODALITY_COLUMNS):
            report.errors.append(f"Row {row_number}: no trial modality or annotation path is set.")

        record = TrialRecord(paths=paths, **identifiers)
        if record.key in seen:
            first_row = seen[record.key]
            report.errors.append(
                f"Row {row_number}: duplicate trial key {record.key}; "
                f"first seen on row {first_row}."
            )
        else:
            seen[record.key] = row_number

        if check_files:
            for column, value in paths.items():
                if value and not config.resolve_path(value).is_file():
                    report.errors.append(
                        f"Row {row_number}: {column} does not exist: {config.resolve_path(value)}"
                    )

        absent = [name.removesuffix("_path") for name in TRIAL_MODALITY_COLUMNS if not paths[name]]
        if absent:
            report.warnings.append(f"Row {row_number}: optional data absent: {', '.join(absent)}")
        report.records.append(record)

    return report
