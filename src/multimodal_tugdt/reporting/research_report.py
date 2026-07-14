"""Privacy-conscious aggregate Markdown report for completed pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from multimodal_tugdt.config import ProjectConfig
from multimodal_tugdt.io.manifest import TrialRecord

MODALITIES = ("imu", "audio", "video", "footswitch", "annotation", "clinical")
QC_ARTIFACTS = {
    "IMU preprocessing": "imu_preprocessing.csv",
    "Synchronization": "synchronization.csv",
    "Audio processing": "audio_processing.csv",
    "Footswitch processing": "footswitch_processing.csv",
    "Video processing": "video_processing.csv",
}
FEATURE_ARTIFACTS = {
    "IMU": "imu_features.csv",
    "Audio": "audio_features.csv",
    "Footswitch": "footswitch_features.csv",
    "Video": "video_features.csv",
    "Multimodal trial table": "multimodal_features.csv",
}


@dataclass(frozen=True)
class ResearchReport:
    """Generated aggregate report path and core dataset counts."""

    path: Path
    participant_count: int
    trial_count: int


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _safe_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _qc_rows(config: ProjectConfig) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for stage, filename in QC_ARTIFACTS.items():
        frame = _safe_csv(config.output_dir / "qc" / filename)
        if frame is None:
            rows.append((stage, "not run", 0, 0, 0, 0))
            continue
        statuses = (
            frame["qc_status"].astype(str).str.lower().value_counts()
            if "qc_status" in frame
            else pd.Series(dtype=int)
        )
        rows.append(
            (
                stage,
                "available",
                len(frame),
                int(statuses.get("pass", 0)),
                int(statuses.get("warning", 0)),
                int(statuses.get("fail", 0)),
            )
        )
    return rows


def _feature_rows(config: ProjectConfig) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for label, filename in FEATURE_ARTIFACTS.items():
        frame = _safe_csv(config.output_dir / "features" / filename)
        if frame is None:
            rows.append((label, "not run", 0, 0, 0))
            continue
        trial_rows = (
            int(frame["feature_level"].eq("trial").sum())
            if "feature_level" in frame
            else len(frame)
        )
        phase_rows = (
            int(frame["feature_level"].eq("phase").sum()) if "feature_level" in frame else 0
        )
        rows.append((label, "available", len(frame), trial_rows, phase_rows))
    return rows


def _modeling_section(config: ProjectConfig) -> list[str]:
    if not config.modeling.enabled:
        return [
            "Baseline modeling is disabled in the example configuration because the public "
            "synthetic dataset contains only one independent participant group.",
        ]
    summary_path = config.output_dir / "modeling" / "summary_metrics.csv"
    skipped_path = config.output_dir / "modeling" / "skipped_evaluations.csv"
    summary = _safe_csv(summary_path)
    skipped = _safe_csv(skipped_path)
    if summary is None:
        return [
            "Baseline modeling was not run. This is expected when `modeling.enabled` is false.",
        ]
    if summary.empty:
        skipped_count = 0 if skipped is None else len(skipped)
        return [
            "Baseline modeling was attempted, but no valid evaluation was produced.",
            f"Skipped evaluation records: **{skipped_count}**. Review "
            "`outputs/modeling/skipped_evaluations.csv` for explicit reasons.",
        ]
    feature_sets = summary["feature_set"].nunique() if "feature_set" in summary else 0
    models = summary["model"].nunique() if "model" in summary else 0
    metrics = summary["metric"].nunique() if "metric" in summary else 0
    return [
        f"Grouped evaluation artifacts contain **{feature_sets}** feature set(s), "
        f"**{models}** model(s), and **{metrics}** metric(s).",
        "This aggregate report does not select or promote a best model. Review "
        "`summary_metrics.csv`, `predictions.csv`, and `split_audit.csv` together.",
    ]


def generate_research_report(
    config: ProjectConfig,
    records: list[TrialRecord],
    output_path: str | Path | None = None,
) -> ResearchReport:
    """Generate a deterministic aggregate report without participant-level values."""
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else config.output_dir / "reports" / "research_summary.md"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    participant_count = len({record.participant_id for record in records})
    trial_count = len(records)
    conditions = pd.Series([record.condition for record in records]).value_counts().sort_index()
    condition_rows = [(condition, int(count)) for condition, count in conditions.items()]
    availability_rows = []
    for modality in MODALITIES:
        available = sum(bool(record.paths.get(f"{modality}_path", "")) for record in records)
        availability_rows.append((modality, available, trial_count - available))

    lines = [
        "# Multimodal TUG-DT Research Summary",
        "",
        "> Aggregate software-generated report. It contains no participant-level measurements and "
        "must not be interpreted as a clinical result.",
        "",
        "## Dataset contract",
        "",
        *_markdown_table(
            ("Measure", "Count"),
            [("Participants", participant_count), ("Trials", trial_count)],
        ),
        "",
        "### Conditions",
        "",
        *_markdown_table(("Condition", "Trials"), condition_rows),
        "",
        "### Modality availability",
        "",
        *_markdown_table(
            ("Modality", "Available trials", "Missing trials"),
            availability_rows,
        ),
        "",
        "## Quality-control status",
        "",
        *_markdown_table(
            ("Stage", "Artifact", "Rows", "Pass", "Warning", "Fail"),
            _qc_rows(config),
        ),
        "",
        "Warnings require review; they are not silently converted to passes. A row count may refer "
        "to trials or modality alignments depending on the stage.",
        "",
        "## Derived feature artifacts",
        "",
        *_markdown_table(
            ("Artifact", "Status", "Total rows", "Trial rows", "Phase rows"),
            _feature_rows(config),
        ),
        "",
        "## Modeling status",
        "",
        *_modeling_section(config),
        "",
        "## Reproducibility record",
        "",
        f"- Configuration file: `{config.source.name}`",
        "- Clock mapping: `reference_time = native_time + offset_seconds`",
        "- Grouped modeling unit: `participant_id`",
        "- Generated artifacts remain excluded from version control by default.",
        "",
        "## Interpretation boundary",
        "",
        "This pipeline is research software, not a medical device. Synthetic fixtures demonstrate "
        "software behavior only. Sensor features, voice-activity intervals, foot-contact events, "
        "video pose proxies, and cross-validated model metrics require protocol-specific "
        "validation before scientific or clinical interpretation.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    return ResearchReport(destination, participant_count, trial_count)
