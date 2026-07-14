"""Validation and application of external manual TUG annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STANDARD_TUG_PHASES = {
    "baseline_sitting",
    "sit_to_stand",
    "outbound_walk",
    "turn_1",
    "return_walk",
    "turn_to_sit",
    "final_sitting",
    "combined_straight_walk",
}
REQUIRED_ANNOTATION_COLUMNS = ("segment_name", "start_time", "end_time")


@dataclass(frozen=True)
class Segment:
    """A validated, half-open time interval on the trial reference clock."""

    name: str
    start_time: float
    end_time: float
    source: str = "manual"
    confidence: float | None = None
    notes: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def load_segments(
    path: str | Path,
    *,
    trial_start: float | None = None,
    trial_end: float | None = None,
) -> list[Segment]:
    """Load annotations and reject invalid or out-of-trial intervals."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Annotation file does not exist: {source}")
    frame = pd.read_csv(source)
    missing = [column for column in REQUIRED_ANNOTATION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing annotation columns in {source}: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"Annotation file contains no segments: {source}")

    segments: list[Segment] = []
    for row_number, row in frame.iterrows():
        name = str(row["segment_name"]).strip()
        if not name:
            raise ValueError(f"Annotation row {row_number + 2} has a blank segment_name.")
        try:
            start = float(row["start_time"])
            end = float(row["end_time"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Annotation row {row_number + 2} has non-numeric start or end time."
            ) from exc
        if start >= end:
            raise ValueError(f"Annotation row {row_number + 2} must have start_time < end_time.")
        tolerance = 1e-6
        if trial_start is not None and start < trial_start - tolerance:
            raise ValueError(
                f"Segment '{name}' starts before the trial ({start:g} < {trial_start:g})."
            )
        if trial_end is not None and end > trial_end + tolerance:
            raise ValueError(f"Segment '{name}' ends after the trial ({end:g} > {trial_end:g}).")
        confidence_value = row.get("confidence")
        confidence = None if pd.isna(confidence_value) else float(confidence_value)
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError(f"Segment '{name}' confidence must be between 0 and 1.")
        segments.append(
            Segment(
                name=name,
                start_time=start,
                end_time=end,
                source=str(row.get("source", "manual")),
                confidence=confidence,
                notes="" if pd.isna(row.get("notes")) else str(row.get("notes", "")),
            )
        )
    return sorted(segments, key=lambda segment: (segment.start_time, segment.end_time))


def slice_segment(frame: pd.DataFrame, segment: Segment) -> pd.DataFrame:
    """Return samples in a segment using a half-open interval [start, end)."""
    if "timestamp" not in frame:
        raise ValueError("Cannot segment data without a timestamp column.")
    mask = (frame["timestamp"] >= segment.start_time) & (frame["timestamp"] < segment.end_time)
    result = frame.loc[mask].copy()
    if result.empty:
        raise ValueError(
            f"Segment '{segment.name}' contains no samples in "
            f"[{segment.start_time:g}, {segment.end_time:g})."
        )
    return result
