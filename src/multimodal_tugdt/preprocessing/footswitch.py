"""Footswitch binarization, debounce, and gait-event extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from multimodal_tugdt.config import FootswitchConfig
from multimodal_tugdt.preprocessing.imu import estimate_sampling_rate


@dataclass(frozen=True)
class FootswitchQualityReport:
    input_sample_count: int
    sampling_rate_hz: float
    changed_sample_count: int
    left_contact_count: int
    right_contact_count: int
    left_contact_ratio: float
    right_contact_ratio: float
    simultaneous_contact_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FootswitchResult:
    frame: pd.DataFrame
    events: pd.DataFrame
    quality: FootswitchQualityReport


def _runs(states: np.ndarray) -> list[tuple[int, int, int]]:
    changes = np.flatnonzero(np.diff(states) != 0) + 1
    boundaries = np.concatenate(([0], changes, [states.size]))
    return [
        (int(start), int(end), int(states[start]))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def _stabilize_contact(
    states: np.ndarray,
    sampling_rate_hz: float,
    config: FootswitchConfig,
) -> np.ndarray:
    stable = states.copy()
    for _ in range(2):
        for start, end, state in _runs(stable):
            duration = (end - start) / sampling_rate_hz
            if state == 1 and duration < config.minimum_contact_duration_s:
                stable[start:end] = 0
            elif (
                state == 0
                and start > 0
                and end < stable.size
                and duration < config.minimum_swing_duration_s
            ):
                stable[start:end] = 1
    return stable


def _events_for_side(
    frame: pd.DataFrame,
    states: np.ndarray,
    side: str,
) -> list[dict[str, object]]:
    changes = np.flatnonzero(np.diff(states) != 0) + 1
    indices = changes.tolist()
    if states[0] == 1:
        indices.insert(0, 0)
    rows: list[dict[str, object]] = []
    for index in indices:
        event = "contact" if states[index] == 1 else "toe_off"
        rows.append(
            {
                "side": side,
                "event": event,
                "timestamp": float(frame["timestamp"].iloc[index]),
                "native_timestamp": float(frame["native_timestamp"].iloc[index]),
            }
        )
    return rows


def process_footswitch(
    frame: pd.DataFrame,
    config: FootswitchConfig,
) -> FootswitchResult:
    """Validate synchronized contacts, debounce binary states, and extract transitions."""
    required = {
        "timestamp",
        "native_timestamp",
        config.left_contact_column,
        config.right_contact_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing synchronized footswitch columns: {', '.join(missing)}")
    working = frame.copy()
    for column in required:
        working[column] = pd.to_numeric(working[column], errors="coerce")
        if working[column].isna().any():
            raise ValueError(f"Footswitch column '{column}' contains invalid values.")
    timestamps = working["timestamp"].to_numpy(dtype=float)
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Synchronized footswitch timestamps are nonmonotonic.")
    sampling_rate = estimate_sampling_rate(timestamps)
    raw_left = (working[config.left_contact_column].to_numpy() >= config.threshold).astype(int)
    raw_right = (working[config.right_contact_column].to_numpy() >= config.threshold).astype(int)
    left = _stabilize_contact(raw_left, sampling_rate, config)
    right = _stabilize_contact(raw_right, sampling_rate, config)
    working["left_contact_binary"] = left
    working["right_contact_binary"] = right
    event_rows = [
        *_events_for_side(working, left, "left"),
        *_events_for_side(working, right, "right"),
    ]
    events = pd.DataFrame(
        event_rows,
        columns=["side", "event", "timestamp", "native_timestamp"],
    ).sort_values(["timestamp", "side"], ignore_index=True)
    changed_sample_count = int(
        np.count_nonzero(left != raw_left) + np.count_nonzero(right != raw_right)
    )
    left_contact_count = int(
        ((events["side"] == "left") & (events["event"] == "contact")).sum()
    )
    right_contact_count = int(
        ((events["side"] == "right") & (events["event"] == "contact")).sum()
    )
    quality = FootswitchQualityReport(
        input_sample_count=len(working),
        sampling_rate_hz=sampling_rate,
        changed_sample_count=changed_sample_count,
        left_contact_count=left_contact_count,
        right_contact_count=right_contact_count,
        left_contact_ratio=float(left.mean()),
        right_contact_ratio=float(right.mean()),
        simultaneous_contact_ratio=float(((left == 1) & (right == 1)).mean()),
    )
    return FootswitchResult(working, events, quality)
