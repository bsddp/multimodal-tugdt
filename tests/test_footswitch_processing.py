import numpy as np
import pandas as pd

from multimodal_tugdt.config import FootswitchConfig
from multimodal_tugdt.preprocessing.footswitch import process_footswitch


def _config() -> FootswitchConfig:
    return FootswitchConfig(
        timestamp_column="timestamp",
        left_contact_column="left_contact",
        right_contact_column="right_contact",
        threshold=0.5,
        minimum_contact_duration_s=0.08,
        minimum_swing_duration_s=0.08,
        event_matching_tolerance_s=0.15,
    )


def test_footswitch_debounce_removes_short_contact_and_extracts_events() -> None:
    timestamps = np.arange(0.0, 2.0, 0.01)
    left = ((timestamps >= 0.2) & (timestamps < 0.8)).astype(float)
    left[(timestamps >= 0.05) & (timestamps < 0.07)] = 1.0
    right = ((timestamps >= 1.0) & (timestamps < 1.6)).astype(float)
    frame = pd.DataFrame(
        {
            "native_timestamp": timestamps,
            "timestamp": timestamps,
            "left_contact": left,
            "right_contact": right,
        }
    )

    result = process_footswitch(frame, _config())

    assert result.quality.changed_sample_count == 2
    assert result.quality.left_contact_count == 1
    assert result.quality.right_contact_count == 1
    assert result.events["event"].tolist() == ["contact", "toe_off", "contact", "toe_off"]


def test_footswitch_all_zero_signal_is_retained_without_events() -> None:
    timestamps = np.arange(0.0, 1.0, 0.01)
    frame = pd.DataFrame(
        {
            "native_timestamp": timestamps,
            "timestamp": timestamps,
            "left_contact": 0,
            "right_contact": 0,
        }
    )

    result = process_footswitch(frame, _config())

    assert result.events.empty
    assert result.quality.left_contact_count == 0
    assert result.quality.right_contact_count == 0
