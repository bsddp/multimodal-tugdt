from pathlib import Path

import pandas as pd
import pytest

from multimodal_tugdt.segmentation.manual import Segment, load_segments, slice_segment


def test_load_and_apply_manual_segments(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "segment_name": ["outbound_walk", "turn_1"],
            "start_time": [1.0, 3.0],
            "end_time": [3.0, 4.0],
            "source": ["manual", "manual"],
            "confidence": [1.0, 0.9],
        }
    ).to_csv(annotations, index=False)
    segments = load_segments(annotations, trial_start=0.0, trial_end=5.0)
    frame = pd.DataFrame({"timestamp": [0.0, 1.0, 2.0, 3.0, 4.0], "x": range(5)})

    outbound = slice_segment(frame, segments[0])

    assert [segment.name for segment in segments] == ["outbound_walk", "turn_1"]
    assert outbound["timestamp"].tolist() == [1.0, 2.0]


def test_invalid_segment_bounds_are_rejected(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame({"segment_name": ["turn_1"], "start_time": [4.0], "end_time": [3.0]}).to_csv(
        annotations, index=False
    )

    with pytest.raises(ValueError, match="start_time < end_time"):
        load_segments(annotations)


def test_empty_segment_slice_is_rejected() -> None:
    frame = pd.DataFrame({"timestamp": [0.0, 1.0], "x": [1, 2]})
    with pytest.raises(ValueError, match="contains no samples"):
        slice_segment(frame, Segment("missing", 3.0, 4.0))
