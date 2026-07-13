from pathlib import Path

import pytest

from multimodal_tugdt.io.video_loader import (
    inspect_video,
    parse_ffprobe_video_metadata,
)


def test_parse_ffprobe_video_metadata_handles_rational_rate_and_exact_count() -> None:
    metadata = parse_ffprobe_video_metadata(
        {
            "streams": [
                {
                    "duration": "20.0",
                    "avg_frame_rate": "30000/1001",
                    "nb_frames": "599",
                    "width": 1920,
                    "height": 1080,
                    "codec_name": "h264",
                }
            ],
            "format": {"duration": "20.0"},
        }
    )

    assert metadata.duration_seconds == 20.0
    assert metadata.frame_rate_hz == pytest.approx(29.97003)
    assert metadata.total_frames == 599
    assert metadata.frame_count_is_estimated is False
    assert metadata.codec_name == "h264"


def test_parse_ffprobe_video_metadata_estimates_missing_frame_count() -> None:
    metadata = parse_ffprobe_video_metadata(
        {
            "streams": [
                {
                    "avg_frame_rate": "25/1",
                    "width": 640,
                    "height": 480,
                }
            ],
            "format": {"duration": "4.0"},
        }
    )

    assert metadata.total_frames == 100
    assert metadata.frame_count_is_estimated is True


def test_parse_ffprobe_video_metadata_uses_fallbacks_for_na_values() -> None:
    metadata = parse_ffprobe_video_metadata(
        {
            "streams": [
                {
                    "duration": "N/A",
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "24/1",
                    "width": 640,
                    "height": 480,
                }
            ],
            "format": {"duration": "2.0"},
        }
    )

    assert metadata.duration_seconds == 2.0
    assert metadata.frame_rate_hz == 24.0
    assert metadata.total_frames == 48


def test_inspect_video_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "recording.mkv"
    path.write_bytes(b"not a video")

    with pytest.raises(ValueError, match="Unsupported video format"):
        inspect_video(path)
