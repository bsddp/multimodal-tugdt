"""Video metadata inspection and optional MediaPipe Tasks pose extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from multimodal_tugdt.config import VideoConfig

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}

POSE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


@dataclass(frozen=True)
class VideoMetadata:
    """Container metadata reported by the first video stream."""

    duration_seconds: float
    frame_rate_hz: float
    total_frames: int
    width_pixels: int
    height_pixels: int
    codec_name: str
    frame_count_is_estimated: bool
    inspection_backend: str = "ffprobe"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable metadata."""
        return asdict(self)


@dataclass(frozen=True)
class PoseExtractionResult:
    """Long-form normalized landmarks plus frame-level extraction counts."""

    landmarks: pd.DataFrame
    frames: pd.DataFrame
    processed_frame_count: int
    detected_frame_count: int
    backend: str

    @property
    def detection_rate(self) -> float:
        """Return the fraction of sampled frames with a detected pose."""
        if self.processed_frame_count == 0:
            return 0.0
        return self.detected_frame_count / self.processed_frame_count


def _positive_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Video metadata field '{field}' is not numeric: {value!r}") from exc
    if number <= 0:
        raise ValueError(f"Video metadata field '{field}' must be positive: {number}")
    return number


def _parse_frame_rate(value: Any) -> float:
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            raise ValueError("Video frame-rate denominator cannot be zero.")
        return _positive_number(float(numerator) / denominator_value, "frame_rate")
    return _positive_number(value, "frame_rate")


def _usable_metadata_value(*values: Any) -> Any:
    for value in values:
        if value not in {None, "", "N/A", "0/0"}:
            return value
    return None


def parse_ffprobe_video_metadata(payload: dict[str, Any]) -> VideoMetadata:
    """Parse a constrained ffprobe JSON response into validated metadata."""
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ValueError("No video stream was found in ffprobe output.")
    stream = streams[0]
    format_values = payload.get("format", {})
    if not isinstance(format_values, dict):
        format_values = {}
    duration = _positive_number(
        _usable_metadata_value(stream.get("duration"), format_values.get("duration")),
        "duration",
    )
    frame_rate = _parse_frame_rate(
        _usable_metadata_value(stream.get("avg_frame_rate"), stream.get("r_frame_rate"))
    )
    frame_count_raw = stream.get("nb_frames")
    estimated = frame_count_raw in {None, "", "N/A"}
    total_frames = round(duration * frame_rate) if estimated else int(frame_count_raw)
    if total_frames <= 0:
        raise ValueError("Video total frame count must be positive.")
    return VideoMetadata(
        duration_seconds=duration,
        frame_rate_hz=frame_rate,
        total_frames=total_frames,
        width_pixels=int(_positive_number(stream.get("width"), "width")),
        height_pixels=int(_positive_number(stream.get("height"), "height")),
        codec_name=str(stream.get("codec_name") or "unknown"),
        frame_count_is_estimated=estimated,
    )


def inspect_video(path: str | Path) -> VideoMetadata:
    """Inspect MP4, MOV, or AVI metadata without decoding all frames."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise ValueError(f"Unsupported video format '{source.suffix}'. Supported: {supported}")
    executable = shutil.which("ffprobe")
    if executable is None:
        raise ValueError(
            "Video metadata inspection requires ffprobe from FFmpeg. "
            "Install FFmpeg and ensure ffprobe is on PATH."
        )
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=duration,avg_frame_rate,r_frame_rate,nb_frames,width,height,codec_name:"
            "format=duration"
        ),
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown ffprobe error"
        raise ValueError(f"Could not inspect video {source}: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid JSON for {source}.") from exc
    return parse_ffprobe_video_metadata(payload)


def extract_mediapipe_pose(
    path: str | Path,
    model_path: str | Path,
    config: VideoConfig,
) -> PoseExtractionResult:
    """Decode sampled frames and extract first-person normalized pose landmarks.

    MediaPipe and its frame decoder are imported only inside this function so the
    remainder of the research pipeline does not depend on the optional video stack.
    """
    source = Path(path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file does not exist: {source}")
    if not model.is_file():
        raise FileNotFoundError(f"MediaPipe pose model does not exist: {model}")
    try:
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "Pose extraction requires the optional video dependencies. "
            "Install them with: python -m pip install -e '.[video]'"
        ) from exc

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {source}")
    frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_rate <= 0:
        capture.release()
        raise ValueError(f"Video reports an invalid frame rate: {frame_rate}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=config.minimum_pose_detection_confidence,
        min_pose_presence_confidence=config.minimum_pose_presence_confidence,
        min_tracking_confidence=config.minimum_tracking_confidence,
        output_segmentation_masks=False,
    )
    rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    processed = detected = frame_index = 0
    previous_timestamp_ms = -1
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                readable, frame = capture.read()
                if not readable:
                    break
                if frame_index % config.frame_step:
                    frame_index += 1
                    continue
                native_timestamp = frame_index / frame_rate
                timestamp_ms = max(round(native_timestamp * 1000), previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp_ms)
                processed += 1
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "native_timestamp": native_timestamp,
                        "pose_detected": bool(result.pose_landmarks),
                    }
                )
                if result.pose_landmarks:
                    detected += 1
                    for landmark_index, landmark in enumerate(result.pose_landmarks[0]):
                        rows.append(
                            {
                                "frame_index": frame_index,
                                "native_timestamp": native_timestamp,
                                "landmark_index": landmark_index,
                                "landmark_name": POSE_LANDMARK_NAMES[landmark_index],
                                "x": float(landmark.x),
                                "y": float(landmark.y),
                                "z": float(landmark.z),
                                "visibility": float(landmark.visibility),
                                "presence": float(landmark.presence),
                            }
                        )
                frame_index += 1
    finally:
        capture.release()

    columns = [
        "frame_index",
        "native_timestamp",
        "landmark_index",
        "landmark_name",
        "x",
        "y",
        "z",
        "visibility",
        "presence",
    ]
    return PoseExtractionResult(
        landmarks=pd.DataFrame(rows, columns=columns),
        frames=pd.DataFrame(
            frame_rows,
            columns=["frame_index", "native_timestamp", "pose_detected"],
        ),
        processed_frame_count=processed,
        detected_frame_count=detected,
        backend="mediapipe_tasks",
    )
