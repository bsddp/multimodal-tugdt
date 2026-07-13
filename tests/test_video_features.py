import pandas as pd
import pytest

from multimodal_tugdt.config import VideoConfig
from multimodal_tugdt.features.video_features import (
    compute_pose_frame_metrics,
    extract_video_features,
)
from multimodal_tugdt.io.manifest import TrialRecord
from multimodal_tugdt.io.video_loader import PoseExtractionResult, VideoMetadata
from multimodal_tugdt.segmentation.manual import Segment


def _config() -> VideoConfig:
    return VideoConfig(
        enable_pose_estimation=True,
        pose_backend="mediapipe",
        pose_model_path="pose.task",
        frame_step=1,
        minimum_visibility=0.5,
        minimum_pose_detection_confidence=0.5,
        minimum_pose_presence_confidence=0.5,
        minimum_tracking_confidence=0.5,
    )


def _pose_result() -> PoseExtractionResult:
    rows = []
    points = {
        "left_shoulder": (0.40, 0.30),
        "right_shoulder": (0.60, 0.30),
        "left_hip": (0.45, 0.60),
        "right_hip": (0.55, 0.60),
        "left_ankle": (0.42, 0.90),
        "right_ankle": (0.58, 0.90),
    }
    for frame_index, timestamp in enumerate((0.0, 1.0, 2.0)):
        for landmark_index, (name, (x, y)) in enumerate(points.items()):
            rows.append(
                {
                    "frame_index": frame_index,
                    "native_timestamp": timestamp,
                    "timestamp": timestamp,
                    "landmark_index": landmark_index,
                    "landmark_name": name,
                    "x": x + (0.03 if "shoulder" in name else 0.02) * frame_index,
                    "y": y + (0.01 * frame_index if "hip" in name else 0.0),
                    "z": 0.0,
                    "visibility": 0.9,
                    "presence": 0.95,
                }
            )
    return PoseExtractionResult(
        landmarks=pd.DataFrame(rows),
        frames=pd.DataFrame(
            {
                "frame_index": [0, 1, 2],
                "native_timestamp": [0.0, 1.0, 2.0],
                "timestamp": [0.0, 1.0, 2.0],
                "pose_detected": [True, True, True],
            }
        ),
        processed_frame_count=3,
        detected_frame_count=3,
        backend="test",
    )


def test_pose_frame_metrics_are_transparent_two_dimensional_proxies() -> None:
    metrics = compute_pose_frame_metrics(_pose_result().landmarks, 0.5)

    assert len(metrics) == 3
    assert metrics["trunk_lean_degrees"].iloc[0] == pytest.approx(0.0, abs=1e-8)
    assert metrics["left_right_step_length_proxy"].iloc[0] == pytest.approx(0.16)
    assert (metrics["lower_limb_symmetry_proxy"] >= 0).all()


def test_video_features_include_trial_phase_detection_and_pose_metrics() -> None:
    metadata = VideoMetadata(3.0, 1.0, 3, 640, 480, "test", False)
    rows = extract_video_features(
        TrialRecord("P001", "S01", "dual_task", "T01", {}),
        metadata,
        _pose_result(),
        [Segment("sit_to_stand", 0.0, 1.5), Segment("outbound_walk", 1.5, 3.0)],
        _config(),
        trial_start=0.0,
        trial_end=3.0,
    )

    trial = rows[0]
    assert len(rows) == 3
    assert trial["video__pose_detection_rate"] == 1.0
    assert trial["video__mean_landmark_confidence"] == pytest.approx(0.9)
    assert trial["video__pelvis_vertical_displacement_proxy"] == pytest.approx(0.02)
    assert trial["video__sit_to_stand_trunk_flexion_degrees"] > 0
