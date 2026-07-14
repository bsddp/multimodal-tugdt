# Video processing

The deliberately optional video interface can process metadata without pose estimation, and an
absent video never blocks the IMU, audio, or footswitch workflows.

## Supported inputs and metadata

The manifest accepts `.mp4`, `.mov`, and `.avi`. The loader uses `ffprobe` to read the first video
stream and records:

- duration;
- average frame rate;
- total frames, with an explicit flag when estimated from duration and rate;
- width and height;
- codec name.

Metadata inspection does not decode every frame. A present video still requires an explicit
synchronization offset. Aligned timestamps follow the project equation:

```text
reference_time = native_time + offset_seconds
```

## Optional MediaPipe Tasks pose interface

Pose inference is disabled by default. Install the optional dependencies with:

```bash
python -m pip install -e '.[video]'
```

Then obtain a compatible Pose Landmarker `.task` model and configure it locally:

```yaml
video:
  enable_pose_estimation: true
  pose_backend: mediapipe
  pose_model_path: models/pose_landmarker.task
  frame_step: 1
  minimum_visibility: 0.5
  minimum_pose_detection_confidence: 0.5
  minimum_pose_presence_confidence: 0.5
  minimum_tracking_confidence: 0.5
```

The model path is resolved from `project.root`. Model assets are ignored by Git and are never
downloaded automatically, which keeps licensing and provenance decisions visible to the researcher.
`frame_step: 2`, for example, evaluates every second decoded frame.

The implementation uses MediaPipe Tasks in video mode and passes an increasing timestamp for every
sampled frame. It exports the first detected person only; multi-person tracking is outside the
current interface.

## Derived tables

`video_pose_frames.csv` contains:

```text
frame_index,native_timestamp,pose_detected,timestamp
```

`video_pose_landmarks.csv` contains one row per landmark per detected frame:

```text
frame_index,native_timestamp,timestamp,landmark_index,landmark_name,x,y,z,visibility,presence
```

`x` and `y` are normalized image coordinates. They are not pixel coordinates or calibrated metric
positions. The exported `z` is the pose model's relative estimate and is not treated as laboratory
3D motion capture by this pipeline.

## Feature definitions and missingness

Trial and annotated phase rows report pose detection rate, mean visibility, signed trunk lean,
pelvis vertical displacement, horizontal ankle separation, and a lower-limb symmetry ratio. The
full formulas and units are listed in the [feature dictionary](feature_dictionary.md).

Only landmarks meeting `minimum_visibility` contribute. If the required points are unavailable, the
corresponding feature is blank. When pose inference is disabled, metadata features remain available,
`video__pose_estimation_enabled` is false, and pose features remain blank. Zero is never used as a
substitute for missing detection.

## Interpretation boundary

The video features are transparent monocular two-dimensional proxies. Camera viewpoint, lens,
occlusion, clothing, mobility aids, and distance to the camera can change them. They must not be
called gait speed, anatomical joint angles, true step length, calibrated displacement, diagnostic
biomarkers, or 3D kinematics without a separate validation study.

Videos commonly contain faces, voices, homes, clinics, timestamps, or other identifiers. Raw video,
pose model files, and real-study derived outputs must remain outside version control unless data
governance and participant consent explicitly permit release.
