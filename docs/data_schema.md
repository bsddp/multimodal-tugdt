# Data schema

## Manifest

The configured manifest is a UTF-8 CSV with one row per trial.

| Column | Required column | Blank allowed | Meaning |
|---|---:|---:|---|
| `participant_id` | yes | no | De-identified participant key |
| `session_id` | yes | no | Data-collection session key |
| `condition` | yes | no | Configured task condition |
| `trial_id` | yes | no | Trial key within a session |
| `imu_path` | yes | yes | IMU or Xsens-derived motion file |
| `video_path` | yes | yes | Video file |
| `audio_path` | yes | yes | Audio file |
| `footswitch_path` | yes | yes | Foot-contact signal file |
| `annotation_path` | yes | yes | Manual or imported phase labels |
| `clinical_path` | yes | yes | Clinical/demographic table |

All columns are required so manifests have a predictable schema. Individual modality values
may be blank because missing modalities are expected in real-world studies. At least one of
IMU, video, audio, footswitch, or annotation must be present for each row.

The unique trial key is `(participant_id, session_id, trial_id)`. Condition is not part of that
key because reusing a trial identifier for two conditions would make file provenance ambiguous.

## Path resolution

Relative paths are interpreted from `project.root` in the YAML file. In
`configs/example.yaml`, root is `..`, which means the repository directory. Absolute paths are
accepted for controlled local studies but should never be committed if they expose usernames,
institutions, or private storage layouts.

## Synthetic files

The demonstration generator creates:

- wide-format pelvis IMU CSV at 100 Hz;
- binary left/right foot-contact CSV at 100 Hz;
- mono WAV containing artificial tones in speech-like activity windows;
- CSV annotations using the seven standard TUG phases;
- a minimal synthetic clinical CSV;
- a manifest joining these files.

These fixtures test software behavior only. Their values are not physiologically calibrated.

## Synchronized trial outputs

The implemented pipelines write the following derived files under each processed trial directory:

- `imu.csv`: processed IMU-reference time series;
- `imu_qc.json`: IMU input and preprocessing evidence;
- `sync_metadata.json`: reference timeline, explicit target alignments, QC thresholds, and errors;
- `segments.csv`: validated annotations already expressed on the IMU reference clock;
- `footswitch_synced.csv`: native footswitch timestamps plus mapped reference timestamps;
- `audio_frames.csv`: native/reference frame bounds, RMS, dBFS, and VAD labels;
- `audio_activity.csv`: merged speech/silence intervals on both clocks;
- `audio_qc.json`: decoding, resampling, clipping, and VAD evidence;
- `footswitch_processed.csv`: raw and stabilized binary contact channels;
- `footswitch_events.csv`: side-specific contact/toe-off events on both clocks;
- `footswitch_qc.json`: sampling, debounce, contact-count, and contact-ratio evidence;
- `video_metadata.json`: container metadata, declared clock offset, optional pose configuration,
  sampled/detected frame counts, and detection rate;
- `video_pose_frames.csv`: one row per sampled frame with native/reference timestamps and a
  pose-detected flag when optional inference is enabled;
- `video_pose_landmarks.csv`: long-form normalized image coordinates, z estimate, visibility, and
  presence confidence for each detected landmark.

Every synchronized time-series CSV repeats participant, session, trial, and condition identifiers.
Derived outputs remain ignored by Git because real-study versions may contain sensitive data.

## Fused and modeling outputs

`outputs/features/multimodal_features.csv` contains one row per manifest trial. It retains the four
trial identifiers, modality-prefixed feature columns, clinical columns prefixed with `clinical__`,
and binary `availability__<modality>` indicators. Fusion is an outer join: an absent modality does
not remove a trial.

`outputs/features/feature_inventory.csv` records each fused column's modality, role, dtype,
observed count, and missing count.

When explicitly enabled, `outputs/features/dual_task_costs.csv` contains one row per complete
single-/dual-task pair with preserved source values and direction-normalized costs. See the
[dual-task cost contract](dual_task_cost.md).

The modeling directory contains:

- `fold_metrics.csv`: one row per feature set, cohort, model, and held-out fold;
- `summary_metrics.csv`: mean and sample standard deviation across valid folds;
- `predictions.csv`: held-out predictions identified by participant/session/trial;
- `split_audit.csv`: train/test participant lists and their overlap count for each fold;
- `skipped_evaluations.csv`: combinations that could not be evaluated and the explicit reason.
- `modeling_metadata.json`: target, grouping, requested folds, seed, models, feature sets,
  preprocessing scope, and scikit-learn version.

These files are derived research outputs and remain ignored by Git.
