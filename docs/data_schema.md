# Milestone 1 data schema

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

