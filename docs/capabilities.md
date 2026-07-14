# Detailed capabilities

This page is the complete implemented-capability inventory. The README keeps only the primary
research workflow and public synthetic results.

## Project and data contract

- YAML configuration with paths resolved from an explicit project root.
- CSV participant/trial manifest with optional missing modalities.
- Validation of schema, identifiers, conditions, duplicate trials, and referenced files.
- Deterministic 20-second synthetic IMU, audio, footswitch, annotation, and clinical fixtures.
- Installable `tugdt` command-line interface with separate stages and `run-all`.
- Structured validation reports and visible warnings for missing optional modalities.

## IMU and phase-aware movement processing

- Wide- and long-format IMU CSV adapters with configurable canonical column mapping.
- Timestamp, sampling interval, duplicate, missing-value, and amplitude quality checks.
- SI unit conversion, optional constant-gravity removal, low-pass filtering, and resampling.
- Quaternion SLERP on complete four-component orientations with normalization and sign continuity.
- Externally annotated TUG phase validation and half-open interval slicing.
- Trial- and phase-level step, acceleration, jerk, and turning features.
- Per-trial QC JSON, project-level QC CSV, processed IMU CSV, and annotated overview plots.

## Synchronization and behavioral modalities

- IMU-reference clock alignment with explicit positive, zero, or negative offsets.
- Native and reference timeline extents, uncertainty, operator, notes, overlap, and duration QC.
- Synchronized footswitch CSV, reference-clock segments, alignment metadata, and coverage plots.
- WAV duration metadata and optional `ffprobe` support for other audio/video containers.
- A strict error when an available target modality lacks an offset declaration.
- WAV and optional FFmpeg audio decoding, mono normalization, resampling, clipping QC, and
  fixed-frame energy VAD.
- Speech duration/ratio, internal pause, and explicitly configured first-response latency features.
- Footswitch thresholding, debounce, left/right contact and toe-off events, stance, swing,
  step-time, and stance-asymmetry features.
- One-to-one IMU/footswitch event matching with precision, recall, F1, and timing error.
- Transcript-dependent response count, correctness, and accuracy remain blank without labels.

## Video and pose interface

- MP4, MOV, and AVI metadata inspection with duration, frame rate, frame count, dimensions, and
  codec.
- Optional MediaPipe Tasks pose extraction with an explicit local model path and configurable
  frame step.
- Long-form normalized landmarks with per-landmark visibility and presence confidence.
- Trial- and phase-level pose detection, trunk lean, pelvis trajectory, ankle-separation, and
  lower-limb symmetry proxy features.
- Metadata-only video processing when pose estimation is disabled.

## Fusion, dual-task cost, and modeling

- Trial-level outer fusion that preserves missing modalities and adds explicit availability flags.
- Machine-readable feature inventory with modality, dtype, observed, and missing counts.
- Configurable single-modality and multimodal feature-set comparisons.
- Config-gated, pair-level single-/dual-task cost with explicit metric direction and preserved
  source values.
- All-sample and complete-modality cohort evaluations.
- Fold-local median imputation and scaling inside scikit-learn Pipelines.
- Participant-grouped regression with linear, ridge, and random-forest baselines.
- Participant-stratified grouped binary classification with logistic and random-forest baselines.
- Fold metrics, summary comparisons, out-of-fold predictions, skipped evaluations, and a split
  audit proving zero participant overlap.

## Reproducibility and presentation

- Deterministic aggregate Markdown reporting without participant IDs, raw paths, or individual
  feature values.
- Executed synthetic-workflow notebook with modality, QC, and interpretation checks.
- Public synthetic report example, reproducibility checklist, and architecture documentation.
- Citation metadata, automated tests, formatting checks, and continuous integration across
  supported Python versions.
