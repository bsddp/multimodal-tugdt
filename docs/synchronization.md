# Synchronization

The synchronization workflow creates an auditable mapping from each available target modality to
the processed IMU reference clock. It never assumes that file time zero is shared merely because
two files have the same duration.

## Clock equation

The implemented convention is:

```text
reference_time = native_time + offset_seconds
```

A positive offset places the target later on the IMU clock. A negative offset places it earlier.
A zero offset is still a declaration and must appear in configuration for every available target.

## Configuration

```yaml
synchronization:
  reference_modality: imu
  method: manual_offset
  offsets_seconds:
    video: 1.42
    audio: 0.85
    footswitch: -0.12
  uncertainty_seconds:
    video: 0.10
    audio: 0.03
    footswitch: 0.01
  timestamp_columns:
    footswitch: timestamp
  operator: researcher_id
  notes: Offsets measured from the documented synchronization event.
  maximum_duration_difference_s: 0.5
  minimum_overlap_ratio: 0.9
  generate_plots: true
```

An available audio, video, or footswitch file without an `offsets_seconds` entry fails alignment.
Missing uncertainty does not invent a value: it is stored as null and produces a QC warning.

## Native timeline readers

- Processed IMU and footswitch extents come from validated timestamp columns.
- WAV duration comes from container frame count and sampling rate.
- Other audio/video containers use `ffprobe` when it is installed.
- Media timelines currently start at native time zero; their configured offset places that zero on
  the IMU clock.

Footswitch timestamps must be finite and monotonic. The workflow does not sort or repair a target
clock because doing so could hide an acquisition problem.

## Recorded evidence

Each successful alignment records:

- reference and target modalities;
- synchronization method and alignment event type;
- offset and estimated uncertainty;
- operator and notes;
- target source path;
- native start/end and mapped reference start/end;
- target and reference durations and their absolute difference;
- overlap duration and overlap ratio;
- pass/warning/fail status and QC notes.

`sync_metadata.json` also stores the exact clock equation, reference source timeline, QC thresholds,
and per-modality errors. `outputs/qc/synchronization.csv` provides one row per attempted target
alignment across the project.

## QC rules

Alignment fails when mapped target and reference have no overlap. It produces a warning when:

- overlap divided by the shorter timeline is below `minimum_overlap_ratio`;
- absolute duration difference exceeds `maximum_duration_difference_s`; or
- estimated uncertainty is not provided.

Warnings preserve the synchronized output for review; structural errors cause the command to exit
nonzero.

## Current boundary

Only declared manual offsets are implemented. Hardware triggers, shared event markers, clap/beep
events, and cross-correlation are planned alignment strategies, not current claims. Manual TUG
annotations must already be on the IMU clock and are copied to `segments.csv` without shifting.
