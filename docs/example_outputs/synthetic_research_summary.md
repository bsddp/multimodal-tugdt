# Multimodal TUG-DT Research Summary

> Aggregate software-generated report. It contains no participant-level measurements and must not be interpreted as a clinical result.

## Dataset contract

| Measure | Count |
|---|---|
| Participants | 1 |
| Trials | 2 |

### Conditions

| Condition | Trials |
|---|---|
| dual_task | 1 |
| single_task | 1 |

### Modality availability

| Modality | Available trials | Missing trials |
|---|---|---|
| imu | 2 | 0 |
| audio | 2 | 0 |
| video | 0 | 2 |
| footswitch | 2 | 0 |
| annotation | 2 | 0 |
| clinical | 2 | 0 |

## Quality-control status

| Stage | Artifact | Rows | Pass | Warning | Fail |
|---|---|---|---|---|---|
| IMU preprocessing | available | 2 | 2 | 0 | 0 |
| Synchronization | available | 4 | 4 | 0 | 0 |
| Audio processing | available | 2 | 2 | 0 | 0 |
| Footswitch processing | available | 2 | 2 | 0 | 0 |
| Video processing | available | 0 | 0 | 0 | 0 |

Warnings require review; they are not silently converted to passes. A row count may refer to trials or modality alignments depending on the stage.

## Derived feature artifacts

| Artifact | Status | Total rows | Trial rows | Phase rows |
|---|---|---|---|---|
| IMU | available | 16 | 2 | 14 |
| Audio | available | 16 | 2 | 14 |
| Footswitch | available | 16 | 2 | 14 |
| Video | available | 0 | 0 | 0 |
| Multimodal trial table | available | 2 | 2 | 0 |
| Dual-task cost pairs | available | 1 | 1 | 0 |

## Modeling status

Baseline modeling is disabled in the example configuration because the public synthetic dataset contains only one independent participant group.

## Reproducibility record

- Configuration file: `example.yaml`
- Clock mapping: `reference_time = native_time + offset_seconds`
- Grouped modeling unit: `participant_id`
- Dual-task cost: `enabled`
- Generated artifacts remain excluded from version control by default.

## Interpretation boundary

This pipeline is research software, not a medical device. Synthetic fixtures demonstrate software behavior only. Sensor features, voice-activity intervals, foot-contact events, video pose proxies, and cross-validated model metrics require protocol-specific validation before scientific or clinical interpretation.
