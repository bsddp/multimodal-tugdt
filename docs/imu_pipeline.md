# IMU pipeline

Milestone 2 converts configured IMU CSV files into uniform SI-unit time series, QC evidence,
phase slices, interpretable features, and overview plots. It is designed to fail visibly when
the input contract is unsafe.

## Canonical signals

Source columns are mapped in YAML. Downstream code only uses canonical names:

```yaml
imu:
  format: wide_csv
  columns:
    timestamp: timestamp
    acc_ap: pelvis_acc_ap
    acc_ml: pelvis_acc_ml
    acc_vertical: pelvis_acc_vertical
    gyro_yaw: pelvis_gyro_yaw
    quat_w: quat_w
    quat_x: quat_x
    quat_y: quat_y
    quat_z: quat_z
```

`wide_csv` expects one column per signal. `long_csv` also requires a `sensor_name` mapping and
selects `target_sensor` before applying the signal mapping. Missing optional signals are logged;
a missing timestamp or a file with no mapped signal is an error.

`mvnx` is an explicit unsupported adapter in this milestone. It raises an error asking for a CSV
export. This prevents a partial parser from silently misinterpreting proprietary schema variants.

## Processing order

1. Coerce timestamps to numeric and remove invalid timestamp rows.
2. Record and sort nonmonotonic timestamps.
3. Record and remove duplicate timestamps, keeping the first occurrence.
4. Coerce signals to numeric and record missing ratios before interpolation.
5. Interpolate internal/edge missing values when a column has at least one observed value.
6. Convert acceleration and angular velocity to `m/s²` and `rad/s`.
7. Optionally subtract a configured constant gravity value from the vertical channel.
8. Record configurable amplitude anomaly ratios.
9. Estimate the original sampling rate from the median positive time interval.
10. Apply a Butterworth zero-phase low-pass filter to acceleration and angular velocity.
11. Resample to a uniform configured rate using linear interpolation.
12. Normalize complete four-component quaternions.

The cutoff is checked against both input and target Nyquist frequencies. Signals too short for
zero-phase filtering are retained unfiltered with a QC warning instead of crashing or silently
switching algorithms.

## QC evidence

Each processed trial receives `imu_qc.json`. The project summary
`outputs/qc/imu_preprocessing.csv` includes:

- input and output sample counts;
- invalid, duplicate, and nonmonotonic timestamp evidence;
- estimated input and output rates;
- sampling-interval coefficient of variation;
- per-column missing and amplitude-anomaly ratios;
- warnings and pass/warning/fail status.

## Phase segmentation

External annotations are the authority. Intervals must have numeric `start_time < end_time`, sit
inside the processed trial, and contain samples. Slicing uses `[start_time, end_time)` so adjacent
phases do not double-count their shared boundary.

Annotation times must already use the IMU reference clock. Milestone 3 validates and copies them
without shifting, while audio, video, and footswitch clocks require explicit offsets.

## Reproducibility commands

```bash
tugdt preprocess --config configs/example.yaml
tugdt extract-features --config configs/example.yaml
```

Or run the implemented stages together:

```bash
tugdt run-all --config configs/example.yaml
```
