# IMU feature dictionary

All exported IMU features use the `imu__` prefix. Rows are identified by participant, session,
trial, condition, `feature_level`, and `segment_name`.

| Feature | Unit | Meaning |
|---|---:|---|
| `imu__sample_count` | samples | Samples in the trial or phase row |
| `imu__duration_s` | s | Trial or annotated phase duration |
| `imu__sampling_rate_hz` | Hz | Rate estimated from processed timestamps |
| `imu__step_count` | steps | Vertical-acceleration peaks in annotated straight walking |
| `imu__cadence_steps_min` | steps/min | Step peaks divided by straight-walking duration |
| `imu__mean_step_time_s` | s | Mean interval between peaks within each walking phase |
| `imu__step_time_sd_s` | s | Sample SD of within-phase step intervals |
| `imu__step_time_cv_pct` | % | Step-time SD divided by mean, times 100 |
| `imu__pelvis_acc_rms_ap_m_s2` | m/s² | AP acceleration root mean square |
| `imu__pelvis_acc_rms_ml_m_s2` | m/s² | ML acceleration root mean square |
| `imu__pelvis_acc_rms_vertical_m_s2` | m/s² | Vertical acceleration root mean square |
| `imu__pelvis_acc_range_ml_m_s2` | m/s² | ML acceleration maximum minus minimum |
| `imu__pelvis_jerk_rms_m_s3` | m/s³ | RMS magnitude of available acceleration derivatives |
| `imu__angular_velocity_rms_rad_s` | rad/s | Yaw angular-velocity RMS |
| `imu__turn_duration_s` | s | Duration of annotated phases whose name begins with `turn` |
| `imu__peak_yaw_velocity_rad_s` | rad/s | Peak absolute yaw velocity in turn phases |
| `imu__mean_abs_yaw_velocity_rad_s` | rad/s | Mean absolute yaw velocity in turn phases |
| `imu__turn_smoothness_rad_s2` | rad/s² | RMS time derivative of yaw velocity in turn phases |

Trial-level step statistics combine `outbound_walk`, `return_walk`, and
`combined_straight_walk` annotations. Intervals are computed within each phase, so the time gap
between outbound and return walking is not treated as a step. Trial-level turning statistics
combine all annotated phase names beginning with `turn`.

Non-applicable phase features are left blank/NaN. For example, a `turn_1` phase does not receive
a fabricated cadence value.

Step peaks and acceleration-derived range features are software baselines, not validated gait
events or clinical biomarkers. Stance, swing, double support, stride, gait speed, and laterality
are intentionally absent until suitable sensor/reference data are implemented.

