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
events or clinical biomarkers. Stance, swing, and laterality are intentionally absent from IMU
features; Milestone 4 reports separate footswitch-derived estimates. Double support, stride, and
gait speed remain unimplemented.

## Audio features

| Feature | Unit | Meaning |
|---|---:|---|
| `audio__speech_segment_count` | intervals | VAD speech intervals overlapping the row window |
| `audio__speech_duration_s` | s | Total clipped VAD speech duration |
| `audio__silence_duration_s` | s | Window duration minus VAD speech duration |
| `audio__speech_ratio` | ratio | Speech duration divided by window duration |
| `audio__pause_count` | intervals | Internal gaps at least the configured pause duration |
| `audio__mean_pause_duration_s` | s | Mean qualifying internal pause |
| `audio__max_pause_duration_s` | s | Longest qualifying internal pause |
| `audio__first_response_latency_s` | s | First speech start after explicitly configured task start; trial rows only |
| `audio__response_count` | — | Blank until response/transcript labels are provided |
| `audio__correct_response_count` | — | Blank until correctness labels are provided |
| `audio__response_accuracy` | — | Blank until correctness labels are provided |

VAD speech means an energy-qualified interval, not verified human speech or a correct response.

## Footswitch features

| Feature | Unit | Meaning |
|---|---:|---|
| `footswitch__left_contact_count` | events | Stabilized left rising transitions |
| `footswitch__right_contact_count` | events | Stabilized right rising transitions |
| `footswitch__step_count` | events | Sum of left and right contact transitions |
| `footswitch__mean_left_stance_time_s` | s | Left contact-to-toe-off interval mean |
| `footswitch__mean_right_stance_time_s` | s | Right contact-to-toe-off interval mean |
| `footswitch__mean_left_swing_time_s` | s | Left toe-off-to-next-contact interval mean |
| `footswitch__mean_right_swing_time_s` | s | Right toe-off-to-next-contact interval mean |
| `footswitch__stance_time_asymmetry_pct` | % | Absolute left-right stance difference divided by their mean |
| `footswitch__mean_step_time_s` | s | Mean interval between consecutive contacts across sides |
| `footswitch__step_time_sd_s` | s | Sample SD of within-phase step intervals |
| `footswitch__step_time_cv_pct` | % | Step-time SD divided by mean, times 100 |
| `footswitch__imu_matched_event_count` | pairs | One-to-one matches inside the configured tolerance |
| `footswitch__imu_event_precision` | ratio | Matches divided by IMU peak events |
| `footswitch__imu_event_recall` | ratio | Matches divided by footswitch contact events |
| `footswitch__imu_event_agreement_f1` | ratio | Harmonic mean of event precision and recall |
| `footswitch__imu_event_mean_abs_error_s` | s | Mean absolute timing error of matched pairs |

Trial timing and agreement aggregate annotated straight-walking phases without treating the gap
between outbound and return walking as a step interval.
