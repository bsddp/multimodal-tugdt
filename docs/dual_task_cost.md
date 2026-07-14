# Dual-task cost

Dual-task cost is an optional, pair-level derived artifact. Enable it explicitly in configuration:

```yaml
dual_task_cost:
  enabled: true
  group_columns: [participant_id, session_id]
  single_condition: single_task
  dual_condition: dual_task
  metrics:
    imu__cadence_steps_min: higher_is_better
    imu__step_time_cv_pct: higher_is_worse
    imu__turn_duration_s: higher_is_worse
```

For metrics where higher values are worse:

```text
cost = (dual - single) / abs(single) * 100
```

For metrics where higher values are better:

```text
cost = (single - dual) / abs(single) * 100
```

Positive values therefore always indicate deterioration under dual-task conditions; negative
values indicate improvement. The output preserves each single-task value, dual-task value, and
normalized cost in `outputs/features/dual_task_costs.csv`.

Pairing is deliberately strict. `participant_id` is mandatory, each configured pair may contain
at most one trial for each condition, duplicate candidate pairs raise an error, and groups missing
either condition are skipped. A zero or missing single-task denominator produces a blank cost
rather than an infinite or fabricated value.

Metric direction is protocol-specific and must be declared rather than inferred from a column
name. The output is a research feature table, not a diagnosis or validated cognitive-motor marker.
