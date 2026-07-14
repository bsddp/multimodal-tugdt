# Feature fusion and baseline modeling

Milestone 6 creates a trial-level research matrix and compares interpretable single- and
multimodal baselines. It does not train deep networks, tune hyperparameters, or claim clinical
validity.

## Fusion contract

The fusion command reads the trial rows from each modality feature table and outer-joins them to the
manifest identifiers:

```text
participant_id + session_id + trial_id + condition
```

Feature names keep their source prefixes:

```text
imu__       audio__       video__       footswitch__       clinical__
```

Every configured modality also receives an availability indicator such as `availability__audio`.
Missing modalities do not cause a trial to be discarded, and missing numeric values are not filled
during fusion.

Clinical tables are joined by `participant_id`. Conflicting duplicate clinical rows fail rather
than selecting one silently. Non-numeric clinical fields remain in the fused CSV for provenance but
are excluded from the current numeric baseline predictors.

Run fusion independently with:

```bash
tugdt fuse-features --config configs/example.yaml
```

## Feature-set and cohort comparisons

Feature sets are declared explicitly:

```yaml
fusion:
  modalities: [imu, audio, video, footswitch, clinical]
  feature_sets:
    imu: [imu]
    imu_audio: [imu, audio]
    imu_clinical: [imu, clinical]
    all_available: [imu, audio, video, footswitch, clinical]
```

Each feature set can be evaluated in two cohorts:

- `all_samples`: retain target-known trials and allow fold-local imputation plus availability flags;
- `complete_modalities`: retain only trials with every modality in that feature set.

Both are reported because complete-case results may describe a different, potentially selected
population from the all-sample analysis.

## Leakage controls

All evaluation uses `participant_id` as the group. Trials from one participant cannot appear in
both training and testing. `split_audit.csv` stores the exact train/test participant lists and an
overlap count that must remain zero.

Median imputation and, for linear models, standardization are inside an sklearn Pipeline. They are
fitted separately on each training fold and then applied to that fold's held-out participants. No
whole-dataset imputation or scaling statistics are used.

Regression uses `GroupKFold`. Binary classification uses shuffled `StratifiedGroupKFold` with a
fixed seed and reduces the requested fold count when participant/class counts are smaller.

## Models and metrics

Regression baselines:

- linear regression;
- ridge regression;
- random-forest regression.

Regression metrics are MAE, RMSE, R², and Spearman correlation.

Binary classification baselines:

- logistic regression with balanced class weights;
- random-forest classification with balanced class weights.

Classification metrics are balanced accuracy, ROC AUC, precision, recall, F1, sensitivity, and
specificity. `positive_label` must be declared explicitly for classification so sensitivity and
specificity cannot silently change direction. ROC AUC is blank when a test fold contains only one
class.

## Configuration and execution

```yaml
modeling:
  enabled: true
  target_column: clinical__moca
  task_type: regression
  group_column: participant_id
  folds: 5
  random_seed: 42
  models: [ridge, random_forest]
  cohort_modes: [all_samples, complete_modalities]
  positive_label: null
```

`modeling.enabled` controls whether baseline evaluation runs as part of `tugdt run-all`. The
explicit command always attempts the configured evaluation:

```bash
tugdt run-baselines --config configs/example.yaml
```

If a feature set has no observed values, a complete-modality cohort is empty, or too few independent
participant groups exist, no metric is fabricated. The combination and reason are written to
`skipped_evaluations.csv`.

## Interpretation boundary

Cross-validation estimates internal performance under the configured splits. It is not external
validation, a diagnostic study, nested model selection, or evidence of clinical utility. Comparing
many feature sets can itself create selection bias; results should be treated as exploratory until
confirmed on a prespecified independent cohort.
