# Multimodal TUG-DT Analysis Pipeline

[![CI](https://github.com/bsddp/multimodal-tugdt/actions/workflows/ci.yml/badge.svg)](https://github.com/bsddp/multimodal-tugdt/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/bsddp/multimodal-tugdt)](https://github.com/bsddp/multimodal-tugdt/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-176B87)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2A9D8F)](LICENSE)

A reproducible research pipeline for organizing, validating, synchronizing, and analyzing
multimodal data collected during single-task and dual-task Timed Up and Go (TUG) assessments.
The data contract covers IMU/Xsens-derived motion, video, audio, footswitch signals, manual
phase annotations, and clinical or demographic metadata.

> Status: The integrated pipeline covers data contracts, privacy-safe synthetic fixtures,
> multimodal quality control, explicit clock synchronization, manual TUG phase segmentation,
> interpretable feature extraction, missing-modality-aware fusion, participant-grouped baseline
> modeling, aggregate reporting, and an executed demonstration notebook. All components are
> research tools rather than clinical claims.

![Synthetic multimodal TUG-DT pipeline output](docs/assets/readme_pipeline_demo.png)

*Generated from the committed synthetic workflow. The values demonstrate software behavior and
are not clinical results.*

## Why this project exists

Human-movement studies often collect files with different clocks, sampling rates, naming
conventions, and missing modalities. Analysis can become difficult to reproduce before any
model is trained. This project starts with an explicit data contract: each trial has stable
identifiers, declared paths, configurable conditions, and validation that fails clearly when
the contract is broken.

The long-term research goal is to support clinically interpretable investigation of
cognitive-motor interference. This software is a research tool. It does not diagnose disease
and currently makes no clinical claims.

## Integrated pipeline capabilities

- Explicit manifest validation, quality control, reference-clock synchronization, and TUG phases.
- IMU, audio, footswitch, optional video pose, annotation, and clinical-data interfaces.
- Interpretable trial/phase features plus IMU–footswitch event-agreement evidence.
- Missing-modality-aware fusion and config-gated single-/dual-task cost features.
- Leakage-aware participant-grouped baselines with split-audit artifacts.
- Reproducible CLI, synthetic fixtures, notebook, aggregate report, tests, and CI.

See the [complete capability inventory](docs/capabilities.md) and
[pipeline architecture](docs/pipeline_diagram.md).

Video remains intentionally absent from the committed synthetic demo. That absence exercises
missing-modality behavior without creating a fake clinical video or normalizing the public release
of identifiable recordings. The video interface and feature mathematics are covered by generated
software fixtures in the test suite.

## What you get

Running `tugdt run-all --config configs/example.yaml` creates processed reference-clock time
series, trial/phase feature tables, QC evidence, synchronization plots, a fused trial matrix,
optional pair-level dual-task costs, and an aggregate research report.

### Synthetic demonstration snapshot

| QC stage | Passing rows | Failed rows |
|---|---:|---:|
| IMU preprocessing | 2/2 | 0 |
| Synchronization | 4/4 | 0 |
| Audio processing | 2/2 | 0 |
| Footswitch processing | 2/2 | 0 |

| Condition | IMU-event precision | Recall | F1 | Mean absolute timing error |
|---|---:|---:|---:|---:|
| Single task | 1.000 | 1.000 | 1.000 | 4.4 ms |
| Dual task | 1.000 | 1.000 | 1.000 | 3.7 ms |

The enabled synthetic dual-task-cost example preserves the source values and reports positive
cost as deterioration: cadence changes from 108 to 96 steps/min (`+11.1%` cost), while mean
footswitch step time changes from 0.556 to 0.661 seconds (`+18.9%` cost).

> **Aggregate report excerpt**
>
> Participants: **1** · Trials: **2** · Conditions: single task and dual task
>
> Modeling: disabled because the public fixture contains only one independent participant group.

These values are deterministic software checks, not performance benchmarks or clinical findings.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For runtime-only installation, use `python -m pip install -e .`.

Video metadata requires `ffprobe` from FFmpeg. Optional pose extraction additionally requires:

```bash
python -m pip install -e '.[video]'
```

A compatible MediaPipe Pose Landmarker `.task` model must be downloaded separately and referenced
in configuration; the repository does not silently download or redistribute model assets.

## Quick start

Generate the public demonstration data:

```bash
tugdt generate-synthetic --output data/synthetic
```

Validate it using the example project configuration:

```bash
tugdt validate-manifest --config configs/example.yaml
```

Run the implemented pipeline:

```bash
tugdt run-all --config configs/example.yaml
```

The same stages can be run separately:

```bash
tugdt preprocess --config configs/example.yaml
tugdt synchronize --config configs/example.yaml
tugdt process-audio --config configs/example.yaml
tugdt process-footswitch --config configs/example.yaml
tugdt process-video --config configs/example.yaml
tugdt extract-features --config configs/example.yaml
tugdt fuse-features --config configs/example.yaml
tugdt run-baselines --config configs/example.yaml
tugdt generate-report --config configs/example.yaml
```

The committed demonstration contains one synthetic participant, so `modeling.enabled` is false in
`configs/example.yaml`. This prevents invalid cross-validation during `run-all`. Explicit baseline
evaluation requires a study with at least two independent participant groups, and binary
classification requires at least two participant groups containing each class.

Generated private/derived outputs are ignored by Git and written under:

```text
data/processed/<participant>/<session>/<trial>/imu.csv
data/processed/<participant>/<session>/<trial>/imu_qc.json
data/processed/<participant>/<session>/<trial>/sync_metadata.json
data/processed/<participant>/<session>/<trial>/segments.csv
data/processed/<participant>/<session>/<trial>/footswitch_synced.csv
data/processed/<participant>/<session>/<trial>/audio_frames.csv
data/processed/<participant>/<session>/<trial>/audio_activity.csv
data/processed/<participant>/<session>/<trial>/audio_qc.json
data/processed/<participant>/<session>/<trial>/footswitch_processed.csv
data/processed/<participant>/<session>/<trial>/footswitch_events.csv
data/processed/<participant>/<session>/<trial>/footswitch_qc.json
data/processed/<participant>/<session>/<trial>/video_metadata.json
data/processed/<participant>/<session>/<trial>/video_pose_frames.csv
data/processed/<participant>/<session>/<trial>/video_pose_landmarks.csv
outputs/qc/imu_preprocessing.csv
outputs/qc/synchronization.csv
outputs/qc/audio_processing.csv
outputs/qc/footswitch_processing.csv
outputs/qc/video_processing.csv
outputs/features/imu_features.csv
outputs/features/audio_features.csv
outputs/features/footswitch_features.csv
outputs/features/video_features.csv
outputs/features/multimodal_features.csv
outputs/features/feature_inventory.csv
outputs/features/dual_task_costs.csv
outputs/modeling/fold_metrics.csv
outputs/modeling/summary_metrics.csv
outputs/modeling/predictions.csv
outputs/modeling/split_audit.csv
outputs/modeling/skipped_evaluations.csv
outputs/modeling/modeling_metadata.json
outputs/reports/research_summary.md
outputs/plots/*_imu.png
outputs/plots/*_synchronization.png
```

The generated report is aggregate by design: it omits participant identifiers, raw paths, and
participant-level measurements. A versioned synthetic example is available at
[the public research summary](docs/example_outputs/synthetic_research_summary.md).

Rebuild and execute the demonstration notebook with:

```bash
python scripts/build_demo_notebook.py
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/01_synthetic_workflow.ipynb
```

The committed [executed notebook](notebooks/01_synthetic_workflow.ipynb) runs the CLI, inspects
aggregate modality and QC artifacts, renders the report, and checks the intended missing-video
behavior. It does not report model performance from the one-participant synthetic fixture.

Run the test suite and code checks:

```bash
pytest
ruff format --check .
ruff check .
```

Use `tugdt --help` or a subcommand's `--help` for all options.

## Repository layout

```text
multimodal-tugdt/
├── configs/example.yaml
├── data/
│   ├── README.md
│   └── synthetic/                 # public generated demonstration only
├── docs/
│   ├── data_schema.md
│   ├── audio_footswitch.md
│   ├── capabilities.md
│   ├── dual_task_cost.md
│   ├── feature_dictionary.md
│   ├── imu_pipeline.md
│   ├── modeling.md
│   ├── pipeline_diagram.md
│   ├── reproducibility.md
│   ├── synchronization.md
│   ├── video_pipeline.md
│   ├── assets/
│   └── example_outputs/
├── notebooks/
│   └── 01_synthetic_workflow.ipynb
├── scripts/
│   ├── build_demo_notebook.py
│   └── build_readme_visual.py
├── src/multimodal_tugdt/
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── logging_utils.py
│   ├── synthetic.py
│   ├── io/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── synchronization/
│   ├── features/
│   ├── fusion/
│   ├── modeling/
│   ├── reporting/
│   └── visualization/
├── tests/
├── CHANGELOG.md
├── CITATION.cff
├── README.md
└── pyproject.toml
```

## Manifest contract

Each row represents one trial and must include:

```csv
participant_id,session_id,condition,trial_id,imu_path,video_path,audio_path,footswitch_path,annotation_path,clinical_path
```

Identifiers may not be blank. The tuple `participant_id + session_id + trial_id` must be
unique. Modality paths may be blank, but each row must reference at least one trial modality or
annotation. Relative paths are resolved from `project.root` in the YAML configuration, not from
the caller's current directory. See [the complete data schema](docs/data_schema.md).

## IMU pipeline

```mermaid
flowchart LR
    A["Configured IMU CSV"] --> B["Canonical column adapter"]
    B --> C["Timestamp and signal QC"]
    C --> D["SI units, filtering, resampling"]
    D --> E["Explicit multimodal clock alignment"]
    E --> H["Synchronization metadata and QC"]
    E --> J["Optional aligned video pose proxies"]
    D --> I["Manual TUG phase slicing"]
    I --> F["Trial and phase features"]
    D --> G["QC metadata and overview plot"]
```

The feature code never uses original vendor column names. `configs/example.yaml` maps source
columns to canonical signals such as `acc_ap`, `acc_ml`, `acc_vertical`, and `gyro_yaw`.
Wide-format files store one signal per column. Long-format files first select the configured
`target_sensor`. Direct MVNX parsing is deliberately rejected with export guidance rather than
silently making assumptions about an Xsens schema.

See [IMU pipeline details](docs/imu_pipeline.md) and the
[feature dictionary](docs/feature_dictionary.md). The clock mapping and QC contract are defined
in [synchronization details](docs/synchronization.md).

Audio VAD, foot-contact event definitions, and IMU agreement metrics are documented in
[audio and footswitch processing](docs/audio_footswitch.md).

Video metadata, optional MediaPipe configuration, landmark tables, and the mathematical limits of
the two-dimensional proxy features are documented in [video processing](docs/video_pipeline.md).

Feature fusion, missing-modality handling, grouped split rules, metrics, and modeling artifacts are
documented in [fusion and baseline modeling](docs/modeling.md).

The [complete architecture diagram](docs/pipeline_diagram.md) shows how configuration, sensor QC,
clock alignment, phase features, fusion, grouped modeling, and aggregate reporting connect. Use
the [reproducibility checklist](docs/reproducibility.md) when adapting the software to a study.

## Synthetic demonstration

The generator creates a pair of single-task and dual-task trials for each requested synthetic
participant. It includes simple periodic signals and known TUG phase boundaries so the complete
workflow has stable software fixtures.

Synthetic data are provided solely for demonstrating the software workflow and should not be
interpreted as clinically valid recordings. The generator creates no names, faces, natural
speech, hospital identifiers, or real participant measurements.

## Privacy

The repository ignores `data/raw`, `data/interim`, `data/processed`, and `outputs/private` by
default. Do not commit identifiable videos, voices, medical-record identifiers, raw clinical
files, restricted Xsens exports, or any research data that lack explicit authorization for
public release. Public examples must be synthetic or appropriately de-identified and approved.

## Integrated workflow

1. Validate a configuration-driven trial manifest and its declared modality files.
2. Load, quality-check, and preprocess IMU, audio, footswitch, video, annotation, and clinical data.
3. Map available signals onto an explicit IMU reference clock and preserve synchronization metadata.
4. Apply validated TUG phase annotations and extract trial- and phase-level interpretable features.
5. Build a missing-modality-aware trial matrix with stable modality prefixes and availability flags.
6. Evaluate configured single- and multimodal baselines with participant-grouped data splits.
7. Export QC evidence, feature inventories, split audits, predictions, aggregate research reports,
   plots, and reproducibility artifacts.

Deep learning, diagnostic claims, automatic silent alignment, and row-level random splitting
are outside the current scope.

## Current limitations

- Step events are estimated from vertical pelvis acceleration with configurable peak detection.
  They require validation against footswitch or another reference before research interpretation.
- Stance, swing, and left-right asymmetry are never inferred from a single pelvis signal; reported
  values come from the explicitly configured footswitch channels. Double support, stride, and gait
  speed remain outside the current feature set.
- Constant gravity subtraction assumes the configured vertical channel includes gravity with a
  known sign. Set `gravity_removal: none` for linear-acceleration inputs.
- Quaternion resampling uses SLERP with sign-continuous, normalized orientations. Missing endpoint
  orientations are held at the nearest valid value and still require protocol-specific validation.
- The current synchronization implementation applies declared manual offsets; it does not yet
  estimate offsets from triggers, events, cross-correlation, or signal content.
- Manual annotation timestamps must already use the IMU reference clock. They are validated and
  copied to processed outputs without an inferred shift.
- Example zero offsets are explicit synthetic-demo declarations and are not defaults for real
  recordings.
- Non-WAV audio decoding requires FFmpeg, and all video metadata inspection requires `ffprobe`.
- Energy VAD detects high-energy waveform intervals, not linguistic speech content. It does not
  perform speaker separation, transcription, or response scoring.
- Footswitch `contact` is a threshold crossing after debounce; it should not be called heel strike
  without validation against the acquisition hardware and protocol.
- IMU/footswitch agreement depends on configurable peak prominence and matching tolerance and must
  be reported with those parameters.
- Video pose extraction requires an externally obtained compatible MediaPipe model. The pipeline
  does not validate that a chosen model is appropriate for a population, camera view, or protocol.
- Video coordinates are monocular normalized image coordinates. Trunk lean, pelvis displacement,
  ankle separation, and symmetry outputs are explicitly proxies, not calibrated 3D kinematics,
  joint angles, step length, or clinical scores.
- Baseline modeling accepts numeric predictors only. Non-numeric clinical fields remain in the
  fused table for provenance but are not silently encoded.
- Classification is binary. ROC AUC is left blank for any test fold containing only one class
  rather than inventing a value.
- Grouped cross-validation prevents participant overlap but is not an external validation cohort,
  nested model-selection study, or guarantee of generalization. No hyperparameter search is run.
- The committed one-participant demo cannot support valid modeling; it demonstrates fusion only.
- Dual-task cost is opt-in and pair-level. It requires one declared single/dual trial per configured
  group; duplicates raise an error and incomplete pairs are skipped.
- The aggregate report summarizes artifact availability and QC counts. It intentionally omits
  individual measurements and cannot replace a protocol-specific statistical analysis.

## Citation

Use the repository commit or release version needed to reproduce an analysis. GitHub can read the
included [citation metadata](CITATION.cff); no DOI or publication is claimed by this repository.

## License

Code is released under the MIT License. This license does not grant permission to use any
third-party or participant data.
