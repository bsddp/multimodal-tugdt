# Multimodal TUG-DT Analysis Pipeline

A reproducible research pipeline for organizing, validating, and ultimately analyzing
multimodal data collected during single-task and dual-task Timed Up and Go (TUG) assessments.
The planned modalities are IMU/Xsens-derived motion, video, audio, footswitch signals, manual
phase annotations, and clinical or demographic metadata.

> Status: Milestones 1 and 2 are implemented. The repository provides the project and manifest
> contracts, privacy-safe synthetic data, configurable IMU CSV adapters, signal quality control,
> filtering, resampling, manual TUG phase segmentation, interpretable IMU features, plots, and
> automated tests. Cross-modal synchronization and modeling remain future work.

## Why this project exists

Human-movement studies often collect files with different clocks, sampling rates, naming
conventions, and missing modalities. Analysis can become difficult to reproduce before any
model is trained. This project starts with an explicit data contract: each trial has stable
identifiers, declared paths, configurable conditions, and validation that fails clearly when
the contract is broken.

The long-term research goal is to support clinically interpretable investigation of
cognitive-motor interference. This software is a research tool. It does not diagnose disease
and currently makes no clinical claims.

## Supported through Milestone 2

- YAML configuration with paths resolved from an explicit project root
- CSV participant/trial manifest with optional missing modalities
- Validation of schema, identifiers, conditions, duplicate trials, and referenced files
- Deterministic 20-second synthetic IMU, audio, footswitch, annotation, and clinical fixtures
- Installable `tugdt` command-line interface
- Structured validation reports and visible warnings for missing optional modalities
- Wide- and long-format IMU CSV adapters with configurable canonical column mapping
- Timestamp, sampling interval, duplicate, missing-value, and amplitude quality checks
- SI unit conversion, optional constant-gravity removal, low-pass filtering, resampling, and
  quaternion normalization
- Externally annotated TUG phase validation and half-open interval slicing
- Trial-level and phase-level step, acceleration, jerk, and turning features
- Per-trial QC JSON, project-level QC CSV, processed IMU CSV, and annotated overview plots
- `preprocess`, `extract-features`, and current-stage `run-all` commands
- Automated unit and integration tests

Video is represented in the manifest contract but intentionally absent from the synthetic
demo. That absence exercises the missing-modality behavior without creating a fake clinical
video.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For runtime-only installation, use `python -m pip install -e .`.

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
tugdt extract-features --config configs/example.yaml
```

Generated private/derived outputs are ignored by Git and written under:

```text
data/processed/<participant>/<session>/<trial>/imu.csv
data/processed/<participant>/<session>/<trial>/imu_qc.json
outputs/qc/imu_preprocessing.csv
outputs/features/imu_features.csv
outputs/plots/*_imu.png
```

Run the test suite and code checks:

```bash
pytest
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
│   ├── feature_dictionary.md
│   └── imu_pipeline.md
├── src/multimodal_tugdt/
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── logging_utils.py
│   ├── synthetic.py
│   ├── io/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── features/
│   └── visualization/
├── tests/
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
    D --> E["Manual TUG phase slicing"]
    E --> F["Trial and phase features"]
    D --> G["QC metadata and overview plot"]
```

The feature code never uses original vendor column names. `configs/example.yaml` maps source
columns to canonical signals such as `acc_ap`, `acc_ml`, `acc_vertical`, and `gyro_yaw`.
Wide-format files store one signal per column. Long-format files first select the configured
`target_sensor`. Direct MVNX parsing is deliberately rejected with export guidance rather than
silently making assumptions about an Xsens schema.

See [IMU pipeline details](docs/imu_pipeline.md) and the
[feature dictionary](docs/feature_dictionary.md).

## Synthetic demonstration

The generator creates a pair of single-task and dual-task trials for each requested synthetic
participant. It includes simple periodic signals and known TUG phase boundaries so future
milestones have stable software fixtures.

Synthetic data are provided solely for demonstrating the software workflow and should not be
interpreted as clinically valid recordings. The generator creates no names, faces, natural
speech, hospital identifiers, or real participant measurements.

## Privacy

The repository ignores `data/raw`, `data/interim`, `data/processed`, and `outputs/private` by
default. Do not commit identifiable videos, voices, medical-record identifiers, raw clinical
files, restricted Xsens exports, or any research data that lack explicit authorization for
public release. Public examples must be synthetic or appropriately de-identified and approved.

## Roadmap

1. **Milestone 1 — foundation (complete):** package, configuration, manifest, synthetic data,
   CLI, logging, and tests.
2. **Milestone 2 — IMU (complete):** configurable CSV adapters, timestamp QC, filtering,
   resampling, segmentation, interpretable features, and plots.
3. **Milestone 3 — synchronization:** explicit offsets, reference timelines, metadata, and
   alignment QC.
4. **Milestone 4 — audio and footswitch:** voice-activity and gait-event features.
5. **Milestone 5 — video interface:** metadata and optional pose extraction.
6. **Milestone 6 — fusion and baselines:** modality-prefixed features and participant-grouped
   scikit-learn evaluation without leakage.
7. **Milestone 7 — research presentation:** reports, example outputs, notebooks, limitations,
   and citation guidance.

Deep learning, diagnostic claims, automatic silent alignment, and row-level random splitting
are outside the current scope.

## Current limitations

- Step events are estimated from vertical pelvis acceleration with configurable peak detection.
  They require validation against footswitch or another reference before research interpretation.
- Stance, swing, double-support, stride, and left-right asymmetry are not inferred from a single
  pelvis signal. Those belong to the footswitch/multi-sensor milestone.
- Constant gravity subtraction assumes the configured vertical channel includes gravity with a
  known sign. Set `gravity_removal: none` for linear-acceleration inputs.
- Quaternion resampling uses component-wise interpolation followed by normalization; direct
  spherical interpolation is not yet implemented.
- Manual annotation timestamps are assumed to use the IMU trial clock until explicit
  cross-modal synchronization is implemented in Milestone 3.

## License

Code is released under the MIT License. This license does not grant permission to use any
third-party or participant data.
