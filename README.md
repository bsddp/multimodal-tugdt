# Multimodal TUG-DT Analysis Pipeline

A reproducible research pipeline for organizing, validating, and ultimately analyzing
multimodal data collected during single-task and dual-task Timed Up and Go (TUG) assessments.
The planned modalities are IMU/Xsens-derived motion, video, audio, footswitch signals, manual
phase annotations, and clinical or demographic metadata.

> Status: Milestone 1 is implemented. The repository currently provides the project package,
> configuration and manifest contracts, privacy-safe synthetic data generation, validation,
> logging, a command-line interface, and automated tests. Signal processing, synchronization,
> feature extraction, and modeling are deliberately reserved for later milestones.

## Why this project exists

Human-movement studies often collect files with different clocks, sampling rates, naming
conventions, and missing modalities. Analysis can become difficult to reproduce before any
model is trained. This project starts with an explicit data contract: each trial has stable
identifiers, declared paths, configurable conditions, and validation that fails clearly when
the contract is broken.

The long-term research goal is to support clinically interpretable investigation of
cognitive-motor interference. This software is a research tool. It does not diagnose disease
and currently makes no clinical claims.

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
│   └── data_schema.md
├── src/multimodal_tugdt/
│   ├── cli.py
│   ├── config.py
│   ├── logging_utils.py
│   ├── synthetic.py
│   └── io/manifest.py
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

1. **Milestone 1 — foundation:** package, configuration, manifest, synthetic data,
   CLI, logging, and tests.
2. **Milestone 2 — IMU:** configurable CSV adapters, timestamp QC, filtering, resampling,
   segmentation, interpretable features, and plots.
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

## License

Code is released under the MIT License. This license does not grant permission to use any
third-party or participant data.

