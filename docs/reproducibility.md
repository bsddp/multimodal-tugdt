# Reproducibility and research-use checklist

## Reproduce the public demonstration

Use Python 3.11 or newer from the repository root:

```bash
python -m pip install -e '.[dev]'
tugdt generate-synthetic --output data/synthetic
tugdt run-all --config configs/example.yaml
pytest -q
ruff format --check .
ruff check .
```

The committed notebook can be rebuilt and executed with:

```bash
python scripts/build_demo_notebook.py
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/01_synthetic_workflow.ipynb
```

## Record for each study

- Repository version or commit hash and Python environment.
- Configuration file and immutable manifest snapshot.
- Acquisition devices, firmware, sampling units, and export procedures.
- Synchronization event, offset, uncertainty, operator, and clock convention.
- Annotation protocol, raters, agreement procedure, and phase definitions.
- Feature definitions and all non-default thresholds.
- Dual-task pairing keys, condition labels, metric directions, and incomplete-pair counts.
- Participant-grouped split audit and any external validation cohort.
- Missing-modality policy and the number of observations in every comparison.
- Approval and governance basis for storing or sharing each modality.

## Public-release boundary

Only synthetic or explicitly approved de-identified examples belong in this repository. Raw and
derived study directories are ignored by default. Faces, natural voices, medical-record identifiers,
restricted vendor exports, and unapproved clinical tables must not be committed.

## Interpretation boundary

Passing software tests demonstrates implementation behavior, not physiological, scientific, or
clinical validity. Energy VAD is not transcription; footswitch threshold crossings are not assumed
to be heel strikes; monocular pose measures are two-dimensional proxies; grouped cross-validation
is not external validation; and the synthetic dataset cannot support a model-performance claim.

Regenerate the README visual only after running the complete synthetic workflow:

```bash
python scripts/build_readme_visual.py
```
