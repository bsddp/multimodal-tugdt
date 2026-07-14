"""Build the committed synthetic-workflow notebook with nbformat."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    """Create a deterministic, reader-facing demonstration notebook."""
    root = Path(__file__).resolve().parents[1]
    destination = root / "notebooks" / "01_synthetic_workflow.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Synthetic Multimodal TUG-DT Workflow

## Goal

Run the public synthetic dataset through the complete pipeline, inspect aggregate quality-control
and modality-availability artifacts, and verify the software contract without presenting any
clinical finding. Synthetic fixtures demonstrate code behavior only."""
        ),
        nbf.v4.new_markdown_cell(
            """## Setup

Install the project with `python -m pip install -e '.[dev]'` before executing this notebook. The
cells locate the repository root, run the same public CLI used in the README, and show only
aggregate or synthetic outputs."""
        ),
        nbf.v4.new_code_cell(
            """import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

project_root = Path.cwd()
if not (project_root / "configs" / "example.yaml").is_file():
    project_root = project_root.parent

assert (project_root / "configs" / "example.yaml").is_file()
print("Repository and example configuration found.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Steps

### 1. Run the end-to-end workflow

The command validates the manifest, preprocesses and synchronizes the supported modalities,
extracts interpretable features, fuses trial-level tables, and writes an aggregate report."""
        ),
        nbf.v4.new_code_cell(
            """completed = subprocess.run(
    ["tugdt", "run-all", "--config", "configs/example.yaml"],
    cwd=project_root,
    text=True,
    capture_output=True,
    check=True,
)
print("Pipeline completed successfully.")"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. Inspect modality availability and quality control

Video is intentionally absent from the committed demo. This tests missing-modality handling
without publishing a face recording or manufacturing a fake clinical video."""
        ),
        nbf.v4.new_code_cell(
            """features = pd.read_csv(project_root / "outputs/features/multimodal_features.csv")
availability_columns = sorted(
    column for column in features.columns if column.startswith("availability__")
)
availability = (
    features[availability_columns]
    .sum()
    .rename_axis("modality")
    .reset_index(name="available_trials")
)
availability["modality"] = availability["modality"].str.replace("availability__", "", regex=False)
display(availability)"""
        ),
        nbf.v4.new_code_cell(
            """qc_files = {
    "IMU": "imu_preprocessing.csv",
    "Synchronization": "synchronization.csv",
    "Audio": "audio_processing.csv",
    "Footswitch": "footswitch_processing.csv",
    "Video": "video_processing.csv",
}
qc_rows = []
for stage, filename in qc_files.items():
    frame = pd.read_csv(project_root / "outputs/qc" / filename)
    counts = frame.get("qc_status", pd.Series(dtype=str)).value_counts()
    qc_rows.append(
        {
            "stage": stage,
            "rows": len(frame),
            "pass": int(counts.get("pass", 0)),
            "warning": int(counts.get("warning", 0)),
            "fail": int(counts.get("fail", 0)),
        }
    )
qc_summary = pd.DataFrame(qc_rows)
display(qc_summary)"""
        ),
        nbf.v4.new_code_cell(
            """plot_data = availability.set_index("modality")["available_trials"]
ax = plot_data.plot.bar(color="#2A6F97", figsize=(7, 3.5))
ax.set_title("Available synthetic trials by modality")
ax.set_xlabel("")
ax.set_ylabel("Trial count")
ax.set_ylim(0, max(2.2, float(plot_data.max()) + 0.2))
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 3. Read the aggregate report

The report deliberately omits participant identifiers, raw paths, and participant-level feature
values. It summarizes whether required software artifacts exist and states the interpretation
boundary."""
        ),
        nbf.v4.new_code_cell(
            'report_text = (project_root / "outputs/reports/research_summary.md")'
            '.read_text(encoding="utf-8")\n'
            "display(Markdown(report_text))"
        ),
        nbf.v4.new_markdown_cell(
            """## Checks

These executable checks make the demo's intended behavior auditable: two trials are fused, the
three generated sensor modalities and clinical metadata are available, video remains absent, and
no QC stage reports a failure."""
        ),
        nbf.v4.new_code_cell(
            """assert len(features) == 2
assert set(features["condition"]) == {"single_task", "dual_task"}
assert features["availability__imu"].eq(1).all()
assert features["availability__audio"].eq(1).all()
assert features["availability__footswitch"].eq(1).all()
assert features["availability__clinical"].eq(1).all()
assert features["availability__video"].eq(0).all()
assert qc_summary["fail"].sum() == 0
print("All synthetic workflow checks passed.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Next Steps

- Replace synthetic paths with authorized, protocol-specific data through the manifest contract.
- Record real synchronization offsets and uncertainty rather than copying the demo's explicit
  zero offsets.
- Validate sensor and pose proxy features against an appropriate reference standard.
- Enable grouped baseline modeling only after collecting enough independent participant groups.
- Treat all outputs as research artifacts, not diagnostic or medical-device results."""
        ),
    ]
    nbf.write(notebook, destination)
    print(destination.relative_to(root))


if __name__ == "__main__":
    main()
