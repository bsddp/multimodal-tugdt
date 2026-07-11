"""Timeline coverage plots for synchronization quality control."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from multimodal_tugdt.synchronization.timeline import (  # noqa: E402
    AlignmentResult,
    Timeline,
)


def plot_synchronization_overview(
    reference: Timeline,
    alignments: list[AlignmentResult],
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    """Plot aligned modality extents on the declared reference clock."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [(reference.modality, reference.native_start_seconds, reference.native_end_seconds)]
    rows.extend(
        (item.target_modality, item.reference_start_seconds, item.reference_end_seconds)
        for item in alignments
    )
    figure_height = max(2.8, 1.0 + 0.65 * len(rows))
    figure, axis = plt.subplots(figsize=(11, figure_height), constrained_layout=True)
    status_colors = {"pass": "#009E73", "warning": "#E69F00", "fail": "#D55E00"}
    for index, (_modality, start, end) in enumerate(rows):
        if index == 0:
            color = "#0072B2"
            label = "reference"
        else:
            alignment = alignments[index - 1]
            color = status_colors[alignment.qc_status]
            label = f"offset {alignment.offset_seconds:+.3f} s"
        axis.barh(index, end - start, left=start, height=0.52, color=color, alpha=0.85)
        axis.text(end, index, f"  {label}", va="center", fontsize=9)
    axis.axvline(reference.native_start_seconds, color="#333333", linestyle="--", linewidth=1)
    axis.axvline(reference.native_end_seconds, color="#333333", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(rows)), [row[0] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel(f"{reference.modality.upper()} reference time (s)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
