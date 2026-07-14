"""Build the README hero visual from synthetic pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1F2933"
MUTED = "#667085"
BLUE = "#176B87"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
GRID = "#D8DEE6"
PANEL = "#FFFFFF"
BACKGROUND = "#F6F8FB"


def _read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required pipeline artifact is missing: {path}. Run "
            "'tugdt run-all --config configs/example.yaml' first."
        )
    return pd.read_csv(path)


def _style_panel(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    for spine in axis.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)


def main() -> None:
    """Render synchronized signals, QC counts, and event agreement."""
    root = Path(__file__).resolve().parents[1]
    trial_dir = root / "data/processed/P001/S01/T02"
    imu = _read_required(trial_dir / "imu.csv")
    segments = _read_required(trial_dir / "segments.csv")
    events = _read_required(trial_dir / "footswitch_events.csv")
    audio = _read_required(trial_dir / "audio_activity.csv")
    agreement = _read_required(root / "outputs/features/footswitch_features.csv")

    qc_files = {
        "IMU": "imu_preprocessing.csv",
        "Synchronization": "synchronization.csv",
        "Audio": "audio_processing.csv",
        "Footswitch": "footswitch_processing.csv",
    }
    qc_rows = []
    for stage, filename in qc_files.items():
        frame = _read_required(root / "outputs/qc" / filename)
        qc_rows.append(
            {
                "stage": stage,
                "rows": len(frame),
                "passed": int(frame["qc_status"].eq("pass").sum()),
            }
        )
    qc = pd.DataFrame(qc_rows)
    trial_agreement = agreement.loc[agreement["feature_level"].eq("trial")].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    figure = plt.figure(figsize=(16, 7.5), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        2,
        5,
        left=0.055,
        right=0.97,
        bottom=0.12,
        top=0.82,
        wspace=0.55,
        hspace=0.5,
    )
    signal_axis = figure.add_subplot(grid[:, :3])
    qc_axis = figure.add_subplot(grid[0, 3:])
    agreement_axis = figure.add_subplot(grid[1, 3:])
    for axis in (signal_axis, qc_axis, agreement_axis):
        _style_panel(axis)

    figure.text(
        0.055,
        0.93,
        "Synthetic multimodal TUG-DT pipeline output",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.055,
        0.875,
        "Phase-aware motion · explicit reference-clock alignment · auditable quality control",
        fontsize=12,
        color=MUTED,
    )

    phase_palette = ["#E8F3F7", "#F5F1E5"]
    for index, segment in segments.iterrows():
        signal_axis.axvspan(
            segment["start_time"],
            segment["end_time"],
            color=phase_palette[index % 2],
            alpha=0.8,
            linewidth=0,
            zorder=0,
        )
        midpoint = (segment["start_time"] + segment["end_time"]) / 2
        short_name = str(segment["segment_name"]).replace("_", "\n")
        signal_axis.text(
            midpoint,
            0.97,
            short_name,
            transform=signal_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=MUTED,
        )

    speech = audio.loc[audio["activity"].eq("speech")]
    for interval in speech.itertuples(index=False):
        signal_axis.fill_between(
            [interval.start_time, interval.end_time],
            0.0,
            0.035,
            transform=signal_axis.get_xaxis_transform(),
            color=GOLD,
            alpha=0.95,
            linewidth=0,
        )
    signal_axis.plot(
        imu["timestamp"],
        imu["acc_vertical"],
        color=BLUE,
        linewidth=1.4,
        label="Pelvis vertical acceleration",
        zorder=3,
    )
    contacts = events.loc[events["event"].eq("contact")].copy()
    contacts["signal"] = np.interp(
        contacts["timestamp"],
        imu["timestamp"],
        imu["acc_vertical"],
    )
    marker_map = {"left": "o", "right": "^"}
    for side, marker in marker_map.items():
        subset = contacts.loc[contacts["side"].eq(side)]
        signal_axis.scatter(
            subset["timestamp"],
            subset["signal"],
            marker=marker,
            s=38,
            facecolor=PANEL,
            edgecolor=TEAL,
            linewidth=1.3,
            label=f"{side.title()} contact",
            zorder=4,
        )
    signal_axis.set_title("Synchronized dual-task trial", loc="left", color=INK, pad=12)
    signal_axis.set_xlabel("IMU reference time (s)")
    signal_axis.set_ylabel("Vertical acceleration (m/s²)")
    signal_axis.set_xlim(float(imu["timestamp"].min()), float(imu["timestamp"].max()))
    signal_axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    signal_axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    signal_axis.text(
        0.995,
        0.02,
        "Gold bands: energy-qualified audio activity",
        transform=signal_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )

    positions = np.arange(len(qc))
    qc_axis.barh(positions, qc["rows"], color="#E8EDF2", height=0.52)
    qc_axis.barh(positions, qc["passed"], color=BLUE, height=0.52)
    for position, row in zip(positions, qc.itertuples(index=False), strict=True):
        qc_axis.text(
            row.rows + 0.08,
            position,
            f"{row.passed}/{row.rows} pass",
            va="center",
            fontsize=9,
            color=INK,
        )
    qc_axis.set_yticks(positions, qc["stage"])
    qc_axis.invert_yaxis()
    qc_axis.set_xlim(0, max(4.8, float(qc["rows"].max()) + 0.8))
    qc_axis.set_title("Quality-control artifacts", loc="left", color=INK, pad=12)
    qc_axis.set_xlabel("Artifact rows")
    qc_axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)

    agreement_axis.axis("off")
    agreement_axis.set_title("IMU–footswitch event agreement", loc="left", color=INK, pad=12)
    headers = ["Condition", "Precision", "Recall", "F1", "Mean |error|"]
    x_positions = [0.02, 0.35, 0.53, 0.68, 0.82]
    for x, header in zip(x_positions, headers, strict=True):
        agreement_axis.text(
            x,
            0.78,
            header,
            transform=agreement_axis.transAxes,
            fontsize=8,
            fontweight="semibold",
            color=MUTED,
        )
    for row_index, row in enumerate(trial_agreement.itertuples(index=False)):
        y = 0.52 - row_index * 0.28
        condition = str(row.condition).replace("_", " ")
        values = [
            condition,
            f"{row.footswitch__imu_event_precision:.3f}",
            f"{row.footswitch__imu_event_recall:.3f}",
            f"{row.footswitch__imu_event_agreement_f1:.3f}",
            f"{row.footswitch__imu_event_mean_abs_error_s * 1000:.1f} ms",
        ]
        for x, value in zip(x_positions, values, strict=True):
            agreement_axis.text(
                x,
                y,
                value,
                transform=agreement_axis.transAxes,
                fontsize=10,
                fontweight="semibold" if x > 0.02 else "normal",
                color=TEAL if x > 0.02 else INK,
            )
        agreement_axis.plot(
            [0.02, 0.98],
            [y - 0.1, y - 0.1],
            transform=agreement_axis.transAxes,
            color=GRID,
            linewidth=0.7,
        )

    figure.text(
        0.055,
        0.035,
        "Synthetic software fixture only — values demonstrate pipeline behavior and are not "
        "clinical results.",
        fontsize=9,
        color=MUTED,
    )
    destination = root / "docs/assets/readme_pipeline_demo.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(destination.relative_to(root))


if __name__ == "__main__":
    main()
