"""Non-interactive IMU overview plots with annotation overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from multimodal_tugdt.segmentation.manual import Segment  # noqa: E402


def plot_imu_overview(
    frame: pd.DataFrame,
    segments: list[Segment],
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    """Save acceleration and yaw-velocity traces with external phase labels."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    colors = {"acc_ap": "#0072B2", "acc_ml": "#D55E00", "acc_vertical": "#009E73"}
    for column, color in colors.items():
        if column in frame:
            axes[0].plot(frame["timestamp"], frame[column], label=column, color=color, linewidth=1)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].grid(alpha=0.2)

    if "gyro_yaw" in frame:
        axes[1].plot(
            frame["timestamp"],
            frame["gyro_yaw"],
            color="#CC79A7",
            linewidth=1,
            label="gyro_yaw",
        )
        axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Yaw velocity (rad/s)")
    axes[1].set_xlabel("Trial time (s)")
    axes[1].grid(alpha=0.2)

    phase_colors = ("#E69F00", "#56B4E9", "#F0E442", "#009E73", "#0072B2")
    for index, segment in enumerate(segments):
        color = phase_colors[index % len(phase_colors)]
        for axis in axes:
            axis.axvspan(segment.start_time, segment.end_time, color=color, alpha=0.08)
        center = (segment.start_time + segment.end_time) / 2
        axes[0].text(
            center,
            1.01,
            segment.name,
            rotation=35,
            ha="left",
            va="bottom",
            fontsize=7,
            transform=axes[0].get_xaxis_transform(),
        )
    figure.suptitle(title)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination

