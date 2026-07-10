"""Deterministic, non-clinical multimodal data for workflow demonstrations."""

from __future__ import annotations

import csv
import math
import random
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

STANDARD_PHASES = (
    ("baseline_sitting", 0.0, 2.0),
    ("sit_to_stand", 2.0, 3.0),
    ("outbound_walk", 3.0, 8.0),
    ("turn_1", 8.0, 10.0),
    ("return_walk", 10.0, 15.0),
    ("turn_to_sit", 15.0, 17.0),
    ("final_sitting", 17.0, 20.0),
)


@dataclass(frozen=True)
class SyntheticDataset:
    """Paths created by :func:`generate_synthetic_dataset`."""

    root: Path
    manifest: Path
    clinical: Path
    trial_count: int


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_imu(path: Path, condition: str, sampling_rate: int, seed: int) -> None:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    samples = 20 * sampling_rate + 1
    condition_scale = 0.84 if condition == "dual_task" else 1.0
    for index in range(samples):
        time = index / sampling_rate
        walking = 3.0 <= time < 8.0 or 10.0 <= time < 15.0
        turning = 8.0 <= time < 10.0
        gait = math.sin(2 * math.pi * 1.8 * condition_scale * time) if walking else 0.0
        noise = rng.gauss(0.0, 0.015)
        mediolateral = 0.12 * math.cos(2 * math.pi * 1.8 * time) * int(walking) + noise
        yaw_velocity = 1.4 * math.sin(math.pi * (time - 8) / 2) if turning else noise
        rows.append(
            {
                "timestamp": f"{time:.3f}",
                "pelvis_acc_ap": f"{0.35 * gait + noise:.6f}",
                "pelvis_acc_ml": f"{mediolateral:.6f}",
                "pelvis_acc_vertical": f"{9.81 + 0.55 * abs(gait) + noise:.6f}",
                "pelvis_gyro_yaw": f"{yaw_velocity:.6f}",
                "quat_w": "1.0",
                "quat_x": "0.0",
                "quat_y": "0.0",
                "quat_z": "0.0",
            }
        )
    _write_csv(path, list(rows[0]), rows)


def _contact_state(time: float, side: str, condition: str) -> int:
    walking = 3.0 <= time < 8.0 or 10.0 <= time < 15.0
    if not walking:
        return 0
    cycle = 1.15 if condition == "dual_task" else 1.0
    phase = ((time - 3.0) % cycle) / cycle
    if side == "left":
        return int(phase < 0.58)
    return int(0.42 <= phase < 0.98)


def _write_footswitch(path: Path, condition: str, sampling_rate: int = 100) -> None:
    rows = [
        {
            "timestamp": f"{index / sampling_rate:.3f}",
            "left_contact": _contact_state(index / sampling_rate, "left", condition),
            "right_contact": _contact_state(index / sampling_rate, "right", condition),
        }
        for index in range(20 * sampling_rate + 1)
    ]
    _write_csv(path, list(rows[0]), rows)


def _write_annotations(path: Path) -> None:
    rows = [
        {
            "segment_name": name,
            "start_time": start,
            "end_time": end,
            "source": "synthetic",
            "confidence": 1.0,
            "notes": "Demonstration only; not a clinical annotation.",
        }
        for name, start, end in STANDARD_PHASES
    ]
    _write_csv(path, list(rows[0]), rows)


def _write_audio(path: Path, condition: str, sampling_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    amplitude = 5_000
    speech_windows = [(4.0, 4.5), (6.0, 6.5), (11.0, 11.5), (13.0, 13.5)]
    if condition == "dual_task":
        speech_windows = [(4.2, 4.7), (7.0, 7.5), (11.8, 12.3), (14.0, 14.5)]
    samples = array("h")
    for index in range(20 * sampling_rate):
        time = index / sampling_rate
        active = any(start <= time < end for start, end in speech_windows)
        value = int(amplitude * math.sin(2 * math.pi * 220 * time)) if active else 0
        samples.append(value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sampling_rate)
        handle.writeframes(samples.tobytes())


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def generate_synthetic_dataset(
    output_dir: str | Path,
    *,
    participants: int = 1,
    seed: int = 42,
    project_root: str | Path | None = None,
) -> SyntheticDataset:
    """Generate a deterministic 20-second demo for two TUG conditions per participant.

    The signals are software fixtures, not physiologically or clinically valid recordings.
    """
    if participants < 1:
        raise ValueError("participants must be at least 1")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    clinical_path = root / "clinical.csv"
    clinical_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for participant_index in range(1, participants + 1):
        participant_id = f"P{participant_index:03d}"
        clinical_rows.append(
            {
                "participant_id": participant_id,
                "age": 70 + participant_index,
                "sex": "synthetic",
                "moca": 25 - participant_index,
                "falls_12m": participant_index % 2,
            }
        )
        for trial_index, condition in enumerate(("single_task", "dual_task"), start=1):
            trial_id = f"T{trial_index:02d}"
            trial_dir = root / participant_id / "S01" / trial_id
            imu_path = trial_dir / "imu.csv"
            footswitch_path = trial_dir / "footswitch.csv"
            annotation_path = trial_dir / "annotations.csv"
            audio_path = trial_dir / "audio.wav"
            _write_imu(imu_path, condition, 100, seed + participant_index * 10 + trial_index)
            _write_footswitch(footswitch_path, condition)
            _write_annotations(annotation_path)
            _write_audio(audio_path, condition)
            manifest_rows.append(
                {
                    "participant_id": participant_id,
                    "session_id": "S01",
                    "condition": condition,
                    "trial_id": trial_id,
                    "imu_path": _portable_path(imu_path, base),
                    "video_path": "",
                    "audio_path": _portable_path(audio_path, base),
                    "footswitch_path": _portable_path(footswitch_path, base),
                    "annotation_path": _portable_path(annotation_path, base),
                    "clinical_path": _portable_path(clinical_path, base),
                }
            )

    _write_csv(clinical_path, list(clinical_rows[0]), clinical_rows)
    manifest_path = root / "participants.csv"
    _write_csv(manifest_path, list(manifest_rows[0]), manifest_rows)
    (root / "README.md").write_text(
        "# Synthetic demonstration data\n\n"
        "These generated signals exist solely to demonstrate the software workflow. "
        "They are not physiologically or clinically valid recordings and must not be "
        "used to make health claims. No real participant data are included.\n",
        encoding="utf-8",
    )
    return SyntheticDataset(
        root=root,
        manifest=manifest_path,
        clinical=clinical_path,
        trial_count=len(manifest_rows),
    )
