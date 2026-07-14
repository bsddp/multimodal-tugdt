"""Command-line interface for reproducible project workflows."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from multimodal_tugdt import __version__
from multimodal_tugdt.config import ConfigurationError, load_config
from multimodal_tugdt.io.manifest import validate_manifest
from multimodal_tugdt.logging_utils import configure_logging
from multimodal_tugdt.pipeline import (
    extract_project_features,
    fuse_project_features,
    generate_report_project,
    preprocess_project,
    process_audio_project,
    process_footswitch_project,
    process_video_project,
    run_baselines_project,
    synchronize_project,
)
from multimodal_tugdt.synthetic import generate_synthetic_dataset

LOGGER = logging.getLogger("multimodal_tugdt")


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""
    parser = argparse.ArgumentParser(
        prog="tugdt",
        description="Reproducible multimodal TUG-DT research workflows.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser(
        "generate-synthetic",
        help="Generate deterministic, non-clinical multimodal demonstration data.",
    )
    synthetic.add_argument("--output", type=Path, default=Path("data/synthetic"))
    synthetic.add_argument("--participants", type=int, default=1)
    synthetic.add_argument("--seed", type=int, default=42)

    validate = subparsers.add_parser(
        "validate-manifest",
        help="Validate manifest schema, trial identifiers, conditions, and file paths.",
    )
    validate.add_argument("--config", type=Path, default=Path("configs/example.yaml"))
    validate.add_argument(
        "--no-check-files",
        action="store_true",
        help="Validate schema without checking referenced files on disk.",
    )

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Validate, filter, resample, and quality-check configured IMU trials.",
    )
    preprocess.add_argument("--config", type=Path, default=Path("configs/example.yaml"))

    synchronize = subparsers.add_parser(
        "synchronize",
        help=(
            "Map configured modalities to the IMU reference clock and generate synchronization QC."
        ),
    )
    synchronize.add_argument("--config", type=Path, default=Path("configs/example.yaml"))

    audio = subparsers.add_parser(
        "process-audio",
        help="Run waveform loading, energy VAD, QC, and audio behavior features.",
    )
    audio.add_argument("--config", type=Path, default=Path("configs/example.yaml"))

    footswitch = subparsers.add_parser(
        "process-footswitch",
        help="Debounce contacts and extract footswitch timing and IMU agreement features.",
    )
    footswitch.add_argument("--config", type=Path, default=Path("configs/example.yaml"))

    video = subparsers.add_parser(
        "process-video",
        help="Inspect video metadata and optionally extract aligned MediaPipe pose landmarks.",
    )
    video.add_argument("--config", type=Path, default=Path("configs/example.yaml"))

    features = subparsers.add_parser(
        "extract-features",
        help="Extract implemented trial- and phase-level multimodal features.",
    )
    features.add_argument("--config", type=Path, default=Path("configs/example.yaml"))

    fusion = subparsers.add_parser(
        "fuse-features",
        help="Build a trial-level multimodal feature matrix with availability indicators.",
    )
    fusion.add_argument("--config", type=Path, default=Path("configs/example.yaml"))

    baselines = subparsers.add_parser(
        "run-baselines",
        help="Evaluate participant-grouped classification or regression baselines.",
    )
    baselines.add_argument("--config", type=Path, default=Path("configs/example.yaml"))

    report = subparsers.add_parser(
        "generate-report",
        help="Generate a privacy-conscious aggregate Markdown research report.",
    )
    report.add_argument("--config", type=Path, default=Path("configs/example.yaml"))
    report.add_argument("--output", type=Path, default=None)

    run_all = subparsers.add_parser(
        "run-all",
        help="Run all implemented stages and generate an aggregate research report.",
    )
    run_all.add_argument("--config", type=Path, default=Path("configs/example.yaml"))
    return parser


def _generate_command(args: argparse.Namespace) -> int:
    dataset = generate_synthetic_dataset(
        args.output,
        participants=args.participants,
        seed=args.seed,
    )
    LOGGER.info("Generated %d synthetic trials in %s", dataset.trial_count, dataset.root)
    LOGGER.info("Manifest: %s", dataset.manifest)
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = validate_manifest(config, check_files=not args.no_check_files)
    for warning in report.warnings:
        LOGGER.warning(warning)
    for error in report.errors:
        LOGGER.error(error)
    if not report.is_valid:
        LOGGER.error("Manifest validation failed with %d error(s).", len(report.errors))
        return 1
    LOGGER.info(
        "Manifest is valid: %d trial(s), %d warning(s).",
        len(report.records),
        len(report.warnings),
    )
    return 0


def _validated_records(config_path: Path):
    config = load_config(config_path)
    report = validate_manifest(config)
    for warning in report.warnings:
        LOGGER.warning(warning)
    for error in report.errors:
        LOGGER.error(error)
    if not report.is_valid:
        raise ValueError(f"Manifest validation failed with {len(report.errors)} error(s).")
    return config, report.records


def _preprocess_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    result = preprocess_project(config, records)
    LOGGER.info(
        "IMU preprocessing complete: %d succeeded, %d failed, %d skipped. QC: %s",
        result.succeeded,
        result.failed,
        result.skipped,
        result.output_path,
    )
    return 1 if result.failed else 0


def _features_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    imu_result = extract_project_features(config, records)
    audio_result = process_audio_project(config, records)
    footswitch_result = process_footswitch_project(config, records)
    video_result = process_video_project(config, records)
    LOGGER.info(
        "Feature extraction complete. IMU: %s; audio: %s; footswitch: %s; video: %s",
        imu_result.output_path,
        audio_result.output_path,
        footswitch_result.output_path,
        video_result.output_path,
    )
    return (
        1
        if any(item.failed for item in (imu_result, audio_result, footswitch_result, video_result))
        else 0
    )


def _synchronize_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    result = synchronize_project(config, records)
    LOGGER.info(
        "Synchronization complete: %d succeeded, %d failed, %d skipped. QC: %s",
        result.succeeded,
        result.failed,
        result.skipped,
        result.output_path,
    )
    return 1 if result.failed else 0


def _audio_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    result = process_audio_project(config, records)
    LOGGER.info(
        "Audio processing complete: %d succeeded, %d failed, %d skipped. Features: %s",
        result.succeeded,
        result.failed,
        result.skipped,
        result.output_path,
    )
    return 1 if result.failed else 0


def _footswitch_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    result = process_footswitch_project(config, records)
    LOGGER.info(
        "Footswitch processing complete: %d succeeded, %d failed, %d skipped. Features: %s",
        result.succeeded,
        result.failed,
        result.skipped,
        result.output_path,
    )
    return 1 if result.failed else 0


def _video_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    result = process_video_project(config, records)
    LOGGER.info(
        "Video processing complete: %d succeeded, %d failed, %d skipped. Features: %s",
        result.succeeded,
        result.failed,
        result.skipped,
        result.output_path,
    )
    return 1 if result.failed else 0


def _fusion_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    result = fuse_project_features(config, records)
    LOGGER.info("Fused %d trial rows. Features: %s", result.succeeded, result.output_path)
    return 0


def _baselines_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    result = run_baselines_project(config, records)
    LOGGER.info(
        "Baseline evaluation complete: %d successful comparisons, %d skipped. Summary: %s",
        result.succeeded,
        result.skipped,
        result.output_path,
    )
    return 1 if result.failed else 0


def _report_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    result = generate_report_project(config, records, args.output)
    LOGGER.info(
        "Research summary generated for %d trials: %s", result.succeeded, result.output_path
    )
    return 0


def _run_all_command(args: argparse.Namespace) -> int:
    config, records = _validated_records(args.config)
    preprocessing = preprocess_project(config, records)
    if preprocessing.failed:
        LOGGER.error("Later stages were not run because IMU preprocessing failed.")
        return 1
    synchronization = synchronize_project(config, records)
    if synchronization.failed:
        LOGGER.error("Feature extraction was not run because synchronization failed.")
        return 1
    audio = process_audio_project(config, records)
    footswitch = process_footswitch_project(config, records)
    video = process_video_project(config, records)
    features = extract_project_features(config, records)
    feature_results = (audio, footswitch, video, features)
    if any(item.failed for item in feature_results):
        LOGGER.error("Fusion was not run because one or more feature stages failed.")
        return 1
    fusion = fuse_project_features(config, records)
    baseline = run_baselines_project(config, records) if config.modeling.enabled else None
    report = generate_report_project(config, records)
    LOGGER.info(
        "Milestone 7 pipeline complete. IMU QC: %s; synchronization QC: %s; "
        "IMU features: %s; audio features: %s; footswitch features: %s; video features: %s; "
        "fused features: %s; modeling: %s; report: %s",
        preprocessing.output_path,
        synchronization.output_path,
        features.output_path,
        audio.output_path,
        footswitch.output_path,
        video.output_path,
        fusion.output_path,
        baseline.output_path if baseline is not None else "disabled in configuration",
        report.output_path,
    )
    return 1 if baseline is not None and baseline.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        if args.command == "generate-synthetic":
            return _generate_command(args)
        if args.command == "validate-manifest":
            return _validate_command(args)
        if args.command == "preprocess":
            return _preprocess_command(args)
        if args.command == "synchronize":
            return _synchronize_command(args)
        if args.command == "process-audio":
            return _audio_command(args)
        if args.command == "process-footswitch":
            return _footswitch_command(args)
        if args.command == "process-video":
            return _video_command(args)
        if args.command == "extract-features":
            return _features_command(args)
        if args.command == "fuse-features":
            return _fusion_command(args)
        if args.command == "run-baselines":
            return _baselines_command(args)
        if args.command == "generate-report":
            return _report_command(args)
        if args.command == "run-all":
            return _run_all_command(args)
    except (ConfigurationError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
