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
    except (ConfigurationError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
