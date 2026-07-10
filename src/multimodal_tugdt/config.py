"""Typed project configuration loading and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a project configuration is missing or malformed."""


@dataclass(frozen=True)
class ProjectConfig:
    """Validated configuration plus reproducible path-resolution rules."""

    source: Path
    project_root: Path
    manifest_path: Path
    interim_dir: Path
    processed_dir: Path
    output_dir: Path
    allowed_conditions: tuple[str, ...]
    values: dict[str, Any]

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a manifest or config path against the configured project root."""
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section '{section}' must be a mapping.")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML configuration and validate Milestone 1 requirements."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {source}")

    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {source}: {exc}") from exc

    root = _mapping(raw, "root")
    project = _mapping(root.get("project", {}), "project")
    paths = _mapping(root.get("paths"), "paths")
    study = _mapping(root.get("study", {}), "study")

    manifest_value = paths.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ConfigurationError("'paths.manifest' must be a non-empty path string.")

    root_value = project.get("root", "..")
    if not isinstance(root_value, str):
        raise ConfigurationError("'project.root' must be a path string.")
    project_root = (source.parent / root_value).resolve()

    def project_path(key: str, default: str) -> Path:
        value = paths.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"'paths.{key}' must be a non-empty path string.")
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (project_root / candidate).resolve()

    allowed = study.get(
        "allowed_conditions",
        ["single_task", "dual_task", "motor_only", "cognitive_only"],
    )
    if not isinstance(allowed, list) or not allowed or not all(isinstance(x, str) for x in allowed):
        raise ConfigurationError("'study.allowed_conditions' must be a non-empty string list.")

    return ProjectConfig(
        source=source,
        project_root=project_root,
        manifest_path=project_path("manifest", "data/metadata/participants.csv"),
        interim_dir=project_path("interim_dir", "data/interim"),
        processed_dir=project_path("processed_dir", "data/processed"),
        output_dir=project_path("output_dir", "outputs"),
        allowed_conditions=tuple(allowed),
        values=root,
    )
