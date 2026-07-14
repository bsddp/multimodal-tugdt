"""Multimodal TUG-DT research pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multimodal-tugdt")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.7.0"

__all__ = ["__version__"]
