"""SpatialCliff console package."""
from __future__ import annotations

from pathlib import Path

__all__ = ["create_app", "repo_root"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def create_app():
    from spatialcliff.ui.app import create_app as _create

    return _create()
