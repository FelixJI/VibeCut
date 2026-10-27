"""Render layer: static fixed-view export pipeline (task #2 baseline).

The static renderer is the baseline for future dynamic rendering - there is
exactly one render path (no duplicated renderer).
"""

from vibecut.render.static import (
    ArtifactProcessError,
    ExportError,
    OutputExistsError,
    RenderResult,
    SourceChangedError,
    StaticRenderConfig,
    render_static,
)

__all__ = [
    "ArtifactProcessError",
    "ExportError",
    "OutputExistsError",
    "RenderResult",
    "SourceChangedError",
    "StaticRenderConfig",
    "render_static",
]
