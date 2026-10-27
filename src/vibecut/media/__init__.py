"""VibeCut media layer: ffmpeg/ffprobe discovery and bounded invocation."""

from vibecut.media.ffmpeg import FFmpegTools, FFmpegMissingError, find_tools
from vibecut.media.probe import (
    ArtifactValidationError,
    ProbeError,
    UnsupportedSourceError,
    VFRDetectedError,
    content_fingerprint,
    probe_source,
    validate_output,
)
from vibecut.media.proxy import ensure_proxy

__all__ = [
    "FFmpegTools",
    "FFmpegMissingError",
    "find_tools",
    "ProbeError",
    "UnsupportedSourceError",
    "VFRDetectedError",
    "ArtifactValidationError",
    "content_fingerprint",
    "probe_source",
    "validate_output",
    "ensure_proxy",
]
