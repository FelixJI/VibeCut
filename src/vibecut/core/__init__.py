"""VibeCut core: versioned contracts, geometry, project persistence, jobs, cache, process management.

Runtime dependencies: Python standard library only (no Qt, no SDK, no models).
Time values use :class:`fractions.Fraction` (exact rational arithmetic) everywhere
source time or output time is represented.
"""

from vibecut.core.contracts import (
    CONTRACT_VERSION,
    AudioStream,
    CameraPath,
    CameraView,
    ColorContract,
    EditPlan,
    JobEvent,
    MediaSource,
    NormalizedAsset,
    TimeMap,
    VideoStream,
)
from vibecut.core.geometry import (
    direction_to_equirect_px,
    equirect_px_to_direction,
    v360_args,
    vfov_from_hfov,
    wrap_yaw,
)

__all__ = [
    "CONTRACT_VERSION",
    "AudioStream",
    "CameraPath",
    "CameraView",
    "ColorContract",
    "EditPlan",
    "JobEvent",
    "MediaSource",
    "NormalizedAsset",
    "TimeMap",
    "VideoStream",
    "direction_to_equirect_px",
    "equirect_px_to_direction",
    "v360_args",
    "vfov_from_hfov",
    "wrap_yaw",
]
