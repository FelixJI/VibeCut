"""Canonical panorama geometry semantics for VibeCut (frozen for task #2).

Canonical conventions (these are the contract; render maps to FFmpeg v360):
  * yaw 0         : the centre column of the equirectangular source
                    (the "forward" direction of the initial view).
  * positive yaw  : rotating the view to the RIGHT (east) in the source image.
  * positive pitch: looking UP (toward the zenith / north pole).
  * positive roll : camera rolls clockwise (horizon appears to rotate
                    counter-clockwise inside the output image).
  * h_fov         : HORIZONTAL field of view in degrees of the rectilinear
                    (pinhole) output image; the vertical FOV is derived from
                    the output aspect ratio, never specified independently.

Source-relative time 0 is the first displayed frame (see contracts.MediaSource).
Canonical directions are absolute; per-project manual locks (explicit user
values in CameraView) always take priority over any automatically derived
orientation - this pipeline never derives, it only carries explicit values.
"""

from __future__ import annotations

import math
from typing import Any

from vibecut.core.contracts import CameraView


def wrap_yaw(yaw_deg: float) -> float:
    """Wrap yaw into [-180, 180)."""
    y = (yaw_deg + 180.0) % 360.0 - 180.0
    # normalise -180.0 to +180-edge so the range is half-open
    if y == -180.0:
        y = 180.0 - 1e-12
    return y


def direction_to_equirect_px(yaw_deg: float, pitch_deg: float, width: int, height: int) -> tuple[float, float]:
    """Map a canonical direction to equirectangular pixel coordinates.

    The pixel centre convention is used: pixel (x, y) covers the direction
    whose continuous coordinates are x + 0.5, y + 0.5.
    """
    yaw = wrap_yaw(yaw_deg)
    pitch = max(-90.0, min(90.0, pitch_deg))
    x = (yaw / 360.0 + 0.5) * width - 0.5
    y = (0.5 - pitch / 180.0) * height - 0.5
    return x, y


def equirect_px_to_direction(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Inverse of :func:`direction_to_equirect_px`."""
    yaw = wrap_yaw(((x + 0.5) / width - 0.5) * 360.0)
    pitch = (0.5 - (y + 0.5) / height) * 180.0
    return yaw, pitch


def vfov_from_hfov(h_fov_deg: float, width: int, height: int) -> float:
    """Derive vertical FOV from horizontal FOV for a pinhole camera model.

    tan(vfov/2) = tan(hfov/2) * height/width
    """
    if not 0.0 < h_fov_deg < 180.0:
        raise ValueError("h_fov_deg must be in (0, 180)")
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    t = math.tan(math.radians(h_fov_deg) / 2.0) * (height / width)
    return math.degrees(2.0 * math.atan(t))


def hfov_from_vfov(v_fov_deg: float, width: int, height: int) -> float:
    """Inverse of :func:`vfov_from_hfov` (used to sanity-check tool output)."""
    if not 0.0 < v_fov_deg < 180.0:
        raise ValueError("v_fov_deg must be in (0, 180)")
    t = math.tan(math.radians(v_fov_deg) / 2.0) * (width / height)
    return math.degrees(2.0 * math.atan(t))


def v360_args(view: CameraView, out_width: int, out_height: int) -> dict[str, str]:
    """Map a canonical CameraView to FFmpeg ``v360`` filter options.

    FFmpeg v360 sign conventions assumed (documented in the v360 filter docs):
    positive yaw turns the view right, positive pitch turns up, positive roll
    rotates the camera clockwise - identical to the canonical semantics above,
    so the mapping is the identity. Both h_fov and v_fov are passed explicitly
    (v_fov derived from the output aspect) so the projection is fully defined.
    """
    return {
        "input": "equirect",
        "output": "flat",
        "w": str(out_width),
        "h": str(out_height),
        "yaw": f"{view.yaw_deg:.6f}",
        "pitch": f"{view.pitch_deg:.6f}",
        "roll": f"{view.roll_deg:.6f}",
        "h_fov": f"{view.h_fov_deg:.6f}",
        "v_fov": f"{vfov_from_hfov(view.h_fov_deg, out_width, out_height):.6f}",
    }


def v360_filter(view: CameraView, out_width: int, out_height: int) -> str:
    """Build the ``v360=equirect:flat:...`` filter string (single argv element)."""
    opts = v360_args(view, out_width, out_height)
    parts = [f"v360={opts['input']}:{opts['output']}"]
    for key in ("w", "h", "yaw", "pitch", "roll", "h_fov", "v_fov"):
        parts.append(f"{key}={opts[key]}")
    return ":".join(parts)
