"""Bounded ffmpeg extraction from the canonical panorama asset.

Two extractors:
  * a full-span cheap luma stream (gray, tiny, fps-filtered) for the
    prompt-independent scene/motion/quality features;
  * one small perspective JPEG per (sample time, view) using the frozen
    ``v360`` mapping from ``core.geometry`` (canonical semantics).

All ffmpeg runs go through the repo's bounded process runner (argv direct,
timeout + cancellation, no shell). Intermediates live in a unique
``tempfile.mkdtemp`` directory owned by the caller and are always removed.
"""

from __future__ import annotations

import tempfile
import threading
from fractions import Fraction
from pathlib import Path
from typing import Optional

from vibecut.analysis.errors import ExtractionError, ModelCancelledError
from vibecut.analysis.views import ViewSpec

_THUMB_WIDTH = 192
_THUMB_HEIGHT = 96


def _require_ffmpeg_ok(res, what: str) -> None:
    if res.cancelled:
        # user cancellation during ffmpeg is a cancellation, not a failure
        raise ModelCancelledError(f"{what} cancelled")
    if res.timed_out:
        raise ExtractionError(f"{what} timed out", code="timeout")
    tail = res.stderr_tail.decode("utf-8", "replace").strip()[-300:]
    if res.returncode != 0 or tail:
        raise ExtractionError(f"{what} failed rc={res.returncode}: {tail}")


def extract_gray_stream(
    tools,
    media_path: str,
    *,
    fps: Fraction,
    cancel_event: Optional[threading.Event],
    timeout_s: float = 300.0,
) -> tuple[bytes, int, int]:
    """Decode the whole media as a small gray raw stream (bounded size)."""
    import os

    fps = Fraction(fps)
    if fps <= 0:
        raise ExtractionError("cheap fps must be positive")
    out_dir = tempfile.mkdtemp(prefix="vibecut-analysis-gray-")
    raw_path = Path(out_dir) / "gray.raw"
    argv = tools.ffmpeg_argv(
        "-i", str(media_path), "-map", "0:v:0", "-an",
        "-vf", f"fps={fps.numerator}/{fps.denominator},scale={_THUMB_WIDTH}:{_THUMB_HEIGHT},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    )
    try:
        with open(raw_path, "wb") as fh:
            from vibecut.core.proc import run_bounded

            res = run_bounded(argv, timeout_s=timeout_s, cancel_event=cancel_event, stdout_file=fh)
        _require_ffmpeg_ok(res, "gray stream extraction")
        data = raw_path.read_bytes()
        frame = _THUMB_WIDTH * _THUMB_HEIGHT
        if len(data) % frame != 0:
            raise ExtractionError(f"gray stream size {len(data)} not a frame multiple")
        return data, _THUMB_WIDTH, _THUMB_HEIGHT
    finally:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)


def extract_perspective_jpeg(
    tools,
    media_path: str,
    t_rel: Fraction,
    view: ViewSpec,
    out_dir: Path,
    *,
    cancel_event: Optional[threading.Event],
    timeout_s: float = 120.0,
) -> Path:
    """One perspective JPEG for (time, view) via the frozen v360 mapping."""
    from vibecut.core.contracts import CameraView
    from vibecut.core.geometry import v360_filter

    cv = CameraView(yaw_deg=view.yaw_deg, pitch_deg=view.pitch_deg,
                    roll_deg=view.roll_deg, h_fov_deg=view.h_fov_deg)
    vf = v360_filter(cv, view.width, view.height)
    safe_view = "".join(c if c.isalnum() or c in "+-._" else "_" for c in view.view_id)
    out = out_dir / f"s{float(t_rel):09.3f}_v{safe_view}.jpg"
    argv = tools.ffmpeg_argv(
        "-ss", f"{float(t_rel):.3f}", "-i", str(media_path), "-map", "0:v:0", "-an",
        "-frames:v", "1", "-vf", vf, "-q:v", "4", "-y", str(out),
    )
    res = tools.run(argv, timeout_s=timeout_s, cancel_event=cancel_event)
    _require_ffmpeg_ok(res, f"perspective jpeg {view.view_id}@{float(t_rel):.3f}s")
    if not out.is_file() or out.stat().st_size == 0:
        raise ExtractionError(f"perspective jpeg missing for {view.view_id}@{float(t_rel):.3f}s")
    return out
