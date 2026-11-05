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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AssetTiming:
    first_pts: int
    time_base: Fraction
    timeline_origin: Fraction
    fps: Fraction


def probe_asset_timing(tools, media_path, *, expected_fps, expected_frames):
    """Reuse #2's packet validation once per asset, without hashing/decoding it."""
    from vibecut.media.probe import _ffprobe_json, _video_packet_pts, _verify_cfr, ProbeError, UnsupportedSourceError
    try:
        data = _ffprobe_json(tools, Path(media_path), ["-show_format", "-show_streams"])
        videos = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if len(videos) != 1:
            raise ExtractionError("analysis asset must contain exactly one video stream")
        video = videos[0]
        time_base = Fraction(video["time_base"])
        pts = _video_packet_pts(tools, Path(media_path), int(video["index"]))
        first, fps = _verify_cfr(pts, time_base, Fraction(video["avg_frame_rate"]),
                                 int(video["start_pts"]) if video.get("start_pts") is not None else None)
        if fps != expected_fps or len(pts) != expected_frames:
            raise ExtractionError("asset displayed packet timing disagrees with verified source")
        origin = Fraction(data.get("format", {}).get("start_time", str(first * time_base)))
        return AssetTiming(first, time_base, origin, fps)
    except (ProbeError, UnsupportedSourceError, KeyError, ValueError, ZeroDivisionError) as exc:
        raise ExtractionError("cannot verify analysis asset displayed timing") from exc


def extract_perspective_jpeg(
    tools,
    media_path: str,
    t_rel: Fraction,
    view: ViewSpec,
    out_dir: Path,
    *,
    cancel_event: Optional[threading.Event],
    timing: AssetTiming,
    timeout_s: float = 120.0,
) -> Path:
    """One perspective JPEG for (time, view) via the frozen v360 mapping."""
    from vibecut.core.contracts import CameraView
    from vibecut.core.geometry import v360_filter

    cv = CameraView(yaw_deg=view.yaw_deg, pitch_deg=view.pitch_deg,
                    roll_deg=view.roll_deg, h_fov_deg=view.h_fov_deg)
    vf = v360_filter(cv, view.width, view.height)
    safe_view = "".join(c if c.isalnum() or c in "+-._" else "_" for c in view.view_id)
    ticks = Fraction(t_rel) / timing.time_base
    if t_rel < 0 or ticks.denominator != 1 or (t_rel * timing.fps).denominator != 1:
        raise ExtractionError("image time is not a verified displayed frame")
    target_pts = timing.first_pts + int(ticks)
    # Input seek avoids decoding from the beginning; retain original PTS for exact trim.
    seek = max(0, int(target_pts * timing.time_base - timing.timeline_origin) - 1)
    vf = (f"settb=expr={timing.time_base.numerator}/{timing.time_base.denominator},"
          f"trim=start_pts={target_pts}:end_pts={target_pts + 1}," + vf)
    out = out_dir / f"pts{target_pts}_v{safe_view}.jpg"
    argv = tools.ffmpeg_argv(
        "-copyts", "-ss", str(seek), "-i", str(media_path), "-map", "0:v:0", "-an",
        "-frames:v", "1", "-fps_mode", "passthrough", "-vf", vf, "-q:v", "4", "-y", str(out),
    )
    res = tools.run(argv, timeout_s=timeout_s, cancel_event=cancel_event)
    _require_ffmpeg_ok(res, f"perspective jpeg {view.view_id}@{float(t_rel):.3f}s")
    if not out.is_file() or out.stat().st_size == 0:
        raise ExtractionError(f"perspective jpeg missing for {view.view_id}@{float(t_rel):.3f}s")
    return out
