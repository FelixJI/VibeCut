"""Shared fixtures: tool discovery + synthetic media generation.

Synthetic equirectangular sources are generated through ffmpeg stdin/stdout
pipes with exact pixel patterns written by this test suite - no user media is
ever read. ffmpeg/ffprobe come from PATH or VIBECUT_FFMPEG/VIBECUT_FFPROBE.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.media.ffmpeg import FFmpegTools  # noqa: E402


def _which_all() -> tuple[bool, bool]:
    ff = os.environ.get("VIBECUT_FFMPEG") or shutil.which("ffmpeg")
    fp = os.environ.get("VIBECUT_FFPROBE") or shutil.which("ffprobe")
    return bool(ff), bool(fp)


HAS_FFMPEG, HAS_FFPROBE = _which_all()
if os.environ.get("VIBECUT_REQUIRE_FFMPEG") and not (HAS_FFMPEG and HAS_FFPROBE):
    # CI must never pass by silently skipping every media test
    raise SystemExit(
        "VIBECUT_REQUIRE_FFMPEG=1 but ffmpeg/ffprobe are unavailable "
        "(PATH / VIBECUT_FFMPEG / VIBECUT_FFPROBE); refusing to green-run on skips"
    )
requires_ffmpeg = pytest.mark.skipif(
    not (HAS_FFMPEG and HAS_FFPROBE),
    reason="ffmpeg/ffprobe not available on PATH or via VIBECUT_FFMPEG/VIBECUT_FFPROBE",
)


@pytest.fixture(scope="session")
def tools() -> FFmpegTools:
    if not (HAS_FFMPEG and HAS_FFPROBE):
        pytest.skip("ffmpeg/ffprobe unavailable")
    return FFmpegTools.discover()


def _ffmpeg_env(tools: FFmpegTools) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_ffmpeg(tools: FFmpegTools, argv: list[str], *, stdin_data: bytes | None = None, timeout: float = 60.0) -> subprocess.CompletedProcess:
    full = [str(tools.ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", *argv]
    return subprocess.run(full, input=stdin_data, capture_output=True, timeout=timeout,
                          env=_ffmpeg_env(tools), check=False)


def run_ffprobe_json(tools: FFmpegTools, path: Path) -> dict:
    import json

    full = [str(tools.ffprobe), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-i", str(path)]
    res = subprocess.run(full, capture_output=True, timeout=30, check=True)
    return json.loads(res.stdout.decode("utf-8"))


def decode_gray_frames(tools: FFmpegTools, path: Path, width: int, height: int) -> list[bytes]:
    """Decode a video to full-range-ish gray frames (list of W*H bytes)."""
    res = subprocess.run(
        [str(tools.ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", "format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=60, check=True,
    )
    data = res.stdout
    frame_size = width * height
    assert len(data) % frame_size == 0, f"raw size {len(data)} not multiple of {frame_size}"
    return [data[i:i + frame_size] for i in range(0, len(data), frame_size)]


def _encode_gray_video(
    tools: FFmpegTools,
    out_path: Path,
    frames: list[bytes],
    width: int,
    height: int,
    fps: Fraction,
    *,
    crf: int = 8,
    extra_out: list[str] | None = None,
    ts_offset: float | None = None,
) -> Path:
    raw = b"".join(frames)
    argv = ["-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
            "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
    if ts_offset is not None:
        argv += ["-output_ts_offset", f"{ts_offset:.6f}"]
    argv += ["-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
             "-colorspace", "bt709", "-color_range", "tv",
             "-map_metadata", "-1"]
    if extra_out:
        argv += extra_out
    argv += ["-y", str(out_path)]
    res = run_ffmpeg(tools, argv, stdin_data=raw)
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    return out_path


def make_marker_equirect(
    tools: FFmpegTools,
    out_path: Path,
    *,
    yaw_deg: float,
    pitch_deg: float,
    width: int = 640,
    height: int = 320,
    fps: Fraction = Fraction(30),
    frames: int = 3,
    band: bool = False,
    span_deg: float = 3.0,
) -> Path:
    """Equirect source: black frame + white marker patch/band at a direction.

    band=True -> full-width horizontal band (pole tests; yaw irrelevant).
    """
    from vibecut.core.geometry import direction_to_equirect_px

    x0, y0 = direction_to_equirect_px(yaw_deg, pitch_deg, width, height)
    buf = bytearray(width * height)  # black
    if band:
        half = span_deg / 2 / 180 * height
        y_lo, y_hi = int(y0 - half), int(y0 + half + 1)
        for y in range(max(0, y_lo), min(height, y_hi)):
            for x in range(width):
                buf[y * width + x] = 235
    else:
        # patch spans `span_deg` in yaw; widen near poles where longitude shrinks
        import math

        yaw_span = min(span_deg / max(0.15, math.cos(math.radians(pitch_deg))), 90.0)
        half_x = yaw_span / 2 / 360 * width
        half_y = span_deg / 2 / 180 * height
        x_lo, x_hi = int(x0 - half_x), int(x0 + half_x + 1)
        y_lo, y_hi = int(y0 - half_y), int(y0 + half_y + 1)
        for y in range(max(0, y_lo), min(height, y_hi)):
            for xi in range(x_lo, x_hi):  # horizontal wrap (seam-safe patches)
                x = xi % width
                buf[y * width + x] = 235
    frame = bytes(buf)
    return _encode_gray_video(tools, out_path, [frame] * frames, width, height, fps, crf=6)


def make_luma_step_video(
    tools: FFmpegTools,
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: Fraction,
    frames: int,
    step_frame: int,
    dark: int = 16,
    bright: int = 200,
    ts_offset: float | None = None,
) -> Path:
    """Video whose luma steps from `dark` to `bright` at frame `step_frame`."""
    frames_data = []
    for i in range(frames):
        v = dark if i < step_frame else bright
        frames_data.append(bytes([v]) * (width * height))
    return _encode_gray_video(tools, out_path, frames_data, width, height, fps, ts_offset=ts_offset)


def make_tone_audio(tools: FFmpegTools, out_path: Path, *, seconds: float, onset_s: float, freq: int = 440, amplitude: float = 0.9) -> Path:
    expr = f"if(lt(t\\,{onset_s:.6f})\\,0\\,{amplitude}*sin(2*PI*{freq}*t))"
    res = run_ffmpeg(tools, [
        "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=48000:d={seconds:.6f}",
        "-c:a", "aac", "-b:a", "128k", "-y", str(out_path),
    ])
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    return out_path


def mux_av(tools: FFmpegTools, video: Path, audio: Path, out_path: Path, *, audio_offset_s: float) -> Path:
    res = run_ffmpeg(tools, [
        "-i", str(video), "-itsoffset", f"{audio_offset_s:.6f}", "-i", str(audio),
        "-map", "0:v", "-map", "1:a", "-c", "copy", "-y", str(out_path),
    ])
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    return out_path


def first_bright_frame_index(frames: list[bytes], width: int, height: int, threshold: int = 100) -> int | None:
    """Index of the first frame whose centre patch mean exceeds threshold."""
    for idx, f in enumerate(frames):
        cx, cy = width // 2, height // 2
        patch = [f[y * width + x] for y in range(cy - 4, cy + 4) for x in range(cx - 4, cx + 4)]
        if sum(patch) / len(patch) > threshold:
            return idx
    return None


def stream_start_time(tools: FFmpegTools, path: Path, stream_index: int) -> float:
    """Container start_time (s) of a stream - the player-timeline anchor
    that a raw decode of the stream does not include (AAC priming / elst)."""
    probe = run_ffprobe_json(tools, path)
    for s in probe.get("streams", []):
        if int(s.get("index", -1)) == stream_index:
            v = s.get("start_time")
            return float(v) if v not in (None, "N/A", "-") else 0.0
    return 0.0


def audio_onset_seconds(tools: FFmpegTools, path: Path, *, threshold: float = 0.05) -> float:
    """First time the decoded PCM exceeds threshold*peak (s16le stereo/mono agnostic)."""
    res = subprocess.run(
        [str(tools.ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-map", "0:a:0", "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
        capture_output=True, timeout=60, check=True,
    )
    import array

    samples = array.array("h")
    samples.frombytes(res.stdout[: len(res.stdout) - len(res.stdout) % 2])
    peak = max(1, max(abs(s) for s in samples) if samples else 1)
    thr = peak * threshold
    for i, s in enumerate(samples):
        if abs(s) > thr:
            return i / 48000.0
    return float("inf")


def centroid_of_bright(frames: list[bytes], width: int, height: int, *, threshold: int = 120, window: float = 0.15) -> tuple[float, float]:
    """Brightness centroid (col, row) over the central window of frame 0.

    The window is deliberately narrow: near the poles a panorama's content is
    mirrored across the pole inside the view, and a wider window would let
    the mirrored copy drag the centroid.
    """
    f = frames[0]
    c0 = int(width * (0.5 - window / 2))
    c1 = int(width * (0.5 + window / 2))
    r0 = int(height * (0.5 - window / 2))
    r1 = int(height * (0.5 + window / 2))
    total = 0.0
    sx = sy = 0.0
    for y in range(r0, r1):
        for x in range(c0, c1):
            v = f[y * width + x]
            if v > threshold:
                total += v
                sx += x * v
                sy += y * v
    if total == 0:
        return float("nan"), float("nan")
    return sx / total, sy / total


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    return tmp_path
