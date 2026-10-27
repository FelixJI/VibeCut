"""Source probing (MediaSource construction) and artifact validation.

Fail-fast rules (invariants, per task #2 contract correction):
  * colour: ONLY explicitly tagged BT.709 / tv-range / 8-bit SDR is accepted.
    unknown/untagged or non-709 values are rejected - VibeCut never silently
    relabels colour it did not actually convert;
  * frame timing: CFR is verified from the REAL frame packet PTS on the wire
    (uniform spacing), never inferred from r_frame_rate==avg_frame_rate alone,
    and the first DISPLAYED frame PTS comes from the packets, not blindly
    from stream.start_pts; non-uniform PTS => VFRDetectedError;
  * projection: only precise structured metadata (stream side data / tags) or
    an explicit user declaration is trusted; conflicting metadata, and known
    stereo/dual-fisheye layouts are rejected. No whole-JSON string scanning;
  * projection/colour/container/codec/audio mismatches are raised before any
    render - nothing enters silent-error rendering.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from vibecut.core.contracts import (
    CONTRACT_VERSION,
    AudioStream,
    ColorContract,
    MediaSource,
    VideoStream,
    parse_frac,
)
from vibecut.media.ffmpeg import FFmpegTools

PROBE_TIMEOUT_S = 30.0

_ALLOWED_VIDEO_CODECS = {"h264", "hevc"}
_ALLOWED_AUDIO_CODECS_PREFIX = ("aac", "mp3", "opus", "flac", "pcm_")
_EQUIRECT_PROJECTION_VALUES = {"equirectangular", "equirect"}
_STEREO_MONO_VALUES = {"", "none", "2d", "mono"}


class ProbeError(RuntimeError):
    pass


class UnsupportedSourceError(RuntimeError):
    pass


class VFRDetectedError(UnsupportedSourceError):
    pass


class ArtifactValidationError(RuntimeError):
    pass


def content_fingerprint(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of the raw file bytes (producer; see contracts module docstring)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _ffprobe_json(tools: FFmpegTools, path: Path, argv_extra: list[str], *, timeout_s: float = PROBE_TIMEOUT_S) -> dict[str, Any]:
    argv = tools.ffprobe_argv("-print_format", "json", *argv_extra, "-i", str(path))
    res = tools.run(argv, timeout_s=timeout_s)
    if res.returncode != 0:
        tail = res.stderr_tail.decode("utf-8", "replace")[-500:].strip()
        raise ProbeError(f"ffprobe failed for {path.name} (rc={res.returncode}): {tail}")
    try:
        return json.loads(res.stdout_tail.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe produced unparsable JSON for {path.name}: {exc}") from exc


def _bit_depth(pix_fmt: str) -> int:
    """Extract component bit depth from a pix_fmt name ('yuv420p' -> 8,
    'yuv420p10le' -> 10); digits from subsampling (420/422/444) are NOT depth."""
    pf = pix_fmt or ""
    m = re.search(r"(?:^|p)(\d+)(?:l[ea])?$", pf) or re.search(r"(\d+)(?:l[ea])?$", pf)
    if m:
        d = int(m.group(1))
        return d if d in (9, 10, 12, 14, 16, 32) else 8
    return 8


def structured_projection(probe: dict[str, Any]) -> Optional[str]:
    """Determine projection from precise structured metadata only.

    Trusted locations (nothing else is scanned - filenames, comments and
    free-text never influence detection):
      * stream ``side_data_list`` entries whose ``side_data_type`` denotes
        spherical mapping (value of ``projection`` checked);
      * stream tag ``ProjectionType`` (common MP4 spherical tag).

    Returns "equirect" when equirectangular metadata is found, None when no
    structured marker exists. Raises UnsupportedSourceError for structured
    markers that name a DIFFERENT projection, for stereo layouts, or for
    contradictory markers.
    """
    findings: list[str] = []
    for stream in probe.get("streams", []):
        for sd in stream.get("side_data_list") or []:
            sd_type = str(sd.get("side_data_type", ""))
            if "spherical" not in sd_type.lower():
                continue
            proj = str(sd.get("projection", "") or "").lower()
            stereo = str(sd.get("stereo", sd.get("stereo_format", "")) or "").lower()
            if stereo not in _STEREO_MONO_VALUES:
                raise UnsupportedSourceError(
                    f"stereo spherical layout {stereo!r} in metadata; only monoscopic sources are supported"
                )
            findings.append(proj or "spherical-unspecified")
        tag = str((stream.get("tags") or {}).get("ProjectionType", "") or "").lower()
        if tag:
            findings.append(tag)
    distinct = {f for f in findings}
    if not distinct:
        return None
    if len(distinct) == 1:
        only = distinct.pop()
        if only in _EQUIRECT_PROJECTION_VALUES:
            return "equirect"
        if only == "spherical-unspecified":
            raise UnsupportedSourceError("spherical metadata present but projection unspecified; declare explicitly if truly equirectangular")
        raise UnsupportedSourceError(f"structured metadata names projection {only!r}, not equirectangular")
    raise UnsupportedSourceError(f"contradictory projection metadata: {sorted(distinct)}")


def _run_ffprobe_to_tempfile(argv: list[str], *, timeout_s: float) -> Path:
    """Run ffprobe with stdout redirected to a controlled temp file.

    Used instead of a pipe when the output may exceed the bounded pipe tail
    (e.g. full packet tables): the temp file lives in the system temp dir,
    belongs exclusively to this call, is removed afterwards, and ffprobe is a
    leaf tool - a hard kill on timeout cannot orphan work we depend on.
    Bounded: argv only (no shell), overall timeout, stderr read is capped.
    """
    import subprocess
    import tempfile
    import time

    fd, tmp_name = tempfile.mkstemp(prefix="vibecut-ffprobe-", suffix=".txt")
    os.close(fd)
    tmp = Path(tmp_name)
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(argv, stdout=open(tmp, "wb"), stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, creationflags=creationflags)
    deadline = time.monotonic() + timeout_s
    while proc.poll() is None:
        if time.monotonic() > deadline:
            proc.kill()
            proc.wait(timeout=5)
            tmp.unlink(missing_ok=True)
            raise ProbeError(f"ffprobe timed out after {timeout_s}s: {argv[:2]}")
        time.sleep(0.02)
    err = proc.stderr.read(65536) if proc.stderr else b""
    proc.stderr.close() if proc.stderr else None
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = err.decode("utf-8", "replace")[-500:].strip()
        raise ProbeError(f"ffprobe failed (rc={proc.returncode}): {tail}")
    return tmp


def _video_packet_pts(tools: FFmpegTools, path: Path, stream_index: int) -> list[int]:
    """Exact PTS list (stream timebase units) of the video stream's packets.

    ffprobe writes bare CSV values (one per line) to a controlled temp file;
    parsing is streamed line-by-line so memory stays bounded per line no
    matter how many packets the file has (pipe tails would truncate).
    """
    argv = list(tools.ffprobe_argv(
        "-of", "csv=p=0", "-select_streams", str(stream_index),
        "-show_packets", "-show_entries", "packet=pts", "-i", str(path),
    ))
    tmp = _run_ffprobe_to_tempfile(argv, timeout_s=PROBE_TIMEOUT_S)
    pts: list[int] = []
    try:
        with open(tmp, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:  # streaming: bounded memory per line
                value = line.strip()
                if not value or value in ("N/A", "AV_NOPTS_VALUE"):
                    continue
                try:
                    pts.append(int(value))
                except ValueError:
                    continue
    finally:
        tmp.unlink(missing_ok=True)
    return sorted(pts)


def _verify_cfr(pts: list[int], time_base: Fraction, avg_frame_rate: Fraction, stream_start_pts: Optional[int]) -> tuple[int, Fraction]:
    """Verify constant frame timing from real packet PTS.

    Returns (first_displayed_pts, exact_fps).

    * non-uniform PTS spacing => VFRDetectedError regardless of any rate
      metadata (r/avg frame-rate fields are recorded but never trusted);
    * exact fps implied by the PTS lattice must equal avg_frame_rate;
    * the first displayed frame comes from the edit-list-aware stream
      start_pts when ffprobe provides one that is consistent with the actual
      packets (min <= start_pts <= max); otherwise from min packet PTS.
      stream.start_pts is verified, never blindly trusted.
    """
    if len(pts) < 2:
        raise UnsupportedSourceError("fewer than 2 timestamped video packets; cannot verify frame timing")
    deltas = {b - a for a, b in zip(pts, pts[1:])}
    if len(deltas) != 1:
        d_sorted = sorted(deltas)
        raise VFRDetectedError(
            "variable frame timing verified from packet PTS "
            f"(deltas {d_sorted[0]}..{d_sorted[-1]} in timebase units are non-uniform); "
            "VFR sources are rejected before rendering"
        )
    frame_duration = next(iter(deltas)) * time_base
    if frame_duration <= 0:
        raise UnsupportedSourceError("non-positive frame duration from packet PTS")
    exact_fps = Fraction(1, 1) / frame_duration
    if avg_frame_rate > 0 and exact_fps != avg_frame_rate:
        raise UnsupportedSourceError(
            f"frame PTS imply fps {exact_fps} but container reports avg_frame_rate {avg_frame_rate}; "
            "time mapping is inconsistent and the source is rejected"
        )
    if stream_start_pts is not None and pts[0] - 1 <= stream_start_pts <= pts[-1]:
        first = stream_start_pts
    elif stream_start_pts is not None:
        raise UnsupportedSourceError(
            f"stream start_pts {stream_start_pts} inconsistent with packet PTS range [{pts[0]}, {pts[-1]}]; "
            "cannot establish the first displayed frame"
        )
    else:
        first = pts[0]
    return first, exact_fps


def _check_color(vs: dict[str, Any]) -> ColorContract:
    """Strict BT.709 SDR gate: unknown or non-709 values are rejected."""
    color = ColorContract(
        primaries=str(vs.get("color_primaries") or "").lower(),
        transfer=str(vs.get("color_transfer") or "").lower(),
        matrix=str(vs.get("color_space") or "").lower(),
        range=str(vs.get("color_range") or "").lower().replace("mpeg", "tv").replace("jpeg", "pc"),
        bit_depth=_bit_depth(str(vs.get("pix_fmt") or "")),
    )
    problems = []
    if color.primaries != "bt709":
        problems.append(f"primaries {color.primaries or 'unknown/untagged'!r}")
    if color.transfer != "bt709":
        problems.append(f"transfer {color.transfer or 'unknown/untagged'!r}")
    if color.matrix != "bt709":
        problems.append(f"matrix {color.matrix or 'unknown/untagged'!r}")
    if color.range not in ("tv", "mpeg"):
        problems.append(f"range {color.range or 'unknown/untagged'!r} (only limited/tv is converted honestly; full/pc would need real range conversion)")
    if color.bit_depth != 8:
        problems.append(f"bit depth {color.bit_depth} (only 8-bit SDR is supported)")
    if problems:
        raise UnsupportedSourceError(
            "source colour contract is not tagged BT.709/tv/8-bit SDR: " + "; ".join(problems)
            + ". VibeCut does not silently relabel colour it did not convert; establish the input colour space and perform a real BT.709 SDR conversion first."
        )
    return ColorContract("bt709", "bt709", "bt709", "tv", 8)


def probe_source(
    path: Path,
    tools: FFmpegTools,
    *,
    declared_projection: Optional[str] = None,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> MediaSource:
    """Build a MediaSource for a monocular equirectangular SDR MP4/MOV."""
    path = Path(path)
    if not path.is_file():
        raise ProbeError(f"source file not found: {path}")
    if declared_projection is not None and declared_projection != "equirect":
        raise UnsupportedSourceError(f"unsupported declared projection: {declared_projection}")

    fingerprint = content_fingerprint(path)

    probe = _ffprobe_json(tools, path, ["-show_format", "-show_streams"])
    fmt = probe.get("format", {})
    container = str(fmt.get("format_name", ""))
    if not any(c in container for c in ("mp4", "mov")):
        raise UnsupportedSourceError(f"container {container!r} is not supported (need MP4/MOV)")

    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not video_streams:
        raise UnsupportedSourceError("no displayable video stream found")
    if len(video_streams) > 1:
        raise UnsupportedSourceError("multiple displayable video streams are not supported")
    vs = video_streams[0]
    codec = str(vs.get("codec_name", ""))
    if codec not in _ALLOWED_VIDEO_CODECS:
        raise UnsupportedSourceError(f"video codec {codec!r} is not supported (need H.264/HEVC)")

    width, height = int(vs["width"]), int(vs["height"])
    if width != 2 * height:
        raise UnsupportedSourceError(f"equirectangular source must be 2:1, got {width}x{height}")

    # explicit boundary rejection: Dolby Vision layers / stereoscopic 3D side data
    for sd in vs.get("side_data_list") or []:
        sd_type = str(sd.get("side_data_type", "")).lower()
        if "dolby vision" in sd_type or "dvc c" in sd_type or sd_type.startswith("dovi"):
            raise UnsupportedSourceError("Dolby Vision side data detected; HDR-derived sources are out of scope")
        if "stereo 3d" in sd_type or "stereoscopic" in sd_type:
            raise UnsupportedSourceError("stereoscopic 3D side data detected; only planar mono video is supported")

    time_base = parse_frac(vs.get("time_base") or "0", name="time_base")
    if time_base <= 0:
        raise UnsupportedSourceError("stream time_base missing")
    avg_rate = parse_frac(vs.get("avg_frame_rate") or "0", name="avg_frame_rate")

    # frame timing verified from real packet PTS (not rate metadata alone)
    packet_pts = _video_packet_pts(tools, path, int(vs["index"]))
    stream_start_pts = int(vs["start_pts"]) if vs.get("start_pts") not in (None, "N/A") else None
    first_display_pts, exact_fps = _verify_cfr(packet_pts, time_base, avg_rate, stream_start_pts)

    color = _check_color(vs)

    # projection: structured markers or explicit declaration only
    detected = structured_projection(probe)
    declared = declared_projection == "equirect"
    if detected is None and not declared:
        raise UnsupportedSourceError(
            "no structured spherical/equirectangular metadata found; declare the projection explicitly "
            "(e.g. --projection equirect) if this is a monocular equirectangular source"
        )
    if detected is not None and detected != "equirect":
        raise UnsupportedSourceError(f"structured metadata names {detected!r}, not equirectangular")

    audio: Optional[AudioStream] = None
    for s in streams:
        if s.get("codec_type") != "audio":
            continue
        acodec = str(s.get("codec_name", ""))
        if not acodec.startswith(_ALLOWED_AUDIO_CODECS_PREFIX):
            raise UnsupportedSourceError(f"audio codec {acodec!r} cannot be cut precisely on the shared timeline")
        audio = AudioStream(
            index=int(s["index"]),
            codec=acodec,
            channels=int(s.get("channels") or 2),
            sample_rate=int(s.get("sample_rate") or 48000),
            start_pts=int(s["start_pts"]) if s.get("start_pts") not in (None, "N/A") else None,
            time_base=parse_frac(s.get("time_base") or "0", name="audio time_base"),
        )
        break

    video = VideoStream(
        index=int(vs["index"]),
        codec=codec,
        width=width,
        height=height,
        start_pts=first_display_pts,  # first displayed frame, verified from packets
        time_base=time_base,
        fps=exact_fps,
        r_frame_rate=parse_frac(vs.get("r_frame_rate") or "0", name="r_frame_rate"),
        nb_frames=len(packet_pts),
    )

    if content_fingerprint(path) != fingerprint:
        raise ProbeError(f"source changed while probing: {path}")
    origin = Fraction(0)
    raw_start = fmt.get("start_time")
    if raw_start not in (None, "N/A", "-"):
        try:
            origin = max(Fraction(0), parse_frac(str(raw_start), name="format start_time"))
        except Exception:
            origin = Fraction(0)
    return MediaSource(
        schema_version=CONTRACT_VERSION,
        source_id=f"v1:{fingerprint[:16]}",
        path=str(path.resolve()),
        content_fingerprint=fingerprint,
        size_bytes=path.stat().st_size,
        container=container.split(",")[0],
        projection="equirect",
        stereo="mono",
        color=color,
        video=video,
        audio=audio,
        vfr=False,
        declared=declared,
        stabilization_provenance="none",
        timeline_origin=origin,
    )


def _spherical_markers(probe: dict[str, Any]) -> list[str]:
    """Collect structural spherical markers from streams (side data + tags).

    Deliberately does NOT scan free text (filenames, comments) - only precise
    structured fields, so e.g. a directory named *spherical* can never
    trigger a false positive.
    """
    markers: list[str] = []
    for stream in probe.get("streams", []):
        for sd in stream.get("side_data_list") or []:
            sd_type = str(sd.get("side_data_type", ""))
            if "spherical" in sd_type.lower():
                markers.append(sd_type)
        tag = (stream.get("tags") or {}).get("ProjectionType")
        if tag:
            markers.append(str(tag))
    return markers


def validate_output(
    path: Path,
    tools: FFmpegTools,
    *,
    expected_width: int,
    expected_height: int,
    expected_fps: Fraction,
    expected_duration_s: float,
    expect_audio: bool,
    tolerance_frames: float = 1.0,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> dict[str, Any]:
    """Probe a rendered artifact and assert the export contract.

    Must pass before an artifact is committed or a success event published.
    """
    probe = _ffprobe_json(tools, path, ["-show_format", "-show_streams"])
    streams = probe.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if len(video) != 1:
        raise ArtifactValidationError(f"expected exactly 1 video stream, found {len(video)}")
    v = video[0]
    if int(v["width"]) != expected_width or int(v["height"]) != expected_height:
        raise ArtifactValidationError(f"resolution {v['width']}x{v['height']} != {expected_width}x{expected_height}")
    if str(v.get("codec_name")) != "h264":
        raise ArtifactValidationError(f"video codec {v.get('codec_name')!r} != h264")
    out_fps = parse_frac(v.get("avg_frame_rate") or "0", name="out fps")
    if out_fps != expected_fps:
        raise ArtifactValidationError(f"output fps {v.get('avg_frame_rate')} != {expected_fps}")
    if expect_audio:
        if not audio:
            raise ArtifactValidationError("audio stream missing but was expected")
        if str(audio[0].get("codec_name")) != "aac":
            raise ArtifactValidationError(f"audio codec {audio[0].get('codec_name')!r} != aac")
    elif audio:
        raise ArtifactValidationError("audio stream present but not expected")

    for key, want in (("color_primaries", "bt709"), ("color_transfer", "bt709"), ("color_space", "bt709")):
        got = str(v.get(key) or "")
        if got != want:
            raise ArtifactValidationError(f"{key} is {got!r}, expected {want!r}")
    if str(v.get("color_range") or "tv") not in ("tv", "mpeg"):
        raise ArtifactValidationError(f"color_range {v.get('color_range')!r} != tv")

    markers = _spherical_markers(probe)
    if markers:
        raise ArtifactValidationError(f"output still carries spherical markers: {markers}")

    duration = float(probe.get("format", {}).get("duration") or 0.0)
    frame_tol_s = tolerance_frames / float(expected_fps) + 0.05
    if abs(duration - expected_duration_s) > frame_tol_s:
        raise ArtifactValidationError(
            f"duration {duration:.4f}s deviates from expected {expected_duration_s:.4f}s by more than {frame_tol_s:.4f}s"
        )
    nb = v.get("nb_frames")
    if nb not in (None, "N/A"):
        expected_frames = round(expected_duration_s * float(expected_fps))
        if abs(int(nb) - expected_frames) > max(1, int(tolerance_frames)):
            raise ArtifactValidationError(f"frame count {nb} != expected ~{expected_frames}")
    return probe
