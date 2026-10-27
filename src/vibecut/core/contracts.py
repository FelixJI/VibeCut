"""Versioned data contracts shared by core, media and render (schema version 1).

All timestamps that originate from, or map onto, media streams are exact
rationals (:class:`fractions.Fraction`). Serialisation uses ``"num/den"``
strings so no precision is lost through JSON round-trips.

Content fingerprint policy (explicit contract of FelixJI/VibeCut#2 only)
-------------------------------------------------------------------------
Producer:
  * :func:`vibecut.media.probe.probe_source` hashes the raw file bytes
    (SHA-256) and stores the digest in ``MediaSource.content_fingerprint``.
Consumers:
  * :mod:`vibecut.core.cache` includes the fingerprint in cache keys, so any
    content change under the same filename invalidates derived caches
    (same-name files are never treated as the same asset).
  * :mod:`vibecut.render.static` re-hashes the source immediately before and
    after every render; a mismatch aborts/fails the job (source immutability).
Failure handling:
  * unreadable/unhashable files fail probing with ``ProbeError`` (no
    MediaSource is produced, nothing is cached);
  * a changed hash during render raises ``SourceChangedError`` and no artifact
    is committed.
No other component computes hashes and no hashes are computed for downloads
or deliveries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

# Schema v2 (frozen with task #2):
#   v1 fields + explicit stabilization provenance on MediaSource/NormalizedAsset
#   + manual-lock/provenance semantics on CameraPath ("user" manual locks win)
#   + manual cut semantics flag on EditPlan.
CONTRACT_VERSION = 2


class ContractError(ValueError):
    """Raised when a contract object fails validation or parsing."""


def parse_frac(value: Any, *, name: str = "value") -> Fraction:
    """Parse ``"num/den"``, int, float or Fraction into an exact Fraction."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ContractError(f"{name}: bool is not a time value")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(10_000_000)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ContractError(f"{name}: cannot parse {value!r} as rational") from exc
    raise ContractError(f"{name}: unsupported type {type(value).__name__}")


def frac_str(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


@dataclass(frozen=True)
class ColorContract:
    """Colour space contract of a stream or render target."""

    primaries: str  # e.g. "bt709"
    transfer: str  # e.g. "bt709" (SDR); HDR transfers are rejected upstream
    matrix: str  # e.g. "bt709"
    range: str  # "tv" | "pc"
    bit_depth: int  # e.g. 8

    def to_json(self) -> dict[str, Any]:
        return {
            "primaries": self.primaries,
            "transfer": self.transfer,
            "matrix": self.matrix,
            "range": self.range,
            "bit_depth": self.bit_depth,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ColorContract":
        c = cls(
            primaries=str(data["primaries"]),
            transfer=str(data["transfer"]),
            matrix=str(data["matrix"]),
            range=str(data["range"]),
            bit_depth=int(data["bit_depth"]),
        )
        _require(c.range in ("tv", "pc"), "range must be tv|pc")
        _require(c.bit_depth in (8, 10, 12, 16), f"unsupported bit depth {c.bit_depth}")
        return c


@dataclass(frozen=True)
class VideoStream:
    """First displayed video stream of a source, with exact rational times.

    ``start_pts``/``time_base`` describe the PTS of the first *displayed* frame
    (after edit lists) in stream timebase units. ``fps`` is the exact average
    frame rate as a rational (e.g. 30000/1001).
    """

    index: int  # absolute stream index in the container
    codec: str
    width: int
    height: int
    start_pts: Optional[int]
    time_base: Fraction
    fps: Fraction
    r_frame_rate: Fraction
    nb_frames: Optional[int]

    def first_frame_seconds(self) -> Fraction:
        """Absolute container time of the first displayed frame (exact)."""
        if self.start_pts is None:
            return Fraction(0)
        return self.start_pts * self.time_base

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "start_pts": self.start_pts,
            "time_base": frac_str(self.time_base),
            "fps": frac_str(self.fps),
            "r_frame_rate": frac_str(self.r_frame_rate),
            "nb_frames": self.nb_frames,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "VideoStream":
        return cls(
            index=int(data["index"]),
            codec=str(data["codec"]),
            width=int(data["width"]),
            height=int(data["height"]),
            start_pts=None if data.get("start_pts") is None else int(data["start_pts"]),
            time_base=parse_frac(data["time_base"], name="time_base"),
            fps=parse_frac(data["fps"], name="fps"),
            r_frame_rate=parse_frac(data["r_frame_rate"], name="r_frame_rate"),
            nb_frames=None if data.get("nb_frames") is None else int(data["nb_frames"]),
        )


@dataclass(frozen=True)
class AudioStream:
    """Primary audio stream used for cut export (same absolute timeline)."""

    index: int
    codec: str
    channels: int
    sample_rate: int
    start_pts: Optional[int]
    time_base: Fraction

    def first_frame_seconds(self) -> Fraction:
        if self.start_pts is None:
            return Fraction(0)
        return self.start_pts * self.time_base

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "codec": self.codec,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "start_pts": self.start_pts,
            "time_base": frac_str(self.time_base),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AudioStream":
        return cls(
            index=int(data["index"]),
            codec=str(data["codec"]),
            channels=int(data["channels"]),
            sample_rate=int(data["sample_rate"]),
            start_pts=None if data.get("start_pts") is None else int(data["start_pts"]),
            time_base=parse_frac(data["time_base"], name="audio time_base"),
        )


@dataclass(frozen=True)
class MediaSource:
    """Immutable description of an ingested source asset.

    ``source_id`` is stable and content-addressed:
    ``"v1:" + content_fingerprint[:16]``. Two files with different content
    never share a source_id, and the same content under a different filename
    does share it (cache identity is content, not path).
    """

    schema_version: int
    source_id: str
    path: str
    content_fingerprint: str  # sha256 hex; producer: media.probe, see module docstring
    size_bytes: int
    container: str
    projection: str  # "equirect" for this pipeline; anything else is rejected
    stereo: str  # "mono"
    color: ColorContract
    video: VideoStream
    audio: Optional[AudioStream]
    vfr: bool  # if True the source must be rejected before rendering
    declared: bool  # True if projection was user-declared rather than detected
    stabilization_provenance: str = "none"  # "none" for task #2; SDK-stabilized sources appear in later tasks
    timeline_origin: Fraction = Fraction(0)  # demuxer media-timeline origin (elst shift) in absolute seconds; ffmpeg -ss is media-relative (verified empirically), absolute cuts subtract this

    def __post_init__(self) -> None:
        _require(self.schema_version == CONTRACT_VERSION, "unsupported schema_version")
        _require(self.projection == "equirect", "static pipeline needs equirect sources")
        _require(self.stereo == "mono", "stereo panorama layouts are not supported")
        _require(len(self.content_fingerprint) == 64, "content_fingerprint must be sha256 hex")
        _require(self.source_id == f"v1:{self.content_fingerprint[:16]}", "source_id/fingerprint mismatch")
        _require(self.video.width == 2 * self.video.height, "equirect sources must be 2:1")
        _require(self.stabilization_provenance in ("none",), "only stabilization_provenance='none' exists in task #2")

    @property
    def first_video_frame_seconds(self) -> Fraction:
        return self.video.first_frame_seconds()

    def cut_window_absolute(self, cut_in_rel: Fraction, cut_out_rel: Fraction) -> tuple[Fraction, Fraction]:
        """Map source-relative cut points to the absolute container timeline.

        Source-relative time 0 is defined as the first *displayed* video frame
        (``start_pts * time_base``). All streams (audio + video) are cut on the
        same absolute timeline via the returned [start, end) seconds.
        """
        base = self.first_video_frame_seconds
        return base + cut_in_rel, base + cut_out_rel

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "path": self.path,
            "content_fingerprint": self.content_fingerprint,
            "size_bytes": self.size_bytes,
            "container": self.container,
            "projection": self.projection,
            "stereo": self.stereo,
            "color": self.color.to_json(),
            "video": self.video.to_json(),
            "audio": self.audio.to_json() if self.audio is not None else None,
            "vfr": self.vfr,
            "declared": self.declared,
            "stabilization_provenance": self.stabilization_provenance,
            "timeline_origin": frac_str(self.timeline_origin),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MediaSource":
        return cls(
            schema_version=int(data["schema_version"]),
            source_id=str(data["source_id"]),
            path=str(data["path"]),
            content_fingerprint=str(data["content_fingerprint"]),
            size_bytes=int(data["size_bytes"]),
            container=str(data["container"]),
            projection=str(data["projection"]),
            stereo=str(data["stereo"]),
            color=ColorContract.from_json(data["color"]),
            video=VideoStream.from_json(data["video"]),
            audio=AudioStream.from_json(data["audio"]) if data.get("audio") else None,
            vfr=bool(data["vfr"]),
            declared=bool(data.get("declared", False)),
            stabilization_provenance=str(data.get("stabilization_provenance", "none")),
            timeline_origin=parse_frac(data.get("timeline_origin", "0"), name="timeline_origin"),
        )


@dataclass(frozen=True)
class TimeMap:
    """Mapping between asset-relative time and source-relative time.

    source_rel(t_asset) = asset_start_source_rel + t_asset (seconds, exact).
    """

    asset_start_source_rel: Fraction
    fps: Fraction

    def to_json(self) -> dict[str, Any]:
        return {"asset_start_source_rel": frac_str(self.asset_start_source_rel), "fps": frac_str(self.fps)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TimeMap":
        return cls(
            asset_start_source_rel=parse_frac(data["asset_start_source_rel"]),
            fps=parse_frac(data["fps"]),
        )


@dataclass(frozen=True)
class NormalizedAsset:
    """Derived asset (proxy/canonical master) with exact time mapping."""

    schema_version: int
    asset_id: str  # "v1:" + cache key prefix of the producing step
    kind: str  # "proxy" | "canonical"
    source_id: str
    path: str
    time_map: TimeMap
    color: ColorContract
    stabilization_provenance: str = "none"  # mirrors the source provenance for derived assets

    def __post_init__(self) -> None:
        _require(self.schema_version == CONTRACT_VERSION, "unsupported schema_version")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "kind": self.kind,
            "source_id": self.source_id,
            "path": self.path,
            "time_map": self.time_map.to_json(),
            "color": self.color.to_json(),
            "stabilization_provenance": self.stabilization_provenance,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "NormalizedAsset":
        return cls(
            schema_version=int(data["schema_version"]),
            asset_id=str(data["asset_id"]),
            kind=str(data["kind"]),
            source_id=str(data["source_id"]),
            path=str(data["path"]),
            time_map=TimeMap.from_json(data["time_map"]),
            color=ColorContract.from_json(data["color"]),
            stabilization_provenance=str(data.get("stabilization_provenance", "none")),
        )


@dataclass(frozen=True)
class CameraView:
    """A single camera orientation, canonical semantics (see core.geometry).

    yaw/pitch/roll in degrees; h_fov is the *horizontal* field of view of the
    rectilinear output. Manual locks (explicit user values) always win over
    any derived/automatic values; this pipeline only carries explicit values.
    """

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    h_fov_deg: float = 90.0

    def to_json(self) -> dict[str, Any]:
        return {"yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg, "roll_deg": self.roll_deg, "h_fov_deg": self.h_fov_deg}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CameraView":
        return cls(
            yaw_deg=float(data["yaw_deg"]),
            pitch_deg=float(data["pitch_deg"]),
            roll_deg=float(data["roll_deg"]),
            h_fov_deg=float(data["h_fov_deg"]),
        )


@dataclass(frozen=True)
class CameraPath:
    """Ordered camera keyframes over source-relative time.

    ``provenance`` records the manual-lock semantics: "user" values are
    explicit manual locks and always win over any derived/automatic values;
    "derived" paths are produced by automation (not created by task #2 - the
    static render only accepts "user").

    The static export of this task requires exactly one key (time 0).
    """

    schema_version: int
    keys: tuple[tuple[Fraction, CameraView], ...]
    provenance: str = "user"

    def __post_init__(self) -> None:
        _require(self.schema_version == CONTRACT_VERSION, "unsupported schema_version")
        _require(len(self.keys) >= 1, "camera path needs at least one key")
        _require(self.provenance in ("user", "derived"), "camera provenance must be 'user' or 'derived'")
        times = [t for t, _ in self.keys]
        _require(all(times[i] < times[i + 1] for i in range(len(times) - 1)), "camera keys must be strictly increasing")

    @property
    def static_view(self) -> CameraView:
        _require(len(self.keys) == 1, "static export requires exactly one camera key")
        t, view = self.keys[0]
        _require(t == 0, "static export camera key must be at source-relative time 0")
        return view

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "keys": [[frac_str(t), v.to_json()] for t, v in self.keys],
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CameraPath":
        return cls(
            schema_version=int(data["schema_version"]),
            keys=tuple((parse_frac(t), CameraView.from_json(v)) for t, v in data["keys"]),
            provenance=str(data.get("provenance", "user")),
        )


@dataclass(frozen=True)
class EditPlan:
    """A cut plan on the source-relative timeline of one MediaSource."""

    schema_version: int
    source_id: str
    cut_in_rel: Fraction  # seconds relative to first displayed frame (>= 0)
    cut_out_rel: Fraction  # seconds relative to first displayed frame (> cut_in)
    project_fps: Fraction  # output frame rate, e.g. 30000/1001
    camera: CameraPath
    audio: bool = True  # False forces silent export even if audio exists
    cut_manual: bool = True  # True = cut in/out are manual locks; automation must not move them

    def __post_init__(self) -> None:
        _require(self.schema_version == CONTRACT_VERSION, "unsupported schema_version")
        _require(self.cut_in_rel >= 0, "cut_in_rel must be >= 0")
        _require(self.cut_out_rel > self.cut_in_rel, "cut_out_rel must be > cut_in_rel")
        _require(self.project_fps > 0, "project_fps must be > 0")

    @property
    def duration_rel(self) -> Fraction:
        return self.cut_out_rel - self.cut_in_rel

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "cut_in_rel": frac_str(self.cut_in_rel),
            "cut_out_rel": frac_str(self.cut_out_rel),
            "project_fps": frac_str(self.project_fps),
            "camera": self.camera.to_json(),
            "audio": self.audio,
            "cut_manual": self.cut_manual,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EditPlan":
        return cls(
            schema_version=int(data["schema_version"]),
            source_id=str(data["source_id"]),
            cut_in_rel=parse_frac(data["cut_in_rel"], name="cut_in_rel"),
            cut_out_rel=parse_frac(data["cut_out_rel"], name="cut_out_rel"),
            project_fps=parse_frac(data["project_fps"], name="project_fps"),
            camera=CameraPath.from_json(data["camera"]),
            audio=bool(data.get("audio", True)),
            cut_manual=bool(data.get("cut_manual", True)),
        )


@dataclass(frozen=True)
class JobEvent:
    """Append-only job lifecycle event tied to a project revision.

    An event whose ``project_revision`` is older than the current project
    revision must be rejected (never overwrite newer state with late events).
    A ``succeeded`` event may only be published after the artifact was
    validated and committed (render enforces the ordering).
    """

    schema_version: int
    job_id: str
    project_revision: int
    status: str  # queued | running | succeeded | failed | cancelled
    at: str  # ISO-8601 UTC
    artifact: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        _require(self.schema_version == CONTRACT_VERSION, "unsupported schema_version")
        _require(self.status in ("queued", "running", "succeeded", "failed", "cancelled"), f"bad status {self.status!r}")
        _require(self.project_revision >= 0, "project_revision must be >= 0")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "project_revision": self.project_revision,
            "status": self.status,
            "at": self.at,
            "artifact": self.artifact,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "JobEvent":
        return cls(
            schema_version=int(data["schema_version"]),
            job_id=str(data["job_id"]),
            project_revision=int(data["project_revision"]),
            status=str(data["status"]),
            at=str(data["at"]),
            artifact=data.get("artifact"),
            error=data.get("error"),
        )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to a temp file in the same directory, then os.replace (atomic)."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
