"""Static export: monocular equirectangular SDR MP4 -> fixed-view flat MP4.

Pipeline order (invariant - never reordered):
  1. verify tools + source fingerprint BEFORE rendering (source is read-only);
  2. reject output collisions (never overwrite source or existing outputs);
  3. write artifact to a temp file in the OUTPUT directory;
  4. run ffmpeg via bounded process management and wait for the real exit;
  5. probe-validate the temp artifact (contract assertions);
  6. re-verify the source fingerprint (detect mid-render source mutation);
  7. commit with a no-clobber link (atomic on both Windows and POSIX);
  8. only then report success (the caller publishes the succeeded JobEvent).

Cancellation, timeouts and non-zero exits abort before step 5 - no partial
artifact is ever committed and no success event is emitted.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from vibecut.core.cache import Cache, normalized_key, render_key
from vibecut.core.contracts import EditPlan, MediaSource
from vibecut.core.geometry import v360_filter
from vibecut.core.proc import run_bounded
from vibecut.media.ffmpeg import FFmpegTools
from vibecut.media.probe import ArtifactValidationError, content_fingerprint, validate_output


class ExportError(RuntimeError):
    pass


class OutputExistsError(ExportError):
    pass


class SourceChangedError(ExportError):
    pass


class ArtifactProcessError(ExportError):
    pass


@dataclass(frozen=True)
class StaticRenderConfig:
    out_width: int = 1920
    out_height: int = 1080
    vcodec: str = "libx264"
    crf: int = 18
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    timeout_s: float = 600.0
    # Render-configuration identity used for cache keys (NOT edit revision):
    def identity(self) -> dict[str, Any]:
        return {
            "pipeline": "static-v1",
            "out_width": self.out_width,
            "out_height": self.out_height,
            "vcodec": self.vcodec,
            "crf": self.crf,
            "preset": self.preset,
            "audio_bitrate": self.audio_bitrate,
        }


@dataclass(frozen=True)
class RenderResult:
    job_id: str
    status: str  # succeeded | failed | cancelled
    artifact: Optional[Path] = None
    error: Optional[str] = None
    cache_key: Optional[str] = None
    events: tuple[tuple[str, str], ...] = ()  # (status, note) trace


def _dec(f: Fraction, digits: int = 6) -> str:
    """Exact-ish decimal rendering of a rational for ffmpeg args (µs precision)."""
    scale = 10**digits
    scaled = f.numerator * scale // f.denominator
    sign = "-" if scaled < 0 else ""
    scaled = abs(scaled)
    return f"{sign}{scaled // scale}.{scaled % scale:0{digits}d}"


def _unique_destination(final: Path) -> Path:
    """First free ``stem-N.ext`` name next to ``final`` (never overwrite)."""
    if not final.exists():
        return final
    for i in range(1, 1000):
        cand = final.with_name(f"{final.stem}-{i}{final.suffix}")
        if not cand.exists():
            return cand
    raise ExportError(f"cannot find a free unique name next to {final}")


def commit_artifact(tmp: Path, final: Path) -> Path:
    """No-clobber commit via hard link.

    ``os.link`` fails with FileExistsError when the destination exists (never
    overwriting an existing artifact). If hard links are unsupported on the
    filesystem we fail explicitly instead of falling back to a racy
    check+rename - the original file is always preserved.
    """
    try:
        os.link(tmp, final)
    except FileExistsError as exc:
        raise OutputExistsError(f"output already exists: {final}") from exc
    except OSError as exc:
        raise ExportError(
            f"atomic no-clobber commit unavailable on this filesystem ({exc}); "
            "refusing to risk overwriting an existing artifact"
        ) from exc
    tmp.unlink(missing_ok=True)
    return final


def build_ffmpeg_argv(
    source: MediaSource,
    plan: EditPlan,
    config: StaticRenderConfig,
    tmp_out: Path,
) -> tuple[str, ...]:
    """Assemble the single-pass ffmpeg argv (list, never a shell string).

    Empirically verified seek semantics (task #2 evidence): ``-ss`` seeks in
    the demuxer's MEDIA timeline, which excludes edit-list shifts - on a
    +1.5s-start file, ``-ss 1.6`` selects nothing while ``-ss 0.1`` selects
    absolute 1.6s..2.2s. The contract's absolute window is therefore
    translated by subtracting ``source.timeline_origin`` (the file's demuxer
    origin) before being passed to ffmpeg.
    """
    abs_in, abs_out = source.cut_window_absolute(plan.cut_in_rel, plan.cut_out_rel)
    seek_target = abs_in - source.timeline_origin
    if seek_target < 0:
        raise ExportError(
            f"absolute cut start {abs_in}s precedes the file timeline origin {source.timeline_origin}s"
        )
    duration = abs_out - abs_in
    view = plan.camera.static_view
    vf = v360_filter(view, config.out_width, config.out_height)
    args = [
        "-xerror", "-err_detect", "explode",
        "-ss", _dec(seek_target),
        "-i", source.path,
        "-t", _dec(duration),
        "-map", f"0:{source.video.index}",
    ]
    if plan.audio and source.audio is not None:
        args += ["-map", f"0:{source.audio.index}"]
    args += [
        "-vf", vf,
        "-r", f"{plan.project_fps.numerator}/{plan.project_fps.denominator}",
        "-c:v", config.vcodec,
        "-crf", str(config.crf),
        "-preset", config.preset,
        "-pix_fmt", "yuv420p",
        # write the full BT.709 VUI so the artifact's colour tags are real,
        # not just container relabelling (input colour was gated to BT.709)
        "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
    ]
    if plan.audio and source.audio is not None:
        args += ["-c:a", "aac", "-b:a", config.audio_bitrate]
    else:
        args += ["-an"]
    args += [
        "-map_metadata", "-1",
        "-movflags", "+faststart",
        "-f", "mp4",  # temp output uses a .partial suffix; muxer must not be guessed
        "-y",  # only ever overwrites our own temp file
        str(tmp_out),
    ]
    return tuple(args)


def render_config_identity(config: StaticRenderConfig, plan: EditPlan) -> dict[str, Any]:
    """Cache identity of a render: config + plan content that shapes the output.

    Deliberately contains no project/edit revision and no manual-lock flags -
    bumping a revision never invalidates caches.
    """
    return {
        **config.identity(),
        "cut_in_rel": str(plan.cut_in_rel),
        "cut_out_rel": str(plan.cut_out_rel),
        "project_fps": str(plan.project_fps),
        "camera": plan.camera.static_view.to_json(),
        "audio": plan.audio,
    }


def render_static(
    plan: EditPlan,
    source: MediaSource,
    tools: FFmpegTools,
    out_path: Path,
    config: StaticRenderConfig = StaticRenderConfig(),
    *,
    on_exists: str = "fail",  # "fail" | "unique"
    cache: Optional[Cache] = None,
    cancel_event: Optional[threading.Event] = None,
    job_id: Optional[str] = None,
) -> RenderResult:
    """Run the full static export; see module docstring for ordering."""
    job_id = job_id or uuid.uuid4().hex
    events: list[tuple[str, str]] = []

    def _fail(err: Exception, status: str = "failed") -> RenderResult:
        events.append((status, str(err)))
        return RenderResult(job_id=job_id, status=status, error=f"{type(err).__name__}: {err}", events=tuple(events))

    if plan.source_id != source.source_id:
        return _fail(ExportError(f"plan is for source {plan.source_id}, got {source.source_id}"))
    if plan.camera.provenance != "user":
        return _fail(ExportError("static render requires explicit manual ('user') camera locks"))

    # 1. source immutability (pre) - also the anchor for cache identity
    try:
        current_fp = content_fingerprint(Path(source.path))
    except OSError as exc:
        return _fail(ExportError(f"source unreadable: {exc}"))
    if current_fp != source.content_fingerprint:
        return _fail(SourceChangedError(f"source content changed on disk since ingest: {source.path}"))

    out_path = Path(out_path).resolve()
    src_path = Path(source.path).resolve()
    if out_path == src_path or out_path.name == src_path.name and out_path.parent == src_path.parent:
        return _fail(OutputExistsError("output path must differ from the source path"))
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _fail(ExportError(f"output directory unavailable: {exc}"))
    if out_path.exists():
        if on_exists == "unique":
            out_path = _unique_destination(out_path)
            events.append(("unique-output", str(out_path)))
        else:
            return _fail(OutputExistsError(f"output already exists: {out_path}"))

    # 2. build argv; temp file in the OUTPUT directory (same volume -> atomic).
    # The temp name carries the FULL job_id (uuid) and is exclusively ours:
    # we never delete or overwrite a path we did not create ourselves.
    tmp_out = out_path.with_name(f".{out_path.name}.tmp-{job_id}.partial")
    try:
        fd = os.open(tmp_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
    except OSError as exc:
        return _fail(ExportError(f"cannot exclusively create temp artifact: {exc}"))
    try:
        argv = tools.ffmpeg_argv(*build_ffmpeg_argv(source, plan, config, tmp_out))

        # cache key: source content + tool + render config + colour contract (never edit revision)
        cache_key: Optional[str] = None
        if cache is not None:
            nkey = normalized_key(
                source_fingerprint=source.content_fingerprint,
                source_projection=source.projection,
                tool_identity=tools.identity(),
                kind="none",
                config={"space": "canonical-equirect-mono", "stabilization_provenance": source.stabilization_provenance},
                color=source.color.to_json(),
            )
            cache_key = render_key(normalized=nkey, config_identity=render_config_identity(config, plan))

        # 3-4. run and wait for the REAL process exit
        res = tools.run(argv, timeout_s=config.timeout_s, cancel_event=cancel_event)
        if res.cancelled:
            events.append(("cancelled", f"duration={res.duration_s:.2f}s"))
            return RenderResult(job_id=job_id, status="cancelled", cache_key=cache_key, events=tuple(events))
        if res.timed_out or res.returncode != 0:
            tail = res.stderr_tail.decode("utf-8", "replace")[-800:].strip()
            return _fail(ArtifactProcessError(f"ffmpeg exited rc={res.returncode}: {tail}"))
        if not tmp_out.exists():
            return _fail(ArtifactProcessError("ffmpeg exited 0 but no artifact was produced"))
        events.append(("process-exit", f"rc=0 duration={res.duration_s:.2f}s"))

        # cancellation between exit and validation must not publish success
        if cancel_event is not None and cancel_event.is_set():
            events.append(("cancelled", "after process exit"))
            return RenderResult(job_id=job_id, status="cancelled", cache_key=cache_key, events=tuple(events))

        # 5. probe-validate BEFORE any commit
        try:
            validate_output(
                tmp_out,
                tools,
                expected_width=config.out_width,
                expected_height=config.out_height,
                expected_fps=plan.project_fps,
                expected_duration_s=float(plan.duration_rel),
                expect_audio=plan.audio and source.audio is not None,
            )
        except ArtifactValidationError as exc:
            return _fail(exc)
        except Exception as exc:  # probe infrastructure failure - still a clean abort
            return _fail(ExportError(f"validation did not run cleanly: {exc}"))
        events.append(("validated", "probe contract ok"))

        # 6. source immutability (post)
        if content_fingerprint(Path(source.path)) != source.content_fingerprint:
            return _fail(SourceChangedError(f"source changed during render: {source.path}"))

        # cancellation between validation and commit must not publish success
        if cancel_event is not None and cancel_event.is_set():
            events.append(("cancelled", "before commit"))
            return RenderResult(job_id=job_id, status="cancelled", cache_key=cache_key, events=tuple(events))

        # 7. no-clobber commit
        try:
            final = commit_artifact(tmp_out, out_path)
        except (OutputExistsError, ExportError) as exc:
            return _fail(exc)
        events.append(("committed", str(final)))

        # 8. record cache manifest (metadata only - payloads are never duplicated)
        if cache is not None and cache_key is not None:
            try:
                cache.put_atomic(
                    cache_key,
                    meta={"kind": "render-manifest", "artifact": str(final), "job_id": job_id},
                )
            except OSError:
                pass  # cache bookkeeping is best-effort; artifact is already committed

        return RenderResult(job_id=job_id, status="succeeded", artifact=final, cache_key=cache_key, events=tuple(events))

    except Exception as exc:
        return _fail(exc)
    finally:
        tmp_out.unlink(missing_ok=True)
