"""Normalised proxy generation with content-verified cache reuse.

The proxy cache key binds source CONTENT + tool build + colour contract +
proxy configuration (dimensions). It is never bound to an edit/project
revision. Cache hits are re-verified against the CURRENT source bytes so a
stale object handed in after the file changed cannot produce a false hit.
"""

from __future__ import annotations

import shutil
import threading
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Optional

from vibecut.core.cache import Cache, normalized_key
from vibecut.core.contracts import CONTRACT_VERSION, MediaSource, NormalizedAsset, TimeMap
from vibecut.media.ffmpeg import FFmpegTools
from vibecut.media.probe import content_fingerprint, validate_output


def _source_duration_s(source: MediaSource) -> Fraction:
    """Source duration from the verified packet count (exact for CFR)."""
    if source.video.nb_frames:
        return Fraction(source.video.nb_frames) / source.video.fps
    return Fraction(0)


def _proxy_key(source: MediaSource, tools: FFmpegTools, width: int, height: int) -> str:
    return normalized_key(
        source_fingerprint=source.content_fingerprint,
        source_projection=source.projection,
        tool_identity=tools.identity(),
        kind="proxy",
        color=source.color.to_json(),
        config={"pipeline": "proxy-v4-error-level-stderr", "width": width, "height": height, "scale": "bicubic",
                "video": "libx264-crf23", "space": "canonical-equirect-mono",
                "stabilization_provenance": source.stabilization_provenance},
    )


def ensure_proxy(
    source: MediaSource,
    tools: FFmpegTools,
    cache: Cache,
    *,
    width: int = 960,
    height: int = 480,
    timeout_s: float = 300.0,
    cancel_event: Optional[threading.Event] = None,
) -> NormalizedAsset:
    """Create (or reuse) the cached normalised proxy for a source."""
    from vibecut.render.static import ArtifactProcessError, ExportError, SourceChangedError

    def check_source() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise ExportError("proxy render cancelled")
        if content_fingerprint(Path(source.path)) != source.content_fingerprint:
            raise SourceChangedError(f"source content changed since ingest: {source.path}")

    def validate(target: Path) -> None:
        validate_output(
            target, tools, expected_width=width, expected_height=height,
            expected_fps=source.video.fps,
            expected_duration_s=float(_source_duration_s(source)),
            expect_audio=False, tolerance_frames=2.0,
        )
        decoded = tools.run(
            tools.ffmpeg_argv("-xerror", "-i", str(target), "-map", "0:v:0", "-f", "null", "-"),
            timeout_s=timeout_s, cancel_event=cancel_event,
        )
        # -loglevel error is pinned by ffmpeg_argv: any stderr line proves a
        # decode error even on builds (6.1.x) that exit rc=0.
        diag = decoded.stderr_tail.decode("utf-8", "replace").strip()[-300:]
        if decoded.returncode != 0 or decoded.cancelled or decoded.timed_out or diag:
            raise ExportError(
                "proxy payload could not be completely decoded" + (f": {diag}" if diag else "")
            )
        check_source()

    def asset(payload: Path) -> NormalizedAsset:
        return NormalizedAsset(
            schema_version=CONTRACT_VERSION, asset_id=f"v1:{key[:16]}",
            kind="proxy", source_id=source.source_id, path=str(payload),
            time_map=TimeMap(asset_start_source_rel=Fraction(0), fps=source.video.fps),
            color=source.color, stabilization_provenance=source.stabilization_provenance,
        )

    check_source()
    key = _proxy_key(source, tools, width, height)
    hit = cache.get(key)
    if hit is not None and hit[1] is not None:
        validate(hit[1])
        return asset(hit[1])

    cache.root.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"{key}.tmp-", suffix=".mp4", dir=cache.root)
    os.close(fd)
    tmp = Path(name)
    try:
        argv = tools.ffmpeg_argv(
            "-xerror", "-err_detect", "explode",
            "-i", source.path, "-map", f"0:{source.video.index}", "-an",
            "-vf", f"scale={width}:{height}:flags=bicubic",
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-colorspace", "bt709", "-color_range", "tv", "-map_metadata", "-1",
            "-f", "mp4", "-y", str(tmp),
        )
        res = tools.run(argv, timeout_s=timeout_s, cancel_event=cancel_event)
        if res.cancelled:
            raise ExportError("proxy render cancelled")
        # rc=0 + non-empty stderr is still a failure: ffmpeg_argv pins
        # -loglevel error, so any stderr byte is error-level evidence.
        tail = res.stderr_tail.decode("utf-8", "replace").strip()[-500:]
        if res.timed_out or res.returncode != 0 or tail:
            raise ArtifactProcessError(f"proxy ffmpeg rc={res.returncode}: {tail}")
        check_source()
        payload = cache.put_atomic(
            key,
            meta={"kind": "proxy", "source_id": source.source_id,
                  "width": width, "height": height, "color": source.color.to_json()},
            payload_writer=lambda target: shutil.copyfile(tmp, target),
            validator=validate,
        )
        return asset(payload)
    finally:
        tmp.unlink(missing_ok=True)
