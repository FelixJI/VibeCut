"""Real synthetic-media tests (AC2/AC3/AC4, ffmpeg-required).

Everything here runs on generated fixtures only: labeled marker panoramas,
luma-step videos and synthesized tone tracks piped through real ffmpeg.
Skipped automatically when ffmpeg/ffprobe are unavailable.
"""

import hashlib
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from conftest import (
    audio_onset_seconds,
    centroid_of_bright,
    decode_gray_frames,
    first_bright_frame_index,
    make_luma_step_video,
    make_marker_equirect,
    make_tone_audio,
    mux_av,
    requires_ffmpeg,
    run_ffmpeg,
    run_ffprobe_json,
    stream_start_time,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.core.cache import Cache  # noqa: E402
from vibecut.core.contracts import CONTRACT_VERSION, CameraPath, CameraView, EditPlan  # noqa: E402
from vibecut.core.geometry import vfov_from_hfov  # noqa: E402
from vibecut.media.ffmpeg import FFmpegTools  # noqa: E402
from vibecut.media.probe import (  # noqa: E402
    ProbeError,
    UnsupportedSourceError,
    VFRDetectedError,
    content_fingerprint,
    probe_source,
)
from vibecut.media.proxy import ensure_proxy  # noqa: E402
from vibecut.render.static import (  # noqa: E402
    OutputExistsError,
    SourceChangedError,
    StaticRenderConfig,
    render_static,
)

HFOV = 90.0
OUT_W, OUT_H = 1920, 1080
VFOV = vfov_from_hfov(HFOV, OUT_W, OUT_H)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(source, *, cut_in=Fraction(0), cut_out=None, fps=Fraction(30),
              yaw=0.0, pitch=0.0, roll=0.0, hfov=HFOV, audio=True) -> EditPlan:
    if cut_out is None:
        cut_out = Fraction(source.video.nb_frames, 1) / source.video.fps if source.video.nb_frames else Fraction(1)
    return EditPlan(
        schema_version=CONTRACT_VERSION,
        source_id=source.source_id,
        cut_in_rel=cut_in,
        cut_out_rel=cut_out,
        project_fps=fps,
        camera=CameraPath(schema_version=CONTRACT_VERSION, keys=((Fraction(0), CameraView(
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, h_fov_deg=hfov,
        )),)),
        audio=audio,
    )


def render_ok(tools, source, plan, out: Path, config=StaticRenderConfig(), **kw):
    res = render_static(plan, source, tools, out, config, **kw)
    assert res.status == "succeeded", f"render failed: {res.error}"
    return res


def angular_centroid_error(tools, out: Path, hfov=HFOV, w=OUT_W, h=OUT_H) -> tuple[float, float]:
    """(yaw_err, pitch_err) degrees of the bright marker vs view centre."""
    frames = decode_gray_frames(tools, out, w, h)
    assert frames, "no frames decoded"
    cx, cy = centroid_of_bright(frames, w, h, window=1.0)
    assert not math.isnan(cx), "no bright marker found in the central window"
    focal = w / (2 * math.tan(math.radians(hfov) / 2))
    yaw_err = math.degrees(math.atan((cx + 0.5 - w / 2) / focal))
    pitch_err = math.degrees(math.atan((h / 2 - cy - 0.5) / focal))
    return yaw_err, pitch_err


# ---------------------------------------------------------------- AC2: geometry
@requires_ffmpeg
@pytest.mark.parametrize("yaw", [0.0, 90.0, 180.0, -90.0])
@pytest.mark.parametrize("w,h", [(1920, 1080), (1080, 1920)])
def test_four_azimuths(tools, tmp_path, yaw, w, h):
    """Marker placed at the canonical azimuth must appear at the view centre."""
    src = make_marker_equirect(tools, tmp_path / f"pano_{int(yaw)}.mp4", yaw_deg=yaw, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    out = tmp_path / f"flat_{int(yaw)}.mp4"
    render_ok(tools, source, make_plan(source, yaw=yaw, pitch=0.0), out,
              config=StaticRenderConfig(out_width=w, out_height=h))
    yaw_err, pitch_err = angular_centroid_error(tools, out, w=w, h=h)
    assert abs(yaw_err) <= 1.0, f"yaw error {yaw_err:.3f} deg at yaw={yaw}"
    assert abs(pitch_err) <= 1.0, f"pitch error {pitch_err:.3f} deg at yaw={yaw}"


@requires_ffmpeg
@pytest.mark.parametrize("pitch", [90.0, -90.0])
@pytest.mark.parametrize("w,h", [(1920, 1080), (1080, 1920)])
def test_poles(tools, tmp_path, pitch, w, h):
    """A cap centred on the true pole stays centred at the usual 90deg HFOV.

    A full latitude band at 85deg is a ring around the pole, not a marker
    centred on that viewing direction; its centroid is biased.
    """
    hfov = 90.0
    src = make_marker_equirect(tools, tmp_path / f"pole_{int(pitch)}.mp4", yaw_deg=0.0, pitch_deg=pitch, band=True)
    source = probe_source(src, tools, declared_projection="equirect")
    out = tmp_path / f"pole_flat_{int(pitch)}.mp4"
    render_ok(tools, source, make_plan(source, yaw=0.0, pitch=pitch, hfov=hfov), out,
              config=StaticRenderConfig(out_width=w, out_height=h))
    _, pitch_err = angular_centroid_error(tools, out, hfov=hfov, w=w, h=h)
    assert abs(pitch_err) <= 1.0, f"pitch error {pitch_err:.3f} deg at pole {pitch}"


@requires_ffmpeg
def test_hfov_edges_16x9(tools, tmp_path):
    """h_fov=90 on 16:9: a marker centred at yaw 43.5 (span 1.5deg) is inside
    the view near the right edge; one at 47.5 is fully outside."""
    src = make_marker_equirect(tools, tmp_path / "edge.mp4", yaw_deg=43.5, pitch_deg=0.0, span_deg=1.5)
    source = probe_source(src, tools, declared_projection="equirect")
    out = tmp_path / "flat43.mp4"
    render_ok(tools, source, make_plan(source), out)
    frames = decode_gray_frames(tools, out, OUT_W, OUT_H)
    row = frames[0]
    right = max(row[y * OUT_W + x] for y in range(OUT_H) for x in range(OUT_W - 200, OUT_W))
    assert right > 100, "marker at 43.5 deg must be visible near the right edge"

    src47 = make_marker_equirect(tools, tmp_path / "edge47.mp4", yaw_deg=47.5, pitch_deg=0.0, span_deg=1.5)
    source47 = probe_source(src47, tools, declared_projection="equirect")
    out47 = tmp_path / "flat47.mp4"
    render_ok(tools, source47, make_plan(source47), out47)
    frames47 = decode_gray_frames(tools, out47, OUT_W, OUT_H)
    assert not frames47 or max(frames47[0]) < 100, "marker at 47.5 deg must fall outside the 90 deg h_fov"


@requires_ffmpeg
def test_output_has_no_spherical_metadata(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "p.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    out = tmp_path / "flat.mp4"
    render_ok(tools, source, make_plan(source), out)
    probe = run_ffprobe_json(tools, out)
    from vibecut.media.probe import _spherical_markers
    assert _spherical_markers(probe) == [], "no structural spherical markers may survive the export"
    v = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
    assert (v["width"], v["height"]) == (OUT_W, OUT_H)
    assert v["codec_name"] == "h264"
    assert v.get("color_primaries") == "bt709"
    assert v.get("color_transfer") == "bt709"
    assert v.get("color_space") == "bt709"


# ---------------------------------------------------------------- AC3: timing
@requires_ffmpeg
def test_nonzero_start_pts_cut_precision(tools, tmp_path):
    """First displayed frame at +1.5s; cut [0.1, 0.7) must land within 1 frame."""
    src = make_luma_step_video(tools, tmp_path / "offset.mp4", width=320, height=160,
                               fps=Fraction(30), frames=45, step_frame=10, ts_offset=1.5)
    source = probe_source(src, tools, declared_projection="equirect")
    first = source.video.first_frame_seconds()
    assert first > Fraction(0), "fixture must have nonzero start PTS"
    assert abs(float(first) - 1.5) < 1e-3, f"first displayed frame PTS must be ~1.5s, got {float(first):.6f}s"
    out = tmp_path / "cut.mp4"
    render_ok(tools, source, make_plan(source, cut_in=Fraction(1, 10), cut_out=Fraction(7, 10)), out)
    frames = decode_gray_frames(tools, out, OUT_W, OUT_H)
    idx = first_bright_frame_index(frames, OUT_W, OUT_H)
    assert idx is not None, "luma step must appear in the cut"
    # step at source-rel 10/30; output frame 7 expected (0.2333s), 1-frame tolerance
    assert abs(idx - 7) <= 1, f"step landed on frame {idx}, expected ~7"
    assert len(frames) in (18, 19), f"0.6s at 30fps should be 18 frames, got {len(frames)}"


@requires_ffmpeg
def test_30000_over_1001_roundtrip(tools, tmp_path):
    fps = Fraction(30000, 1001)
    src = make_luma_step_video(tools, tmp_path / "ntsc.mp4", width=320, height=160,
                               fps=fps, frames=30, step_frame=5)
    source = probe_source(src, tools, declared_projection="equirect")
    assert source.video.fps == fps, f"probe fps {source.video.fps} != {fps}"
    out = tmp_path / "ntsc_flat.mp4"
    render_ok(tools, source, make_plan(source, fps=fps), out)
    probe = run_ffprobe_json(tools, out)
    v = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
    assert Fraction(v["avg_frame_rate"]) == fps
    assert Fraction(v["r_frame_rate"]) == fps


@requires_ffmpeg
def test_audio_video_offset_and_sync(tools, tmp_path):
    """Audio starting 0.25s after video, both marked at abs t=1/3s: the export
    must keep them aligned within 80ms on the shared absolute timeline.

    Evidence is measured on real decoded content, both on the INPUT fixture
    (sanity, <=20ms) and the OUTPUT (contract, <=80ms); thresholds are not
    loosened to make tests pass.
    """
    video = make_luma_step_video(tools, tmp_path / "v.mp4", width=320, height=160,
                                 fps=Fraction(30), frames=45, step_frame=10)
    onset = 10 / 30  # 0.3333s absolute marker
    audio = make_tone_audio(tools, tmp_path / "a.m4a", seconds=1.5, onset_s=onset - 0.25)
    src = mux_av(tools, video, audio, tmp_path / "av.mp4", audio_offset_s=0.25)
    source = probe_source(src, tools, declared_projection="equirect")
    assert source.audio is not None
    audio_first = source.audio.first_frame_seconds()
    # AAC encoder priming (~1024 samples = 21.3ms @48k) is written before the
    # requested start; the edit list compensates it on decode. Evidence:
    # measured 343/1500 = 0.25 - 0.02133s.
    assert Fraction(22, 100) <= audio_first <= Fraction(1, 4), \
        f"fixture audio stream must actually start ~0.25s (got {audio_first})"
    assert source.video.first_frame_seconds() == 0

    # input-side sanity: markers are aligned in the muxed fixture itself.
    # A raw stream decode starts at the stream's first sample, so the tone
    # onset must be anchored by the stream's container start_time (AAC
    # priming ~21.3ms is inside that anchor) - the same math a player uses.
    in_probe = run_ffprobe_json(tools, src)
    a_idx = source.audio.index
    a_start = stream_start_time(tools, src, a_idx)
    in_v = first_bright_frame_index(decode_gray_frames(tools, video, 320, 160), 320, 160) or 0
    in_t_video = in_v / 30.0
    in_t_audio = a_start + audio_onset_seconds(tools, src)
    assert abs(in_t_audio - in_t_video) <= 0.020, \
        f"fixture must be aligned pre-export (audio {in_t_audio:.4f}s vs video {in_t_video:.4f}s)"

    out = tmp_path / "av_flat.mp4"
    render_ok(tools, source, make_plan(source, cut_in=Fraction(1, 10), cut_out=Fraction(13, 10)), out)
    frames = decode_gray_frames(tools, out, OUT_W, OUT_H)
    v_idx = first_bright_frame_index(frames, OUT_W, OUT_H)
    assert v_idx is not None
    t_video = v_idx / 30.0
    out_probe = run_ffprobe_json(tools, out)
    out_a = [s for s in out_probe["streams"] if s.get("codec_type") == "audio"][0]
    t_audio = float(out_a["start_time"]) + audio_onset_seconds(tools, out)
    assert abs(t_audio - t_video) <= 0.080, \
        f"audio at {t_audio:.4f}s vs video step at {t_video:.4f}s (must be <=80ms)"


@requires_ffmpeg
def test_no_audio_source_succeeds(tools, tmp_path):
    src = make_luma_step_video(tools, tmp_path / "silent.mp4", width=320, height=160,
                               fps=Fraction(30), frames=10, step_frame=5)
    source = probe_source(src, tools, declared_projection="equirect")
    assert source.audio is None
    out = tmp_path / "silent_flat.mp4"
    render_ok(tools, source, make_plan(source), out)
    probe = run_ffprobe_json(tools, out)
    assert not [s for s in probe["streams"] if s["codec_type"] == "audio"]


@requires_ffmpeg
def test_unicode_and_spaces_paths(tools, tmp_path):
    deep = tmp_path / "测试 项目" / "sub dir with spaces"
    deep.mkdir(parents=True)
    src = make_marker_equirect(tools, deep / "源 文件 360度.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    out = deep / "导出 成片.mp4"
    render_ok(tools, source, make_plan(source), out)
    assert out.is_file()


@requires_ffmpeg
def test_vfr_rejected_fail_fast(tools, tmp_path):
    out = tmp_path / "vfr.mp4"
    # -fps_mode passthrough keeps the setpts timestamps: FFmpeg 6.x's default
    # regularises them back to CFR and the fixture stops being VFR (CI 6.1.1).
    res = run_ffmpeg(tools, [
        "-f", "lavfi", "-i", "testsrc2=s=320x160:r=30",
        "-vf", "setpts='(N+floor(N/6)*0.5)/30/TB'",
        "-fps_mode", "passthrough",
        "-frames:v", "30", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-colorspace", "bt709", "-color_range", "tv",
        "-map_metadata", "-1", "-y", str(out),
    ])
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    # self-check: displayed packet PTS intervals must really be non-uniform,
    # else the VFRDetectedError assertion below would be vacuously green.
    packets = subprocess.run(tools.ffprobe_argv(
        "-show_packets", "-show_entries", "packet=pts,flags", "-of", "json", str(out)),
        capture_output=True, check=True, timeout=15)
    pts = sorted(int(p["pts"]) for p in json.loads(packets.stdout)["packets"]
                 if "D" not in p["flags"])
    deltas = {b - a for a, b in zip(pts, pts[1:])}
    assert len(deltas) > 1, f"fixture must be genuinely VFR, got uniform PTS deltas {deltas}"
    with pytest.raises(VFRDetectedError):
        probe_source(out, tools, declared_projection="equirect")


@requires_ffmpeg
def test_vfr_with_identical_avg_and_r_rates_rejected(tools, tmp_path):
    """avg_frame_rate equals the CFR value (30/1) yet the PTS are truly VFR.

    Regression for the contract correction: rate metadata alone must never
    establish CFR; this file reports the same avg_frame_rate as an honest
    30fps CFR file, and only the packet-PTS verification catches it.
    """
    cfr = tmp_path / "cfr60.mp4"
    res = run_ffmpeg(tools, [
        "-f", "lavfi", "-i", "testsrc2=s=320x160:r=30",
        "-frames:v", "60", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-x264-params", "bframes=0:colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-colorspace", "bt709", "-color_range", "tv",
        "-map_metadata", "-1", "-y", str(cfr),
    ])
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    # jitter only interior odd frames; first/last stay on the CFR lattice so
    # the container still reports avg_frame_rate = 30/1
    out = tmp_path / "vfr_same_avg.mp4"
    res = run_ffmpeg(tools, [
        "-i", str(cfr), "-c:v", "copy",
        "-bsf:v", r"setts=ts=PTS+128*mod(N\,2)*lt(N\,59)",
        "-map_metadata", "-1", "-y", str(out),
    ])
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    probe = run_ffprobe_json(tools, out)
    v = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
    assert v["avg_frame_rate"] == "30/1", \
        f"fixture must keep the CFR average rate (got {v['avg_frame_rate']})"
    with pytest.raises(VFRDetectedError, match="non-uniform"):
        probe_source(out, tools, declared_projection="equirect")


@requires_ffmpeg
def test_untagged_color_rejected(tools, tmp_path):
    """Unknown/unlabelled colour must be rejected, never silently treated as SDR."""
    out = tmp_path / "untagged.mp4"
    res = run_ffmpeg(tools, [
        "-f", "lavfi", "-i", "testsrc2=s=320x160:r=30",
        "-frames:v", "10", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-map_metadata", "-1", "-y", str(out),
    ])
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    with pytest.raises(UnsupportedSourceError, match="BT.709"):
        probe_source(out, tools, declared_projection="equirect")


@requires_ffmpeg
def test_hdr_transfer_rejected(tools, tmp_path):
    """Even a fully tagged bt2020/pq file is rejected (no HDR mapping in MVP)."""
    out = tmp_path / "hdr.mp4"
    res = run_ffmpeg(tools, [
        "-f", "lavfi", "-i", "testsrc2=s=320x160:r=30",
        "-frames:v", "10", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-x264-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
        "-colorspace", "bt2020nc", "-color_range", "tv",
        "-map_metadata", "-1", "-y", str(out),
    ])
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    with pytest.raises(UnsupportedSourceError, match="BT.709"):
        probe_source(out, tools, declared_projection="equirect")


# ---------------------------------------------------------------- AC4: safety
@requires_ffmpeg
def test_source_hash_unchanged_by_export(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0.0, pitch_deg=0.0)
    before = sha256(src)
    source = probe_source(src, tools, declared_projection="equirect")
    assert before == source.content_fingerprint
    render_ok(tools, source, make_plan(source), tmp_path / "o.mp4")
    assert sha256(src) == before, "export must never write to the source"


@requires_ffmpeg
def test_source_mutation_detected(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "s2.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    src.write_bytes(src.read_bytes() + b"trailing mutation")  # mutate after ingest
    res = render_static(make_plan(source), source, tools, tmp_path / "o2.mp4", StaticRenderConfig())
    assert res.status == "failed" and res.error.startswith("SourceChangedError")


@requires_ffmpeg
def test_output_collision_never_overwrites(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "s3.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    out = tmp_path / "existing.mp4"
    out.write_bytes(b"pre-existing artifact - must survive")
    res = render_static(make_plan(source), source, tools, out, StaticRenderConfig())
    assert res.status == "failed" and res.error.startswith("OutputExistsError")
    assert out.read_bytes() == b"pre-existing artifact - must survive"
    # unique mode creates a sibling instead
    res2 = render_static(make_plan(source), source, tools, out, StaticRenderConfig(), on_exists="unique")
    assert res2.status == "succeeded" and res2.artifact == out.with_name("existing-1.mp4")
    assert out.read_bytes() == b"pre-existing artifact - must survive"


@requires_ffmpeg
def test_unavailable_output_dir_fails_cleanly(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "s4.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    out = blocker / "child" / "out.mp4"
    res = render_static(make_plan(source), source, tools, out, StaticRenderConfig())
    assert res.status == "failed"
    assert not list(tmp_path.glob("**/*.partial")), "no temp artifacts may survive a failed run"


@requires_ffmpeg
def test_corrupt_input_rejected_before_render(tools, tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x00\x00\x00\x18ftypisom some garbage" + b"\x5a" * 200)
    with pytest.raises(ProbeError):
        probe_source(corrupt, tools, declared_projection="equirect")


@requires_ffmpeg
def test_undeclared_projection_rejected(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "no_meta.mp4", yaw_deg=0.0, pitch_deg=0.0)
    with pytest.raises(UnsupportedSourceError):
        probe_source(src, tools)  # no spherical metadata, no declaration


# ---------------------------------------------------------------- AC6: cache
@requires_ffmpeg
def test_proxy_cache_survives_revision_bump_and_invalidates_on_content(tools, tmp_path):
    cache = Cache(tmp_path / "cache")
    src = make_marker_equirect(tools, tmp_path / "c1.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    asset = ensure_proxy(source, tools, cache)
    first = Path(asset.path)
    assert cache.has(_key_of(cache, asset))
    mtime1 = first.stat().st_mtime_ns

    # an "edit revision bump" (project saved again) must not rebuild the master
    asset2 = ensure_proxy(source, tools, cache)
    assert Path(asset2.path) == first
    assert first.stat().st_mtime_ns == mtime1, "normalised master must not be needlessly rebuilt"

    # changed content under the same name -> new cache identity, new file
    make_marker_equirect(tools, src, yaw_deg=90.0, pitch_deg=0.0)
    source_b = probe_source(src, tools, declared_projection="equirect")
    asset_b = ensure_proxy(source_b, tools, cache)
    assert Path(asset_b.path) != first
    assert asset_b.source_id != source.source_id


def _key_of(cache: Cache, asset) -> str:
    for manifest in cache.root.glob("*.json"):
        hit = cache.get(manifest.stem)
        if hit is not None and hit[1] == Path(asset.path):
            return manifest.stem
    raise AssertionError("asset has no committed cache manifest")


@requires_ffmpeg
def test_render_cache_key_changes_with_config_not_revision(tools, tmp_path):
    cache = Cache(tmp_path / "cache2")
    src = make_marker_equirect(tools, tmp_path / "c2.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    out1 = tmp_path / "r1.mp4"
    r1 = render_ok(tools, source, make_plan(source, hfov=90.0), out1, cache=cache)
    out2 = tmp_path / "r2.mp4"
    r2 = render_ok(tools, source, make_plan(source, hfov=90.0), out2, cache=cache)
    assert r1.cache_key == r2.cache_key  # identical config -> identical key
    out3 = tmp_path / "r3.mp4"
    r3 = render_ok(tools, source, make_plan(source, hfov=60.0), out3, cache=cache)
    assert r3.cache_key != r1.cache_key  # config change invalidates


# ------------------------------------------------- AC2/AC4 evidence additions
@requires_ffmpeg
def test_hfov_vertical_edges_9x16(tools, tmp_path):
    """9:16 canvas, h_fov=90: the derived vertical FOV is 2*atan(16/9)=121.28deg,
    so a marker at pitch +58 must appear near the top edge while +64 is out."""
    w, h = 1080, 1920
    vfov = vfov_from_hfov(90.0, w, h)
    assert vfov == pytest.approx(121.28449, abs=1e-3)

    def max_luma_in_top_third(frames):
        frame = frames[0]
        top = frame[: h // 3 * w]
        return max(top) if top else 0

    src58 = make_marker_equirect(tools, tmp_path / "p58.mp4", yaw_deg=0.0, pitch_deg=58.0, span_deg=2.0)
    source58 = probe_source(src58, tools, declared_projection="equirect")
    cfg = StaticRenderConfig(out_width=w, out_height=h)
    out58 = tmp_path / "o58.mp4"
    res = render_static(make_plan(source58), source58, tools, out58, cfg)
    assert res.status == "succeeded", res.error
    assert max_luma_in_top_third(decode_gray_frames(tools, out58, w, h)) > 100, \
        "marker at pitch 58 must be visible near the top edge of the 9:16 view"

    src64 = make_marker_equirect(tools, tmp_path / "p64.mp4", yaw_deg=0.0, pitch_deg=64.0, span_deg=2.0)
    source64 = probe_source(src64, tools, declared_projection="equirect")
    out64 = tmp_path / "o64.mp4"
    res = render_static(make_plan(source64), source64, tools, out64, cfg)
    assert res.status == "succeeded", res.error
    frames64 = decode_gray_frames(tools, out64, w, h)
    assert not frames64 or max(frames64[0]) < 100, \
        "marker at pitch 64 must fall outside the derived vertical FOV (60.64deg half-angle)"


@requires_ffmpeg
def test_input_seek_is_absolute_container_time(tools, tmp_path):
    """Nonzero start PTS evidence: cut [0, 0.5) on a +1.5s-start source must
    keep frame N at output position N. ffmpeg -ss is media-relative (verified:
    -ss 1.5 on this file selects nothing, -ss 0.1 selects absolute 1.6s+), so
    the renderer translates the contract's absolute window by the file's
    timeline_origin; this test proves the translation end-to-end."""
    src = make_luma_step_video(tools, tmp_path / "off.mp4", width=320, height=160,
                               fps=Fraction(30), frames=45, step_frame=10, ts_offset=1.5)
    source = probe_source(src, tools, declared_projection="equirect")
    assert abs(float(source.video.first_frame_seconds()) - 1.5) < 1e-3
    out = tmp_path / "cut0.mp4"
    render_ok(tools, source, make_plan(source, cut_in=Fraction(0), cut_out=Fraction(1, 2)), out)
    frames = decode_gray_frames(tools, out, OUT_W, OUT_H)
    idx = first_bright_frame_index(frames, OUT_W, OUT_H)
    assert idx is not None and abs(idx - 10) <= 1, \
        f"step must stay at source frame 10 in the output (got {idx})"


@requires_ffmpeg
@pytest.mark.parametrize("stage", ["process", "commit"])
def test_disk_full_fails_cleanly(tools, tmp_path, monkeypatch, stage):
    """Deterministic ENOSPC at write/commit without mounts or elevated rights."""
    import errno
    import vibecut.render.static as renderer
    from vibecut.core.project import Project

    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    project = Project.create(tmp_path / "project")
    project.add_source(source)
    project.save()
    before = project.path.read_bytes()
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"keep existing artifact")
    out = tmp_path / "out.mp4"

    def full(*args, **kwargs):
        if stage == "process":
            Path(args[1][-1]).write_bytes(b"partial")
        raise OSError(errno.ENOSPC, "injected disk full")

    if stage == "process":
        monkeypatch.setattr(FFmpegTools, "run", full)
    else:
        monkeypatch.setattr(renderer, "commit_artifact", full)
    res = render_static(make_plan(source), source, tools, out)
    assert res.status == "failed" and "disk full" in res.error
    assert not out.exists() and not list(tmp_path.glob("*.partial"))
    assert existing.read_bytes() == b"keep existing artifact"
    assert project.path.read_bytes() == before
    assert Project.load(project.path).sources[source.source_id] == source
    assert content_fingerprint(src) == source.content_fingerprint


@requires_ffmpeg
def test_proxy_dimension_change_invalidates_and_stale_source_rejected(tools, tmp_path):
    cache = Cache(tmp_path / "cache3")
    src = make_marker_equirect(tools, tmp_path / "c.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    a1 = ensure_proxy(source, tools, cache, width=960, height=480)
    first = Path(a1.path)
    mtime1 = first.stat().st_mtime_ns
    # same dims -> reuse without rebuilding
    a2 = ensure_proxy(source, tools, cache, width=960, height=480)
    assert Path(a2.path) == first and first.stat().st_mtime_ns == mtime1
    # dimension change -> different key -> new payload, old entry untouched
    a3 = ensure_proxy(source, tools, cache, width=640, height=320)
    assert Path(a3.path) != first and Path(a3.path).is_file()
    assert first.is_file()
    # stale MediaSource after on-disk mutation must never false-hit
    make_marker_equirect(tools, src, yaw_deg=90.0, pitch_deg=0.0)
    with pytest.raises(SourceChangedError):
        ensure_proxy(source, tools, cache, width=960, height=480)


@requires_ffmpeg
def test_cold_proxy_rejects_stale_source(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0, pitch_deg=0)
    source = probe_source(src, tools, declared_projection="equirect")
    src.write_bytes(src.read_bytes() + b"changed")
    cache = Cache(tmp_path / "cold")
    with pytest.raises(SourceChangedError):
        ensure_proxy(source, tools, cache)
    assert not cache.root.exists()


@requires_ffmpeg
def test_source_changed_during_probe_is_rejected(tools, tmp_path, monkeypatch):
    import vibecut.media.probe as module
    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0, pitch_deg=0)
    original = module._video_packet_pts
    def packets(*args):
        result = original(*args)
        src.write_bytes(src.read_bytes() + b"changed during probe")
        return result
    monkeypatch.setattr(module, "_video_packet_pts", packets)
    with pytest.raises(ProbeError, match="changed while probing"):
        probe_source(src, tools, declared_projection="equirect")


@requires_ffmpeg
@pytest.mark.parametrize("failure", ["exception", "mutation", "cancel"])
def test_proxy_abort_never_commits_or_leaves_temps(tools, tmp_path, monkeypatch, failure):
    import threading
    from vibecut.render.static import ExportError
    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0, pitch_deg=0)
    source = probe_source(src, tools, declared_projection="equirect")
    cache = Cache(tmp_path / "cache")
    cancel = threading.Event()
    real_run = FFmpegTools.run

    def run(self, argv, **kwargs):
        generating = str(argv[-1]).endswith(".mp4")
        if generating and failure == "exception":
            Path(argv[-1]).write_bytes(b"partial")
            raise OSError("injected process launch failure")
        result = real_run(self, argv, **kwargs)
        if generating and failure == "mutation":
            src.write_bytes(src.read_bytes() + b"changed while rendering")
        if generating and failure == "cancel":
            cancel.set()
        return result

    monkeypatch.setattr(FFmpegTools, "run", run)
    with pytest.raises((ExportError, OSError)):
        ensure_proxy(source, tools, cache, cancel_event=cancel)
    assert not list(cache.root.iterdir())


@requires_ffmpeg
def test_corrupt_proxy_hit_is_rejected(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0, pitch_deg=0)
    source = probe_source(src, tools, declared_projection="equirect")
    cache = Cache(tmp_path / "cache")
    asset = ensure_proxy(source, tools, cache)
    Path(asset.path).write_bytes(b"not a media payload")
    with pytest.raises(ProbeError):
        ensure_proxy(source, tools, cache)


@requires_ffmpeg
def test_foreign_temporary_artifact_is_preserved(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0, pitch_deg=0)
    source = probe_source(src, tools, declared_projection="equirect")
    out = tmp_path / "out.mp4"
    foreign = tmp_path / ".out.mp4.tmp-same-job.partial"
    foreign.write_bytes(b"another owner's data")
    result = render_static(make_plan(source), source, tools, out, job_id="same-job")
    assert result.status == "failed" and not out.exists()
    assert foreign.read_bytes() == b"another owner's data"


@requires_ffmpeg
@pytest.mark.parametrize("phase", ["process-exit", "validated", "post-fingerprint"])
def test_render_abort_after_work_cleans_own_temp(tools, tmp_path, monkeypatch, phase):
    import threading
    import vibecut.render.static as renderer
    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0, pitch_deg=0)
    source = probe_source(src, tools, declared_projection="equirect")
    cancel = threading.Event()
    if phase == "process-exit":
        real_run = FFmpegTools.run
        def run(self, argv, **kwargs):
            result = real_run(self, argv, **kwargs)
            cancel.set()
            return result
        monkeypatch.setattr(FFmpegTools, "run", run)
    elif phase == "validated":
        real_validate = renderer.validate_output
        def validate(*args, **kwargs):
            result = real_validate(*args, **kwargs)
            cancel.set()
            return result
        monkeypatch.setattr(renderer, "validate_output", validate)
    else:
        real_fp = renderer.content_fingerprint
        calls = 0
        def fingerprint(path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("source disappeared after render")
            return real_fp(path)
        monkeypatch.setattr(renderer, "content_fingerprint", fingerprint)
    result = render_static(make_plan(source), source, tools, tmp_path / "out.mp4",
                           cancel_event=cancel)
    assert result.status == ("failed" if phase == "post-fingerprint" else "cancelled")
    assert not (tmp_path / "out.mp4").exists()
    assert not list(tmp_path.glob("*.partial"))
    assert all(event[0] != "committed" for event in result.events)


@requires_ffmpeg
@pytest.mark.parametrize("offset", [Fraction(0), Fraction(3, 2)])
def test_edit_list_preroll_is_not_display_duration(tools, tmp_path, offset):
    original = tmp_path / "original.mp4"
    result = run_ffmpeg(tools, [
        "-f", "lavfi", "-i", "testsrc2=size=320x160:rate=30", "-frames:v", "60",
        "-c:v", "libx264", "-g", "60", "-pix_fmt", "yuv420p",
        "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-colorspace", "bt709", "-color_range", "tv", str(original),
    ])
    assert result.returncode == 0, result.stderr
    trimmed = tmp_path / "trimmed.mp4"
    result = run_ffmpeg(tools, ["-ss", "0.5", "-i", str(original), "-c", "copy", str(trimmed)])
    assert result.returncode == 0, result.stderr
    if offset:
        # Add a leading empty edit while preserving the real preroll edit.
        # FFmpeg's output_ts_offset instead exposes those preroll packets.
        import struct
        data = bytearray(trimmed.read_bytes())
        def box(name):
            index = data.index(name)
            assert index == data.rindex(name), "fixture requires one unfragmented video track"
            return index
        moov, trak, edts, elst = [box(name) for name in (b"moov", b"trak", b"edts", b"elst")]
        mvhd, tkhd = box(b"mvhd"), box(b"tkhd")
        assert moov > box(b"mdat"), "fixture moov must follow media data"
        assert data[elst + 4] == data[mvhd + 4] == data[tkhd + 4] == 0
        assert int.from_bytes(data[elst + 8:elst + 12], "big") == 1
        timescale = int.from_bytes(data[mvhd + 16:mvhd + 20], "big")
        delay = offset * timescale
        assert delay.denominator == 1
        for index in (moov, trak, edts, elst):
            size = int.from_bytes(data[index - 4:index], "big")
            data[index - 4:index] = (size + 12).to_bytes(4, "big")
        for index in (mvhd + 20, tkhd + 24):
            duration = int.from_bytes(data[index:index + 4], "big")
            data[index:index + 4] = (duration + int(delay)).to_bytes(4, "big")
        data[elst + 8:elst + 12] = (2).to_bytes(4, "big")
        data[elst + 12:elst + 12] = struct.pack(">IiHH", int(delay), -1, 1, 0)
        trimmed.write_bytes(data)
    packets = subprocess.run(tools.ffprobe_argv(
        "-show_packets", "-show_entries", "packet=pts,flags", "-of", "json", str(trimmed)),
        capture_output=True, check=True, timeout=15)
    discard = [p for p in json.loads(packets.stdout)["packets"] if "D" in p["flags"]]
    assert len(discard) == 15
    minimum = min(int(p["pts"]) for p in discard)
    assert minimum > 0 if offset else minimum < 0
    source = probe_source(trimmed, tools, declared_projection="equirect")
    assert source.video.first_frame_seconds() == offset
    assert source.timeline_origin == offset
    assert source.video.nb_frames == 45, "discard preroll packets are not displayed frames"
    assert make_plan(source).duration_rel == Fraction(3, 2)
    before = content_fingerprint(trimmed)
    render_ok(tools, source, make_plan(source), tmp_path / "flat.mp4")
    proxy = ensure_proxy(source, tools, Cache(tmp_path / "cache"))
    assert Path(proxy.path).is_file()
    assert content_fingerprint(trimmed) == before


@requires_ffmpeg
@pytest.mark.parametrize("entry", ["render", "proxy"])
def test_corrupt_packet_never_becomes_success(tools, tmp_path, entry):
    src = tmp_path / "corrupt-packet.mp4"
    result = run_ffmpeg(tools, [
        "-f", "lavfi", "-i", "testsrc2=size=320x160:rate=30", "-frames:v", "60",
        "-c:v", "libx264", "-g", "60", "-pix_fmt", "yuv420p",
        "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-colorspace", "bt709", "-color_range", "tv", str(src),
    ])
    assert result.returncode == 0, result.stderr
    packets = subprocess.run(tools.ffprobe_argv(
        "-show_packets", "-show_entries", "packet=pos,size", "-of", "json", str(src)),
        capture_output=True, check=True, timeout=15)
    packet = json.loads(packets.stdout)["packets"][20]
    data = bytearray(src.read_bytes())
    start, size = int(packet["pos"]), int(packet["size"])
    data[start:start + size] = bytes(size)
    src.write_bytes(data)
    # Prove corruption from error-level stderr, not the exit code: rc for
    # recoverable decode errors is version-dependent (6.1.1 rc=0, 9.0.2 non-zero).
    rejected = run_ffmpeg(tools, ["-xerror", "-i", str(src), "-f", "null", "-"])
    diag = rejected.stderr.decode("utf-8", "replace").strip()
    assert rejected.returncode != 0 or diag, \
        "fixture must exhibit a real decode error (non-zero exit or error-level stderr)"
    source = probe_source(src, tools, declared_projection="equirect")
    before = src.read_bytes()
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"previous valid artifact")
    cache = Cache(tmp_path / "cache")
    if entry == "render":
        result = render_static(make_plan(source), source, tools, tmp_path / "out.mp4", cache=cache)
        assert result.status == "failed", result
        assert not (tmp_path / "out.mp4").exists()
    else:
        from vibecut.render.static import ExportError
        with pytest.raises(ExportError):
            ensure_proxy(source, tools, cache)
    assert src.read_bytes() == before and existing.read_bytes() == b"previous valid artifact"
    assert not list(tmp_path.glob("*.partial"))
    assert not cache.root.exists() or not list(cache.root.iterdir())


@requires_ffmpeg
@pytest.mark.parametrize("entry", ["render", "proxy", "cachehit"])
def test_rc0_with_error_stderr_never_succeeds(tools, tmp_path, monkeypatch, entry):
    """rc=0 plus error-level stderr must be rejected (simulated exit contract).

    Why: some FFmpeg builds (6.1.x on CI) exit rc=0 while stderr carries real
    decode errors, and ffmpeg_argv pins -loglevel error, so any stderr line
    must fail the run. The diagnostics are SIMULATED onto a real, clean local
    9.0.2 run; this is not a Linux 6.1.1 real run (that stays with CI).
    """
    from dataclasses import replace

    from vibecut.render.static import ArtifactProcessError, ExportError

    simulated_stderr = (
        b"[h264 @ 0x55f54c2b4e00] Invalid NAL unit size (0 > 644)\n"
        b"[h264 @ 0x55f54c2b4e00] Error submitting packet to decoder\n"
        b"[h264 @ 0x55f54c2b4e00] Decoder thread returned error\n"
    )
    src = make_marker_equirect(tools, tmp_path / "good.mp4", yaw_deg=0.0, pitch_deg=0.0)
    source = probe_source(src, tools, declared_projection="equirect")
    before = src.read_bytes()
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"previous valid artifact")
    cache = Cache(tmp_path / "cache")
    real_run = FFmpegTools.run
    if entry == "cachehit":
        # build a REAL hot cache first; only its full-decode validation is poisoned below
        payload = Path(ensure_proxy(source, tools, cache).path)
        payload_bytes = payload.read_bytes()

    def run(self, argv, **kwargs):
        result = real_run(self, argv, **kwargs)
        # inject only into production calls: the encode (.partial/.mp4 output)
        # or, for cachehit, the full-decode validation (output "-")
        production = str(argv[-1]).endswith((".partial", ".mp4")) or (
            entry == "cachehit" and str(argv[-1]) == "-"
        )
        if str(argv[0]) == str(self.ffmpeg) and production:
            assert result.returncode == 0 and not result.stderr_tail, \
                "simulation precondition: the real run must be clean"
            return replace(result, returncode=0, stderr_tail=simulated_stderr)
        return result

    monkeypatch.setattr(FFmpegTools, "run", run)
    if entry == "render":
        result = render_static(make_plan(source), source, tools, tmp_path / "out.mp4", cache=cache)
        assert result.status == "failed" and "rc=0" in result.error and "Invalid NAL unit size" in result.error
        assert not (tmp_path / "out.mp4").exists()
        assert not cache.root.exists() or not list(cache.root.iterdir())
    elif entry == "proxy":
        with pytest.raises(ArtifactProcessError, match="Invalid NAL unit size"):
            ensure_proxy(source, tools, cache)
        assert not cache.root.exists() or not list(cache.root.iterdir())
    else:  # cachehit: the hot cache's full decode is poisoned, not its bytes
        with pytest.raises(ExportError, match="Invalid NAL unit size"):
            ensure_proxy(source, tools, cache)
        assert payload.is_file() and payload.read_bytes() == payload_bytes, \
            "rejection must preserve the existing cached payload"
        assert list(cache.root.glob("*.json")), "cache manifest must survive the rejected hit"
    assert src.read_bytes() == before and existing.read_bytes() == b"previous valid artifact"
    assert not list(tmp_path.glob("*.partial"))
