"""Contract tests: rational time, serialization round-trips, cut mapping."""

from fractions import Fraction

import pytest

from vibecut.core.contracts import (
    CONTRACT_VERSION,
    AudioStream,
    CameraPath,
    CameraView,
    ColorContract,
    ContractError,
    EditPlan,
    JobEvent,
    MediaSource,
    NormalizedAsset,
    TimeMap,
    VideoStream,
    parse_frac,
)


def make_source(*, start_pts=0, fps=Fraction(30), audio_start_pts=None):
    video = VideoStream(
        index=0, codec="h264", width=1920, height=960,
        start_pts=start_pts, time_base=Fraction(1, 15360),
        fps=fps, r_frame_rate=fps, nb_frames=90,
    )
    audio = None
    if audio_start_pts is not None:
        audio = AudioStream(
            index=1, codec="aac", channels=2, sample_rate=48000,
            start_pts=audio_start_pts, time_base=Fraction(1, 48000),
        )
    fp = "ab" * 32
    return MediaSource(
        schema_version=CONTRACT_VERSION, source_id=f"v1:{fp[:16]}", path="C:/x/y.mp4",
        content_fingerprint=fp, size_bytes=123, container="mp4", projection="equirect",
        stereo="mono", color=ColorContract("bt709", "bt709", "bt709", "tv", 8),
        video=video, audio=audio, vfr=False, declared=True,
    )


def test_parse_frac_forms():
    assert parse_frac("30000/1001") == Fraction(30000, 1001)
    assert parse_frac("1.5") == Fraction(3, 2)
    assert parse_frac(3) == 3
    assert parse_frac(0.25) == Fraction(1, 4)
    with pytest.raises(Exception):
        parse_frac("abc")
    with pytest.raises(Exception):
        parse_frac(True)


def test_fraction_roundtrip_through_json():
    src = make_source(start_pts=15360, fps=Fraction(30000, 1001), audio_start_pts=2400)
    again = MediaSource.from_json(src.to_json())
    assert again.video.fps == Fraction(30000, 1001)
    assert again.video.time_base == Fraction(1, 15360)
    assert again.audio.start_pts == 2400
    assert again.source_id == src.source_id


def test_cut_window_absolute_uses_first_displayed_frame():
    src = make_source(start_pts=15360)  # first display frame at 1.0s
    a, b = src.cut_window_absolute(Fraction(0), Fraction(1, 2))
    assert a == Fraction(1) and b == Fraction(3, 2)


def test_cut_window_nonzero_start_and_30000_1001():
    src = make_source(start_pts=0, fps=Fraction(30000, 1001))
    a, b = src.cut_window_absolute(Fraction(1001, 30000), Fraction(2002, 30000))
    assert a == Fraction(1001, 30000)
    assert b == Fraction(2002, 30000)


def test_media_source_rejects_non_2to1_and_bad_ids():
    video = VideoStream(0, "h264", 1280, 960, 0, Fraction(1, 15360), Fraction(30), Fraction(30), 10)
    fp = "cd" * 32
    with pytest.raises(Exception):
        MediaSource(1, f"v1:{fp[:16]}", "p", fp, 1, "mp4", "equirect", "mono",
                    ColorContract("bt709", "bt709", "bt709", "tv", 8), video, None, False, True)


def test_media_source_stabilization_provenance_frozen():
    src = make_source()
    assert src.stabilization_provenance == "none"
    again = MediaSource.from_json(src.to_json())
    assert again.stabilization_provenance == "none"
    bad = {**src.to_json(), "stabilization_provenance": "sdk-v9"}
    with pytest.raises(Exception):
        MediaSource.from_json(bad)  # unknown provenance must not round-trip into a valid contract


def test_edit_plan_roundtrip():
    cam = CameraPath(schema_version=CONTRACT_VERSION, keys=((Fraction(0), CameraView(yaw_deg=90.0)),), provenance="user")
    plan = EditPlan(CONTRACT_VERSION, "v1:ab" * 6 + "ab", Fraction(1, 3), Fraction(2, 3), Fraction(24000, 1001), cam, True)
    again = EditPlan.from_json(plan.to_json())
    assert again.cut_in_rel == Fraction(1, 3)
    assert again.project_fps == Fraction(24000, 1001)
    assert again.camera.static_view.yaw_deg == 90.0
    assert again.camera.provenance == "user"
    assert again.cut_manual is True


def test_job_event_roundtrip_and_statuses():
    ev = JobEvent(CONTRACT_VERSION, "job", 3, "succeeded", "2026-01-01T00:00:00+00:00", "out.mp4", None)
    again = JobEvent.from_json(ev.to_json())
    assert again.status == "succeeded" and again.project_revision == 3
    with pytest.raises(Exception):
        JobEvent(CONTRACT_VERSION, "job", 3, "exploded", "t")
    with pytest.raises(Exception):
        JobEvent(CONTRACT_VERSION + 1, "job", 3, "succeeded", "t")  # wrong schema


def test_normalized_asset_time_map():
    tm = TimeMap(asset_start_source_rel=Fraction(1, 2), fps=Fraction(30000, 1001))
    asset = NormalizedAsset(CONTRACT_VERSION, "v1:xy", "proxy", "v1:ab", "p.mp4", tm,
                            ColorContract("bt709", "bt709", "bt709", "tv", 8))
    again = NormalizedAsset.from_json(asset.to_json())
    assert again.time_map.asset_start_source_rel == Fraction(1, 2)
    assert again.time_map.fps == Fraction(30000, 1001)
    assert again.stabilization_provenance == "none"


def test_camera_path_rejects_unknown_schema_version():
    view = CameraView(yaw_deg=0.0)
    with pytest.raises(ContractError):
        CameraPath(schema_version=999, keys=((Fraction(0), view),), provenance="user")
    bad = {
        "schema_version": 999,
        "keys": [["0/1", view.to_json()]],
        "provenance": "user",
    }
    with pytest.raises(ContractError):
        CameraPath.from_json(bad)  # unknown versions must not parse silently


def test_normalized_asset_rejects_unknown_schema_version():
    tm = TimeMap(asset_start_source_rel=Fraction(0), fps=Fraction(30))
    color = ColorContract("bt709", "bt709", "bt709", "tv", 8)
    with pytest.raises(ContractError):
        NormalizedAsset(999, "v1:xy", "proxy", "v1:ab", "p.mp4", tm, color)
    good = NormalizedAsset(CONTRACT_VERSION, "v1:xy", "proxy", "v1:ab", "p.mp4", tm, color)
    bad = {**good.to_json(), "schema_version": 999}
    with pytest.raises(ContractError):
        NormalizedAsset.from_json(bad)
