"""Structural projection detection (pure unit tests - no ffmpeg needed).

Regression guards for the contract correction: detection must trust only
precise structured markers, never free text (filenames, comments, tag soup),
and must reject stereo layouts and contradictory metadata.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.media.probe import UnsupportedSourceError, structured_projection  # noqa: E402


def probe_with(streams):
    return {"streams": streams, "format": {"filename": "whatever.mp4"}}


def test_no_metadata_returns_none():
    assert structured_projection(probe_with([
        {"index": 0, "codec_type": "video", "tags": {"language": "und"},
         "side_data_list": [{"side_data_type": "Display Matrix"}]},
    ])) is None


def test_equirect_side_data_detected():
    probe = probe_with([
        {"index": 0, "codec_type": "video",
         "side_data_list": [{"side_data_type": "Spherical Mapping", "projection": "equirectangular"}]},
    ])
    assert structured_projection(probe) == "equirect"


def test_projection_type_tag_detected():
    probe = probe_with([
        {"index": 0, "codec_type": "video", "tags": {"ProjectionType": "equirectangular"}},
    ])
    assert structured_projection(probe) == "equirect"


def test_filename_or_comment_text_is_never_trusted():
    """A path/comment containing 'equirect' must NOT be treated as metadata."""
    probe = {
        "streams": [{"index": 0, "codec_type": "video", "tags": {"comment": "equirect test"}}],
        "format": {"filename": "my_equirect_panorama_360.mp4"},
    }
    assert structured_projection(probe) is None


def test_non_equirect_structured_marker_rejected():
    probe = probe_with([
        {"index": 0, "codec_type": "video",
         "side_data_list": [{"side_data_type": "Spherical Mapping", "projection": "cubemap"}]},
    ])
    with pytest.raises(UnsupportedSourceError):
        structured_projection(probe)


def test_spherical_unspecified_rejected():
    probe = probe_with([
        {"index": 0, "codec_type": "video",
         "side_data_list": [{"side_data_type": "Spherical Mapping"}]},
    ])
    with pytest.raises(UnsupportedSourceError):
        structured_projection(probe)


def test_stereo_layout_rejected():
    probe = probe_with([
        {"index": 0, "codec_type": "video",
         "side_data_list": [{"side_data_type": "Spherical Mapping", "projection": "equirectangular",
                             "stereo": "top-bottom"}]},
    ])
    with pytest.raises(UnsupportedSourceError, match="stereo"):
        structured_projection(probe)


def test_contradictory_markers_rejected():
    probe = probe_with([
        {"index": 0, "codec_type": "video", "tags": {"ProjectionType": "equirectangular"}},
        {"index": 1, "codec_type": "video",
         "side_data_list": [{"side_data_type": "Spherical Mapping", "projection": "cubemap"}]},
    ])
    with pytest.raises(UnsupportedSourceError, match="contradictory"):
        structured_projection(probe)


def test_dual_fisheye_tag_rejected():
    probe = probe_with([
        {"index": 0, "codec_type": "video", "tags": {"ProjectionType": "equirect"}},
    ])
    assert structured_projection(probe) == "equirect"
    other = probe_with([
        {"index": 0, "codec_type": "video", "tags": {"ProjectionType": "fisheye"}},
    ])
    with pytest.raises(UnsupportedSourceError):
        structured_projection(other)


def test_equivalent_projection_aliases_agree():
    probe = probe_with([{
        "codec_type": "video", "tags": {"ProjectionType": "equirect"},
        "side_data_list": [{"side_data_type": "Spherical Mapping", "projection": "equirectangular"}],
    }])
    assert structured_projection(probe) == "equirect"


def test_first_display_pts_comes_from_packets_not_a_later_stream_start():
    from fractions import Fraction
    from vibecut.media.probe import _verify_cfr
    pts = [0, 512, 1024]
    with pytest.raises(UnsupportedSourceError, match="inconsistent"):
        _verify_cfr(pts, Fraction(1, 15360), Fraction(30), 512)
    first, fps = _verify_cfr(pts, Fraction(1, 15360), Fraction(30), 1)
    assert first == 0 and fps == 30


def test_probe_interrupt_reaps_process_and_removes_temp(tmp_path, monkeypatch):
    import subprocess
    import tempfile
    import vibecut.media.probe as module

    ready = tmp_path / "child-ready"
    children = []
    original_popen = subprocess.Popen
    original_mkstemp = tempfile.mkstemp
    interrupted = False
    def temporary(*args, **kwargs):
        kwargs["dir"] = tmp_path
        return original_mkstemp(*args, **kwargs)
    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        original_poll = child.poll
        def poll():
            nonlocal interrupted
            if ready.exists() and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("injected probe interruption")
            return original_poll()
        child.poll = poll
        return child
    monkeypatch.setattr(tempfile, "mkstemp", temporary)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    script = f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(8)"
    try:
        with pytest.raises(KeyboardInterrupt, match="injected probe interruption"):
            module._run_ffprobe_to_tempfile([sys.executable, "-c", script], timeout_s=10)
        assert len(children) == 1 and children[0].returncode is not None
        assert children[0].poll() is not None, "interrupted probe child survived"
        assert not list(tmp_path.glob("vibecut-ffprobe-*"))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("mode", ["timeout", "cancel", "nonzero"])
def test_probe_tempfile_failure_cleanup(tmp_path, monkeypatch, mode):
    import subprocess
    import tempfile
    import threading
    import vibecut.media.probe as module

    original_mkstemp = tempfile.mkstemp
    original_popen = subprocess.Popen
    children = []
    def temporary(*args, **kwargs):
        kwargs["dir"] = tmp_path
        return original_mkstemp(*args, **kwargs)
    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(tempfile, "mkstemp", temporary)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    cancel = threading.Event()
    if mode == "cancel":
        cancel.set()
    script = "import sys,time; sys.stderr.buffer.write(b'e'*200000); sys.stdout.buffer.write(b'x'*200000); "
    script += "sys.exit(3)" if mode == "nonzero" else "time.sleep(8)"
    with pytest.raises(module.ProbeError):
        module._run_ffprobe_to_tempfile([sys.executable, "-c", script], timeout_s=0.4,
                                       cancel_event=cancel)
    assert children and children[0].returncode is not None
    assert not list(tmp_path.glob("vibecut-ffprobe-*"))
