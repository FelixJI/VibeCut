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
