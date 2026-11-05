"""Business validation: bad bbox / unknown IDs / NaN / malformed JSON / URLs."""

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.analysis import validate
from vibecut.analysis.errors import StructuredOutputError, ValidationFailureError


def _sel(**over):
    item = {"candidate_id": "c0001", "selected": True, "relevance": 0.9, "confidence": 0.8,
            "reason": "画面中有跳舞的人", "uncertainty": None, "bbox": None, "event_time_s": None}
    item.update(over)
    return {"selections": [item]}


def test_malformed_json_rejected():
    with pytest.raises(StructuredOutputError):
        validate.parse_json_strict("{not json")


def test_nan_infinity_literals_rejected():
    with pytest.raises(StructuredOutputError):
        validate.parse_json_strict('{"selections": [{"relevance": NaN}]}')
    with pytest.raises(StructuredOutputError):
        validate.parse_json_strict('{"x": Infinity}')


def test_unknown_candidate_id_dropped():
    raw = _sel(candidate_id="c9999")
    valid, dropped = validate.validate_selections(raw, {"c0001": (Fraction(0), Fraction(2))})
    assert valid == [] and dropped


@pytest.mark.parametrize("bbox", [
    {"x": -0.1, "y": 0.0, "w": 0.5, "h": 0.5},   # x < 0
    {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.5},    # zero extent
    {"x": 0.8, "y": 0.0, "w": 0.5, "h": 0.5},    # x+w > 1 (out of frame)
    {"x": 0.0, "y": 0.9, "w": 0.5, "h": 0.5},    # y+h > 1
    {"x": "0", "y": 0, "w": 0.5, "h": 0.5},      # non-numeric
])
def test_bad_bbox_dropped(bbox):
    raw = _sel(bbox=bbox)
    valid, dropped = validate.validate_selections(raw, {"c0001": (Fraction(0), Fraction(2))})
    assert valid == []
    assert any("bbox" in d for d in dropped)


def test_out_of_range_time_dropped():
    raw = _sel(event_time_s=5.0)
    valid, dropped = validate.validate_selections(raw, {"c0001": (Fraction(0), Fraction(2))})
    assert valid == [] and any("window" in d for d in dropped)


@pytest.mark.parametrize("bad", [
    {"relevance": 1.5, "confidence": 0.5},   # > 1
    {"relevance": 0.5, "confidence": -0.1},  # < 0
    {"relevance": True, "confidence": 0.5},  # bool masquerading as number
    {"reason": "see https://evil.example/x"},  # URL in reason
    {"reason": "run C:\\tools\\boom.exe"},     # path
    {"reason": "cat a | b; rm -rf"},           # shell-like
    {"reason": ""},                            # empty reason
    {"selected": "yes"},                       # non-boolean
])
def test_invalid_fields_dropped(bad):
    over = {"candidate_id": "c0001"}
    over.update(bad)
    raw = _sel(**over)
    valid, dropped = validate.validate_selections(raw, {"c0001": (Fraction(0), Fraction(2))})
    assert valid == []
    assert dropped


def test_valid_selection_passes():
    raw = _sel(bbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}, event_time_s=1.0)
    valid, dropped = validate.validate_selections(raw, {"c0001": (Fraction(0), Fraction(2))})
    assert not dropped and len(valid) == 1
    assert valid[0]["bbox"] == (0.1, 0.2, 0.3, 0.4)


def test_observation_unknown_img_id_rejected():
    raw = {"observations": [{"img_id": "s00.ghost", "subjects": ["x"], "scene": "y",
                             "quality": "good", "motion_hint": "static"}]}
    with pytest.raises(ValidationFailureError):
        validate.validate_observations(raw, frozenset({"s00.eq+030"}))


def test_observation_url_subject_rejected():
    raw = {"observations": [{"img_id": "s00.eq+030", "subjects": ["www.evil.io"], "scene": "ok",
                             "quality": "good", "motion_hint": "static"}]}
    with pytest.raises(ValidationFailureError):
        validate.validate_observations(raw, frozenset({"s00.eq+030"}))


def test_observation_bad_enum_rejected():
    raw = {"observations": [{"img_id": "s00.eq+030", "subjects": ["x"], "scene": "y",
                             "quality": "excellent", "motion_hint": "static"}]}
    with pytest.raises(ValidationFailureError):
        validate.validate_observations(raw, frozenset({"s00.eq+030"}))


def test_json_payload_string_roundtrip():
    payload = {"selections": []}
    parsed = validate.parse_json_strict(json.dumps(payload))
    assert parsed == payload



def test_missing_or_duplicate_observations_cannot_be_cached_as_full_coverage():
    from analysis_offline_server import observe_payload
    for ids in (["a"], ["a", "a", "b"]):
        with pytest.raises(ValidationFailureError):
            validate.validate_observations(observe_payload(ids), frozenset({"a", "b"}))
