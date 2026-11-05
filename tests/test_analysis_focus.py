"""Focus parsing: Chinese text -> bounded fields (pure CPU)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.analysis.focus import FocusError, parse_focus


def test_person_priority_and_duration():
    f = parse_focus("人物优先，重点拍跳舞的人，不要夜市摊位，剪出45秒")
    assert f.priority == "person"
    assert f.person_weight > f.scenery_weight
    assert f.target_duration_s == Fraction(45)
    assert f.exclude_terms == ("夜市摊位",)
    assert "跳舞的人" in f.include_terms


def test_scenery_priority_minutes():
    f = parse_focus("风景优先，包含海边日落，排除游客，目标1分钟")
    assert f.priority == "scenery"
    assert f.scenery_weight > f.person_weight
    assert f.target_duration_s == Fraction(60)
    assert f.exclude_terms == ("游客",)


def test_explicit_args_win():
    f = parse_focus("人物优先 剪出45秒", priority="scenery", target_duration_s=Fraction(10))
    assert f.priority == "scenery"
    assert f.target_duration_s == Fraction(10)


def test_balanced_default_duration():
    f = parse_focus("随便看看")
    assert f.priority == "balanced"
    assert f.target_duration_s == Fraction(30)


def test_bounds_rejected():
    with pytest.raises(FocusError):
        parse_focus("")
    with pytest.raises(FocusError):
        parse_focus("x" * 501)
    with pytest.raises(FocusError):
        parse_focus("剪出0秒的片段")


def test_json_roundtrip_bounded():
    f = parse_focus("人物优先，不要广告牌")
    data = f.to_json()
    assert data["priority"] == "person"
    assert data["target_duration_s"] == "30/1"
    assert len(data["exclude_terms"]) <= 4
