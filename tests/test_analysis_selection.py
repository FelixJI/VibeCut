"""Selection algorithm: merge/dedupe, duration budget <=1s, short/empty plans."""

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.analysis.focus import parse_focus
from vibecut.analysis.plan import CandidateWindow, ModelSelection
from vibecut.analysis.select import SelectionParams, build_selection_plan
from vibecut.analysis.plan import to_edit_plan
from vibecut.core.contracts import CONTRACT_VERSION


def _cand(cid, start, end, view="eq+030", score=0.5, source_id="v1:abc"):
    return CandidateWindow(candidate_id=cid, source_id=source_id, start_rel=Fraction(start),
                           end_rel=Fraction(end), view_id=view, origin="motion", score=score)


def _sel(cid, relevance=0.9, confidence=0.9, reason="有人物活动", selected=True, **kw):
    return ModelSelection(candidate_id=cid, selected=selected, relevance=relevance,
                          confidence=confidence, reason=reason, **kw)


def _focus(duration=30, text="人物优先"):
    return parse_focus(text, target_duration_s=Fraction(duration))


def test_merge_adjacent_and_dedupe_overlap():
    cands = (_cand("c1", 0, 5), _cand("c2", 4.5, 8), _cand("c3", 4.8, 8.2), _cand("c4", 20, 25))
    sels = (_sel("c1"), _sel("c2", 0.5), _sel("c3", 0.4), _sel("c4", 0.8))
    plan = build_selection_plan(cands, sels, _focus(duration=25), Fraction(60))
    spans = [(s.cut_in_rel, s.cut_out_rel) for s in plan.segments]
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 <= s2, "segments must never overlap"
    assert len(plan.segments) == 2  # c1+c2+c3 merged, c4 separate


def test_duration_tolerance_one_second():
    # 12s + 12s = 24s vs target 24.5s -> within 1s tolerance, complete
    cands = (_cand("c1", 0, 12), _cand("c2", 20, 32))
    plan = build_selection_plan(cands, (_sel("c1"), _sel("c2")), _focus(duration=24.5), Fraction(60))
    assert plan.status == "complete"
    assert abs(float(plan.total_duration_s) - 24.0) <= 1.0


def test_short_plan_explicit_no_padding():
    cands = (_cand("c1", 0, 6),)
    plan = build_selection_plan(cands, (_sel("c1"),), _focus(duration=30), Fraction(60))
    assert plan.status == "short"
    assert plan.total_duration_s == Fraction(6)
    assert plan.shortfall_reason and "不足" in plan.shortfall_reason
    assert len(plan.segments) == 1  # no fabrication, no loop padding


def test_zero_match_returns_empty_for_manual_review():
    cands = (_cand("c1", 0, 6),)
    plan = build_selection_plan(cands, (_sel("c1", selected=False),), _focus(duration=30), Fraction(60))
    assert plan.status == "empty"
    assert plan.segments == ()
    assert plan.shortfall_reason


def test_never_exceeds_max_segment_length():
    cands = (_cand("c1", 0, 60),)
    plan = build_selection_plan(cands, (_sel("c1"),), _focus(duration=30), Fraction(60),
                                SelectionParams(max_segment_s=Fraction(10)))
    assert plan.segments and all(s.duration_rel <= Fraction(10) for s in plan.segments)


def test_evidence_is_traceable():
    cands = (_cand("c1", 2, 7),)
    sel = _sel("c1", bbox=(0.2, 0.3, 0.4, 0.5), event_time_s=4.0)
    plan = build_selection_plan(cands, (sel,), _focus(duration=5), Fraction(60))
    seg = plan.segments[0]
    assert seg.evidence[0].candidate_id == "c1"
    assert seg.evidence[0].time_rel == Fraction(4)
    assert seg.evidence[0].view_id == "eq+030"
    assert seg.source_id == "v1:abc"


def test_direction_callback_populates_evidence():
    cands = (_cand("c1", 2, 7),)
    sel = _sel("c1", bbox=(0.25, 0.25, 0.5, 0.5))
    plan = build_selection_plan(cands, (sel,), _focus(duration=5), Fraction(60),
                                direction_of=lambda c, s: (123.0, 4.0))
    assert plan.segments[0].evidence[0].direction == (123.0, 4.0)


def test_to_edit_plan_automation_semantics():
    from vibecut.analysis.views import ViewSpec

    cands = (_cand("c1", 2, 7),)
    plan = build_selection_plan(cands, (_sel("c1"),), _focus(duration=5), Fraction(60))
    view = ViewSpec("eq+030", 30.0, 0.0)
    edit = to_edit_plan(plan.segments[0], view, Fraction(30000, 1001))
    assert edit.schema_version == CONTRACT_VERSION
    assert edit.cut_manual is False
    assert edit.camera.provenance == "derived"
    assert edit.cut_in_rel == plan.segments[0].cut_in_rel
    # static export (task #2) must refuse derived views: human confirms first
    with __import__("pytest").raises(Exception):
        _ = edit.camera.static_view



def test_trimmed_plan_keeps_actual_evidence_inside_retained_segment():
    cands = (_cand("early", 0, 4), _cand("late", 4, 60))
    sels = (_sel("early", relevance=1, event_time_s=0.123456789), _sel("late", relevance=0.2, event_time_s=50))
    plan = build_selection_plan(cands, sels, _focus(duration=2), Fraction(60))
    segment = plan.segments[0]
    assert [e.candidate_id for e in segment.evidence] == ["early"]
    assert segment.evidence[0].time_rel == Fraction("0.123456789")
    assert segment.cut_in_rel <= segment.evidence[0].time_rel < segment.cut_out_rel


def test_one_second_target_does_not_return_empty_when_material_exists():
    plan = build_selection_plan((_cand("c1", 0, 1),), (_sel("c1"),), _focus(duration=1), Fraction(1))
    assert plan.segments and plan.total_duration_s == 1
