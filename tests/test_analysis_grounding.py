"""Candidate directions and same-window evidence are part of the model boundary."""
from fractions import Fraction
from types import SimpleNamespace
import threading

from vibecut.analysis.candidates import discover_candidates
from vibecut.analysis.focus import parse_focus
from vibecut.analysis.plan import MeteringReport, ModelSelection, Observation
from vibecut.analysis.sampling import plan_sampling
from vibecut.analysis.service import AnalysisConfig, _select
from vibecut.core.cache import Cache


def test_every_direction_including_poles_and_seam_is_selectable():
    sampling = plan_sampling(Fraction(120), samples=12)
    candidates = discover_candidates((), sampling, source_id="source")
    assert len(candidates) <= 24
    assert {c.view_id for c in candidates} == {v.view_id for v in sampling.views}
    for c in candidates:
        assert any(c.start_rel <= t < c.end_rel for t in sampling.times)
    assert any(c.view_id.startswith("p+") for c in candidates)
    assert any(c.view_id.startswith("p-") for c in candidates)
    assert {"eq-150", "eq+150"} <= {c.view_id for c in candidates}
    assert {int(c.start_rel * 3 / 120) for c in candidates} == {0, 1, 2}


def test_selection_cannot_borrow_same_view_evidence_from_another_window(tmp_path):
    sampling = plan_sampling(Fraction(120), samples=12)
    candidates = discover_candidates((), sampling, source_id="source")
    candidate = candidates[0]
    t = next(t for t in sampling.times if candidate.start_rel <= t < candidate.end_rel)
    observations = [Observation("real", t, candidate.view_id, ("person",), "scene", "good", "static"),
                    Observation("late", Fraction(119), candidate.view_id, ("wrong",), "elsewhere", "good", "static")]
    class Client:
        model_id = "fixture"
        max_tokens = 2048
        def select(self, focus, payload, windows, **kwargs):
            assert set(windows) == {candidate.candidate_id}
            assert len(payload) == 1
            assert payload[0]["observed"][0]["img_id"] == "real"
            assert payload[0]["observed"][0]["time_rel"] == f"{t.numerator}/{t.denominator}"
            assert "wrong" not in str(payload)
            return [ModelSelection(candidate.candidate_id, True, 1, 1, "person", img_ids=("real",), times_rel=(t,))], [], []
    source = SimpleNamespace(content_fingerprint="fixture")
    asset = SimpleNamespace(asset_id="fixture")
    result, _ = _select(source, asset, Cache(tmp_path), Client(), AnalysisConfig(), parse_focus("人物"),
                        candidates, observations, sampling, MeteringReport(), threading.Event(), None)
    assert result.segments[0].evidence[0].candidate_id == candidate.candidate_id
    assert result.segments[0].evidence[0].direction is not None
