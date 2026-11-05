"""Exact independent-review reproductions, using no model or media processes."""
from fractions import Fraction as F
from types import SimpleNamespace as NS
import threading

from vibecut.analysis import service
from vibecut.analysis.candidates import discover_candidates
from vibecut.analysis.features import SampleFeatures, features_to_json
from vibecut.analysis.focus import parse_focus
from vibecut.analysis.plan import MeteringReport, SelectionPlan
from vibecut.analysis.sampling import plan_sampling
from vibecut.core.cache import Cache


def test_final_source_read_error_discards_unverified_plan(tmp_path, monkeypatch):
    import vibecut.media.probe as probe
    count = 0
    def fingerprint(path):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("final source read failed")
        return "fp"
    monkeypatch.setattr(probe, "content_fingerprint", fingerprint)
    monkeypatch.setattr(service, "_cheap_features", lambda *args: ())
    monkeypatch.setattr(service.ModelClient, "probe", lambda *args, **kwargs: {"id": "fixture"})
    monkeypatch.setattr(service, "_observe", lambda *args: ([], None))
    monkeypatch.setattr(service, "_select", lambda *args: (SelectionPlan("complete", (), F(2), F(2)), []))
    source = NS(source_id="source", path="synthetic", content_fingerprint="fp", video=NS(nb_frames=120, fps=F(30)))
    import vibecut.analysis.extract as extract
    monkeypatch.setattr(extract, "probe_asset_timing", lambda *args, **kwargs: NS(fps=F(30)))
    asset = NS(path="synthetic", source_id="source", kind="proxy", asset_id="asset", time_map=NS(asset_start_source_rel=F(0), fps=F(30)))
    result = service.analyze(source, asset, None, Cache(tmp_path), service.AnalysisConfig(), parse_focus("red square"))
    assert count == 2 and result.status == "failed"
    assert result.plan is None and result.plan_status is None
    assert result.diagnostics[0]["code"] == "os-error"
    assert result.to_json()["plan"] is None


def test_motion_threshold_matches_cold_and_existing_rounded_cache(tmp_path, monkeypatch):
    import vibecut.analysis.extract as extract
    from vibecut.analysis.cachekeys import cheap_features_key
    raw_features = (SampleFeatures(0, F(10), (), (0.0099999,) * 6),)
    monkeypatch.setattr(service, "compute_features", lambda *args: raw_features)
    monkeypatch.setattr(extract, "extract_gray_stream", lambda *args, **kwargs: (b"", 1, 1))
    source, asset, tools = NS(content_fingerprint="fp"), NS(path="synthetic", asset_id="proxy960"), NS(identity=lambda: "tool")
    sampling, config, cancel = plan_sampling(F(60), samples=3), service.AnalysisConfig(), threading.Event()
    cache = Cache(tmp_path)
    cold = service._cheap_features(source, asset, tools, cache, sampling, config, cancel, MeteringReport())
    # This is the manifest shape/precision written by the original v1 producer.
    key = cheap_features_key(source_fingerprint="fp", asset_id="proxy960", tool_identity="tool", sampling_identity=sampling.identity())
    cache.put_atomic(key, {"kind": "analysis-cheap-features", "schema_version": 1, "features": features_to_json(raw_features)})
    hot = service._cheap_features(source, asset, tools, cache, sampling, config, cancel, MeteringReport())
    assert cold == hot and cold[0].motion[0] == 0.01
    cold_candidates = discover_candidates(cold, sampling, source_id="source")
    hot_candidates = discover_candidates(hot, sampling, source_id="source")
    assert cold_candidates == hot_candidates
    assert any(c.origin == "motion" for c in cold_candidates)



def test_cheap_features_are_cold_for_different_normalized_asset_ids(tmp_path,monkeypatch):
    import vibecut.analysis.extract as extract
    calls = []
    def gray(*args, **kwargs):
        calls.append(args[1])
        return b"",1,1
    monkeypatch.setattr(extract,"extract_gray_stream",gray)
    monkeypatch.setattr(service,"compute_features",lambda *args: ())
    source, tools = NS(content_fingerprint="same-source"), NS(identity=lambda:"tool")
    sampling, cache = plan_sampling(F(60),samples=3), Cache(tmp_path)
    for identity, expected in (("proxy960","miss"),("proxy960","hit"),("proxy480","miss"),("canonical","miss")):
        meter = MeteringReport()
        service._cheap_features(source,NS(path=identity,asset_id=identity),tools,cache,sampling,service.AnalysisConfig(),threading.Event(),meter)
        assert meter.cache["cheap_features"] == expected
    assert calls == ["proxy960","proxy480","canonical"]
