"""End-to-end analysis over synthetic media + offline loopback model fixtures.

Covers: full-pipeline status, evidence traceability, AC7 cache separation
(hot run / changed focus / model requests), ai-unavailable path, cancellation
with partial cache retention, budget-bounded partial results. Synthetic
fixtures only (conftest generators); the "model" is the 127.0.0.1 fixture.
"""

import sys
import threading
from fractions import Fraction
from pathlib import Path

import pytest
from conftest import make_luma_step_video, requires_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis_offline_server import OfflineLM, observe_payload, select_payload

from vibecut.analysis import AnalysisConfig, analyze, parse_focus
from vibecut.analysis.transport import LoopbackTransport
from vibecut.core.cache import Cache


@pytest.fixture()
def luma_source(tools, tmp_path):
    """3 s equirect luma-step source: motion event at ~1.5 s."""
    src = make_luma_step_video(tools, tmp_path / "pano.mp4", width=640, height=320,
                               fps=Fraction(30), frames=90, step_frame=45)
    from vibecut.media.probe import probe_source

    return probe_source(src, tools, declared_projection="equirect")


@pytest.fixture()
def lm():
    server = OfflineLM().start()
    yield server
    server.stop()


def _config(**over):
    cfg = AnalysisConfig(samples=3, max_model_requests=24, model_timeout_s=10.0)
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def _select_all(candidates):
    items = []
    times = (Fraction(1, 2), Fraction(3, 2), Fraction(5, 2))
    for c in candidates:
        index = next(i for i, t in enumerate(times) if c.start_rel <= t < c.end_rel)
        items.append({"candidate_id": c.candidate_id, "img_ids": [f"s{index:02d}.{c.view_id}"],
                      "relevance": 0.9, "confidence": 0.85, "reason": "检测到人物活动"})
    return select_payload(items)


def fill_observe_script(lm, samples: int, *, delay: float = 0.0) -> None:
    """Script observe responses exactly as the client batches them (8 imgs/request)."""
    views = _views()
    chunks = [views[j:j + 8] for j in range(0, len(views), 8)]
    for i in range(samples):
        for ch in chunks:
            lm.chat_script.append({"content": observe_payload([f"s{i:02d}." + v.view_id for v in ch]),
                                   **({"delay": delay} if delay else {})})


@requires_ffmpeg
def test_full_flow_with_offline_model(luma_source, lm, tools, tmp_path):
    cache = Cache(tmp_path / "cache")
    # pre-build the proxy through the real #2 pipeline so analysis reuses it
    from vibecut.media.proxy import ensure_proxy

    asset = ensure_proxy(luma_source, tools, cache)

    fill_observe_script(lm, 3)
    lm.chat_script.append({"content": {"selections": []}})

    focus = parse_focus("人物优先，剪出2秒")
    result = analyze(luma_source, asset, tools, cache, _config(), focus,
                     transport=LoopbackTransport(lm.base_url, request_timeout_s=10.0))
    assert result.status in ("succeeded", "partial")
    assert result.asset_id == asset.asset_id
    assert result.model_info["id"] == "qwen3.8-27b@q4_k_xl"
    assert result.candidates, "cheap candidates must exist for manual review"
    # observations arrived for every sampled view (prompt-independent)
    assert len(result.observations) == 3 * 14
    # metering records requests; images per request bounded
    obs_requests = [r for r in result.metering.requests if r["kind"] in ("observe", "retry")]
    assert all(r["images"] <= 8 for r in obs_requests)
    assert result.metering.request_count >= 7
    # empty selection -> empty plan with explicit reason (never fabricated)
    assert result.plan is not None and result.plan.status == "empty"


def _views():
    from vibecut.analysis.views import default_view_set

    return default_view_set()


@requires_ffmpeg
def test_hot_run_and_focus_change_cache_semantics(luma_source, lm, tools, tmp_path):
    cache = Cache(tmp_path / "cache")
    from vibecut.media.proxy import ensure_proxy

    asset = ensure_proxy(luma_source, tools, cache)

    def fill_scripts():
        fill_observe_script(lm, 3)
        lm.chat_script.append({"content": {"selections": []}})

    tr = LoopbackTransport(lm.base_url, request_timeout_s=10.0)
    fill_observe_script(lm, 3)
    lm.chat_script.append({"content": {"selections": []}})
    r1 = analyze(luma_source, asset, tools, cache, _config(), parse_focus("人物优先"), transport=tr)
    cold_requests = len(lm.chat_calls)
    assert cold_requests == 7  # 6 observe (2 chunks x 3 samples) + 1 select

    # hot run: same source, same params -> zero model requests (AC7);
    # no script is queued: any model call would hit the default (invalid)
    # response and fail loudly instead of silently "passing".
    r2 = analyze(luma_source, asset, tools, cache, _config(), parse_focus("人物优先"), transport=tr)
    assert len(lm.chat_calls) == cold_requests
    assert r2.metering.cache["cheap_features"] == "hit"
    assert r2.metering.cache.get("observation_batches_hit", 0) == 6
    assert r2.metering.cache["selection"] == "hit"

    # changed focus: observations reused (hit), selection recomputed (miss -> 1 request)
    lm.chat_script.clear()  # r1 script fully consumed; nothing may be left over
    lm.chat_script.append({"content": {"selections": []}})
    r3 = analyze(luma_source, asset, tools, cache, _config(), parse_focus("风景优先，看日落"), transport=tr)
    assert len(lm.chat_calls) == cold_requests + 1
    assert r3.metering.cache.get("observation_batches_hit", 0) == 6
    assert r3.metering.cache["selection"] == "miss"


@requires_ffmpeg
def test_ai_unavailable_keeps_manual_path(luma_source, lm, tools, tmp_path):
    lm.model_state = "not-loaded"
    cache = Cache(tmp_path / "cache")
    from vibecut.media.proxy import ensure_proxy

    asset = ensure_proxy(luma_source, tools, cache)
    result = analyze(luma_source, asset, tools, cache, _config(), parse_focus("人物优先"),
                     transport=LoopbackTransport(lm.base_url, request_timeout_s=10.0))
    assert result.status == "ai-unavailable"
    assert result.plan is None  # ai-unavailable never fabricates segments
    assert result.candidates  # manual-review windows still delivered
    assert result.diagnostics[0]["code"] == "model-not-loaded"
    assert len(lm.chat_calls) == 0  # no inference attempted when not loaded


@requires_ffmpeg
def test_cancellation_mid_observation_retains_batches(luma_source, lm, tools, tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache")
    from vibecut.media.proxy import ensure_proxy

    asset = ensure_proxy(luma_source, tools, cache)
    # Cancel after two client-validated cache commits, not after a server send:
    # server send completion does not establish that the client accepted a batch.
    cancel = threading.Event()
    put_atomic = cache.put_atomic
    committed = 0

    def cancel_after_two(*args, **kwargs):
        nonlocal committed
        result = put_atomic(*args, **kwargs)
        if kwargs.get("meta", {}).get("kind") == "analysis-observation-batch":
            committed += 1
            if committed == 2:
                cancel.set()
        return result

    monkeypatch.setattr(cache, "put_atomic", cancel_after_two)
    fill_observe_script(lm, 3)
    result = analyze(luma_source, asset, tools, cache, _config(samples=3), parse_focus("人物优先"),
                     cancel_event=cancel,
                     transport=LoopbackTransport(lm.base_url, request_timeout_s=10.0))
    assert result.status == "cancelled"
    assert result.plan is None
    assert len(result.observations) == 14  # both confirmed batches survive cancellation
    # completed batches before cancellation are cached (confirmed work retained)
    assert result.metering.cache.get("observation_batches_miss", 0) >= 1

    # rerun without cancel: cached batches are hits, the remaining ones are
    # requested, then exactly one select; the leftover script entries align
    # with whatever the cancelled run actually consumed.
    calls_after_cancel = len(lm.chat_calls)
    lm.chat_script.append({"content": {"selections": []}})
    result2 = analyze(luma_source, asset, tools, cache, _config(samples=3), parse_focus("人物优先"),
                      transport=LoopbackTransport(lm.base_url, request_timeout_s=10.0))
    assert result2.status in ("succeeded", "partial")
    hits = result2.metering.cache.get("observation_batches_hit", 0)
    rerun_misses = result2.metering.cache.get("observation_batches_miss", 0)
    assert hits >= 1  # confirmed batches from the cancelled run were reused
    assert hits + rerun_misses == 6  # every batch accounted for exactly once
    assert len(lm.chat_calls) - calls_after_cancel == rerun_misses + 1  # + 1 select


@requires_ffmpeg
def test_request_budget_yields_partial_result(luma_source, lm, tools, tmp_path):
    cache = Cache(tmp_path / "cache")
    from vibecut.media.proxy import ensure_proxy

    asset = ensure_proxy(luma_source, tools, cache)
    fill_observe_script(lm, 3)
    lm.chat_script.append({"content": {"selections": []}})
    cfg = _config(max_model_requests=2)
    result = analyze(luma_source, asset, tools, cache, cfg, parse_focus("人物优先"),
                     transport=LoopbackTransport(lm.base_url, request_timeout_s=10.0))
    assert result.status == "partial"
    assert any(d["code"] == "budget-exceeded" for d in result.diagnostics)
    assert result.metering.request_count <= 2  # hard ceiling includes repairs


@requires_ffmpeg
def test_selection_plan_duration_and_evidence(luma_source, lm, tools, tmp_path):
    cache = Cache(tmp_path / "cache")
    from vibecut.media.proxy import ensure_proxy

    asset = ensure_proxy(luma_source, tools, cache)
    fill_observe_script(lm, 3)

    # cheap candidates first (no model) to script a matching selection
    cfg = _config()
    focus = parse_focus("人物优先，剪出3秒")
    lm.model_state = "not-loaded"
    draft = analyze(luma_source, asset, tools, cache, _config(), focus,
                    transport=LoopbackTransport(lm.base_url, request_timeout_s=10.0))
    assert draft.status == "ai-unavailable"
    lm.model_state = "loaded"
    # now script selection over the real candidate ids
    lm.chat_script.append({"content": _select_all(draft.candidates)})
    result = analyze(luma_source, asset, tools, cache, cfg, focus,
                     transport=LoopbackTransport(lm.base_url, request_timeout_s=10.0))
    assert result.plan is not None
    if result.plan.segments:
        seg = result.plan.segments[0]
        assert 0 <= seg.cut_in_rel < seg.cut_out_rel <= Fraction(3)
        for ev in seg.evidence:
            matching = [c for c in result.candidates if c.candidate_id == ev.candidate_id]
            assert matching, "evidence must reference real candidates"
            assert matching[0].view_id == ev.view_id
        total = float(result.plan.total_duration_s)
        assert total <= 3.0 + 1.0  # duration budget respected (<=1s tolerance)


@requires_ffmpeg
def test_rejected_model_text_never_leaks_into_results(luma_source, lm, tools, tmp_path):
    cache = Cache(tmp_path / "cache")
    from vibecut.media.proxy import ensure_proxy

    asset = ensure_proxy(luma_source, tools, cache)
    fill_observe_script(lm, 1)
    lm.chat_script.append({"content": select_payload([
        {"candidate_id": "cXXXX", "img_ids": ["unknown"], "relevance": 0.9, "confidence": 0.9,
         "reason": "see https://evil.example/x"},
    ])})
    result = analyze(luma_source, asset, tools, cache, _config(samples=1), parse_focus("人物优先"),
                     transport=LoopbackTransport(lm.base_url, request_timeout_s=10.0))
    import json

    blob = json.dumps(result.to_json(), ensure_ascii=False)
    assert "evil.example" not in blob  # rejected content never leaks into results
    assert any(d["code"] == "item-dropped" for d in result.diagnostics)


@requires_ffmpeg
def test_cancel_after_selection_discards_late_plan(luma_source, lm, tools, tmp_path, monkeypatch):
    import vibecut.analysis.service as service
    from vibecut.media.proxy import ensure_proxy
    cache = Cache(tmp_path / "cache")
    asset = ensure_proxy(luma_source, tools, cache)
    fill_observe_script(lm, 3)
    lm.chat_script.append({"content": {"selections": []}})
    cancel = threading.Event()
    select = service._select
    def late_selection(*args, **kwargs):
        result = select(*args, **kwargs)
        cancel.set()
        return result
    monkeypatch.setattr(service, "_select", late_selection)
    result = analyze(luma_source, asset, tools, cache, _config(), parse_focus("人物"), cancel_event=cancel,
                     transport=LoopbackTransport(lm.base_url, request_timeout_s=10))
    assert result.status == "cancelled"
    assert result.plan is None and result.plan_status is None
    assert result.observations  # confirmed evidence is retained



def test_cancel_during_proxy_preparation_is_not_reported_as_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import vibecut.media.proxy as proxy
    from vibecut.render.static import ExportError
    cancel = threading.Event()
    cancel.set()
    def stopped(*args, **kwargs):
        raise ExportError("proxy cancelled")
    monkeypatch.setattr(proxy, "ensure_proxy", stopped)
    result = analyze(SimpleNamespace(source_id="source"), None, None, Cache(tmp_path), _config(),
                     parse_focus("人物"), cancel_event=cancel)
    assert result.status == "cancelled" and result.plan is None
    assert result.diagnostics[0]["code"] == "cancelled"
