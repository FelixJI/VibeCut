"""Analysis orchestration: cheap features -> candidates -> observe -> select.

Budget/cancel/caching semantics (task #5):
  * every stage is timed and metered; request count, retry count, cache hits
    and durations end up in the public :class:`MeteringReport`;
  * global budgets (max model requests, total model time) are checked
    before every request and during active HTTP - exceeding them yields a partial, reviewable result,
    never an unbounded run;
  * cancellation is honoured between requests/stages and inside ffmpeg/HTTP;
  * two-level caching: prompt-independent visual observations (cheap
    features + per-batch image observations) vs prompt/duration-dependent
    selections - a changed focus reuses observations but never old selections;
  * when the model is unavailable the run returns ``ai-unavailable`` with
    candidates for the manual path - never a rule-based fallback disguised
    as AI success.

The analysis never writes the project file: no revision can be created or
overwritten by a late response, and manual locks stay untouched.
"""

from __future__ import annotations

import math
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Optional

from vibecut.analysis import cachekeys
from vibecut.analysis.candidates import discover_candidates
from vibecut.analysis.errors import AIUnavailableError, AnalysisError, BudgetExhaustedError, ExtractionError, ModelCancelledError, SourceUnavailableError
from vibecut.analysis.features import compute_features, features_from_json, features_to_json
from vibecut.analysis.focus import FocusSpec
from vibecut.analysis.model import ModelClient
from vibecut.analysis.plan import ANALYSIS_SCHEMA_VERSION, AnalysisResult, CandidateWindow, MeteringReport, Observation, SelectionPlan
from vibecut.analysis.sampling import plan_sampling
from vibecut.analysis.select import SelectionParams, build_selection_plan
from vibecut.analysis.transport import LoopbackTransport, validate_endpoint
from vibecut.core.cache import Cache
from vibecut.core.contracts import MediaSource, NormalizedAsset

DEFAULT_ENDPOINT = "http://127.0.0.1:1234"
DEFAULT_MODEL = "qwen3.8-27b@q4_k_xl"  # user-selected local model; probe-verified before use


@dataclass(frozen=True)
class AnalysisConfig:
    endpoint: str = DEFAULT_ENDPOINT
    model_id: Optional[str] = DEFAULT_MODEL
    samples: int = 12
    image_width: int = 768
    image_height: int = 432
    max_model_requests: int = 32
    max_images_per_request: int = 8
    max_request_input_bytes: int = 6_000_000
    model_timeout_s: float = 300.0
    total_model_budget_s: float = 900.0
    max_repair_retries: int = 1
    max_candidates: int = 24
    ffmpeg_timeout_s: float = 300.0
    temperature: float = 0.2
    allow_jit_load: bool = False  # explicit opt-in: first chat request may load the model

    def __post_init__(self):
        for name, low, high in (("samples", 1, 64), ("max_model_requests", 0, 1024),
                                ("max_images_per_request", 1, 8), ("max_request_input_bytes", 1, 64_000_000),
                                ("max_repair_retries", 0, 1), ("max_candidates", 14, 256),
                                ("image_width", 16, 4096), ("image_height", 16, 4096)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        for name in ("model_timeout_s", "total_model_budget_s", "ffmpeg_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be finite and in 0..2")

    def budget_json(self) -> dict:
        return {
            "endpoint": self.endpoint, "model_id": self.model_id, "samples": self.samples,
            "max_model_requests": self.max_model_requests, "max_images_per_request": self.max_images_per_request,
            "max_request_input_bytes": self.max_request_input_bytes,
            "model_timeout_s": self.model_timeout_s, "total_model_budget_s": self.total_model_budget_s,
            "max_repair_retries": self.max_repair_retries, "max_candidates": self.max_candidates,
            "temperature": self.temperature, "allow_jit_load": self.allow_jit_load,
        }


def _source_duration_s(source: MediaSource) -> Fraction:
    if source.video.nb_frames:
        return Fraction(source.video.nb_frames) / source.video.fps
    raise SourceUnavailableError("source has no verified frame count; refusing to guess duration")


def _cheap_stream_fps(duration_s: Fraction, *, max_frames: int = 1200) -> Fraction:
    fps = Fraction(2)
    frames = fps * duration_s
    if frames > max_frames:
        fps = Fraction(max_frames) / duration_s
    return fps.limit_denominator(1000) if fps.denominator != 1 else fps


class _StageTimer:
    def __init__(self, metering: MeteringReport, name: str) -> None:
        self.metering = metering
        self.name = name

    def __enter__(self) -> "_StageTimer":
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        self.metering.stage_timings[self.name] = self.metering.stage_timings.get(self.name, 0.0) + (time.monotonic() - self.start)


def analyze(
    source: MediaSource,
    asset: Optional[NormalizedAsset],
    tools,
    cache: Cache,
    config: AnalysisConfig,
    focus: FocusSpec,
    *,
    cancel_event: Optional[threading.Event] = None,
    transport=None,
    selection_params: SelectionParams | None = None,
) -> AnalysisResult:
    """Run the full bounded analysis; returns a result object, never raises."""
    cancel_event = cancel_event or threading.Event()
    metering = MeteringReport(budget=config.budget_json())
    diagnostics: list[dict] = []
    observations: list[Observation] = []
    plan: Optional[SelectionPlan] = None
    plan_status: Optional[str] = None
    status = "failed"
    model_info: Optional[dict] = None
    candidates: tuple[CandidateWindow, ...] = ()
    asset_id = ""
    partial_reason: Optional[str] = None

    try:
        validate_endpoint(config.endpoint)

        with _StageTimer(metering, "asset"):
            if asset is None:
                from vibecut.media.proxy import ensure_proxy
                from vibecut.render.static import ExportError

                try:
                    asset = ensure_proxy(source, tools, cache, cancel_event=cancel_event)
                except ModelCancelledError:
                    raise
                except (ExportError, OSError) as exc:
                    if cancel_event.is_set():
                        raise ModelCancelledError("cancelled while preparing proxy") from exc
                    raise ExtractionError(f"proxy unavailable: {exc}") from exc
            if cancel_event.is_set():
                raise ModelCancelledError("cancelled while preparing asset")
            from vibecut.media.probe import content_fingerprint
            if content_fingerprint(Path(source.path)) != source.content_fingerprint:
                raise SourceUnavailableError("source content changed since ingest", code="source-changed")
            if asset.source_id != source.source_id or asset.kind not in ("proxy", "canonical"):
                raise ExtractionError("asset is not a canonical/proxy asset for this source")
            if asset.time_map.fps != source.video.fps:
                raise ExtractionError("asset/source frame rates disagree")
            if asset.time_map.asset_start_source_rel != 0:
                raise ExtractionError("only assets starting at source-relative 0 are analysable in this task")
            asset_id = asset.asset_id
            duration = _source_duration_s(source)

        from vibecut.analysis.views import default_view_set
        batches_per_sample = (len(default_view_set()) + config.max_images_per_request - 1) // config.max_images_per_request
        # Redistribute affordable samples across the whole source, reserving selection + one repair.
        effective_samples = min(config.samples, max(1, (config.max_model_requests - 2) // batches_per_sample))
        metering.budget["effective_samples"] = effective_samples
        sampling = plan_sampling(duration, samples=effective_samples, views=tuple(
            replace(v, width=config.image_width, height=config.image_height) for v in default_view_set()
        ))

        with _StageTimer(metering, "cheap-features"):
            features = _cheap_features(source, asset, tools, cache, sampling, config, cancel_event, metering)

        candidates = discover_candidates(
            features, sampling, source_id=source.source_id, max_candidates=config.max_candidates,
        )

        transport = transport or LoopbackTransport(config.endpoint, request_timeout_s=config.model_timeout_s)
        client = ModelClient(
            transport, config.model_id, metering=metering,
            request_timeout_s=config.model_timeout_s, temperature=config.temperature,
            max_model_requests=config.max_model_requests, total_model_budget_s=config.total_model_budget_s,
            max_request_input_bytes=config.max_request_input_bytes,
            max_images_per_request=config.max_images_per_request, max_repair_retries=config.max_repair_retries,
        )

        try:
            with _StageTimer(metering, "model-probe"):
                model_info = client.probe(cancel_event=cancel_event, allow_jit_load=config.allow_jit_load)
        except AIUnavailableError as exc:
            diagnostics.append(exc.diagnostic)
            return AnalysisResult(
                status="ai-unavailable", plan_status=None, source_id=source.source_id, asset_id=asset_id,
                focus=focus.to_json(), plan=None, candidates=candidates, observations=(),
                model_info={"requested_id": config.model_id}, diagnostics=tuple(diagnostics), metering=metering,
            )

        with _StageTimer(metering, "observe"):
            observations, partial_reason = _observe(
                source, asset, tools, cache, client, config, sampling, metering, cancel_event, observations,
            )

        with _StageTimer(metering, "select"):
            plan, select_diagnostics = _select(
                source, asset, cache, client, config, focus, candidates, observations,
                sampling, metering, cancel_event, selection_params,
            )
        if cancel_event.is_set():
            raise ModelCancelledError("cancelled before result validation")
        if content_fingerprint(Path(source.path)) != source.content_fingerprint:
            plan = None
            raise SourceUnavailableError("source content changed during analysis", code="source-changed")
        diagnostics.extend(select_diagnostics)
        plan_status = plan.status if plan is not None else None
        status = "partial" if partial_reason or any(d["code"] == "budget-exceeded" for d in select_diagnostics) else "succeeded"
        if partial_reason:
            diagnostics.append({"stage": "budget", "code": "budget-exceeded", "message": partial_reason})
    except BudgetExhaustedError as exc:
        status = "partial"
        diagnostics.append(exc.diagnostic)
    except ModelCancelledError as exc:
        plan = None
        plan_status = None
        status = "cancelled"
        diagnostics.append(exc.diagnostic)
        for msg in getattr(exc, "dropped_items", ()):
            diagnostics.append({"stage": "validation", "code": "item-dropped", "message": str(msg)[:200]})
    except AnalysisError as exc:
        plan = None
        plan_status = None
        status = "failed"
        diagnostics.append(exc.diagnostic)
        for msg in getattr(exc, "dropped_items", ()):
            # rejected-but-observable model items from a failed stage
            diagnostics.append({"stage": "validation", "code": "item-dropped", "message": str(msg)[:200]})
    except OSError as exc:
        status = "failed"
        diagnostics.append({"stage": "io", "code": "os-error", "message": str(exc)[:200]})

    return AnalysisResult(
        status=status, plan_status=plan_status, source_id=source.source_id, asset_id=asset_id,
        focus=focus.to_json(), plan=plan, candidates=candidates,
        observations=tuple(observations), model_info=model_info,
        diagnostics=tuple(diagnostics), metering=metering,
    )


# -- stage helpers -----------------------------------------------------------


def _tool_identity(tools) -> str:
    return tools.identity()


def _cheap_features(source, asset, tools, cache, sampling, config, cancel_event, metering):
    key = cachekeys.cheap_features_key(
        source_fingerprint=source.content_fingerprint,
        tool_identity=_tool_identity(tools),
        sampling_identity=sampling.identity(),
    )
    hit = cache.get(key)
    if hit is not None:
        metering.cache["cheap_features"] = "hit"
        return features_from_json(hit[0]["features"])
    metering.cache["cheap_features"] = "miss"
    from vibecut.analysis.extract import extract_gray_stream

    fps = _cheap_stream_fps(sampling.duration_s)
    raw, w, h = extract_gray_stream(
        tools, asset.path, fps=fps, cancel_event=cancel_event, timeout_s=config.ffmpeg_timeout_s,
    )
    features = compute_features(raw, w, h, sampling, fps)
    cache.put_atomic(key, meta={"kind": "analysis-cheap-features", "schema_version": ANALYSIS_SCHEMA_VERSION,
                                "features": features_to_json(features)})
    return features


def _observe(source, asset, tools, cache, client, config, sampling, metering, cancel_event, observations):
    """Per-sample batches of perspective JPEGs; cached per batch (prompt-free)."""
    from vibecut.analysis.extract import extract_perspective_jpeg

    hits = misses = 0
    partial_reason: Optional[str] = None
    out_dir = Path(tempfile.mkdtemp(prefix="vibecut-analysis-views-"))
    try:
        for sample_idx, t in enumerate(sampling.times):
            if cancel_event.is_set():
                raise ModelCancelledError("cancelled between observation batches")
            if metering.request_count >= config.max_model_requests:
                partial_reason = f"observation stopped: request budget {config.max_model_requests} reached"
                break
            if metering.model_time_s >= config.total_model_budget_s:
                partial_reason = f"observation stopped: model time budget {config.total_model_budget_s}s reached"
                break
            views = sampling.views
            for chunk_start in range(0, len(views), config.max_images_per_request):
                chunk = views[chunk_start:chunk_start + config.max_images_per_request]
                img_metas = [(f"s{sample_idx:02d}.{v.view_id}", t, v) for v in chunk]
                batch = [{"img_id": iid, "time_rel": f"{t.numerator}/{t.denominator}", "view_id": v.view_id}
                         for iid, _tt, v in img_metas]
                key = cachekeys.observation_key(
                    source_fingerprint=source.content_fingerprint, asset_id=asset.asset_id,
                    tool_identity=_tool_identity(tools), sampling_identity=sampling.identity(),
                    model_id=client.model_id or "", batch=batch,
                    inference_parameters={"temperature": config.temperature, "max_tokens": client.max_tokens},
                )
                hit = cache.get(key)
                if hit is not None:
                    hits += 1
                    observations.extend(_observations_from_meta(hit[0]["observations"]))
                    continue
                misses += 1
                if metering.request_count >= config.max_model_requests:
                    partial_reason = f"observation stopped: request budget {config.max_model_requests} reached"
                    break
                images: list[tuple[str, bytes]] = []
                for iid, tt, v in img_metas:
                    jpeg = extract_perspective_jpeg(
                        tools, asset.path, tt, v, out_dir,
                        cancel_event=cancel_event, timeout_s=config.ffmpeg_timeout_s,
                    )
                    images.append((iid, jpeg.read_bytes()))
                meta = {iid: (t, v.view_id) for iid, t, v in img_metas}
                obs, _repair = client.observe(images, meta, cancel_event=cancel_event)
                observations.extend(obs)
                cache.put_atomic(key, meta={
                    "kind": "analysis-observation-batch", "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "observations": [o.to_json() for o in obs],
                })
            if partial_reason:
                break
    finally:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
        # counters survive cancellation/failure: confirmed batches stay
        # observable in the metering report (AC5/AC7)
        metering.cache["observation_batches_hit"] = hits
        metering.cache["observation_batches_miss"] = misses
    return observations, partial_reason


def _observations_from_meta(items: list[dict]) -> list[Observation]:
    return [
        Observation(
            img_id=str(i["img_id"]), time_rel=Fraction(str(i["time_rel"])), view_id=str(i["view_id"]),
            subjects=tuple(i["subjects"]), scene=str(i["scene"]),
            quality=str(i["quality"]), motion_hint=str(i["motion_hint"]),
        )
        for i in items
    ]


def _select(source, asset, cache, client, config, focus, candidates, observations, sampling, metering, cancel_event, selection_params):
    diagnostics: list[dict] = []
    obs_by_view_time: dict[tuple[str, Fraction], Observation] = {}
    for o in observations:
        obs_by_view_time[(o.view_id, o.time_rel)] = o

    def summarize(c: CandidateWindow) -> dict:
        near = sorted(
            (o for (vid, _t), o in obs_by_view_time.items()
             if vid == c.view_id and c.start_rel <= o.time_rel < c.end_rel),
            key=lambda o: min(abs(o.time_rel - c.start_rel), abs(o.time_rel - c.end_rel)),
        )
        top = near[:2]
        return {
            "candidate_id": c.candidate_id, "view_id": c.view_id,
            "time_start_s": round(float(c.start_rel), 3), "time_end_s": round(float(c.end_rel), 3),
            "origin": c.origin, "score": c.score,
            "observed": [{"img_id": o.img_id, "time_rel": f"{o.time_rel.numerator}/{o.time_rel.denominator}",
                          "subjects": list(o.subjects), "scene": o.scene, "quality": o.quality,
                          "motion_hint": o.motion_hint, "sample_time_s": round(float(o.time_rel), 3)} for o in top],
        }

    candidates_payload = [summarize(c) for c in candidates]
    candidates_payload = [c for c in candidates_payload if c["observed"]]
    grounded_ids = {c["candidate_id"] for c in candidates_payload}
    key = cachekeys.selection_key(
        source_fingerprint=source.content_fingerprint, asset_id=asset.asset_id,
        model_id=client.model_id or "", focus_json=focus.to_json(),
        candidates=[c.to_json() for c in candidates],
        observations=[o.to_json() for o in observations],
        inference_parameters={"temperature": config.temperature, "max_tokens": client.max_tokens},
    )
    hit = cache.get(key)
    if hit is not None:
        metering.cache["selection"] = "hit"
        selections, dropped = _selections_from_meta(hit[0]["selections"]), []
    else:
        metering.cache["selection"] = "miss"
        if cancel_event.is_set():
            raise ModelCancelledError("cancelled before selection request")
        if metering.request_count >= config.max_model_requests:
            # budget checked between requests here too: selection is skipped
            # (never silently fired over budget); the partial result keeps
            # its reviewable candidates and the budget diagnostic.
            diagnostics.append({"stage": "budget", "code": "budget-exceeded",
                                "message": f"selection skipped: request budget {config.max_model_requests} reached"})
            return None, diagnostics
        selections, repair, dropped = client.select(
            focus.prompt_identity(), candidates_payload,
            {c.candidate_id: (c.start_rel, c.end_rel) for c in candidates if c.candidate_id in grounded_ids},
            cancel_event=cancel_event,
        )
        for msg in repair:
            diagnostics.append({"stage": "model-request", "code": "repair-retry", "message": msg})
        for msg in dropped:
            diagnostics.append({"stage": "validation", "code": "item-dropped", "message": msg[:200]})
        cache.put_atomic(key, meta={
            "kind": "analysis-selection", "schema_version": ANALYSIS_SCHEMA_VERSION,
            "selections": [s.to_json() for s in selections],
        })

    views_by_id = {v.view_id: v for v in sampling.views}

    def direction_of(cand: CandidateWindow, sel):
        view = views_by_id.get(cand.view_id)
        if view is None:
            return None
        if sel.bbox is None:
            return (view.yaw_deg, view.pitch_deg)
        yaw, pitch = view.bbox_center_direction(sel.bbox)
        return (round(yaw, 3), round(pitch, 3))

    duration = sampling.duration_s
    plan = build_selection_plan(candidates, tuple(selections), focus, duration, selection_params,
                                direction_of=direction_of)
    return plan, diagnostics


def _selections_from_meta(items: list[dict]) -> list:
    from vibecut.analysis.plan import ModelSelection

    return [
        ModelSelection(
            candidate_id=i["candidate_id"], selected=bool(i["selected"]), relevance=float(i["relevance"]),
            confidence=float(i["confidence"]), reason=i["reason"], uncertainty=i.get("uncertainty"),
            bbox=tuple(i["bbox"]) if i.get("bbox") else None,
            event_time_s=float(i["event_time_s"]) if i.get("event_time_s") is not None else None,
        )
        for i in items
    ]
