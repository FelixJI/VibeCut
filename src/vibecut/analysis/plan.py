"""Data objects of the analysis pipeline (task #5, analysis-local schema v1).

These are *analysis-local* contracts; shared repo contracts (EditPlan etc.)
stay frozen in :mod:`vibecut.core.contracts`. ``to_edit_plan`` maps one
draft segment onto the frozen core contract with automation semantics:
``cut_manual=False`` and ``CameraPath.provenance="derived"`` (manual locks
always win; static export only accepts user-locked views, so AI drafts must
be explicitly confirmed by a human before export - by design).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

from vibecut.core.contracts import CONTRACT_VERSION, CameraPath, CameraView, EditPlan, frac_str

ANALYSIS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Observation:
    """Prompt-independent visual observation of one (sample time, view)."""

    img_id: str
    time_rel: Fraction
    view_id: str
    subjects: tuple[str, ...]
    scene: str
    quality: str
    motion_hint: str

    def to_json(self) -> dict:
        return {
            "img_id": self.img_id, "time_rel": frac_str(self.time_rel), "view_id": self.view_id,
            "subjects": list(self.subjects), "scene": self.scene,
            "quality": self.quality, "motion_hint": self.motion_hint,
        }


@dataclass(frozen=True)
class CandidateWindow:
    """Cheap-layer candidate window referencing one view (evidence-backed)."""

    candidate_id: str
    source_id: str
    start_rel: Fraction
    end_rel: Fraction
    view_id: str
    origin: str  # "motion" | "coverage"
    score: float  # bounded [0, 1]

    def to_json(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "source_id": self.source_id,
            "start_rel": frac_str(self.start_rel), "end_rel": frac_str(self.end_rel),
            "view_id": self.view_id, "origin": self.origin, "score": self.score,
        }


@dataclass(frozen=True)
class ModelSelection:
    """One validated model decision about a provided candidate."""

    candidate_id: str
    selected: bool
    relevance: float
    confidence: float
    reason: str
    uncertainty: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    event_time_s: Optional[float] = None

    @property
    def weight(self) -> float:
        return self.relevance * self.confidence

    def to_json(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "selected": self.selected,
            "relevance": self.relevance, "confidence": self.confidence,
            "reason": self.reason, "uncertainty": self.uncertainty,
            "bbox": list(self.bbox) if self.bbox else None, "event_time_s": self.event_time_s,
        }


@dataclass(frozen=True)
class EvidenceRef:
    """Source/time/view evidence of one segment (one-to-one traceable)."""

    candidate_id: str
    time_rel: Fraction
    view_id: str
    bbox: Optional[tuple[float, float, float, float]] = None
    direction: Optional[tuple[float, float]] = None  # canonical yaw/pitch deg

    def to_json(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "time_rel": frac_str(self.time_rel),
            "view_id": self.view_id,
            "bbox": list(self.bbox) if self.bbox else None,
            "direction_yaw_pitch_deg": list(self.direction) if self.direction else None,
        }


@dataclass(frozen=True)
class SegmentDraft:
    """AI draft segment on the source-relative timeline (never manual-locked)."""

    segment_id: str
    source_id: str
    cut_in_rel: Fraction
    cut_out_rel: Fraction
    view_id: str
    reason: str
    evidence: tuple[EvidenceRef, ...]
    relevance: float
    confidence: float
    provenance: str = "ai-draft"

    @property
    def duration_rel(self) -> Fraction:
        return self.cut_out_rel - self.cut_in_rel

    def to_json(self) -> dict:
        return {
            "segment_id": self.segment_id, "source_id": self.source_id,
            "cut_in_rel": frac_str(self.cut_in_rel), "cut_out_rel": frac_str(self.cut_out_rel),
            "duration_rel": frac_str(self.duration_rel), "view_id": self.view_id,
            "reason": self.reason, "evidence": [e.to_json() for e in self.evidence],
            "relevance": self.relevance, "confidence": self.confidence, "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SelectionPlan:
    """Bounded merge/dedupe/budget result over validated model selections."""

    status: str  # "complete" | "short" | "empty"
    segments: tuple[SegmentDraft, ...]
    target_duration_s: Fraction
    total_duration_s: Fraction
    shortfall_reason: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "status": self.status, "segments": [s.to_json() for s in self.segments],
            "target_duration_s": frac_str(self.target_duration_s),
            "total_duration_s": frac_str(self.total_duration_s),
            "shortfall_reason": self.shortfall_reason,
        }


@dataclass
class MeteringReport:
    """Public record of budgets, request counts, retries and timings (AC7)."""

    budget: dict[str, Any] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)

    @property
    def request_count(self) -> int:
        return len([r for r in self.requests if r.get("kind") != "probe"])

    @property
    def retry_count(self) -> int:
        return len([r for r in self.requests if r.get("kind") == "retry"])

    @property
    def model_time_s(self) -> float:
        return sum(r.get("duration_s", 0.0) for r in self.requests)

    def to_json(self) -> dict:
        return {
            "budget": self.budget,
            "requests": self.requests,
            "request_count": self.request_count,
            "probe_count": sum(r.get("kind") == "probe" for r in self.requests),
            "total_request_count": len(self.requests),
            "retry_count": self.retry_count,
            "model_time_s": round(self.model_time_s, 3),
            "cache": self.cache,
            "stage_timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Terminal outcome of one analysis run (never raises past this point)."""

    status: str  # "succeeded" | "partial" | "ai-unavailable" | "cancelled" | "failed"
    plan_status: Optional[str]
    source_id: str
    asset_id: str
    focus: dict
    plan: Optional[SelectionPlan]
    candidates: tuple[CandidateWindow, ...]  # manual-reviewable windows
    observations: tuple[Observation, ...]
    model_info: Optional[dict]
    diagnostics: tuple[dict, ...]
    metering: MeteringReport

    def to_json(self) -> dict:
        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "status": self.status,
            "plan_status": self.plan_status,
            "source_id": self.source_id,
            "asset_id": self.asset_id,
            "focus": self.focus,
            "plan": self.plan.to_json() if self.plan else None,
            "candidates": [c.to_json() for c in self.candidates],
            "observations": [o.to_json() for o in self.observations],
            "model": self.model_info,
            "diagnostics": [dict(d) for d in self.diagnostics],
            "metering": self.metering.to_json(),
        }


def to_edit_plan(segment: SegmentDraft, view, project_fps: Fraction, *, audio: bool = True) -> EditPlan:
    """Map one draft segment to the frozen core EditPlan (automation semantics).

    The camera path is ``provenance="derived"`` and cuts are not manual locks;
    static export requires user confirmation, which is exactly the intended
    human-in-the-loop boundary.
    """
    camera = CameraPath(
        schema_version=CONTRACT_VERSION,
        keys=((segment.cut_in_rel, CameraView(
            yaw_deg=view.yaw_deg, pitch_deg=view.pitch_deg,
            roll_deg=view.roll_deg, h_fov_deg=view.h_fov_deg,
        )),),
        provenance="derived",
    )
    return EditPlan(
        schema_version=CONTRACT_VERSION,
        source_id=segment.source_id,
        cut_in_rel=segment.cut_in_rel,
        cut_out_rel=segment.cut_out_rel,
        project_fps=project_fps,
        camera=camera,
        audio=audio,
        cut_manual=False,
    )
