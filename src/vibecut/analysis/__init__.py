"""Local panorama analysis (task #5): Chinese focus -> evidenced draft plan.

Public surface (stdlib-only core, heavy media imports stay lazy):
  * :func:`analyze` - bounded orchestration with metering/caching/cancel;
  * :class:`AnalysisConfig`, :class:`FocusSpec`, :func:`parse_focus`;
  * view/sampling geometry with back-computable evidence (``views``);
  * data objects + :func:`to_edit_plan` mapping to the frozen core contract.
"""

from vibecut.analysis.errors import AnalysisError
from vibecut.analysis.focus import FocusError, FocusSpec, parse_focus
from vibecut.analysis.plan import (
    ANALYSIS_SCHEMA_VERSION,
    AnalysisResult,
    CandidateWindow,
    EvidenceRef,
    MeteringReport,
    ModelSelection,
    Observation,
    SegmentDraft,
    SelectionPlan,
    to_edit_plan,
)
from vibecut.analysis.sampling import SamplingPlan, plan_sampling
from vibecut.analysis.select import SelectionParams, build_selection_plan
from vibecut.analysis.service import DEFAULT_ENDPOINT, DEFAULT_MODEL, AnalysisConfig, analyze
from vibecut.analysis.transport import LoopbackTransport, validate_endpoint
from vibecut.analysis.views import ViewSpec, default_view_set, find_covering_view

__all__ = [
    "AnalysisConfig", "AnalysisError", "AnalysisResult", "ANALYSIS_SCHEMA_VERSION",
    "CandidateWindow", "DEFAULT_ENDPOINT", "DEFAULT_MODEL", "EvidenceRef", "FocusError",
    "FocusSpec", "LoopbackTransport", "MeteringReport", "ModelSelection", "Observation",
    "SamplingPlan", "SelectionParams", "SelectionPlan", "SegmentDraft", "ViewSpec",
    "analyze", "build_selection_plan", "default_view_set", "find_covering_view",
    "parse_focus", "plan_sampling", "to_edit_plan", "validate_endpoint",
]
