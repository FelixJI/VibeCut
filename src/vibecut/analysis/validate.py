"""Application-layer business validation of model output (task #5).

Schema legality (enforced by the model service) is never trusted as semantic
correctness. Everything is re-validated here:
  * observation ``img_id`` / selection ``candidate_id`` must be among the IDs
    *we* provided (the model can only pick, never invent);
  * numbers must be finite and in range; ``NaN``/``Infinity`` literals are
    rejected at the JSON parse layer;
  * bbox is normalized in-view ``[0,1]^2`` with positive extent inside the
    frame;
  * optional ``event_time_s`` must fall inside the referenced candidate's
    window (no out-of-range times);
  * free text (reason/uncertainty/subjects/scene) must not look like URLs,
    absolute paths or shell command fragments - such content is rejected,
    never executed, and never reaches a plan.

One bounded repair retry is driven by :mod:`vibecut.analysis.model`; this
module only decides valid/invalid and reports bounded issue lists.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

from vibecut.analysis.errors import StructuredOutputError, ValidationFailureError

MAX_REASON_LEN = 200
MAX_SUBJECTS = 8
MAX_SUBJECT_LEN = 40
MAX_SCENE_LEN = 60
QUALITY_VALUES = ("good", "acceptable", "poor")
MOTION_VALUES = ("static", "slight", "strong")
_SHELL_CHARS = "`|;&<>$"
_URL_MARKERS = ("http://", "https://", "ftp://", "www.")
_WIN_PATH_RE = __import__("re").compile(r"[A-Za-z]:[\\\\/]")


def parse_json_strict(text: str) -> Any:
    """json.loads that rejects NaN/Infinity/-Infinity constants."""

    def _reject(value: str) -> float:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        return json.loads(text, parse_constant=_reject)
    except (json.JSONDecodeError, ValueError) as exc:
        raise StructuredOutputError(f"model content is not strict JSON: {exc}") from exc


def sanitize_text(value: Any, *, max_len: int = MAX_REASON_LEN) -> str | None:
    """Return cleaned text, or None if it must be rejected (executable-like)."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or len(s) > max_len:
        return None
    low = s.lower()
    for marker in _URL_MARKERS:
        if marker in low:
            return None
    if s.startswith("/") or s.startswith("\\"):
        return None
    if _WIN_PATH_RE.search(s):
        return None
    if any(ch in s for ch in _SHELL_CHARS):
        return None
    if any(ord(ch) < 0x20 for ch in s):
        return None
    return s


def _finite_number(value: Any, *, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        return None
    return v


def _validate_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, dict):
        return None
    try:
        x, y = _finite_number(raw["x"], low=0.0, high=1.0), _finite_number(raw["y"], low=0.0, high=1.0)
        w, h = _finite_number(raw["w"], low=1e-6, high=1.0), _finite_number(raw["h"], low=1e-6, high=1.0)
    except KeyError:
        return None
    if x is None or y is None or w is None or h is None:
        return None
    if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        return None
    return (min(x, 1.0), min(y, 1.0), min(w, 1.0 - x + 1e-9), min(h, 1.0 - y + 1e-9))


def validate_observations(raw: Any, provided_img_ids: frozenset[str]) -> list[dict]:
    """Validate an observe-stage payload; raises with bounded issues."""
    issues: list[str] = []
    if not isinstance(raw, dict) or not isinstance(raw.get("observations"), list):
        raise ValidationFailureError("observe payload must be {'observations': [...]}")
    out: list[dict] = []
    for i, item in enumerate(raw["observations"][:64]):
        if not isinstance(item, dict):
            issues.append(f"item {i}: not an object")
            continue
        img_id = item.get("img_id")
        if not isinstance(img_id, str) or img_id not in provided_img_ids:
            issues.append(f"item {i}: img_id not among provided IDs")
            continue
        subjects_raw = item.get("subjects")
        if not isinstance(subjects_raw, list) or len(subjects_raw) > MAX_SUBJECTS:
            issues.append(f"item {i}: subjects must be a list of <= {MAX_SUBJECTS}")
            continue
        subjects = [sanitize_text(s, max_len=MAX_SUBJECT_LEN) for s in subjects_raw]
        if any(s is None for s in subjects):
            issues.append(f"item {i}: invalid subject text")
            continue
        scene = sanitize_text(item.get("scene"), max_len=MAX_SCENE_LEN)
        if scene is None:
            issues.append(f"item {i}: invalid scene text")
            continue
        if item.get("quality") not in QUALITY_VALUES:
            issues.append(f"item {i}: quality must be one of {QUALITY_VALUES}")
            continue
        if item.get("motion_hint") not in MOTION_VALUES:
            issues.append(f"item {i}: motion_hint must be one of {MOTION_VALUES}")
            continue
        out.append({"img_id": img_id, "subjects": subjects, "scene": scene,
                    "quality": item["quality"], "motion_hint": item["motion_hint"]})
    if issues:
        raise ValidationFailureError("; ".join(issues[:5]) + (f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""))
    returned_ids = [item["img_id"] for item in out]
    if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != provided_img_ids:
        raise ValidationFailureError("observations must describe every provided img_id exactly once")
    return out


def validate_selections(
    raw: Any,
    candidates: dict[str, tuple],  # candidate_id -> (start_rel, end_rel) Fractions
) -> tuple[list[dict], list[str]]:
    """Validate a select-stage payload.

    Returns ``(valid_items, dropped_reasons)``. Invalid single items are
    dropped (recorded, never entering a plan); if *nothing* is valid the
    caller treats the stage as failed after its bounded repair retry.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("selections"), list):
        raise ValidationFailureError("select payload must be {'selections': [...]}")
    valid: list[dict] = []
    dropped: list[str] = []
    for i, item in enumerate(raw["selections"][:128]):
        if not isinstance(item, dict):
            dropped.append(f"item {i}: not an object")
            continue
        cid = item.get("candidate_id")
        if not isinstance(cid, str) or cid not in candidates:
            dropped.append(f"item {i}: candidate_id not among provided candidates")
            continue
        start_rel, end_rel = candidates[cid]
        if not isinstance(item.get("selected"), bool):
            dropped.append(f"{cid}: selected must be boolean")
            continue
        relevance = _finite_number(item.get("relevance"), low=0.0, high=1.0)
        confidence = _finite_number(item.get("confidence"), low=0.0, high=1.0)
        if relevance is None or confidence is None:
            dropped.append(f"{cid}: relevance/confidence must be finite in [0,1]")
            continue
        reason = sanitize_text(item.get("reason"))
        if reason is None:
            dropped.append(f"{cid}: reason missing/invalid (URL/path/command-like text rejected)")
            continue
        uncertainty = item.get("uncertainty")
        if uncertainty is not None:
            uncertainty = sanitize_text(uncertainty)
            if uncertainty is None:
                dropped.append(f"{cid}: invalid uncertainty text")
                continue
        bbox = item.get("bbox")
        if bbox is not None:
            bbox = _validate_bbox(bbox)
            if bbox is None:
                dropped.append(f"{cid}: bbox invalid (must be normalized [0,1] inside frame)")
                continue
        event_time = item.get("event_time_s")
        if event_time is not None:
            event_time = _finite_number(event_time, low=float(start_rel), high=float(end_rel))
            if event_time is None or event_time >= float(end_rel):
                dropped.append(f"{cid}: event_time_s outside candidate window [{float(start_rel):.3f},{float(end_rel):.3f}]")
                continue
        valid.append({
            "candidate_id": cid, "selected": item["selected"], "relevance": relevance,
            "confidence": confidence, "reason": reason, "uncertainty": uncertainty,
            "bbox": bbox, "event_time_s": event_time,
        })
    return valid, dropped


def provided_ids(items: Iterable[str]) -> frozenset[str]:
    return frozenset(items)
