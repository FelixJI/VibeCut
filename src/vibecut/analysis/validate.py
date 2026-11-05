"""Application-layer business validation of model output (task #5).

Schema legality (enforced by the model service) is never trusted as semantic
correctness. Everything is re-validated here:
  * observation ``img_id`` / selection ``candidate_id`` must be among the IDs
    *we* provided (the model can only pick, never invent);
  * numbers must be finite and in range; ``NaN``/``Infinity`` literals are
    rejected at the JSON parse layer;
  * selection img_ids must belong to the exact provided candidate; location
    and arbitrary event-time fields are rejected;
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
    try:
        v = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(v) or not low <= v <= high:
        return None
    return v


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


def validate_selections(raw: Any, candidates: dict[str, tuple], evidence: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Selection may cite only provided images; it cannot invent time or location."""
    if not isinstance(raw, dict) or set(raw) != {"selections"} or not isinstance(raw["selections"], list):
        raise ValidationFailureError("select payload must be {'selections': [...]}")
    if len(raw["selections"]) > 128:
        raise ValidationFailureError("too many selections")
    valid, dropped, seen = [], [], set()
    fields = {"candidate_id", "img_ids", "relevance", "confidence", "reason"}
    for i, item in enumerate(raw["selections"]):
        if not isinstance(item, dict) or set(item) != fields:
            dropped.append(f"item {i}: selection fields invalid; bbox/time/location are forbidden")
            continue
        cid = item["candidate_id"]
        if not isinstance(cid, str) or cid not in candidates or cid in seen:
            dropped.append(f"item {i}: unknown or duplicate candidate_id")
            continue
        ids = item["img_ids"]
        allowed = evidence.get(cid, {})
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 4
                or any(not isinstance(iid, str) or iid not in allowed for iid in ids)
                or len(set(ids)) != len(ids)):
            dropped.append(f"{cid}: img_ids must cite this candidate's provided images exactly once")
            continue
        start, end = candidates[cid]
        if any(not start <= allowed[iid] < end for iid in ids):
            dropped.append(f"{cid}: image evidence is outside candidate window")
            continue
        relevance = _finite_number(item["relevance"], low=0, high=1)
        confidence = _finite_number(item["confidence"], low=0, high=1)
        reason = sanitize_text(item["reason"])
        if relevance is None or confidence is None or reason is None:
            dropped.append(f"{cid}: invalid relevance/confidence/reason")
            continue
        seen.add(cid)
        valid.append({"candidate_id": cid, "img_ids": ids, "relevance": relevance,
                      "confidence": confidence, "reason": reason})
    return valid, dropped


def provided_ids(items: Iterable[str]) -> frozenset[str]:
    return frozenset(items)
