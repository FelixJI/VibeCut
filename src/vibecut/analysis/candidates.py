"""Bounded candidate discovery from cheap features.

Guarantees (task #5 contract):
  * full-time-span coverage - every third of the timeline contributes at
    least sampled coverage candidates (plus motion windows within the remaining budget);
  * bounded candidate count (``max_candidates``);
  * each candidate carries a view id (equatorial view whose yaw centre is
    nearest the sector) and a bounded [0,1] score;
  * stable ids (c0001...) ordered by (start, view) so caches and evidence
    stay deterministic.
"""

from __future__ import annotations

from fractions import Fraction

from vibecut.analysis.features import SECTOR_COUNT, SampleFeatures, sector_yaw_center
from vibecut.analysis.plan import CandidateWindow
from vibecut.analysis.sampling import SamplingPlan
from vibecut.analysis.views import ViewSpec, equatorial_views, wrap_yaw_deg


def _nearest_view(views: tuple[ViewSpec, ...], yaw: float) -> ViewSpec:
    best, best_d = views[0], 1e9
    for v in views:
        d = abs(wrap_yaw_deg(v.yaw_deg - yaw))
        if d < best_d:
            best, best_d = v, d
    return best


def discover_candidates(
    features: tuple[SampleFeatures, ...],
    plan: SamplingPlan,
    *,
    source_id: str,
    max_candidates: int = 24,
    min_len_s: Fraction = Fraction(2),
    max_len_s: Fraction = Fraction(12),
    motion_floor: float = 0.01,
) -> tuple[CandidateWindow, ...]:
    duration = plan.duration_s
    eq = equatorial_views(plan.views)
    if not eq:
        raise ValueError("sampling plan has no equatorial views")
    min_len = min(min_len_s, duration)
    max_len = min(max_len_s, duration)

    windows: list[tuple[Fraction, Fraction, int, str, float]] = []  # start, end, sector, origin, score

    for s in range(SECTOR_COUNT):
        motions = [f.motion[s] if s < len(f.motion) else 0.0 for f in features]
        if not motions or max(motions) <= 0.0:
            continue
        peak_thr = max(motion_floor, 0.5 * max(motions))
        for f in features:
            m = f.motion[s] if s < len(f.motion) else 0.0
            if m < peak_thr or m <= 0:
                continue
            half = min(max(min_len, Fraction(4)), max_len) / 2
            start = max(Fraction(0), f.time - half)
            end = min(duration, f.time + half)
            if end - start <= 0:
                continue
            windows.append((start, end, s, "motion", min(1.0, 0.25 + m * 3.0)))

    # merge overlapping windows of the same sector
    merged: list[dict] = []
    for start, end, s, origin, score in sorted(windows, key=lambda w: (w[2], w[0])):
        for m in merged:
            if m["sector"] == s and start <= m["end"] and end >= m["start"]:
                m["start"] = min(m["start"], start)
                m["end"] = max(m["end"], end)
                m["score"] = max(m["score"], score)
                m["origin"] = "motion" if origin == "motion" or m["origin"] == "motion" else "coverage"
                break
        else:
            merged.append({"start": start, "end": end, "sector": s, "origin": origin, "score": score})

    # Reserve one sampled coverage window per direction, including the poles.
    # Motion candidates use only the remaining slots; all candidates stay bounded.
    coverage = []
    count = min(max_candidates, len(plan.views))
    if count == 0:
        return ()
    for index, view in enumerate(plan.views[:count]):
        sample = plan.times[min(len(plan.times) - 1, index * len(plan.times) // count)]
        half = max(min_len, min(max_len, duration / len(plan.times))) / 2
        coverage.append((max(Fraction(0), sample - half), min(duration, sample + half), view, "coverage", 0.3))
    motion = []
    for m in sorted(merged, key=lambda m: -m["score"]):
        if m["origin"] != "motion":
            continue
        view = _nearest_view(eq, sector_yaw_center(m["sector"]))
        # Do not offer a temporal window that has no actual sampled image.
        if any(m["start"] <= t < m["end"] for t in plan.times):
            motion.append((m["start"], m["end"], view, m["origin"], m["score"]))
    windows = sorted(coverage + motion[:max_candidates - len(coverage)], key=lambda m: (m[0], m[2].view_id))
    return tuple(
        CandidateWindow(candidate_id=f"c{i + 1:04d}", source_id=source_id,
                        start_rel=start, end_rel=end, view_id=view.view_id, origin=origin, score=round(score, 4))
        for i, (start, end, view, origin, score) in enumerate(windows)
    )
