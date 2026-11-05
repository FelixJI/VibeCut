"""Bounded selection: merge, dedupe, duration budget, explicit shortfalls.

Rules (task #5 contract):
  * adjacent/highly-overlapping selected candidates merge (union window,
    max weight, concatenated evidence);
  * a segment never overlaps another; total duration respects the target
    within a <=1 s tolerance;
  * no material is fabricated and nothing loops to pad duration - if the
    available relevant material is short, the plan is *short* with an
    explicit reason; zero matches yield an *empty* plan for manual review.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from vibecut.analysis.focus import FocusSpec
from vibecut.analysis.plan import CandidateWindow, EvidenceRef, ModelSelection, SegmentDraft, SelectionPlan


@dataclass(frozen=True)
class SelectionParams:
    min_segment_s: Fraction = Fraction(2)
    max_segment_s: Fraction = Fraction(15)
    max_segments: int = 8
    tolerance_s: Fraction = Fraction(1)
    min_take_s: Fraction = Fraction(1)
    merge_gap_s: Fraction = Fraction(1)


def _clip_window(center: Fraction, length: Fraction, duration: Fraction) -> tuple[Fraction, Fraction]:
    length = min(length, duration)
    start = max(Fraction(0), center - length / 2)
    end = min(duration, start + length)
    start = max(Fraction(0), end - length)
    return start, end


def build_selection_plan(
    candidates: tuple[CandidateWindow, ...],
    selections: tuple[ModelSelection, ...],
    focus: FocusSpec,
    duration_s: Fraction,
    params: SelectionParams | None = None,
    *,
    direction_of=None,
) -> SelectionPlan:
    params = params or SelectionParams()
    by_id = {c.candidate_id: c for c in candidates}
    items = []
    for sel in selections:
        if not sel.selected:
            continue
        cand = by_id.get(sel.candidate_id)
        if cand is None:
            continue
        items.append((cand.start_rel, cand.end_rel, cand, sel))
    items.sort(key=lambda it: (it[0], it[1]))

    merged: list[dict] = []
    for start, end, cand, sel in items:
        if merged and start <= merged[-1]["end"] + params.merge_gap_s:
            m = merged[-1]
            m["end"] = max(m["end"], end)
            m["start"] = min(m["start"], start)
            m["items"].append((cand, sel))
        else:
            merged.append({"start": start, "end": end, "items": [(cand, sel)]})

    chosen: list[tuple[Fraction, Fraction, dict]] = []
    total = Fraction(0)
    target = focus.target_duration_s
    for m in sorted(merged, key=lambda m: -max(s.weight for _, s in m["items"])):
        if len(chosen) >= params.max_segments:
            break
        remaining = target - total
        if remaining <= 0 or (total > 0 and remaining <= params.tolerance_s):
            break  # within tolerance - done, no padding
        span = m["end"] - m["start"]
        take = min(max(span, params.min_segment_s), params.max_segment_s, remaining + params.tolerance_s)
        if take < min(params.min_take_s, duration_s):
            continue  # cannot form a meaningful segment; try next
        best_cand, best_sel = max(m["items"], key=lambda cs: cs[1].weight)
        center = (Fraction(str(best_sel.event_time_s)) if best_sel.event_time_s is not None
                  else (best_cand.start_rel + best_cand.end_rel) / 2)
        start, end = _clip_window(center - m["start"], take, span)
        start, end = start + m["start"], end + m["start"]
        if end - start < min(params.min_take_s, duration_s):
            continue
        if any(start < e and s < end for s, e, _ in chosen):
            continue  # never overlapping/duplicated spans
        chosen.append((start, end, m))
        total += end - start

    chosen.sort(key=lambda c: c[0])
    segments: list[SegmentDraft] = []
    for i, (start, end, m) in enumerate(chosen):
        def evidence_time(item):
            cand, sel = item
            return Fraction(str(sel.event_time_s)) if sel.event_time_s is not None else (cand.start_rel + cand.end_rel) / 2
        top = sorted((item for item in m["items"] if start <= evidence_time(item) < end), key=lambda cs: -cs[1].weight)
        evidence = []
        for cand, sel in top[:4]:
            direction = direction_of(cand, sel) if direction_of else None
            t = evidence_time((cand, sel))
            evidence.append(EvidenceRef(
                candidate_id=cand.candidate_id, time_rel=t,
                view_id=cand.view_id, bbox=sel.bbox, direction=direction,
            ))
        reason = "；".join(dict.fromkeys(sel.reason for _, sel in top[:2]))[:240]
        segments.append(SegmentDraft(
            segment_id=f"seg{i + 1:03d}", source_id=(top[0][0].source_id if top else ""),
            cut_in_rel=start, cut_out_rel=end, view_id=(top[0][0].view_id if top else ""),
            reason=reason, evidence=tuple(evidence),
            relevance=max(s.relevance for _, s in m["items"]),
            confidence=max(s.confidence for _, s in m["items"]),
        ))

    if not segments:
        return SelectionPlan(
            status="empty", segments=(), target_duration_s=target, total_duration_s=Fraction(0),
            shortfall_reason="零匹配：模型未选中任何相关候选，请人工审阅候选窗口后手动选择",
        )
    if target - total > params.tolerance_s:
        shortfall = target - total
        return SelectionPlan(
            status="short", segments=tuple(segments), target_duration_s=target, total_duration_s=total,
            shortfall_reason=(
                f"可用相关素材不足：目标 {float(target):.1f}s，实际可选 {float(total):.1f}s，"
                f"缺 {float(shortfall):.1f}s；不虚构、不循环凑时长"
            ),
        )
    return SelectionPlan(status="complete", segments=tuple(segments), target_duration_s=target, total_duration_s=total)
