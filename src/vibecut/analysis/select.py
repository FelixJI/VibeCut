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
        cand = by_id.get(sel.candidate_id)
        if cand is None or not sel.selected or not sel.img_ids or len(sel.img_ids) != len(sel.times_rel):
            continue
        if any(not cand.start_rel <= t < cand.end_rel for t in sel.times_rel):
            continue
        items.append((cand, sel))
    items.sort(key=lambda item: (item[0].start_rel, item[0].end_rel))
    groups = []
    for cand, sel in items:
        if (groups and cand.start_rel <= groups[-1]["end"]
                and cand.source_id == groups[-1]["items"][0][0].source_id
                and cand.view_id == groups[-1]["items"][0][0].view_id):
            groups[-1]["end"] = max(groups[-1]["end"], cand.end_rel)
            groups[-1]["items"].append((cand, sel))
        else:
            groups.append({"start": cand.start_rel, "end": cand.end_rel, "items": [(cand, sel)]})

    chosen = []
    total = Fraction(0)
    target = focus.target_duration_s
    for group in sorted(groups, key=lambda g: -max(s.weight for _, s in g["items"])):
        intervals = [(group["start"], group["end"])]
        # Remove already consumed source intervals, including alternative views.
        for used_start, used_end, *_ in chosen:
            intervals = [(lo, hi) for start, end in intervals
                         for lo, hi in ((start, min(end, used_start)), (max(start, used_end), end)) if lo < hi]
        while intervals and len(chosen) < params.max_segments:
            remaining = target - total
            if remaining <= 0 or (total > 0 and remaining <= params.tolerance_s):
                break
            options = []
            for index, (lo, hi) in enumerate(intervals):
                for cand, sel in group["items"]:
                    for iid, t in zip(sel.img_ids, sel.times_rel):
                        if lo <= t < hi:
                            options.append((sel.weight, index, cand, sel, iid, t))
            if not options:
                break  # no actual image supports another retained segment
            _, index, anchor, selected, _, time_rel = max(options, key=lambda option: option[0])
            lo, hi = intervals.pop(index)
            take = min(hi - lo, params.max_segment_s, remaining)
            if take < min(params.min_take_s, duration_s):
                continue
            start, end = _clip_window(time_rel - lo, take, hi - lo)
            start, end = start + lo, end + lo
            refs = [(cand, sel, iid, t) for cand, sel in group["items"]
                    if cand.view_id == anchor.view_id and cand.source_id == anchor.source_id
                    for iid, t in zip(sel.img_ids, sel.times_rel) if start <= t < end]
            if not refs:
                continue
            chosen.append((start, end, anchor, selected, refs))
            total += end - start
            intervals.extend((a, b) for a, b in ((lo, start), (end, hi)) if a < b)
        if target - total <= params.tolerance_s and chosen:
            break

    segments = []
    for index, (start, end, anchor, selected, refs) in enumerate(sorted(chosen, key=lambda item: item[0])):
        evidence = tuple(EvidenceRef(candidate_id=cand.candidate_id, time_rel=t, view_id=cand.view_id,
                                     img_id=iid, source_id=cand.source_id,
                                     direction=direction_of(cand, sel) if direction_of else None)
                         for cand, sel, iid, t in refs[:4])
        segments.append(SegmentDraft(segment_id=f"seg{index + 1:03d}", source_id=anchor.source_id,
                                     cut_in_rel=start, cut_out_rel=end, view_id=anchor.view_id,
                                     reason=selected.reason, evidence=evidence,
                                     relevance=selected.relevance, confidence=selected.confidence))

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
