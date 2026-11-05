"""Chinese focus text -> bounded focus fields (task #5).

Deterministic rule-based conversion (no model inference): topic weights,
include/exclude preferences, target duration. All outputs are bounded so a
hostile/overlong focus can never blow up prompts or budgets. 人优先/风景优先
are explicit; visible-target terms stay short strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

MAX_FOCUS_TEXT = 500
MAX_TERMS = 4
MAX_TERM_LEN = 16
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(秒钟?|秒|s|S|分钟|min|分)")
_SPLIT_RE = re.compile(r"[，,。.、;；\s]+")

_PERSON_TERMS = ("人物", "人们", "有人", "人像", "人物优先", "舞者", "演员", "表演", "跳舞", "舞", "脸", "面孔", "身影",
                 "小孩", "孩子", "朋友", "同伴", "说话", "唱歌", "讲话", "走动的人", "玩家")
_SCENERY_TERMS = ("风景", "景色", "景", "山水", "山", "天空", "云", "云海", "海", "湖", "日落", "夕阳", "日出", "朝霞",
                  "晚霞", "建筑", "城市", "夜景", "星空", "银河", "雪", "花", "树", "森林", "草原", "海边")
_ACTION_TERMS = ("运动", "动作", "跳", "跑", "奔跑", "滑", "滑雪", "滑板", "冲浪", "骑行", "驾驶", "飞", "飞行")
_EXCLUDE_PREFIX = ("不要", "排除", "不含", "别拍", "避免")
_INCLUDE_PREFIX = ("包含", "要包括", "需要", "优先拍", "重点拍")


class FocusError(ValueError):
    pass


@dataclass(frozen=True)
class FocusSpec:
    """Bounded, prompt-facing focus fields."""

    text: str
    priority: str  # "person" | "scenery" | "balanced"
    include_terms: tuple[str, ...]
    exclude_terms: tuple[str, ...]
    target_duration_s: Fraction
    person_weight: float
    scenery_weight: float
    action_weight: float
    aspect: str = "16:9"

    def __post_init__(self) -> None:
        if self.priority not in ("person", "scenery", "balanced"):
            raise FocusError(f"bad priority {self.priority!r}")
        if not (0.0 <= self.person_weight <= 1.0 and 0.0 <= self.scenery_weight <= 1.0 and 0.0 <= self.action_weight <= 1.0):
            raise FocusError("topic weights must be within [0, 1]")
        if self.target_duration_s <= 0:
            raise FocusError("target_duration_s must be positive")
        if len(self.include_terms) > MAX_TERMS or len(self.exclude_terms) > MAX_TERMS:
            raise FocusError("too many include/exclude terms")

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "priority": self.priority,
            "include_terms": list(self.include_terms),
            "exclude_terms": list(self.exclude_terms),
            "target_duration_s": f"{self.target_duration_s.numerator}/{self.target_duration_s.denominator}",
            "topic_weights": {"person": self.person_weight, "scenery": self.scenery_weight, "action": self.action_weight},
            "aspect": self.aspect,
        }

    def prompt_identity(self) -> dict:
        """Fields that define selection-cache identity (target duration included)."""
        return self.to_json()


def _extract_terms_after(text: str, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    for prefix in prefixes:
        idx = text.find(prefix)
        while idx >= 0:
            rest = text[idx + len(prefix):]
            term = _SPLIT_RE.split(rest, maxsplit=1)[0].strip("：:!?！？的与和及 ")
            if term and len(term) <= MAX_TERM_LEN and term not in terms:
                terms.append(term)
            if len(terms) >= MAX_TERMS:
                return tuple(terms)
            idx = text.find(prefix, idx + len(prefix))
    return tuple(terms)


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(t) for t in terms)


def parse_focus(
    text: str,
    *,
    priority: str | None = None,
    target_duration_s: Fraction | None = None,
    default_duration_s: Fraction = Fraction(30),
) -> FocusSpec:
    """Convert a Chinese focus description into bounded :class:`FocusSpec` fields.

    Precedence for duration: explicit argument > text-embedded duration >
    default. Priority: explicit argument > keyword majority > "balanced".
    """
    text = (text or "").strip()
    if not text or len(text) > MAX_FOCUS_TEXT:
        raise FocusError(f"focus text must be 1..{MAX_FOCUS_TEXT} chars (got {len(text)})")

    include = _extract_terms_after(text, _INCLUDE_PREFIX)
    exclude = _extract_terms_after(text, _EXCLUDE_PREFIX)

    person = _count_terms(text, _PERSON_TERMS)
    scenery = _count_terms(text, _SCENERY_TERMS)
    action = _count_terms(text, _ACTION_TERMS)
    total = person + scenery + action
    if total == 0:
        pw = sw = aw = 1.0 / 3.0
    else:
        pw, sw, aw = person / total, scenery / total, action / total
    if priority is None:
        if person > scenery:
            priority = "person"
        elif scenery > person:
            priority = "scenery"
        else:
            priority = "balanced"

    duration = default_duration_s
    m = _DURATION_RE.search(text)
    if m:
        value = Fraction(m.group(1)).limit_denominator(1000)
        if m.group(2).startswith("分") or m.group(2) in ("min",):
            value *= 60
        duration = value
    if target_duration_s is not None:
        duration = target_duration_s
    if not Fraction(1) <= duration <= Fraction(600):
        raise FocusError(f"target duration must be within [1, 600] s (got {duration})")

    return FocusSpec(
        text=text,
        priority=priority,
        include_terms=include,
        exclude_terms=exclude,
        target_duration_s=duration,
        person_weight=round(pw, 4),
        scenery_weight=round(sw, 4),
        action_weight=round(aw, 4),
    )
