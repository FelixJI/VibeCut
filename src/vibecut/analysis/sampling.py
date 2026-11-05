"""Full-time-span sampling plan over the canonical panorama asset."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from vibecut.analysis.views import ViewSpec, default_view_set

MAX_SAMPLES = 64


@dataclass(frozen=True)
class SamplingPlan:
    """Evenly spread sample times x observation views (covers the whole span)."""

    times: tuple[Fraction, ...]
    views: tuple[ViewSpec, ...]
    duration_s: Fraction

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("duration must be positive")
        if not self.times:
            raise ValueError("sampling plan needs at least one sample")
        if len(self.views) < 6:
            raise ValueError("sampling plan needs the full view set (>= 6 views)")
        prev = Fraction(-1)
        for t in self.times:
            if not Fraction(0) <= t < self.duration_s:
                raise ValueError(f"sample time {t} outside [0, {self.duration_s})")
            if t <= prev:
                raise ValueError("sample times must be strictly increasing")
            prev = t

    def identity(self) -> dict:
        """Cache-facing identity of the sampling itself (prompt-independent)."""
        return {
            "duration_s": f"{self.duration_s.numerator}/{self.duration_s.denominator}",
            "times": [f"{t.numerator}/{t.denominator}" for t in self.times],
            "views": [v.to_json() for v in self.views],
        }


def plan_sampling(duration_s: Fraction, *, samples: int, views: tuple[ViewSpec, ...] | None = None, verified_fps: Fraction | None = None) -> SamplingPlan:
    """Evenly spread ``samples`` interior times across the *whole* timeline.

    Times are bin midpoints, never t=0 alone, so the plan structurally cannot
    only look at the first seconds of the source.
    """
    duration_s = Fraction(duration_s)
    if not 1 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be 1..{MAX_SAMPLES} (got {samples})")
    views = views if views is not None else default_view_set()
    times = tuple(Fraction(2 * i + 1) * duration_s / (2 * samples) for i in range(samples))
    if verified_fps is not None:
        # verified_fps comes from uniform displayed packet PTS, never rate tags.
        fps = Fraction(verified_fps)
        frame_count = duration_s * fps
        if fps <= 0 or frame_count.denominator != 1 or frame_count < 1:
            raise ValueError("verified CFR duration must contain whole displayed frames")
        times = tuple(sorted({Fraction(min(int(t * fps), int(frame_count) - 1)) / fps for t in times}))
    return SamplingPlan(times=times, views=views, duration_s=duration_s)
