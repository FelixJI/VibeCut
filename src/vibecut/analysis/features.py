"""Cheap prompt-independent features: sector luma stats, motion, quality.

The full span is decoded once into tiny gray frames (see extract.py), split
into six 60-degree yaw sectors, binned onto the sampling plan, and reduced to
per-(sample, sector) mean/std plus inter-sample motion. Everything is
standard-library pure Python and bounded by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from vibecut.analysis.sampling import SamplingPlan

SECTOR_COUNT = 6
SECTOR_YAW_CENTERS = tuple(-180.0 + (s + 0.5) * (360.0 / SECTOR_COUNT) for s in range(SECTOR_COUNT))


def sector_yaw_center(sector: int) -> float:
    return SECTOR_YAW_CENTERS[sector]


@dataclass(frozen=True)
class SectorStats:
    mean: float  # 0..255
    std: float  # 0..127.5
    frames: int

    def quality(self) -> str:
        if self.frames == 0:
            return "unknown"
        if self.mean < 16.0:
            return "dark"
        if self.mean > 240.0:
            return "blown"
        if self.std < 2.0:
            return "flat"
        return "ok"


@dataclass(frozen=True)
class SampleFeatures:
    index: int
    time: Fraction
    sectors: tuple[SectorStats, ...]
    motion: tuple[float, ...]  # |mean_i - mean_{i-1}|/255 per sector (0 for i=0)

    def to_json(self) -> dict:
        return {
            "index": self.index, "time": f"{self.time.numerator}/{self.time.denominator}",
            "sectors": [{"mean": round(s.mean, 2), "std": round(s.std, 2), "frames": s.frames, "quality": s.quality()}
                        for s in self.sectors],
            "motion": [round(m, 5) for m in self.motion],
        }


def features_to_json(features: tuple[SampleFeatures, ...]) -> dict:
    return {"sector_count": SECTOR_COUNT, "samples": [f.to_json() for f in features]}


def features_from_json(data: dict) -> tuple[SampleFeatures, ...]:
    out = []
    for s in data.get("samples", []):
        sectors = tuple(SectorStats(mean=float(x["mean"]), std=float(x["std"]), frames=int(x["frames"]))
                        for x in s["sectors"])
        motion = tuple(float(m) for m in s["motion"])
        t = Fraction(s["time"])
        out.append(SampleFeatures(index=int(s["index"]), time=t, sectors=sectors, motion=motion))
    return tuple(out)


def compute_features(raw: bytes, width: int, height: int, plan: SamplingPlan, stream_fps: Fraction) -> tuple[SampleFeatures, ...]:
    """Bin decoded gray frames onto the sampling plan and derive features."""
    frame_size = width * height
    n_frames = len(raw) // frame_size
    if n_frames == 0:
        return tuple(
            SampleFeatures(index=i, time=t,
                           sectors=tuple(SectorStats(0.0, 0.0, 0) for _ in range(SECTOR_COUNT)),
                           motion=(0.0,) * SECTOR_COUNT)
            for i, t in enumerate(plan.times)
        )
    sector_w = width // SECTOR_COUNT
    bins: list[list[tuple[float, float, int]]] = [[] for _ in range(len(plan.times))]
    duration = plan.duration_s
    step = Fraction(1) / Fraction(stream_fps)
    for idx in range(n_frames):
        t = Fraction(idx) * step
        # nearest-sample bin over the whole timeline
        bin_idx = min(len(plan.times) - 1, int(Fraction(len(plan.times)) * t / duration))
        acc: list[tuple[float, float, int]] = [None] * SECTOR_COUNT  # type: ignore[list-item]
        base = idx * frame_size
        frame = raw[base:base + frame_size]
        for s in range(SECTOR_COUNT):
            x0, x1 = s * sector_w, (s + 1) * sector_w
            total = 0
            total_sq = 0
            count = 0
            for y in range(height):
                row = y * width
                for x in range(x0, x1):
                    v = frame[row + x]
                    total += v
                    total_sq += v * v
                    count += 1
            mean = total / count
            var = max(0.0, total_sq / count - mean * mean)
            acc[s] = (mean, var ** 0.5, 1)
        bins[bin_idx].append(acc)
    out: list[SampleFeatures] = []
    prev_means: tuple[float, ...] | None = None
    for i, t in enumerate(plan.times):
        frames = bins[i]
        if frames:
            merged = []
            for s in range(SECTOR_COUNT):
                n = len(frames)
                mean = sum(f[s][0] for f in frames) / n
                std = sum(f[s][1] for f in frames) / n
                merged.append(SectorStats(mean=mean, std=std, frames=n))
        else:
            merged = [SectorStats(0.0, 0.0, 0) for _ in range(SECTOR_COUNT)]
        means = tuple(st.mean for st in merged)
        if prev_means is None:
            motion = (0.0,) * SECTOR_COUNT
        else:
            motion = tuple(abs(means[s] - prev_means[s]) / 255.0 for s in range(SECTOR_COUNT))
        out.append(SampleFeatures(index=i, time=t, sectors=tuple(merged), motion=motion))
        prev_means = means
    return tuple(out)
