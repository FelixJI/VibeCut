"""Version 1 dynamic sequence, independent of the existing v2 static contracts."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from vibecut.core.contracts import CameraPath, CameraView, ContractError, MediaSource, frac_str, parse_frac

GEOMETRY_REVISION = 1  # absolute Ry(yaw) Rx(pitch) Rz(roll), v360 rorder=ypr


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


@dataclass(frozen=True)
class BoundCameraPath:
    asset_id: str
    aspect: str
    camera: CameraPath
    cuts: tuple[Fraction, ...] = ()  # right-continuous jumps at existing keys
    geometry_revision: int = GEOMETRY_REVISION
    schema_version: int = 1

    def __post_init__(self):
        _check(self.schema_version == 1, "unsupported bound camera schema")
        _check(self.geometry_revision == GEOMETRY_REVISION, "stale geometry revision")
        _check(self.aspect in ("16:9", "9:16"), "aspect must be 16:9 or 9:16")
        _check(bool(self.asset_id), "camera asset binding missing")
        _check(self.camera.provenance == "user", "dynamic camera requires user locks")
        times = [t for t, _ in self.camera.keys]
        _check(all(isinstance(t, Fraction) and t >= 0 for t in times), "camera times must be nonnegative Fractions")
        _check(all(isinstance(t, Fraction) for t in self.cuts), "cut times must be Fractions")
        _check(tuple(sorted(set(self.cuts))) == self.cuts, "cuts must be strictly increasing")
        _check(all(t in times[1:] for t in self.cuts), "cut must name a camera key after the first")
        for _, view in self.camera.keys:
            _check(all(math.isfinite(v) for v in (view.yaw_deg, view.pitch_deg, view.roll_deg, view.h_fov_deg)), "camera values must be finite")
            _check(-90 <= view.pitch_deg <= 90, "pitch must be in [-90,90]")
            _check(0 < view.h_fov_deg < 180, "horizontal FOV must be in (0,180)")

    def evaluate(self, source_rel: Fraction) -> CameraView:
        """Shared preview/final evaluator; shortest yaw/roll arcs, explicit cuts."""
        keys = self.camera.keys
        _check(keys[0][0] <= source_rel <= keys[-1][0], "camera does not cover requested source time")
        for (ta, a), (tb, b) in zip(keys, keys[1:]):
            if source_rel < tb:
                if tb in self.cuts:
                    return a
                f = float((source_rel - ta) / (tb - ta))
                arc = lambda x, y: x + ((y - x + 180) % 360 - 180) * f
                return CameraView(arc(a.yaw_deg, b.yaw_deg), a.pitch_deg + (b.pitch_deg - a.pitch_deg) * f,
                                  arc(a.roll_deg, b.roll_deg), a.h_fov_deg + (b.h_fov_deg - a.h_fov_deg) * f)
        return keys[-1][1]

    def to_json(self):
        return {"schema_version": 1, "geometry_revision": self.geometry_revision, "asset_id": self.asset_id,
                "aspect": self.aspect, "camera": self.camera.to_json(), "cuts": [frac_str(t) for t in self.cuts]}

    @classmethod
    def from_json(cls, d):
        return cls(d["asset_id"], d["aspect"], CameraPath.from_json(d["camera"]),
                   tuple(parse_frac(t) for t in d.get("cuts", [])), d["geometry_revision"], d["schema_version"])


@dataclass(frozen=True)
class SequenceClip:
    source_id: str
    cut_in_rel: Fraction
    cut_out_rel: Fraction
    path: BoundCameraPath

    @property
    def duration_rel(self):
        return self.cut_out_rel - self.cut_in_rel

    def __post_init__(self):
        _check(isinstance(self.cut_in_rel, Fraction) and isinstance(self.cut_out_rel, Fraction), "clip times must be Fractions")
        _check(0 <= self.cut_in_rel < self.cut_out_rel, "invalid clip window")
        _check(self.path.asset_id == self.source_id, "camera asset binding mismatch")
        _check(self.path.camera.keys[0][0] <= self.cut_in_rel and self.path.camera.keys[-1][0] >= self.cut_out_rel,
               "camera must cover entire clip window")

    def to_json(self):
        return {"source_id": self.source_id, "cut_in_rel": frac_str(self.cut_in_rel),
                "cut_out_rel": frac_str(self.cut_out_rel), "path": self.path.to_json()}

    @classmethod
    def from_json(cls, d):
        return cls(d["source_id"], parse_frac(d["cut_in_rel"]), parse_frac(d["cut_out_rel"]), BoundCameraPath.from_json(d["path"]))


@dataclass(frozen=True)
class SequencePlan:
    clips: tuple[SequenceClip, ...]
    project_fps: Fraction
    aspect: str = "16:9"
    audio: bool = True  # silence is inserted for every missing/disabled source track
    schema_version: int = 1

    def __post_init__(self):
        _check(self.schema_version == 1, "unsupported sequence schema")
        _check(isinstance(self.audio, bool), "audio must be a boolean")
        _check(bool(self.clips), "sequence needs clips")
        _check(isinstance(self.project_fps, Fraction) and self.project_fps > 0, "sequence fps must be a positive Fraction")
        _check(self.aspect in ("16:9", "9:16"), "unsupported aspect")
        _check(all(c.path.aspect == self.aspect for c in self.clips), "each aspect needs its own bound path")
        _check(all((c.duration_rel * self.project_fps).denominator == 1 for c in self.clips),
               "clip durations must contain whole output frames")

    @property
    def duration_rel(self):
        return sum((c.duration_rel for c in self.clips), Fraction(0))

    def validate_sources(self, sources: dict[str, MediaSource]):
        for clip in self.clips:
            _check(clip.source_id in sources, "sequence source missing")
            src = sources[clip.source_id]
            _check(not src.vfr and bool(src.video.nb_frames), "source needs verified CFR packet count")
            duration = Fraction(src.video.nb_frames) / src.video.fps
            _check(clip.cut_out_rel <= duration, "clip exceeds source range")
            _check(clip.path.camera.keys[-1][0] <= duration, "camera key exceeds source range")

    def to_json(self):
        return {"schema_version": 1, "clips": [c.to_json() for c in self.clips], "project_fps": frac_str(self.project_fps),
                "aspect": self.aspect, "audio": self.audio}

    @classmethod
    def from_json(cls, d):
        return cls(tuple(SequenceClip.from_json(c) for c in d["clips"]), parse_frac(d["project_fps"]),
                   d["aspect"], d.get("audio", True), d["schema_version"])
