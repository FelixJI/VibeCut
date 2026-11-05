"""Six-direction overlapping perspective views + back-computable geometry.

Task #5 scope: sample the *canonical panorama asset* (the #2 proxy), never
raw LRV. Every view carries a unique ``view_id``; the spherical view
transform is fully back-computable via :meth:`ViewSpec.project_px` /
:meth:`ViewSpec.direction_at_px` (canonical semantics of ``core.geometry``:
yaw 0 = equirect centre, +yaw right, +pitch up, +roll clockwise, h_fov is
horizontal).

Default observation set (bounded, fixed): six overlapping equatorial views
(yaw -150/-90/-30/+30/+90/+150, hfov 90 with centres 60 degrees apart, seam covered by two views) plus
eight polar-ring views (pitch +-55, hfov 120) so behind / seam-edge / up / down directions are all covered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vibecut.core.geometry import vfov_from_hfov, wrap_yaw

_EPS = 1e-9


def _direction_vector(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    y = math.radians(wrap_yaw(yaw_deg))
    p = math.radians(max(-90.0, min(90.0, pitch_deg)))
    cp = math.cos(p)
    return (cp * math.cos(y), cp * math.sin(y), math.sin(p))


def _yaw_pitch(v: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    yaw = math.degrees(math.atan2(y, x))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, z / n))))
    return wrap_yaw(yaw), pitch


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(_dot(a, a))
    if n < _EPS:
        raise ValueError("zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


@dataclass(frozen=True)
class ViewSpec:
    """One perspective observation view over the canonical panorama."""

    view_id: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0
    h_fov_deg: float = 90.0
    width: int = 768
    height: int = 432

    def __post_init__(self) -> None:
        if not self.view_id or not all(c.isalnum() or c in "+-._" for c in self.view_id):
            raise ValueError(f"invalid view_id {self.view_id!r}")
        if not (0.0 < self.h_fov_deg < 180.0):
            raise ValueError("h_fov_deg must be in (0, 180)")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("view size must be positive")

    # -- geometry ----------------------------------------------------------

    def basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Camera basis (forward, right, up) with canonical sign conventions."""
        f = _direction_vector(self.yaw_deg, self.pitch_deg)
        # Near the poles the world-up reference degenerates; fall back to +Y.
        ref_up = (0.0, 0.0, 1.0) if abs(self.pitch_deg) < 89.0 else (0.0, 1.0, 0.0)
        r = _norm(_cross(ref_up, f))
        u = _norm(_cross(f, r))
        theta = math.radians(self.roll_deg)
        # positive roll = camera rotates clockwise (up tilts toward right)
        u2 = (u[0] * math.cos(theta) + r[0] * math.sin(theta),
              u[1] * math.cos(theta) + r[1] * math.sin(theta),
              u[2] * math.cos(theta) + r[2] * math.sin(theta))
        r2 = (r[0] * math.cos(theta) - u[0] * math.sin(theta),
              r[1] * math.cos(theta) - u[1] * math.sin(theta),
              r[2] * math.cos(theta) - u[2] * math.sin(theta))
        return f, _norm(r2), _norm(u2)

    def vfov_deg(self) -> float:
        return vfov_from_hfov(self.h_fov_deg, self.width, self.height)

    def project_px(self, yaw_deg: float, pitch_deg: float) -> tuple[float, float] | None:
        """Canonical direction -> view pixel coords (None if behind camera)."""
        f, r, u = self.basis()
        d = _direction_vector(yaw_deg, pitch_deg)
        x, a, b = _dot(d, f), _dot(d, r), _dot(d, u)
        if x <= _EPS:
            return None
        th = math.tan(math.radians(self.h_fov_deg) / 2.0)
        tv = math.tan(math.radians(self.vfov_deg()) / 2.0)
        x_ndc = a / (x * th)
        y_ndc = b / (x * tv)
        return (x_ndc + 1.0) / 2.0 * self.width, (1.0 - y_ndc) / 2.0 * self.height

    def direction_at_px(self, px: float, py: float) -> tuple[float, float]:
        """Inverse of :meth:`project_px` (pinhole unprojection)."""
        f, r, u = self.basis()
        x_ndc = 2.0 * px / self.width - 1.0
        y_ndc = 1.0 - 2.0 * py / self.height
        th = math.tan(math.radians(self.h_fov_deg) / 2.0)
        tv = math.tan(math.radians(self.vfov_deg()) / 2.0)
        d = (f[0] + x_ndc * th * r[0] + y_ndc * tv * u[0],
             f[1] + x_ndc * th * r[1] + y_ndc * tv * u[1],
             f[2] + x_ndc * th * r[2] + y_ndc * tv * u[2])
        return _yaw_pitch(_norm(d))

    def covers(self, yaw_deg: float, pitch_deg: float, *, margin_frac: float = 0.1) -> bool:
        """True if the direction projects inside the view (with margin)."""
        px = self.project_px(yaw_deg, pitch_deg)
        if px is None:
            return False
        mx = margin_frac * self.width
        my = margin_frac * self.height
        x, y = px
        return -mx <= x <= self.width + mx and -my <= y <= self.height + my

    def bbox_center_direction(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Normalized in-view bbox (x, y, w, h) -> canonical direction."""
        x, y, w, _h = bbox
        return self.direction_at_px((x + w / 2.0) * self.width, (y + _h / 2.0) * self.height)

    # -- serialisation ------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "view_id": self.view_id, "yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg, "h_fov_deg": self.h_fov_deg,
            "width": self.width, "height": self.height,
        }

    @classmethod
    def from_json(cls, data: dict) -> "ViewSpec":
        return cls(
            view_id=str(data["view_id"]), yaw_deg=float(data["yaw_deg"]), pitch_deg=float(data["pitch_deg"]),
            roll_deg=float(data.get("roll_deg", 0.0)), h_fov_deg=float(data.get("h_fov_deg", 90.0)),
            width=int(data.get("width", 768)), height=int(data.get("height", 432)),
        )


def wrap_yaw_deg(delta_deg: float) -> float:
    """Wrap an angular difference into [-180, 180)."""
    d = (delta_deg + 180.0) % 360.0 - 180.0
    if d == -180.0:
        d = 180.0 - 1e-12
    return d


def default_view_set() -> tuple[ViewSpec, ...]:
    """Six overlapping equatorial views + two polar rings (full sphere).

    Equator: yaw -150/-90/-30/+30/+90/+150 at hfov 90 (30 deg horizontal overlap,
    seam behind covered by two views). Poles: rings of four views at
    pitch +55 / -55 (yaw -90/0/90/180) at hfov 120 so every mid-latitude
    direction - including behind the equirect seam - stays covered.
    """
    eq = tuple(
        ViewSpec(view_id=f"eq{yaw:+03d}", yaw_deg=float(yaw), pitch_deg=0.0, roll_deg=0.0, h_fov_deg=90.0)
        for yaw in (-150, -90, -30, 30, 90, 150)
    )
    polar: list[ViewSpec] = []
    for sign, tag in ((1.0, "+"), (-1.0, "-")):
        for yaw in (-90, 0, 90, 180):
            yid = f"p{tag}{abs(yaw):03d}" + ("L" if yaw < 0 else "")
            polar.append(ViewSpec(view_id=yid, yaw_deg=float(yaw), pitch_deg=55.0 * sign,
                                  roll_deg=0.0, h_fov_deg=120.0))
    return eq + tuple(polar)


def equatorial_views(views: tuple[ViewSpec, ...]) -> tuple[ViewSpec, ...]:
    return tuple(v for v in views if abs(v.pitch_deg) < 45.0)


def find_covering_view(views: tuple[ViewSpec, ...], yaw_deg: float, pitch_deg: float) -> ViewSpec | None:
    for v in views:
        if v.covers(yaw_deg, pitch_deg):
            return v
    return None


def coverage_probe(views: tuple[ViewSpec, ...], *, yaw_step: float = 15.0, pitch_step: float = 10.0) -> list[tuple[float, float]]:
    """Directions (for tests) that no view covers - must be empty for the default set."""
    missed: list[tuple[float, float]] = []
    pitch = -85.0
    while pitch <= 85.0:
        yaw = -180.0
        while yaw < 180.0:
            if find_covering_view(views, yaw, pitch) is None:
                missed.append((yaw, pitch))
            yaw += yaw_step
        pitch += pitch_step
    for pole in (89.9, -89.9):
        if find_covering_view(views, 0.0, pole) is None or find_covering_view(views, 179.0, pole) is None:
            missed.append((0.0 if pole > 0 else 180.0, pole))
    return missed
