"""View coverage + back-computation: behind / seam / up / down (AC3, pure CPU)."""

import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.analysis.sampling import plan_sampling
from vibecut.analysis.views import ViewSpec, coverage_probe, default_view_set, find_covering_view

VIEWS = default_view_set()


def test_full_sphere_coverage_with_overlap():
    missed = coverage_probe(VIEWS, yaw_step=5, pitch_step=5)
    assert missed == []
    ids = [v.view_id for v in VIEWS]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("yaw,pitch", [
    (180.0, 0.0), (179.9, 0.0), (-179.9, 0.0),  # behind + seam edges
    (0.0, 89.0), (0.0, -89.0), (180.0, 70.0), (0.0, -70.0),  # up / down / high latitudes
    (135.0, 45.0), (-120.0, -40.0), (90.0, 0.0), (-90.0, 0.0), (0.0, 0.0),
])
def test_targets_are_covered(yaw, pitch):
    view = find_covering_view(VIEWS, yaw, pitch)
    assert view is not None, f"direction ({yaw}, {pitch}) uncovered"


def test_seam_is_covered_by_two_views():
    covering = [v for v in VIEWS if v.covers(180.0, 0.0)]
    assert len(covering) >= 1
    # the seam neighbourhood is inside at least two equatorial views
    assert any(v.view_id == "eq+150" for v in covering) and any(v.view_id == "eq-150" for v in covering)


def test_projection_round_trip_center_and_offsets():
    for view in VIEWS:
        cx, cy = view.width / 2, view.height / 2
        yaw, pitch = view.direction_at_px(cx, cy)
        assert abs(yaw - view.yaw_deg) < 1e-6
        assert abs(pitch - view.pitch_deg) < 1e-6
        px = view.project_px(yaw, pitch)
        assert px is not None
        assert abs(px[0] - cx) < 1e-6 and abs(px[1] - cy) < 1e-6


def test_projection_round_trip_random_directions():
    import random

    rng = random.Random(42)
    for _ in range(300):
        yaw = rng.uniform(-180, 180)
        pitch = math.degrees(math.asin(rng.uniform(-0.98, 0.98)))
        view = find_covering_view(VIEWS, yaw, pitch)
        assert view is not None
        px = view.project_px(yaw, pitch)
        assert px is not None
        y2, p2 = view.direction_at_px(px[0], px[1])
        # angular error must be negligible (back-computable evidence)
        dy = abs((y2 - yaw + 180) % 360 - 180)
        assert dy < 1e-3, (view.view_id, yaw, pitch, y2)
        assert abs(p2 - pitch) < 1e-3


def test_bbox_center_direction_maps_into_view():
    view = VIEWS[0]
    bbox = (0.25, 0.25, 0.5, 0.5)
    yaw, pitch = view.bbox_center_direction(bbox)
    assert view.covers(yaw, pitch, margin_frac=0.0)
    px = view.project_px(yaw, pitch)
    assert abs(px[0] - view.width * 0.5) < 1e-6
    assert abs(px[1] - view.height * 0.5) < 1e-6


def test_roll_is_back_computable():
    view = ViewSpec("r090", 0.0, 0.0, 90.0, 90.0)
    # +90 deg roll (camera clockwise): the image-top axis u' becomes the old
    # camera-right r, so the top-centre direction lies horizontally at
    # yaw = atan(vfov-tangent) with pitch exactly 0 - back-computable.
    yaw, pitch = view.direction_at_px(view.width / 2, 0.5)
    assert abs(pitch) < 1e-6
    assert abs(yaw - 29.30) < 0.05  # atan(0.9977 * tan(vfov/2)), vfov(90 deg, 16:9) = 58.72
    # and unrolling is the identity-consistent inverse for the centre
    cy = view.direction_at_px(view.width / 2, view.height / 2)
    assert abs(cy[0]) < 1e-6 and abs(cy[1]) < 1e-6


def test_adjacent_equatorial_views_overlap():
    from vibecut.analysis.views import equatorial_views

    eq = equatorial_views(VIEWS)
    yaws = sorted(v.yaw_deg for v in eq)
    for a, b in zip(yaws, yaws[1:] + [yaws[0] + 360]):
        gap = b - a  # spacing
        overlap = 90.0 - gap  # hfov 90
        assert overlap > 20.0, f"views at {a}/{b} overlap only {overlap} deg"


def test_sampling_covers_whole_span():
    plan = plan_sampling(Fraction(60), samples=6)
    assert plan.times[0] == Fraction(5)
    assert plan.times[-1] == Fraction(55)
    for third in range(3):
        lo, hi = Fraction(20 * third), Fraction(20 * (third + 1))
        assert any(lo <= t < hi for t in plan.times)
