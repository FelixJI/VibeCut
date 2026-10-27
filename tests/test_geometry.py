"""Pure core tests: geometry semantics (no ffmpeg needed)."""

import math
from fractions import Fraction

import pytest

from vibecut.core.contracts import CONTRACT_VERSION, CameraPath, CameraView, ContractError, EditPlan
from vibecut.core.geometry import (
    direction_to_equirect_px,
    equirect_px_to_direction,
    hfov_from_vfov,
    v360_filter,
    vfov_from_hfov,
    wrap_yaw,
)


def test_wrap_yaw_range():
    assert wrap_yaw(190.0) == -170.0
    assert wrap_yaw(-190.0) == 170.0
    assert wrap_yaw(360.0) == 0.0
    assert wrap_yaw(-720.0) == 0.0
    assert -180.0 <= wrap_yaw(179.999) < 180.0


@pytest.mark.parametrize("yaw", [0.0, 45.0, 90.0, 180.0 - 1e-9, -90.0, -135.0])
def test_pixel_direction_roundtrip_midlatitude(yaw):
    w, h = 1920, 960
    x, y = direction_to_equirect_px(yaw, 0.0, w, h)
    yaw2, pitch2 = equirect_px_to_direction(x, y, w, h)
    assert abs(wrap_yaw(yaw2 - yaw)) < 1e-9
    assert abs(pitch2) < 1e-9


@pytest.mark.parametrize("pitch", [-80.0, -45.0, 0.0, 45.0, 80.0])
def test_pixel_direction_roundtrip_pitch(pitch):
    w, h = 640, 320
    x, y = direction_to_equirect_px(30.0, pitch, w, h)
    _, pitch2 = equirect_px_to_direction(x, y, w, h)
    assert abs(pitch2 - pitch) < 1e-9


def test_canonical_cardinal_directions_pixel_positions():
    """yaw0 = centre column, +yaw = right half, +pitch = upper half."""
    w, h = 640, 320
    x0, _ = direction_to_equirect_px(0.0, 0.0, w, h)
    x90, _ = direction_to_equirect_px(90.0, 0.0, w, h)
    xm90, _ = direction_to_equirect_px(-90.0, 0.0, w, h)
    _, yup = direction_to_equirect_px(0.0, 45.0, w, h)
    assert abs(x0 - (w / 2 - 0.5)) < 1e-9
    assert x90 > x0 and abs(x90 - (w * 0.75 - 0.5)) < 1e-9
    assert xm90 < x0 and abs(xm90 - (w * 0.25 - 0.5)) < 1e-9
    assert yup < h / 2 - 0.5  # up = smaller row index


def test_poles_map_to_top_and_bottom_rows():
    w, h = 640, 320
    _, y_north = direction_to_equirect_px(0.0, 90.0, w, h)
    _, y_south = direction_to_equirect_px(0.0, -90.0, w, h)
    assert y_north <= 0.0
    assert y_south >= h - 1


@pytest.mark.parametrize("w,h", [(1920, 1080), (1080, 1920), (1000, 1000), (640, 320)])
def test_vfov_hfov_aspect(w, h):
    hfov = 90.0
    vfov = vfov_from_hfov(hfov, w, h)
    assert 0 < vfov < 180
    # pinhole: tan(h/2)*h/w == tan(v/2)
    assert math.isclose(
        math.tan(math.radians(hfov / 2)) * h / w,
        math.tan(math.radians(vfov / 2)),
        rel_tol=1e-12,
    )
    assert math.isclose(hfov_from_vfov(vfov, w, h), hfov, rel_tol=1e-9)


def test_vfov_16x9_value():
    # documented reference value for the default export canvas
    assert math.isclose(vfov_from_hfov(90.0, 1920, 1080), 58.71550, abs_tol=1e-4)


def test_vfov_rejects_degenerate():
    with pytest.raises(ValueError):
        vfov_from_hfov(0.0, 100, 100)
    with pytest.raises(ValueError):
        vfov_from_hfov(180.0, 100, 100)
    with pytest.raises(ValueError):
        vfov_from_hfov(90.0, 0, 100)


def test_v360_filter_string_carries_both_fovs():
    view = CameraView(yaw_deg=10.0, pitch_deg=-20.0, roll_deg=0.0, h_fov_deg=80.0)
    filt = v360_filter(view, 1920, 1080)
    assert filt.startswith("v360=equirect:flat:")
    for part in ("w=1920", "h=1080", "yaw=10.000000", "pitch=-20.000000", "roll=0.000000", "h_fov=80.000000"):
        assert part in filt
    vfov = vfov_from_hfov(80.0, 1920, 1080)
    assert f"v_fov={vfov:.6f}" in filt


def test_static_camera_path_requires_single_key_at_zero():
    view = CameraView()
    path = CameraPath(schema_version=CONTRACT_VERSION, keys=((0, view),))
    assert path.static_view is view
    two = CameraPath(schema_version=CONTRACT_VERSION, keys=((0, view), (1, view)))
    with pytest.raises(ContractError):
        _ = two.static_view  # exactly one key required for static export
    with pytest.raises(ContractError):
        _ = CameraPath(schema_version=CONTRACT_VERSION, keys=((Fraction(1), view),)).static_view


def test_camera_provenance_semantics():
    view = CameraView()
    user = CameraPath(schema_version=CONTRACT_VERSION, keys=((0, view),), provenance="user")
    derived = CameraPath(schema_version=CONTRACT_VERSION, keys=((0, view),), provenance="derived")
    assert user.provenance == "user" and derived.provenance == "derived"
    with pytest.raises(ContractError):
        CameraPath(schema_version=CONTRACT_VERSION, keys=((0, view),), provenance="guessed")
    assert CameraPath.from_json(user.to_json()).provenance == "user"


def test_edit_plan_validation():
    view = CameraView()
    cam = CameraPath(schema_version=CONTRACT_VERSION, keys=((0, view),))
    plan = EditPlan(
        schema_version=CONTRACT_VERSION, source_id="v1:abc", cut_in_rel=Fraction(0),
        cut_out_rel=Fraction("1/2"), project_fps=Fraction(30000, 1001), camera=cam,
    )
    assert plan.duration_rel == Fraction(1, 2)
    assert plan.cut_manual is True  # manual cut locks are the default and survive serialization
    with pytest.raises(ContractError):
        EditPlan(schema_version=CONTRACT_VERSION, source_id="v1:abc", cut_in_rel=Fraction(1),
                 cut_out_rel=Fraction(1), project_fps=Fraction(30), camera=cam)
    with pytest.raises(ContractError):
        EditPlan(schema_version=CONTRACT_VERSION, source_id="v1:abc", cut_in_rel=Fraction(-1),
                 cut_out_rel=Fraction(1), project_fps=Fraction(30), camera=cam)
    with pytest.raises(ContractError):
        EditPlan(schema_version=1, source_id="v1:abc", cut_in_rel=Fraction(0),
                 cut_out_rel=Fraction(1), project_fps=Fraction(30), camera=cam)  # stale schema version
