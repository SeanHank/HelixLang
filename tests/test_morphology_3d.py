"""3D L-system morphogenesis unit tests."""
import math

import pytest

from helixlang.plugins.runtime.morphology_3d import (
    PLANT_PRESETS,
    Line3D,
    LSystem3D,
    Point3D,
    rotate_vector,
)


# -------- Initial state --------
def test_initial_turtle_state():
    ls = LSystem3D(axiom="F", rules={}, angle=90.0, step=1.0)
    t = ls._initial_turtle()
    assert t.position == Point3D(0.0, 0.0, 0.0)
    assert t.heading == Point3D(0.0, 1.0, 0.0)   # +Y forward
    assert t.left == Point3D(-1.0, 0.0, 0.0)    # -X left
    assert t.up == Point3D(0.0, 0.0, 1.0)       # +Z up
    assert t.pen_down is True
    assert t.line_width == 1.0


def test_initial_orthonormal_basis():
    # H x L should equal U (right-handed system)
    ls = LSystem3D(axiom="F", rules={}, angle=90.0)
    t = ls._initial_turtle()
    cross = t.heading.cross(t.left)
    assert cross.x == pytest.approx(t.up.x)
    assert cross.y == pytest.approx(t.up.y)
    assert cross.z == pytest.approx(t.up.z)


# -------- Forward drawing --------
def test_forward_draw_line():
    ls = LSystem3D(axiom="F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 1
    line = lines[0]
    assert line.start == Point3D(0.0, 0.0, 0.0)
    assert line.end == Point3D(0.0, 1.0, 0.0)  # heading +Y
    assert line.width == 1.0


def test_forward_no_draw():
    # f moves forward without drawing; F draws
    ls = LSystem3D(axiom="fF", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 1
    assert lines[0].start == Point3D(0.0, 1.0, 0.0)  # after f moves
    assert lines[0].end == Point3D(0.0, 2.0, 0.0)


def test_step_scales_distance():
    ls = LSystem3D(axiom="F", rules={}, angle=90.0, step=2.5)
    lines = ls.draw(0)
    assert lines[0].end == Point3D(0.0, 2.5, 0.0)


# -------- yaw rotation +/-
def test_yaw_right():
    # +90 yaw turn right: heading +Y -> -X
    ls = LSystem3D(axiom="+F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 1
    end = lines[0].end
    assert end.x == pytest.approx(-1.0)
    assert end.y == pytest.approx(0.0, abs=1e-9)
    assert end.z == pytest.approx(0.0, abs=1e-9)


def test_yaw_left():
    # -90 yaw turn left: heading +Y -> +X
    ls = LSystem3D(axiom="-F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 1
    end = lines[0].end
    assert end.x == pytest.approx(1.0)
    assert end.y == pytest.approx(0.0, abs=1e-9)
    assert end.z == pytest.approx(0.0, abs=1e-9)


def test_yaw_round_trip():
    # + then - should return to the original heading
    ls = LSystem3D(axiom="+-F", rules={}, angle=45.0, step=1.0)
    lines = ls.draw(0)
    assert lines[0].end == Point3D(0.0, 1.0, 0.0)


# -------- pitch rotation &/^
def test_pitch_down():
    # & +90 pitch turn down: heading +Y -> -Z
    ls = LSystem3D(axiom="&F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 1
    end = lines[0].end
    assert end.x == pytest.approx(0.0, abs=1e-9)
    assert end.y == pytest.approx(0.0, abs=1e-9)
    assert end.z == pytest.approx(-1.0)


def test_pitch_up():
    # ^ -90 pitch turn up: heading +Y -> +Z
    ls = LSystem3D(axiom="^F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 1
    end = lines[0].end
    assert end.x == pytest.approx(0.0, abs=1e-9)
    assert end.y == pytest.approx(0.0, abs=1e-9)
    assert end.z == pytest.approx(1.0)


# -------- roll rotation \/
def test_roll_left_then_pitch():
    # \ +90 roll left: L,U rotate around H, then & +90 pitch
    # after roll L=(0,0,1), U=(1,0,0); pitch & rotates +90 around the new L=(0,0,1)
    # H=(0,1,0) rotated +90 around (0,0,1) -> (-1,0,0)
    ls = LSystem3D(axiom="\\&F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 1
    end = lines[0].end
    assert end.x == pytest.approx(-1.0)
    assert end.y == pytest.approx(0.0, abs=1e-9)
    assert end.z == pytest.approx(0.0, abs=1e-9)


def test_roll_right_then_pitch():
    # / -90 roll right: L,U rotate around H, then & +90 pitch
    # after roll L=(0,0,-1), U=(-1,0,0); pitch & rotates +90 around the new L=(0,0,-1)
    # H=(0,1,0) rotated +90 around (0,0,-1) -> (1,0,0)
    ls = LSystem3D(axiom="/&F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 1
    end = lines[0].end
    assert end.x == pytest.approx(1.0)
    assert end.y == pytest.approx(0.0, abs=1e-9)
    assert end.z == pytest.approx(0.0, abs=1e-9)


def test_roll_does_not_change_heading():
    # After a standalone roll, F still goes along +Y
    ls = LSystem3D(axiom="\\F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert lines[0].end == Point3D(0.0, 1.0, 0.0)


# -------- Stack-based branching []
def test_branch_stack():
    # F[+F]F: forward -> branch (yaw+F) -> restore -> forward
    ls = LSystem3D(axiom="F[+F]F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 3
    # Segment 1: (0,0,0) -> (0,1,0)
    assert lines[0].start == Point3D(0.0, 0.0, 0.0)
    assert lines[0].end == Point3D(0.0, 1.0, 0.0)
    # Branch segment: from (0,1,0) yaw right 90 -> (-1,1,0)
    assert lines[1].start == Point3D(0.0, 1.0, 0.0)
    assert lines[1].end.x == pytest.approx(-1.0)
    assert lines[1].end.y == pytest.approx(1.0)
    # After restore: from (0,1,0) along +Y -> (0,2,0)
    assert lines[2].start == Point3D(0.0, 1.0, 0.0)
    assert lines[2].end == Point3D(0.0, 2.0, 0.0)


def test_nested_branch():
    ls = LSystem3D(axiom="F[F[F]F]F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 5


def test_empty_stack_pop_is_safe():
    # Extra ] should not crash
    ls = LSystem3D(axiom="F]F", rules={}, angle=90.0, step=1.0)
    lines = ls.draw(0)
    assert len(lines) == 2


# -------- L-system rewriting --------
def test_derive_no_iteration():
    ls = LSystem3D(axiom="F", rules={"F": "FF"}, angle=25.0)
    assert ls.derive(0) == "F"


def test_derive_growth():
    ls = LSystem3D(axiom="F", rules={"F": "FF"}, angle=25.0)
    assert ls.derive(1) == "FF"
    assert ls.derive(2) == "FFFF"
    assert ls.derive(3) == "FFFFFFFF"


def test_derive_branching_rule():
    ls = LSystem3D(axiom="F", rules={"F": "F[+F]F[-F]F"}, angle=25.0)
    assert ls.derive(1) == "F[+F]F[-F]F"


def test_derive_multi_symbol_rule():
    ls = LSystem3D(axiom="A", rules={"A": "F+A", "F": "FF"}, angle=30.0)
    assert ls.derive(1) == "F+A"
    assert ls.derive(2) == "FF+F+A"


def test_derive_unknown_symbol_unchanged():
    ls = LSystem3D(axiom="FXF", rules={"F": "FF"}, angle=25.0)
    assert ls.derive(1) == "FFXFF"


# -------- 3D tree generation --------
def test_tree3d_generation():
    p = PLANT_PRESETS["tree3d"]
    ls = LSystem3D(axiom=p["axiom"], rules=p["rules"],
                  angle=p["angle"], step=p["step"])
    lines = ls.draw(3)
    assert len(lines) > 0
    points = ls.get_points(3)
    xs = [pt.x for pt in points]
    ys = [pt.y for pt in points]
    zs = [pt.z for pt in points]
    # Should have 3D lateral spread (roll takes branches out of the XY plane)
    assert max(zs) - min(zs) > 0 or max(xs) - min(xs) > 0
    # Should grow upward
    assert max(ys) > 0


def test_tree3d_more_iterations_more_lines():
    p = PLANT_PRESETS["tree3d"]
    ls = LSystem3D(axiom=p["axiom"], rules=p["rules"],
                  angle=p["angle"], step=p["step"])
    n1 = len(ls.draw(1))
    n2 = len(ls.draw(2))
    n3 = len(ls.draw(3))
    assert n2 > n1
    assert n3 > n2


# -------- Bounds calculation --------
def test_bounds_basic():
    ls = LSystem3D(axiom="F+F+F", rules={}, angle=90.0, step=1.0)
    bounds = ls.get_bounds(0)
    # Rotation introduces floating-point error; compare with approx
    assert bounds["min"].x == pytest.approx(-1.0)
    assert bounds["min"].y == pytest.approx(0.0)
    assert bounds["max"].y == pytest.approx(1.0)
    assert bounds["size"].x == pytest.approx(1.0)
    assert bounds["size"].y == pytest.approx(1.0)
    assert bounds["size"].z == pytest.approx(0.0)
    assert bounds["center"].x == pytest.approx(-0.5)
    assert bounds["center"].y == pytest.approx(0.5)


def test_bounds_3d_extent():
    ls = LSystem3D(axiom="F^F&F", rules={}, angle=90.0, step=1.0)
    bounds = ls.get_bounds(0)
    assert bounds["size"].z > 0


def test_bounds_returns_dict_keys():
    ls = LSystem3D(axiom="F", rules={}, angle=25.0)
    bounds = ls.get_bounds(0)
    assert set(bounds.keys()) == {"min", "max", "center", "size"}
    assert isinstance(bounds["min"], Point3D)
    assert isinstance(bounds["max"], Point3D)


# -------- Plant presets --------
def test_presets_keys():
    assert set(PLANT_PRESETS.keys()) >= {"fern", "tree3d", "bush", "algae"}


def test_presets_structure():
    for name, p in PLANT_PRESETS.items():
        assert "axiom" in p, f"{name} missing axiom"
        assert "rules" in p, f"{name} missing rules"
        assert "angle" in p, f"{name} missing angle"
        assert isinstance(p["rules"], dict)


def test_presets_produce_lines():
    for name in PLANT_PRESETS:
        p = PLANT_PRESETS[name]
        ls = LSystem3D(axiom=p["axiom"], rules=p["rules"],
                       angle=p["angle"], step=p.get("step", 1.0))
        lines = ls.draw(2)
        assert isinstance(lines, list)
        assert all(isinstance(ln, Line3D) for ln in lines)


def test_fern_preset_2d_in_xy_plane():
    # Ferns only use the +/- symbols, so they should stay in the XY plane (z always 0)
    p = PLANT_PRESETS["fern"]
    ls = LSystem3D(axiom=p["axiom"], rules=p["rules"],
                   angle=p["angle"], step=p["step"])
    points = ls.get_points(2)
    assert all(abs(pt.z) < 1e-9 for pt in points)


# -------- rotate_vector unit tests --------
def test_rotate_vector_around_z():
    v = Point3D(1.0, 0.0, 0.0)
    axis = Point3D(0.0, 0.0, 1.0)
    r = rotate_vector(v, axis, math.pi / 2)
    assert r.x == pytest.approx(0.0, abs=1e-9)
    assert r.y == pytest.approx(1.0)
    assert r.z == pytest.approx(0.0, abs=1e-9)


def test_rotate_vector_preserves_length():
    v = Point3D(1.0, 2.0, 3.0)
    axis = Point3D(0.0, 1.0, 0.0)
    r = rotate_vector(v, axis, 0.7)
    assert r.norm() == pytest.approx(v.norm())


def test_rotate_vector_zero_angle_identity():
    v = Point3D(1.5, -2.0, 0.7)
    r = rotate_vector(v, Point3D(0.0, 0.0, 1.0), 0.0)
    assert r.x == pytest.approx(v.x)
    assert r.y == pytest.approx(v.y)
    assert r.z == pytest.approx(v.z)
