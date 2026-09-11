"""L-system unit tests."""
import pytest

from helixlang.plugins.runtime.lsystem import LSystem


def test_simple_iteration():
    ls = LSystem(axiom="F", rules={"F": "FF"}, angle=25)
    assert ls.state == "F"
    ls.iterate()
    assert ls.state == "FF"
    ls.iterate()
    assert ls.state == "FFFF"


def test_branching():
    ls = LSystem(axiom="F", rules={"F": "F[+F]F[-F]F"}, angle=25)
    ls.iterate()
    assert ls.state == "F[+F]F[-F]F"


def test_turtle_points():
    ls = LSystem(axiom="F", rules={"F": "FF"}, angle=25, step=1.0)
    pts = ls.iterate()
    # F->FF: two F's, turtle advances 2 times, adds 2 points
    assert len(pts) == 2
    # The first point should be near (0, 1) (pointing up at 90 degrees)
    assert pts[0][1] == pytest.approx(1.0)


def test_state_length_grows():
    ls = LSystem(axiom="F", rules={"F": "F[+F]F[-F]F"}, angle=25)
    lengths = [len(ls.state)]
    for _ in range(3):
        ls.iterate()
        lengths.append(len(ls.state))
    # Length should grow monotonically
    assert lengths == sorted(lengths)
    assert lengths[-1] > lengths[0]


def test_branch_stack():
    ls = LSystem(axiom="F", rules={"F": "F[+F]-F"}, angle=90, step=1.0)
    pts = ls.iterate()
    # Should produce multiple points
    assert len(pts) >= 2


def test_branch_stack_pop_restores():
    """A ']' with a non-empty stack pops and restores position/heading."""
    from helixlang.plugins.runtime.lsystem import LSystem

    ls = LSystem(axiom="F[+F]F", rules={}, angle=90, step=1.0)
    pts = ls.iterate()
    # Trunk (0,0)->(0,1), branch (0,1)->(-1,1), restore, trunk (0,1)->(0,2)
    assert len(ls.points) == 4
    assert len(pts) == 3
    assert pts[0][0] == pytest.approx(0.0, abs=1e-9)
    assert pts[0][1] == pytest.approx(1.0)
    assert pts[1][0] == pytest.approx(-1.0, abs=1e-9)
    assert pts[1][1] == pytest.approx(1.0)
    assert pts[2][0] == pytest.approx(0.0, abs=1e-9)
    assert pts[2][1] == pytest.approx(2.0)


def test_branch_stack_empty_pop_is_safe():
    """A ']' with an empty stack does not crash (defensive branch)."""
    from helixlang.plugins.runtime.lsystem import LSystem

    ls = LSystem(axiom="]F", rules={}, angle=90, step=1.0)
    pts = ls.iterate()
    assert pts[0][1] == pytest.approx(1.0)


def test_state_length():
    from helixlang.plugins.runtime.lsystem import LSystem

    ls = LSystem(axiom="F[+F]", rules={}, angle=25)
    assert ls.state_length() == len("F[+F]")
    ls.iterate()
    assert ls.state_length() == len(ls.state)
