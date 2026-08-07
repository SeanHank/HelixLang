"""3D L-system morphogenesis.

Extends the 2D L-system to 3D, supporting pitch/yaw/roll rotations.

Symbol extensions:
- F: move forward + draw line
- f: move forward without drawing
- +: yaw right (+angle)
- -: yaw left (-angle)
- &: pitch down (+angle)
- ^: pitch up (-angle)
- \\: roll left (+angle)
- /: roll right (-angle)
- [ ]: stack-based branching

Based on:
- Prusinkiewicz & Lindenmayer 1990 "The Algorithmic Beauty of Plants"
- 3D turtle geometry using Euler angles or rotation matrices
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class Point3D:
    """3D point / vector."""
    x: float
    y: float
    z: float

    def dot(self, other: Point3D) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Point3D) -> Point3D:
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def scale(self, s: float) -> Point3D:
        return Point3D(self.x * s, self.y * s, self.z * s)

    def add(self, other: Point3D) -> Point3D:
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def norm(self) -> float:
        return math.sqrt(self.dot(self))


@dataclass(slots=True)
class TurtleState3D:
    """3D turtle state: position + three orthogonal basis vectors (H, L, U)."""
    position: Point3D
    heading: Point3D  # forward direction vector H
    left: Point3D     # left direction vector L
    up: Point3D       # up direction vector U
    pen_down: bool = True
    line_width: float = 1.0


@dataclass(slots=True)
class Line3D:
    """3D line segment."""
    start: Point3D
    end: Point3D
    width: float


def _normalize(v: Point3D) -> Point3D:
    n = v.norm()
    if n == 0.0:
        return Point3D(0.0, 0.0, 0.0)
    return Point3D(v.x / n, v.y / n, v.z / n)


def rotate_vector(v: Point3D, axis: Point3D, angle: float) -> Point3D:
    """Rotate vector v around axis by angle radians (Rodrigues' formula)."""
    a = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    cross = a.cross(v)
    dot = a.dot(v)
    return Point3D(
        v.x * c + cross.x * s + a.x * dot * (1 - c),
        v.y * c + cross.y * s + a.y * dot * (1 - c),
        v.z * c + cross.z * s + a.z * dot * (1 - c),
    )


class LSystem3D:
    """3D L-system: parallel string rewriting + 3D turtle drawing."""

    def __init__(self, axiom: str, rules: dict[str, str],
                 angle: float = 22.5, step: float = 1.0):
        self.axiom = axiom
        self.rules = rules
        self.angle = angle  # degrees
        self.step = step

    def _initial_turtle(self) -> TurtleState3D:
        # Right-handed coordinate system: H × L = U
        # H=+Y (forward), L=-X (left), U=+Z (up)
        return TurtleState3D(
            position=Point3D(0.0, 0.0, 0.0),
            heading=Point3D(0.0, 1.0, 0.0),
            left=Point3D(-1.0, 0.0, 0.0),
            up=Point3D(0.0, 0.0, 1.0),
        )

    def derive(self, iterations: int) -> str:
        """Parallel rewriting over N generations."""
        state = self.axiom
        for _ in range(iterations):
            state = ''.join(self.rules.get(c, c) for c in state)
        return state

    def draw(self, iterations: int) -> list[Line3D]:
        """3D turtle drawing, returns the list of line segments."""
        state = self.derive(iterations)
        turtle = self._initial_turtle()
        stack: list[TurtleState3D] = []
        lines: list[Line3D] = []
        rad = math.radians(self.angle)
        for c in state:
            if c == 'F':
                new_pos = turtle.position.add(turtle.heading.scale(self.step))
                if turtle.pen_down:
                    lines.append(Line3D(
                        start=Point3D(turtle.position.x, turtle.position.y,
                                      turtle.position.z),
                        end=Point3D(new_pos.x, new_pos.y, new_pos.z),
                        width=turtle.line_width,
                    ))
                turtle.position = new_pos
            elif c == 'f':
                turtle.position = turtle.position.add(
                    turtle.heading.scale(self.step))
            elif c == '+':  # yaw right: H, L rotate around U
                turtle.heading = rotate_vector(turtle.heading, turtle.up, rad)
                turtle.left = rotate_vector(turtle.left, turtle.up, rad)
            elif c == '-':  # yaw left: H, L rotate around U
                turtle.heading = rotate_vector(turtle.heading, turtle.up, -rad)
                turtle.left = rotate_vector(turtle.left, turtle.up, -rad)
            elif c == '&':  # pitch down: H, U rotate around L
                turtle.heading = rotate_vector(turtle.heading, turtle.left, rad)
                turtle.up = rotate_vector(turtle.up, turtle.left, rad)
            elif c == '^':  # pitch up: H, U rotate around L
                turtle.heading = rotate_vector(turtle.heading, turtle.left, -rad)
                turtle.up = rotate_vector(turtle.up, turtle.left, -rad)
            elif c == '\\':  # roll left: L, U rotate around H
                turtle.left = rotate_vector(turtle.left, turtle.heading, rad)
                turtle.up = rotate_vector(turtle.up, turtle.heading, rad)
            elif c == '/':  # roll right: L, U rotate around H
                turtle.left = rotate_vector(turtle.left, turtle.heading, -rad)
                turtle.up = rotate_vector(turtle.up, turtle.heading, -rad)
            elif c == '[':
                stack.append(TurtleState3D(
                    position=Point3D(turtle.position.x, turtle.position.y,
                                     turtle.position.z),
                    heading=Point3D(turtle.heading.x, turtle.heading.y,
                                   turtle.heading.z),
                    left=Point3D(turtle.left.x, turtle.left.y, turtle.left.z),
                    up=Point3D(turtle.up.x, turtle.up.y, turtle.up.z),
                    pen_down=turtle.pen_down,
                    line_width=turtle.line_width,
                ))
            elif c == ']':
                if stack:
                    turtle = stack.pop()
        return lines

    def get_points(self, iterations: int) -> list[Point3D]:
        """Return all vertices."""
        lines = self.draw(iterations)
        points: list[Point3D] = [Point3D(0.0, 0.0, 0.0)]
        for line in lines:
            points.append(line.end)
        return points

    def get_bounds(self, iterations: int) -> dict:
        """Return the 3D bounds {min, max, center, size}."""
        points = self.get_points(iterations)
        minx = min(p.x for p in points)
        maxx = max(p.x for p in points)
        miny = min(p.y for p in points)
        maxy = max(p.y for p in points)
        minz = min(p.z for p in points)
        maxz = max(p.z for p in points)
        return {
            "min": Point3D(minx, miny, minz),
            "max": Point3D(maxx, maxy, maxz),
            "center": Point3D((minx + maxx) / 2, (miny + maxy) / 2,
                              (minz + maxz) / 2),
            "size": Point3D(maxx - minx, maxy - miny, maxz - minz),
        }


# Preset plant morphologies
PLANT_PRESETS: dict[str, dict] = {
    "fern": {
        # Fern (2D converted to 3D, staying in the XY plane)
        "axiom": "X",
        "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
        "angle": 22.5,
        "step": 1.0,
    },
    "tree3d": {
        # 3D tree: four-way branching + upward growth
        "axiom": "F",
        "rules": {"F": "F[\\F][/F][&F][^F]F"},
        "angle": 25.0,
        "step": 1.0,
    },
    "bush": {
        # Bush: dense binary branching
        "axiom": "F",
        "rules": {"F": "FF-[-F+F+F]+[+F-F-F]"},
        "angle": 22.5,
        "step": 1.0,
    },
    "algae": {
        # Algae: 3D repeated branching
        "axiom": "A",
        "rules": {"A": "F[+A][-A]F[^A][&A]A", "F": "FF"},
        "angle": 30.0,
        "step": 1.0,
    },
}
