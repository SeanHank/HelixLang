"""L-system morphogenesis: parallel string rewriting + turtle graphics interpretation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class TurtleState:
    x: float
    y: float
    heading: float
    stack: list[tuple[float, float, float]] = field(default_factory=list)


class LSystem:
    """Parametric L-system."""

    def __init__(self, axiom: str, rules: dict[str, str],
                 angle: float = 25.0, step: float = 1.0):
        self.axiom = axiom
        self.rules = rules
        self.angle = angle
        self.step = step
        self.state = axiom
        self.turtle = TurtleState(0.0, 0.0, 90.0)  # facing up
        self.points: list[tuple[float, float]] = [(0.0, 0.0)]
        self.iteration = 0

    def iterate(self) -> list[tuple[float, float]]:
        """One parallel rewrite + turtle interpretation, returning the new points."""
        self.state = ''.join(self.rules.get(c, c) for c in self.state)
        self.iteration += 1
        return self._interpret()

    def _interpret(self) -> list[tuple[float, float]]:
        new_pts: list[tuple[float, float]] = []
        # Reset the turtle to the origin and re-interpret (so the morphology is reproducible)
        self.turtle = TurtleState(0.0, 0.0, 90.0)
        self.points = [(0.0, 0.0)]
        for c in self.state:
            if c == 'F':
                rad = math.radians(self.turtle.heading)
                self.turtle.x += self.step * math.cos(rad)
                self.turtle.y += self.step * math.sin(rad)
                self.points.append((self.turtle.x, self.turtle.y))
                new_pts.append((self.turtle.x, self.turtle.y))
            elif c == '+':
                self.turtle.heading += self.angle
            elif c == '-':
                self.turtle.heading -= self.angle
            elif c == '[':
                self.turtle.stack.append(
                    (self.turtle.x, self.turtle.y, self.turtle.heading))
            elif c == ']':
                if self.turtle.stack:
                    x, y, h = self.turtle.stack.pop()
                    self.turtle.x, self.turtle.y, self.turtle.heading = x, y, h
        return new_pts

    def state_length(self) -> int:
        return len(self.state)
