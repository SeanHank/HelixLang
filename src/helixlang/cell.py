"""Cell state: the VM runtime context."""
from __future__ import annotations

from dataclasses import dataclass, field

# Directions: 0=N, 1=E, 2=S, 3=W
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


@dataclass(slots=True)
class Cell:
    """Single-cell state."""

    name: str = "cell-0"
    x: int = 0
    y: int = 0
    energy: int = 100
    proteins: dict[int | str, float] = field(default_factory=dict)
    slots: list = field(default_factory=lambda: [None] * 256)
    alive: bool = True
    morphology_points: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0)])
    color: tuple[int, int, int] = (255, 255, 255)
    age: int = 0
    divisions: int = 0

    def add_protein(self, kind: int, amount: float = 1.0) -> None:
        self.proteins[kind] = self.proteins.get(kind, 0.0) + amount

    def consume_protein(self, kind: int, amount: float = 1.0) -> float:
        """Consume a protein and return the amount actually consumed."""
        avail = self.proteins.get(kind, 0.0)
        consumed = min(avail, amount)
        self.proteins[kind] = avail - consumed
        if self.proteins[kind] <= 0.0:
            self.proteins.pop(kind, None)
        return consumed

    def move(self, direction: int) -> None:
        dx, dy = DIRECTIONS[direction % 4]
        self.x += dx
        self.y += dy
        if self.energy > 0:
            self.energy -= 1

    def consume_energy(self, n: int = 1) -> bool:
        if self.energy >= n:
            self.energy -= n
            return True
        return False

    def feed(self, amount: int = 10) -> None:
        self.energy += amount

    def divide(self) -> bool:
        """Symmetric division: halve the energy, return whether it succeeded."""
        if self.energy < 2:
            return False
        self.energy //= 2
        self.divisions += 1
        return True

    def die(self) -> None:
        self.alive = False

    def dump(self) -> str:
        return (f"Cell({self.name} pos=({self.x},{self.y}) "
                f"energy={self.energy} alive={self.alive} "
                f"proteins={self.proteins} color={self.color})")
