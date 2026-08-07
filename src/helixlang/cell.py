"""Cell state: the VM runtime context.

.. note::
   **Units disclaimer.** The energy / slot / neighborhood quantities
   below are *gameplay units*, not physical units: they define a
   dimensionless cellular automaton budget, not Joules or molecules.
   Physical quantities used elsewhere in HelixLang (µM signals in
   :mod:`helixlang.population`, metabolic fluxes in
   :mod:`helixlang.metabolism`) are cited at their points of use.  All
   magic numbers are registered as named constants below so future
   calibration against a quantitative model is a one-line edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Directions: 0=N, 1=E, 2=S, 3=W (4-neighborhood, von Neumann)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# ============================================================================
# Gameplay-unit constant registry
# ============================================================================
# All values are dimensionless gameplay units (see module note).  No
# physical measurement is claimed for these defaults.

#: initial cell energy budget (gameplay units)
INITIAL_CELL_ENERGY = 100
#: number of protein slots per cell (gameplay model capacity)
CELL_PROTEIN_SLOT_COUNT = 256
#: energy consumed by one move step (gameplay cost)
MOVE_ENERGY_COST = 1
#: default feed amount added by :meth:`Cell.feed`
FEED_ENERGY_AMOUNT = 10
#: minimum energy required to divide (symmetric division halves energy)
MIN_DIVISION_ENERGY = 2
#: default cell color (white, RGB)
DEFAULT_CELL_COLOR: tuple[int, int, int] = (255, 255, 255)

#: gameplay-unit axis summary (see module docstring)
UNITS: dict[str, str] = {
    "energy": "gameplay units (not Joules); threshold calibration is a "
              "one-line edit via INITIAL_CELL_ENERGY",
    "signals": "physical where cited (e.g. Xavier 2003: 10 uM AI-2 in "
               "helixlang.population); dimensionless elsewhere",
    "neighborhood": "4-connected von Neumann lattice",
}


@dataclass(slots=True)
class Cell:
    """Single-cell state."""

    name: str = "cell-0"
    x: int = 0
    y: int = 0
    energy: int = INITIAL_CELL_ENERGY
    proteins: dict[int | str, float] = field(default_factory=dict)
    slots: list = field(default_factory=lambda: [None] * CELL_PROTEIN_SLOT_COUNT)
    alive: bool = True
    morphology_points: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0)])
    color: tuple[int, int, int] = DEFAULT_CELL_COLOR
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
            self.energy -= MOVE_ENERGY_COST

    def consume_energy(self, n: int = 1) -> bool:
        if self.energy >= n:
            self.energy -= n
            return True
        return False

    def feed(self, amount: int = FEED_ENERGY_AMOUNT) -> None:
        self.energy += amount

    def divide(self) -> bool:
        """Symmetric division: halve the energy, return whether it succeeded."""
        if self.energy < MIN_DIVISION_ENERGY:
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
