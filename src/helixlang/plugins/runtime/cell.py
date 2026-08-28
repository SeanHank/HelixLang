"""Cell state: the VM runtime context.

.. note::
   **Units.** Energy is stored in **ATP molecule counts** (whole-cell
   convention; Karr et al. 2012).  1 energy count = 1 ATP molecule.
   A newborn cell holds ~10^9 ATP (Orth 2010 maintenance flux 8.39
   mmol/gDW/h x ~0.3 pg dry weight x 6e23 / 60 ~= 2.5e7 ATP/min,
   Alberts).  Signals in :mod:`helixlang.plugins.runtime.population` are µM
   concentrations and metabolic fluxes in :mod:`helixlang.plugins.runtime.metabolism`
   are mmol/gDW/h — both cited at their points of use.  All quantities
   are registered as named constants below so future recalibration
   against a quantitative model is a one-line edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Directions: 0=N, 1=E, 2=S, 3=W (4-neighborhood, von Neumann)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# ============================================================================
# Physical constant registry
# ============================================================================

#: initial cell energy budget (ATP molecules; ~10^9 ATP, Orth 2010 +
#: Alberts dry mass ~0.3 pg)
INITIAL_CELL_ENERGY = 1.0e9
#: number of protein slots per cell (symbolic capacity; E. coli ~4300 genes)
CELL_PROTEIN_SLOT_COUNT = 256
#: energy consumed by one move step (ATP; flagellar motor ~10^3-10^4 ATP/rev)
MOVE_ENERGY_COST = 1.0e7
#: default feed amount added by :meth:`Cell.feed` (ATP; aerobic glucose
#: uptake -> ATP, ATP_PER_GLUCOSE = 38)
FEED_ENERGY_AMOUNT = 1.0e8
#: minimum energy required to divide (ATP; symmetric division halves energy)
MIN_DIVISION_ENERGY = 2.0e7
#: default cell color (white, RGB)
DEFAULT_CELL_COLOR: tuple[int, int, int] = (255, 255, 255)
#: membrane permeability scale: 0 = impermeable, MAX = fully permeable
MAX_MEMBRANE_PERMEABILITY = 255
#: default membrane permeability. Fully permeable by default so the
#: unmodeled-membrane ``feed`` behavior is preserved exactly; the
#: ``OP_BUILD_MEMBRANE`` opcode lowers it, which scales nutrient intake.
DEFAULT_MEMBRANE_PERMEABILITY = MAX_MEMBRANE_PERMEABILITY

#: physical-unit axis summary (see module docstring)
UNITS: dict[str, str] = {
    "energy": "ATP molecules (newborn ~1e9; Orth 2010 + Alberts dry mass)",
    "signals": "µM concentrations (e.g. Xavier 2003: 10 µM AI-2 quorum "
               "threshold in helixlang.plugins.runtime.population)",
    "neighborhood": "4-connected von Neumann lattice (site edge = "
                    "LATTICE_SPACING_UM µm)",
}


@dataclass(slots=True)
class Cell:
    """Single-cell state.

    Args:
        name: cell identifier
        x, y: lattice position (site edge = 10 µm)
        energy: energy budget in ATP molecules (default ~10^9 ATP,
            Orth 2010)
        proteins: protein molecule counts by kind/name
        slots: protein slot storage (CELL_PROTEIN_SLOT_COUNT slots)
        membrane_permeability: 0 (impermeable) .. MAX (fully permeable)
    """

    name: str = "cell-0"
    x: int = 0
    y: int = 0
    energy: float = INITIAL_CELL_ENERGY
    proteins: dict[int | str, float] = field(default_factory=dict)
    slots: list = field(default_factory=lambda: [None] * CELL_PROTEIN_SLOT_COUNT)
    alive: bool = True
    morphology_points: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0)])
    color: tuple[int, int, int] = DEFAULT_CELL_COLOR
    age: int = 0
    divisions: int = 0
    membrane_permeability: int = DEFAULT_MEMBRANE_PERMEABILITY

    def add_protein(self, kind: int | str, amount: float = 1.0) -> None:
        self.proteins[kind] = self.proteins.get(kind, 0.0) + amount

    def consume_protein(self, kind: int | str, amount: float = 1.0) -> float:
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

    def consume_energy(self, n: float = 1.0) -> bool:
        if self.energy >= n:
            self.energy -= n
            return True
        return False

    def feed(self, amount: float = FEED_ENERGY_AMOUNT) -> None:
        """Energy intake across the membrane, scaled by permeability.

        ``energy += round(amount * permeability / MAX)``: a fully
        permeable membrane (the default) gains the full ``amount``; an
        impermeable one gains nothing.
        """
        self.energy += round(
            amount * self.membrane_permeability / MAX_MEMBRANE_PERMEABILITY
        )

    def set_membrane_permeability(self, value: int) -> None:
        """Set membrane permeability, clamped to ``[0, MAX_MEMBRANE_PERMEABILITY]``.

        The value models how readily nutrients pass the cell membrane
        (0 = impermeable, ``MAX_MEMBRANE_PERMEABILITY`` = fully permeable).
        """
        self.membrane_permeability = max(
            0, min(MAX_MEMBRANE_PERMEABILITY, int(value)))

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
                f"permeability={self.membrane_permeability} "
                f"proteins={self.proteins} color={self.color})")
