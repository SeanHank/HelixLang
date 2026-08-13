"""Synthetic microbial consortium: quorum consensus + composition control (S1).

A synthetic consortium of three differentiated roles on a 2-D lattice:

- **producer** -- constitutively secretes an autoinducer into the shared
  extracellular field (AI-2-like molecule, µM units).
- **sensor** -- reads the local autoinducer concentration and commits to a
  binary decision only once it crosses a concentration threshold; the
  ensemble of sensors performs a *distributed consensus vote*.
- **actuator** -- produces a chemical output only after the consensus
  fraction (share of sensors that decided) exceeds a configurable quorum.

The simulator couples the same physical machinery as
:mod:`helixlang.population`: autoinducer emission (µM/tick, Xavier &
Bassler 2003), first-order autoinducer decay (Miller & Bassler 2001),
on-lattice diffusion converted from a physical µm^2/s coefficient
(:func:`helixlang.units.diffusion_to_lattice`), an energy budget with
division/death, and per-role growth rates.

Two design patterns from the literature are reproduced:

1. **Density-dependent consensus** (You et al. 2004 Nature 428:868-874;
   di Bernardo & colleagues 2026, arXiv:2602.19666 "engineering consensus
   in synthetic consortia"): the extracellular signal is a *collective*
   quantity, so a colony below the critical density stays below threshold
   (no consensus) while a colony above it flips nearly synchronously.

2. **Composition (ratio) control** (consortium regulation review: Mee &
   Wang 2012; 2020s engineered consortium balance, e.g. McCarty & Ledesma-
   Amaro 2019): a proportional feedback loop modulates each role's growth
   rate from the deviation between the current and target composition,
   driving the population toward the desired ratio.

A ``to_helix()`` renderer emits the consensus circuit as an executable
``.helix`` program (quorum-gated reporter), and
:func:`run_consortium_quorum` compiles and runs that program inside the
real :class:`~helixlang.population.CellPopulation` VM pipeline.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from helixlang.errors import BioError
from helixlang.population import (
    DIVISION_ENERGY_THRESHOLD,
    ENERGY_INTAKE_PER_STEP,
    MAX_SUBSTEP_D_LATTICE,
    METABOLIC_COST_PER_STEP,
    POPULATION_CELL_INITIAL_ENERGY,
    QUORUM_SIGNAL_THRESHOLD,
    SIGNAL_DIFFUSION_UM2_S,
    SIGNAL_EMISSION_PER_STEP,
    signal_diffusion_step,
)
from helixlang.units import (
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    diffusion_to_lattice,
)

ROLE_PRODUCER = "producer"
ROLE_SENSOR = "sensor"
ROLE_ACTUATOR = "actuator"

ROLES = (ROLE_PRODUCER, ROLE_SENSOR, ROLE_ACTUATOR)


# ============================================================================
# Data classes
# ============================================================================

@dataclass(slots=True)
class ConsortiumCell:
    """One cell of the synthetic consortium."""

    role: str
    x: int
    y: int
    energy: float = POPULATION_CELL_INITIAL_ENERGY
    alive: bool = True
    age: int = 0
    division_count: int = 0
    decided: bool = False
    output: float = 0.0
    signal_emitted: float = 0.0
    id: int = 0


@dataclass(slots=True)
class ConsortiumConfig:
    """Simulator configuration.

    Args:
        grid_width, grid_height: lattice dimensions (site edge =
            LATTICE_SPACING_UM µm).
        signal_threshold_um: quorum concentration (µM) a sensor needs
            before committing its decision.
        emission_um_per_tick: autoinducer emitted per producer per tick
            (µM).
        signal_decay_per_tick: first-order autoinducer decay fraction
            per tick (Miller & Bassler 2001; a decay term is what makes
            the threshold genuinely density-dependent).
        signal_diffusion_um2_s: physical autoinducer diffusion
            coefficient (µm^2/s), converted on-lattice.
        metabolic_cost: maintenance energy (ATP) per cell per tick.
        energy_intake: per-role energy intake (ATP/tick); the *effective*
            intake is scaled by the ratio-control feedback when
            ``target_ratios`` is set.
        division_threshold, death_threshold: energy thresholds (ATP).
        initial_energy: newborn cell energy (ATP).
        max_size: hard cap on the number of living cells.
        consensus_fraction: fraction of living sensors that must have
            decided for the consortium consensus to be "reached".
        output_per_actuator: output units per active actuator per tick.
        ratio_control_gain: proportional feedback gain on per-role
            growth (0 disables composition control).
        target_ratios: desired composition (role -> fraction).  When
            None, all roles grow at their base rate.
        seed: RNG seed.
    """

    grid_width: int = 40
    grid_height: int = 40
    signal_threshold_um: float = QUORUM_SIGNAL_THRESHOLD
    emission_um_per_tick: float = SIGNAL_EMISSION_PER_STEP
    signal_decay_per_tick: float = 0.1
    signal_diffusion_um2_s: float = SIGNAL_DIFFUSION_UM2_S
    metabolic_cost: float = METABOLIC_COST_PER_STEP
    energy_intake: dict[str, float] = field(default_factory=dict)
    division_threshold: float = DIVISION_ENERGY_THRESHOLD
    death_threshold: float = 0.0
    initial_energy: float = POPULATION_CELL_INITIAL_ENERGY
    max_size: int = 4000
    consensus_fraction: float = 0.5
    output_per_actuator: float = 1.0
    ratio_control_gain: float = 1.0
    target_ratios: dict[str, float] | None = None
    seed: int | None = None


# ============================================================================
# Simulator
# ============================================================================

class ConsortiumSimulator:
    """Run a synthetic consortium on a 2-D lattice.

    The role interplay follows the quorum circuit of You et al. 2004: an
    autoinducer released by producers diffuses over the lattice, and
    sensors commit once the local concentration crosses the threshold;
    once ``consensus_fraction`` of the sensors have committed, the
    actuators are switched on.  When ``target_ratios`` is configured, a
    proportional controller adjusts each role's growth rate so the
    composition converges to the target.
    """

    def __init__(self, config: ConsortiumConfig | None = None) -> None:
        self.config = config or ConsortiumConfig()
        self.cells: list[ConsortiumCell] = []
        self.rng = random.Random(self.config.seed)
        self.signal_field: list[list[float]] = [
            [0.0] * self.config.grid_width
            for _ in range(self.config.grid_height)
        ]
        self.tick = 0
        self.consensus_reached = False
        self.output_units = 0.0
        self._next_id = 0
        self.history: list[dict[str, float]] = []

    # -- population setup ---------------------------------------------------

    def add_cells(self, count: int, role: str, x: int | None = None,
                  y: int | None = None, *, stack: bool = False) -> None:
        """Add ``count`` cells of ``role``.

        Without ``x``/``y`` the cells are scattered over the lattice.  With
        ``stack=True`` every cell is placed at the same (x, y) site, which
        models a point-source colony (used to probe the density threshold
        deterministically).
        """
        if role not in ROLES:
            raise BioError(f"unknown consortium role {role!r}; "
                           f"choose from {ROLES}")
        for _ in range(count):
            if stack:
                if x is None or y is None:
                    raise BioError(
                        "stack=True requires explicit x and y coordinates")
                cx, cy = x, y
            elif x is not None and y is not None:
                cx, cy = x, y
            else:
                cx = self.rng.randrange(self.config.grid_width)
                cy = self.rng.randrange(self.config.grid_height)
            self.cells.append(ConsortiumCell(
                role=role,
                x=int(cx),
                y=int(cy),
                energy=self.config.initial_energy,
                id=self._next_id,
            ))
            self._next_id += 1

    def _intake_for(self, role: str, fraction: float) -> float:
        """Effective ATP intake for a role under ratio feedback."""
        base = self.config.energy_intake.get(role, ENERGY_INTAKE_PER_STEP)
        cfg = self.config
        if cfg.target_ratios is None or cfg.ratio_control_gain <= 0.0:
            return base
        target = cfg.target_ratios.get(role, 0.0)
        scale = 1.0 + cfg.ratio_control_gain * (target - fraction)
        return base * max(0.0, min(2.0, scale))

    # -- main loop ----------------------------------------------------------

    def _emit_and_diffuse(self) -> None:
        """Producers secrete autoinducer; field diffuses and decays."""
        cfg = self.config
        field = self.signal_field
        for cell in self.cells:
            if not cell.alive or cell.role != ROLE_PRODUCER:
                continue
            if not (0 <= cell.x < cfg.grid_width
                    and 0 <= cell.y < cfg.grid_height):
                continue
            field[cell.y][cell.x] += cfg.emission_um_per_tick
            cell.signal_emitted += cfg.emission_um_per_tick
        d_lattice = diffusion_to_lattice(
            cfg.signal_diffusion_um2_s, DIFFUSION_DT_S, LATTICE_SPACING_UM)
        if d_lattice > 0.0:
            n = max(1, math.ceil(d_lattice / MAX_SUBSTEP_D_LATTICE))
            for _ in range(n):
                field = signal_diffusion_step(field, d_lattice / n)
        decay = cfg.signal_decay_per_tick
        if decay > 0.0:
            self.signal_field = [
                [max(0.0, v * (1.0 - decay)) for v in row]
                for row in field
            ]
        else:
            self.signal_field = field

    def _sense(self) -> None:
        """Sensors commit a decision when local signal >= threshold."""
        cfg = self.config
        for cell in self.cells:
            if not cell.alive or cell.role != ROLE_SENSOR:
                continue
            if cell.decided:
                continue
            if not (0 <= cell.x < cfg.grid_width
                    and 0 <= cell.y < cfg.grid_height):
                continue
            if self.signal_field[cell.y][cell.x] >= cfg.signal_threshold_um:
                cell.decided = True

    def _sensor_fraction(self) -> float:
        alive_sensors = [c for c in self.cells
                         if c.alive and c.role == ROLE_SENSOR]
        if not alive_sensors:
            return 0.0
        return sum(1 for c in alive_sensors if c.decided) / len(alive_sensors)

    def _metabolism_and_division(self) -> None:
        """Per-role growth (ratio feedback) + division + death."""
        cfg = self.config
        fractions = self.role_fractions()
        alive = [c for c in self.cells if c.alive]
        survivors: list[ConsortiumCell] = []
        for cell in alive:
            cell.age += 1
            cell.energy += self._intake_for(cell.role,
                                            fractions.get(cell.role, 0.0))
            cell.energy -= cfg.metabolic_cost
            if cell.energy <= cfg.death_threshold:
                cell.alive = False
                continue
            survivors.append(cell)
        # division (deterministic once the energy threshold is crossed)
        if len(survivors) < cfg.max_size:
            queue = list(survivors)
            for cell in queue:
                if cell.energy < cfg.division_threshold:
                    continue
                if len(survivors) >= cfg.max_size:
                    break
                cell.energy /= 2.0
                daughter = ConsortiumCell(
                    role=cell.role,
                    x=cell.x,
                    y=cell.y,
                    energy=cell.energy,
                    id=self._next_id,
                )
                self._next_id += 1
                daughter.division_count = cell.division_count + 1
                cell.division_count += 1
                survivors.append(daughter)
        self.cells = survivors

    def step(self) -> dict[str, float]:
        """Advance one tick; returns the tick statistics."""
        self._emit_and_diffuse()
        self._sense()
        decided = self._sensor_fraction()
        if not self.consensus_reached and decided >= self.config.consensus_fraction:
            self.consensus_reached = True
        # actuators act only after consensus
        active_actuators = 0
        produced = 0.0
        if self.consensus_reached:
            for cell in self.cells:
                if cell.alive and cell.role == ROLE_ACTUATOR:
                    produced += self.config.output_per_actuator
                    cell.output += self.config.output_per_actuator
                    active_actuators += 1
            self.output_units += produced
        self._metabolism_and_division()
        self.tick += 1
        fracs = self.role_fractions()
        avg_signal = sum(sum(row) for row in self.signal_field) / max(
            1, self.config.grid_width * self.config.grid_height)
        stats = {
            "tick": float(self.tick),
            "alive": float(sum(1 for c in self.cells if c.alive)),
            "consensus_fraction": decided,
            "consensus_reached": 1.0 if self.consensus_reached else 0.0,
            "output_rate": produced,
            "cumulative_output": self.output_units,
            "active_actuators": float(active_actuators),
            "avg_signal": avg_signal,
            "max_signal": max(max(row) for row in self.signal_field),
            "producer_fraction": fracs.get(ROLE_PRODUCER, 0.0),
            "sensor_fraction": fracs.get(ROLE_SENSOR, 0.0),
            "actuator_fraction": fracs.get(ROLE_ACTUATOR, 0.0),
        }
        self.history.append(stats)
        return stats

    def run(self, n_steps: int) -> list[dict[str, float]]:
        """Run ``n_steps`` ticks and return :attr:`history`."""
        for _ in range(n_steps):
            self.step()
        return self.history

    # -- observations -------------------------------------------------------

    def role_fractions(self) -> dict[str, float]:
        """Current composition (role -> fraction of living cells)."""
        alive = [c for c in self.cells if c.alive]
        if not alive:
            return {r: 0.0 for r in ROLES}
        n = len(alive)
        out = {r: 0.0 for r in ROLES}
        for c in alive:
            out[c.role] += 1.0 / n
        return out

    def alive_count(self, role: str) -> int:
        return sum(1 for c in self.cells if c.alive and c.role == role)

    def mean_signal_at(self, cells: list[ConsortiumCell]) -> float:
        """Mean local signal at the given cell positions."""
        values = [
            self.signal_field[c.y][c.x]
            for c in cells
            if 0 <= c.x < self.config.grid_width
            and 0 <= c.y < self.config.grid_height
        ]
        return sum(values) / len(values) if values else 0.0


# ============================================================================
# .helix program rendering + execution on the real VM pipeline
# ============================================================================

def make_consortium_helix(threshold_um: float = 20.0,
                          promoter_strength: float = -0.4,
                          reporter_strength: float = 0.7) -> str:
    """Render the consortium consensus circuit as a ``.helix`` program.

    The emitted source mirrors ``examples/21_quorum_circuit.helix``: a
    constitutive ``signal`` gene (codon ``TCA`` = OP_SIGNAL) releases the
    autoinducer into the population signal field, and a quorum-gated
    ``reporter`` gene is activated only when the field crosses the
    threshold (You et al. 2004 density switch).
    """
    return (
        "# Synthetic consortium consensus circuit (S1)\n"
        "# A constitutive signal gene secretes the autoinducer; the\n"
        "# quorum-gated reporter turns on only once the local field\n"
        "# crosses the density-dependent threshold (You 2004; di Bernardo\n"
        "# 2026 arXiv:2602.19666).\n"
        "\n"
        f"#promoter name=p_signal strength={promoter_strength:g}   "
        "# constitutive signaler\n"
        f"#promoter name=p_reporter strength={reporter_strength:g}  "
        "# quorum-gated reporter\n"
        "\n"
        "#gene name=signal promoter=p_signal\n"
        "ATG TCA TAA\n"
        "#end\n"
        "\n"
        "#gene name=reporter promoter=p_reporter\n"
        "ATG GCT GCT TAA\n"
        "#end\n"
        "\n"
        "#regulate p_signal -> signal strength=+0.8\n"
        "#regulate p_reporter -> reporter strength=+0.9\n"
        "\n"
        f"#config ticks=30 output=stdout threshold={threshold_um:g}\n"
    )


def run_consortium_quorum(cells_per_side: int,
                          threshold_um: float = 20.0,
                          n_ticks: int = 30,
                          grid: int = 24) -> bool:
    """Compile and run the consensus circuit on the real CellPopulation.

    Packs a centered ``cells_per_side x cells_per_side`` colony and
    returns whether quorum was reached (``proteins["quorum"] == 1.0`` on
    the first cell), mirroring ``examples/21_quorum_circuit.helix``.
    """
    from helixlang.codon_table import STANDARD_TABLE
    from helixlang.compiler import Compiler
    from helixlang.lexer import Lexer
    from helixlang.parser import Parser
    from helixlang.population import CellPopulation, PopulationCell, PopulationConfig
    from helixlang.semantic import SemanticAnalyzer

    src = make_consortium_helix(threshold_um=threshold_um)
    program = Parser(list(Lexer(src).tokens())).parse()
    SemanticAnalyzer(program).check()
    chunk = Compiler(STANDARD_TABLE).compile(program)

    config = PopulationConfig(
        program=program,
        chunk=chunk,
        grid_width=grid,
        grid_height=grid,
        signal_diffusion=0.3,
        signal_threshold=threshold_um,
        division_threshold=1e9,
        metabolic_cost=0.0,
        energy_intake=0.0,
    )
    n = cells_per_side
    off = grid // 2 - n // 2
    cells = [
        PopulationCell(id=i, energy=100.0,
                       x=off + i % n, y=off + i // n)
        for i in range(n * n)
    ]
    population = CellPopulation(cells, config)
    for _ in range(n_ticks):
        population.step()
    return bool(population.cells[0].proteins.get("quorum", 0.0) == 1.0)


# ============================================================================
# Convenience report
# ============================================================================

@dataclass(slots=True)
class ConsortiumReport:
    """High-level outcome of a consortium run."""

    ticks: int
    alive: int
    consensus_fraction: float
    consensus_reached: bool
    output_units: float
    composition: dict[str, float]
    max_signal_um: float


def run_consortium(config: ConsortiumConfig | None = None,
                   initial: dict[str, int] | None = None,
                   n_ticks: int = 120) -> ConsortiumReport:
    """Run a consortium to completion and summarize the outcome.

    ``initial`` maps role -> starting cell count (defaults to 30 of each
    role scattered over the lattice).
    """
    sim = ConsortiumSimulator(config)
    counts = initial or {ROLE_PRODUCER: 30, ROLE_SENSOR: 30,
                         ROLE_ACTUATOR: 30}
    for role, count in counts.items():
        sim.add_cells(int(count), role)
    history = sim.run(int(n_ticks))
    last = history[-1]
    return ConsortiumReport(
        ticks=int(last["tick"]),
        alive=int(last["alive"]),
        consensus_fraction=last["consensus_fraction"],
        consensus_reached=bool(last["consensus_reached"]),
        output_units=last["cumulative_output"],
        composition={
            ROLE_PRODUCER: last["producer_fraction"],
            ROLE_SENSOR: last["sensor_fraction"],
            ROLE_ACTUATOR: last["actuator_fraction"],
        },
        max_signal_um=last["max_signal"],
    )


__all__ = [
    "ROLE_PRODUCER", "ROLE_SENSOR", "ROLE_ACTUATOR", "ROLES",
    "ConsortiumCell", "ConsortiumConfig", "ConsortiumSimulator",
    "ConsortiumReport",
    "make_consortium_helix", "run_consortium_quorum", "run_consortium",
]
