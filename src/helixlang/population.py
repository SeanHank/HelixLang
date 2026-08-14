"""Multicellular population simulation: cell division + signal communication + spatial organization.

Based on real biology:
- E. coli doubling time ~20 min (grown in rich medium, 37°C)
- Quorum sensing: AI-2 signal molecule threshold ~10 uM (Xavier 2003)
- Cell division requires a sufficient energy budget (ATP molecules;
  the 1.8e9 ATP threshold is reachable in ~20 rich-medium minutes at
  the +4e7 ATP/tick net intake)
- Intercellular signal diffusion coefficient ~1e-6 cm^2/s = 100 um^2/s
  (order of magnitude; stored in um^2/s and converted to a stable
  on-lattice form at the declared lattice edge)
- Biofilm formation requires a sufficiently high cell density (O'Toole 2000)

.. note::
   **Units.** Energy is in **ATP molecules**, the signal field is in
   **µM concentrations**, the on-lattice diffusion input is a physical
   **µm²/s** coefficient (converted internally via
   :func:`helixlang.units.diffusion_to_lattice` at the declared 10 µm
   lattice edge, D_lattice ~= 60), and one tick is one minute
   (:mod:`helixlang.units`).
"""
from __future__ import annotations

import copy
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field

from helixlang.ast_nodes import Program
from helixlang.bytecode import Chunk
from helixlang.environment import (
    Environment,
    crowding_diffusion_factor,
    monod_uptake,
)
from helixlang.grn import GRN
from helixlang.metabolism import (
    ECOLI_CORE_MODEL,
    DynamicFBAConfig,
    DynamicFluxBalance,
)
from helixlang.morphology_3d import LSystem3D
from helixlang.units import (
    AI2_DIFFUSION_UM2_S,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    diffusion_to_lattice,
)

# numpy is optional (falls back to pure Python if unavailable)
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ============================================================================
# Physical constant registry
# ============================================================================

#: maximum population size
DEFAULT_MAX_POPULATION_SIZE = 10000
#: simulation lattice dimensions (each site edge = LATTICE_SPACING_UM µm;
#: a 100 x 100 grid = 1 mm biofilm patch)
DEFAULT_GRID_WIDTH = 100
DEFAULT_GRID_HEIGHT = 100
#: energy budget required for division (ATP; newborn 1e9 + net 4e7/tick
#: reaches this in ~20 ticks, Neidhardt 1996 rich-medium doubling)
DIVISION_ENERGY_THRESHOLD = 1.8e9
#: energy below which a cell dies (starvation death at 0 ATP)
DEATH_ENERGY_THRESHOLD = 0.0
#: AI-2 intercellular signal diffusion coefficient, µm^2/s
#: (~1e-6 cm^2/s, Miller & Bassler 2001); on-lattice D at the declared
#: 10 µm lattice edge is ~60 via diffusion_to_lattice
SIGNAL_DIFFUSION_UM2_S = AI2_DIFFUSION_UM2_S
#: quorum-sensing signal threshold (µM AI-2, Xavier 2003 ~10 µM)
QUORUM_SIGNAL_THRESHOLD = 10.0
#: metabolic energy cost per time step (ATP; maintenance flux ~2.5e7
#: ATP/min, Orth 2010)
METABOLIC_COST_PER_STEP = 1.0e7
#: nutrient intake per time step, rich medium (ATP; glucose uptake
#: -> ATP, Alberts)
ENERGY_INTAKE_PER_STEP = 5.0e7
#: newborn cell energy (ATP molecules)
POPULATION_CELL_INITIAL_ENERGY = 1.0e9
#: AI-2 signal emitted per cell per tick (µM; 1 lattice unit = 2 µM,
#: Xavier & Bassler 2003)
SIGNAL_EMISSION_PER_STEP = 2.0

#: volume fraction of one 10 µm lattice site occupied by a single cell
#: (~1.5 µm^3 per E. coli cell vs. 1000 µm^3 per site); ~90 cells per
#: site cross the CROMICS 14% critical volume fraction
#: (Angeles-Martinez & Hatzimanikatis 2021)
CELL_VOLUME_FRACTION = 1.5e-3
#: newborn cell volume (um3; Taheri-Araghi 2015)
CELL_NEWBORN_VOLUME_UM3 = 1.6
#: maximum on-lattice diffusion coefficient per substep (explicit
#: 5-point Laplacian stability limit)
MAX_SUBSTEP_D_LATTICE = 0.25
#: protein slot capacity per programmable cell (mirrors cell.py)
CELL_SLOT_COUNT = 256

#: physical-unit axis summary (see module docstring)
UNITS: dict[str, str] = {
    "energy": "ATP molecules (newborn ~1e9; maintenance ~2.5e7 ATP/min, "
              "Orth 2010)",
    "signal": "µM concentrations; quorum threshold ~10 µM AI-2 "
              "(Xavier 2003)",
    "diffusion": "physical D in µm^2/s (~100 µm^2/s AI-2, Miller & "
                 "Bassler 2001); converted to a stable on-lattice form "
                 "at the declared lattice edge (see units.py)",
}


# ============================================================================
# Config and data classes
# ============================================================================
@dataclass(slots=True)
class PopulationConfig:
    """Population simulation config.

    Args:
        max_size: maximum number of cells on the lattice
        grid_width, grid_height: lattice dimensions (site edge =
            LATTICE_SPACING_UM µm)
        division_threshold: energy (ATP) required to divide
        death_threshold: energy (ATP) below which a cell dies
        signaling_enabled: whether cells emit and sense the AI-2 signal
        signal_diffusion: AI-2 diffusion coefficient in µm^2/s
        signal_threshold: quorum threshold in µM AI-2
        metabolic_cost: maintenance ATP per tick
        energy_intake: rich-medium ATP intake per tick
        environment: extracellular medium; when set, per-cell ATP intake
            is scaled by the local glucose concentration (Monod
            saturation, Ks = glucose_half_saturation_mm) and the field
            is depleted by each cell's uptake, then refreshed each tick
        glucose_half_saturation_mm: Monod Ks for the growth-saturating
            uptake term (Kovárová-Kovar & Egli 1998; default 0.1 mM)
        max_glucose_uptake_mm: per-cell per-tick glucose demand cap
            (mM per site); the field depletes by at most this amount
        crowding_enabled: when set, solute diffusion is slowed locally by
            the cell volume fraction (CROMICS effective diffusion,
            Angeles-Martinez & Hatzimanikatis 2021)
        mechanics: spatial mechanics mode, None (no repulsion),
            "shoving" (one cell per site via iDynoMiCS-style shoving) or
            "force" (density-gradient repulsion, iDynoMiCS force-based
            mechanics, Lardon 2011 / Cockx 2024)
        program: HelixLang :class:`~helixlang.ast_nodes.Program` run
            inside every cell (GRN + bytecode); requires ``chunk``
        chunk: compiled bytecode for ``program``
        noise_enabled, noise_seed: per-cell GRN telegraph-promoter noise
            (T1.4), passed to every cell's GRN
        ops_per_tick: bytecode op quota per cell per tick
        program_controlled_division: when True (and a program is
            attached), cells only divide on OP_DIVIDE; otherwise the
            energy threshold governs division
        trace_streaming: append a per-cell snapshot dict to
            ``population.trace`` each tick (T1.5)
        dfba_enabled: per-cell dynamic-FBA metabolism (Phase 5,
            Mahadevan 2002 + Cockx 2024).  When True (and
            ``environment`` is set) every alive cell owns a
            :class:`~helixlang.metabolism.DynamicFluxBalance` that reads
            the local glucose field, integrates one step, writes the
            consumed glucose and deposited acetate back to the
            environment, and converts growth into the energy budget.
            Default False (unchanged energy model).
        dfba_dt_h: integration step (hours) per population tick.
        dfba_energy_scale: energy (ATP) per unit specific growth rate
            (1/h) per hour, coupling the batch growth rate to the cell
            energy budget.
        dfba_initial_biomass_gdw: per-cell batch starting biomass (gDW/L).
        dfba_glucose_half_saturation_mm: Monod Ks (mM) of the per-cell
            glucose uptake (default 0.1, Kovárová-Kovar & Egli 1998).
        dfba_oxygen_max_uptake: maximum O2 uptake rate (mmol O2/gDW/h)
            used to cap the batch's respiratory capacity from the local
            oxygen field; O2-limited respiration is what forces the
            overflow switch to fermentative acetate (spatial
            stratification).
        dfba_oxygen_half_saturation_mm: Monod Ks (mM) of the per-cell
            oxygen uptake (~0.01 mM, on the µM scale of the respiration
            Ks; Stolper 2010).
    """

    max_size: int = DEFAULT_MAX_POPULATION_SIZE
    grid_width: int = DEFAULT_GRID_WIDTH
    grid_height: int = DEFAULT_GRID_HEIGHT
    grid_depth: int = 1
    division_threshold: float = DIVISION_ENERGY_THRESHOLD
    death_threshold: float = DEATH_ENERGY_THRESHOLD
    signaling_enabled: bool = True
    signal_diffusion: float = SIGNAL_DIFFUSION_UM2_S
    signal_threshold: float = QUORUM_SIGNAL_THRESHOLD
    metabolic_cost: float = METABOLIC_COST_PER_STEP
    energy_intake: float = ENERGY_INTAKE_PER_STEP
    environment: Environment | None = None
    glucose_half_saturation_mm: float = 0.1
    max_glucose_uptake_mm: float = 0.5
    crowding_enabled: bool = False
    mechanics: str | None = None
    program: Program | None = None
    chunk: Chunk | None = None
    noise_enabled: bool = False
    noise_seed: int | None = None
    ops_per_tick: int = 100
    program_controlled_division: bool = False
    trace_streaming: bool = False
    dfba_enabled: bool = False
    dfba_dt_h: float = 0.25
    dfba_energy_scale: float = 1.25e8
    dfba_initial_biomass_gdw: float = 0.05
    dfba_glucose_half_saturation_mm: float = 0.1
    dfba_oxygen_max_uptake: float = 40.0
    dfba_oxygen_half_saturation_mm: float = 0.01


@dataclass(slots=True)
class PopulationCell:
    """A single cell in the population.

    Programmable cells (config.program set) also carry a per-cell
    :class:`~helixlang.grn.GRN` and suspended bytecode VM state
    (vm_ip / vm_stack / vm_frames), so every cell executes its own copy
    of the genome with independent expression state.
    """

    id: int = 0
    parent_id: int | None = None
    energy: float = POPULATION_CELL_INITIAL_ENERGY
    x: int = 0
    y: int = 0
    z: int = 0
    proteins: dict[int | str, float] = field(default_factory=dict)
    alive: bool = True
    age: int = 0
    division_count: int = 0                 # division count
    signal_emitted: float = 0.0             # amount of signal emitted
    grn: GRN | None = None                  # per-cell gene regulatory network
    vm_ip: int = 0                          # suspended bytecode instruction pointer
    vm_stack: list[float] = field(default_factory=list)
    vm_frames: list[int] = field(default_factory=list)  # return ips per call
    current_gene: str | None = None         # gene currently executing
    slots: list[float] = field(
        default_factory=lambda: [0.0] * CELL_SLOT_COUNT)
    flag_divide: bool = False               # OP_DIVIDE request, honored next
    color: tuple[int, int, int] = (255, 255, 255)
    membrane_permeability: int = 255        # 0 (impermeable) .. 255 (open)
    volume_um3: float = CELL_NEWBORN_VOLUME_UM3  # physical cell volume (um3)
    dfba: DynamicFluxBalance | None = None  # per-cell batch metabolism (Phase 5)


def cell_radius_um(volume_um3: float) -> float:
    """Equivalent-sphere radius (um) of a cell of the given volume."""
    v: float = max(0.0, volume_um3)
    radius: float = (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)
    return radius


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
    prots_a: dict[int | str, float] = {}
    prots_b: dict[int | str, float] = {}
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

    # Daughters inherit the parent's GRN (expression state carries over)
    # but start with a fresh, empty bytecode execution state.  Each
    # daughter gets its own deep copy so later divergence stays isolated.
    daughter_grn_a = copy.deepcopy(cell.grn) if cell.grn is not None else None
    daughter_grn_b = copy.deepcopy(cell.grn) if cell.grn is not None else None

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
        grn=daughter_grn_a,
        color=cell.color,
        membrane_permeability=cell.membrane_permeability,
        volume_um3=cell.volume_um3 / 2.0,
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
        grn=daughter_grn_b,
        color=cell.color,
        membrane_permeability=cell.membrane_permeability,
        volume_um3=cell.volume_um3 / 2.0,
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


def _crowded_laplacian_step(
    grid: list[list[float]],
    factors: list[list[float]],
    d_sub: float,
    w: int,
    h: int,
) -> list[list[float]]:
    """One explicit diffusion step with a per-site (crowded) coefficient.

    ``factors[i][j]`` is the CROMICS effective-diffusion factor in
    [0, 1] for each site and ``d_sub`` the per-substep on-lattice
    coefficient (``D_lattice / n_substeps``).  Unlike the uniform-D
    :func:`signal_diffusion_step`, the spatially varying coefficient is
    realized with the **flux-conservative** form of Fick's law,
    ``dc/dt = div(D grad c)``: the flux on each face uses the mean of
    the two adjacent site coefficients, so the scheme conserves mass
    exactly (telescoping face fluxes) and reduces to the 5-point
    Laplacian when D is uniform.
    """
    if _HAS_NUMPY:
        import numpy as _np
        a = _np.asarray(grid, dtype=float)
        d = _np.asarray(factors, dtype=float) * d_sub
        dp = _np.pad(d, 1, mode="edge")
        cp = _np.pad(a, 1, mode="edge")
        # face coefficients: arithmetic mean of the two adjacent sites
        fe = 0.5 * (dp[1:-1, 1:-1] + dp[1:-1, 2:]) \
            * (cp[1:-1, 2:] - cp[1:-1, 1:-1])
        fw = 0.5 * (dp[1:-1, 1:-1] + dp[1:-1, :-2]) \
            * (cp[1:-1, 1:-1] - cp[1:-1, :-2])
        fn = 0.5 * (dp[1:-1, 1:-1] + dp[:-2, 1:-1]) \
            * (cp[1:-1, 1:-1] - cp[:-2, 1:-1])
        fs = 0.5 * (dp[1:-1, 1:-1] + dp[2:, 1:-1]) \
            * (cp[2:, 1:-1] - cp[1:-1, 1:-1])
        new = a + fe - fw + fs - fn
        _np.clip(new, 0.0, None, out=new)
        result: list[list[float]] = new.tolist()
        return result
    new_grid: list[list[float]] = []
    for i in range(h):
        row = grid[i]
        new_row: list[float] = []
        for j in range(w):
            cur = row[j]
            # east face flux
            de = 0.5 * (factors[i][j] + factors[i][j + 1]) * d_sub \
                if j + 1 < w else 0.0
            dw = 0.5 * (factors[i][j] + factors[i][j - 1]) * d_sub \
                if j > 0 else 0.0
            dn = 0.5 * (factors[i][j] + factors[i - 1][j]) * d_sub \
                if i > 0 else 0.0
            ds = 0.5 * (factors[i][j] + factors[i + 1][j]) * d_sub \
                if i + 1 < h else 0.0
            fe = de * (grid[i][j + 1] - cur) if j + 1 < w else 0.0
            fw = dw * (cur - grid[i][j - 1]) if j > 0 else 0.0
            fn = dn * (cur - grid[i - 1][j]) if i > 0 else 0.0
            fs = ds * (grid[i + 1][j] - cur) if i + 1 < h else 0.0
            v = cur + fe - fw + fs - fn
            new_row.append(v if v > 0.0 else 0.0)
        new_grid.append(new_row)
    return new_grid


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
        self._template_grn: GRN | None = None
        self._step_start_alive = sum(1 for c in self.cells if c.alive)
        self._generation = 0
        # T1.5: streaming per-cell trace (only when requested)
        self.trace: list[dict] = []
        if config.program is not None:
            if config.chunk is None:
                raise ValueError(
                    "PopulationConfig.chunk is required when program is set")
            self._template_grn = self._build_program_grn(
                config.program, config.noise_enabled, config.noise_seed)
            for c in self.cells:
                if c.alive and c.grn is None:
                    c.grn = copy.deepcopy(self._template_grn)
        else:
            self._template_grn = None
        if config.mechanics not in (None, "shoving", "force"):
            raise ValueError(
                "mechanics must be None, 'shoving' or 'force', "
                f"got {config.mechanics!r}")
        if config.environment is not None and (
            config.environment.config.width != config.grid_width
            or config.environment.config.height != config.grid_height
        ):
            raise ValueError(
                "environment lattice must match the population grid "
                f"({config.grid_width}x{config.grid_height})")
        if config.dfba_enabled and config.environment is None:
            raise ValueError(
                "dfba_enabled requires an environment (shared substrate "
                "fields) to be configured")

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

    @staticmethod
    def _build_program_grn(program: Program, noise_enabled: bool = False,
                           noise_seed: int | None = None) -> GRN:
        """Compile a template GRN from a HelixLang program.

        Mirrors ``CellVM._init_subsystems`` (vm.py): promoters and genes
        become GRN nodes (negative promoter strength = constitutive,
        active from tick 0), regulations become edges.  Each population
        cell deep-copies this template so every cell carries its own
        expression state.
        """
        grn = GRN(noise_enabled=noise_enabled, noise_seed=noise_seed)
        prom_by_name = {p.name: p for p in program.promoters}
        for p in program.promoters:
            initial = 1.0 if p.strength < 0 else 0.0
            grn.add_gene(p.name, threshold=p.strength, initial_level=initial)
        for g in program.genes:
            if g.promoter and g.promoter in prom_by_name:
                threshold = prom_by_name[g.promoter].strength
                initial = 0.0
            else:
                threshold = -1.0
                initial = 1.0
            grn.add_gene(g.name, threshold, initial_level=initial)
        for r in program.regulations:
            grn.add_edge(r.source, r.target, r.strength)
        return grn

    def _read_cell_u8(self, cell: PopulationCell) -> int:
        """Read a 1-byte operand from the cell's bytecode stream."""
        chunk = self.config.chunk
        if chunk is None or cell.vm_ip >= len(chunk.code):
            return 0
        v = chunk.code[cell.vm_ip]
        cell.vm_ip += 1
        return v

    def _read_cell_u16(self, cell: PopulationCell) -> int:
        """Read a 2-byte (big-endian) operand from the cell's bytecode."""
        chunk = self.config.chunk
        if chunk is None:
            return 0
        if cell.vm_ip + 1 >= len(chunk.code):
            cell.vm_ip = len(chunk.code)
            return 0
        v = (chunk.code[cell.vm_ip] << 8) | chunk.code[cell.vm_ip + 1]
        cell.vm_ip += 2
        return v

    # -- Main loop --
    def step(self) -> dict:
        """Advance the population one step and return the statistics.

        - Metabolism of each cell (consumes energy; Monod-coupled to the
          extracellular environment when ``config.environment`` is set)
        - Program execution: each cell's GRN + bytecode (when a program
          is attached)
        - Signal diffusion between cells
        - Cell division (energy above threshold; or OP_DIVIDE when
          ``program_controlled_division`` is set)
        - Cell death (energy exhausted)
        - Spatial mechanics (shoving / force repulsion)
        - Signal response (quorum sensing activates genes)

        When numpy is available and the number of alive cells > 100, the metabolism phase automatically takes the
        numpy vectorized path of :meth:`_step_vectorized`; otherwise pure Python is used.
        """
        alive_before = sum(1 for c in self.cells if c.alive)
        self._step_start_alive = alive_before

        # Large populations take the numpy vectorized metabolism path; small populations use pure Python
        if (_HAS_NUMPY and alive_before > 100
                and self.config.environment is None):
            metabolized, deaths = self._step_vectorized_metabolism()
        else:
            metabolized, deaths = self._step_metabolism_python()

        # 1.5) Program execution (per-cell GRN + bytecode)
        if self.config.program is not None and metabolized:
            metabolized, prog_deaths = self._step_programs(metabolized)
            deaths += prog_deaths

        # 2) Signal diffusion
        config = self.config
        if config.signaling_enabled:
            self.signal_field = self._diffuse(config)

        # 3) quorum sensing + division
        # Each division increases the population by 1; at most (max_size - current alive) divisions are allowed
        divisions_allowed = max(0, config.max_size - len(metabolized))
        divisions_done = 0
        next_cells: list[PopulationCell] = []
        program_controls = (config.program is not None
                            and config.program_controlled_division)
        for cell in metabolized:
            if config.signaling_enabled and self._in_bounds(cell.x, cell.y):
                sig = self.signal_field[cell.y][cell.x]
                quorum_sensing(cell, sig, config.signal_threshold)
            wants_division = (cell.flag_divide
                              if program_controls
                              else cell.energy >= config.division_threshold)
            if (wants_division
                    and cell.energy >= config.division_threshold
                    and divisions_done < divisions_allowed):
                a, b = divide_cell(cell, config, self.rng)
                divisions_done += 1
                next_cells.append(a)
                next_cells.append(b)
            else:
                next_cells.append(cell)

        # 4) Spatial mechanics (exclusion / force-based repulsion)
        if config.mechanics is not None and next_cells:
            self._apply_mechanics(next_cells)

        self.cells = next_cells
        self._assign_ids()
        self._last_divisions = divisions_done
        self._last_deaths = deaths
        self._total_deaths += deaths
        self._generation += 1

        # 5) Refresh the extracellular environment (diffuse + flow)
        if config.environment is not None:
            config.environment.step()

        # 6) Optional per-cell trace streaming
        if config.trace_streaming:
            self._append_trace()

        return self.get_statistics()

    # -- Metabolism phase (pure Python fallback) --
    def _diffuse(self, config: PopulationConfig) -> list[list[float]]:
        """Diffuse the signal field one tick (physical µm²/s → on-lattice).

        The config's ``signal_diffusion`` is a physical diffusion
        coefficient (µm^2/s).  It is converted to the dimensionless
        on-lattice form via :func:`helixlang.units.diffusion_to_lattice`
        at the declared lattice edge and tick length, then realized by
        stable sub-steps so the explicit 5-point-Laplacian scheme never
        blows up and the analytical Gaussian spread matches D_phys.
        """
        D_lattice = diffusion_to_lattice(
            config.signal_diffusion, DIFFUSION_DT_S, LATTICE_SPACING_UM)
        if D_lattice <= 0.0:
            return self.signal_field
        if config.crowding_enabled:
            return self._crowded_diffuse(config, D_lattice)
        n = math.ceil(D_lattice / MAX_SUBSTEP_D_LATTICE)
        field = self.signal_field
        for _ in range(n):
            field = signal_diffusion_step(field, D_lattice / n)
        return field

    def _occupancy(self, cells: list[PopulationCell]) -> list[list[int]]:
        """Alive-cell count per lattice site, indexed [y][x]."""
        grid = [[0] * self.config.grid_width for _ in range(self.config.grid_height)]
        for c in cells:
            if c.alive and self._in_bounds(c.x, c.y):
                grid[c.y][c.x] += 1
        return grid

    def get_volume_fractions(self) -> list[list[float]]:
        """Local biomass volume fraction per site (CELL_VOLUME_FRACTION x
        cell count), indexed [y][x]; input to the CROMICS crowding factor."""
        occ = self._occupancy(self.cells)
        return [
            [min(0.999, o * CELL_VOLUME_FRACTION) for o in row]
            for row in occ
        ]

    def _crowded_diffuse(self, config: PopulationConfig,
                         d_lattice: float) -> list[list[float]]:
        """Diffuse the signal field with CROMICS spatially-varying
        effective diffusion.

        Every site's diffusion coefficient is reduced by the free volume
        fraction ``1 - phi`` (CROMICS, Angeles-Martinez &
        Hatzimanikatis 2021): ``D_eff(x, y) = D_lattice *
        crowding_diffusion_factor(phi(x, y))``.  Sub-step count is set by
        the largest coefficient so the explicit scheme stays stable.
        """
        fracs = self.get_volume_fractions()
        factors = [
            [crowding_diffusion_factor(min(0.999, phi)) for phi in row]
            for row in fracs
        ]
        n = math.ceil(d_lattice / MAX_SUBSTEP_D_LATTICE)
        d_sub = d_lattice / n
        w = config.grid_width
        h = config.grid_height
        field = self.signal_field
        for _ in range(n):
            field = _crowded_laplacian_step(field, factors, d_sub, w, h)
        return field

    # -- Programmable cells (per-cell GRN + bytecode) --
    def _step_programs(self, cells: list[PopulationCell]
                       ) -> tuple[list[PopulationCell], int]:
        """Run each cell's GRN and bytecode program for one tick.

        Returns ``(alive_cells, deaths)``; cells killed by OP_DIE are
        removed and counted.  Triggered genes push a call frame whose
        return ip resumes the suspended stream; execution is limited to
        ``ops_per_tick`` ops per cell per tick (remaining frames resume
        next tick).
        """
        alive: list[PopulationCell] = []
        deaths = 0
        for cell in cells:
            if not cell.alive:
                continue
            grn = cell.grn
            if grn is not None:
                for gene in grn.step():
                    self._push_gene_frame(cell, gene)
            self._execute_cell(cell, self.config.ops_per_tick)
            if cell.alive:
                alive.append(cell)
            else:
                deaths += 1
        return alive, deaths

    def _push_gene_frame(self, cell: PopulationCell, gene: str) -> None:
        """Push a call frame for a triggered gene (from chunk.gene_offsets)."""
        if self.config.chunk is None:
            return
        off = self.config.chunk.gene_offsets.get(gene)
        if off is None:
            return
        if len(cell.vm_frames) >= 256:
            return
        cell.current_gene = gene
        cell.vm_frames.append(cell.vm_ip)
        cell.vm_ip = off

    def _execute_cell(self, cell: PopulationCell, quota: int) -> None:
        """Execute one cell's bytecode until its frames empty or the
        per-tick op quota is exhausted."""
        if self.config.chunk is None:
            return
        from helixlang.cell import (
            DIRECTIONS,
            FEED_ENERGY_AMOUNT,
            MAX_MEMBRANE_PERMEABILITY,
            MOVE_ENERGY_COST,
        )
        from helixlang.codon_table import OP_OPERAND_BYTES, Op
        from helixlang.vm import (
            BIND_LEVEL_BOOST,
            REGULATE_EDGE_WEIGHT,
            SIGNAL_EMISSION_AMOUNT,
        )

        code = self.config.chunk.code
        stack = cell.vm_stack
        frames = cell.vm_frames
        while quota > 0 and frames:
            if len(frames) > 256:
                frames.clear()
                break
            if cell.vm_ip >= len(code):
                frames.pop()
                if frames:
                    cell.vm_ip = frames[-1]
                break
            op_byte = code[cell.vm_ip]
            cell.vm_ip += 1
            try:
                op = Op(op_byte)
            except ValueError:
                continue
            match op:
                case Op.OP_START | Op.OP_NOP | Op.OP_TICK:
                    pass
                case Op.OP_HALT | Op.OP_RETURN:
                    if frames:
                        cell.vm_ip = frames.pop()
                        cell.current_gene = None
                case Op.OP_PUSH_CONST:
                    idx = self._read_cell_u8(cell)
                    constants = self.config.chunk.constants
                    stack.append(constants[idx] if idx < len(constants) else idx)
                case Op.OP_POP:
                    if stack:
                        stack.pop()
                case Op.OP_DUP:
                    if stack:
                        stack.append(stack[-1])
                case Op.OP_SWAP:
                    if len(stack) >= 2:
                        stack[-1], stack[-2] = stack[-2], stack[-1]
                case Op.OP_BUILD_PROTEIN:
                    kind = self._read_cell_u8(cell)
                    cell.proteins[kind] = cell.proteins.get(kind, 0.0) + 1.0
                case Op.OP_BUILD_MEMBRANE:
                    v = self._read_cell_u8(cell)
                    cell.membrane_permeability = max(
                        0, min(MAX_MEMBRANE_PERMEABILITY, v))
                case Op.OP_BUILD_PIGMENT:
                    cell.color = (200, 50, 50)
                case Op.OP_MOVE:
                    d = self._read_cell_u8(cell)
                    dx, dy = DIRECTIONS[d % 4]
                    nx = cell.x + dx
                    ny = cell.y + dy
                    if self._in_bounds(nx, ny):
                        cell.x, cell.y = nx, ny
                    cell.energy -= min(cell.energy, MOVE_ENERGY_COST)
                case Op.OP_SIGNAL:
                    ch = self._read_cell_u8(cell)
                    cell.signal_emitted += SIGNAL_EMISSION_AMOUNT * (1 + ch)
                    if self._in_bounds(cell.x, cell.y):
                        self.signal_field[cell.y][cell.x] += min(
                            1.0, SIGNAL_EMISSION_AMOUNT * (1 + ch))
                case Op.OP_DIVIDE:
                    self._read_cell_u8(cell)
                    cell.flag_divide = True
                case Op.OP_DIE:
                    self._read_cell_u8(cell)
                    cell.alive = False
                case Op.OP_FEED:
                    self._read_cell_u8(cell)
                    cell.energy += round(
                        FEED_ENERGY_AMOUNT
                        * cell.membrane_permeability
                        / MAX_MEMBRANE_PERMEABILITY)
                case Op.OP_GROW_LSYSTEM | Op.OP_DIFFUSE | Op.OP_REACT | \
                        Op.OP_EMIT_MORPHOGEN:
                    self._read_cell_u8(cell)
                case Op.OP_READ_MEM:
                    slot = self._read_cell_u8(cell)
                    stack.append(cell.slots[slot % len(cell.slots)])
                case Op.OP_WRITE_MEM:
                    slot = self._read_cell_u8(cell)
                    if stack:
                        cell.slots[slot % len(cell.slots)] = stack.pop()
                case Op.OP_MODIFY_STATE:
                    f = self._read_cell_u8(cell)
                    if f == 0:
                        cell.color = (100, 200, 50)
                    elif f == 1:
                        cell.age += 1
                    elif f == 2:
                        cell.color = (200, 200, 50)
                    elif f == 3:
                        cell.color = (200, 50, 200)
                case Op.OP_REGULATE:
                    mode = self._read_cell_u8(cell)
                    grn = cell.grn
                    if grn is not None and grn.nodes:
                        names = list(grn.nodes)
                        source = (cell.current_gene
                                  if cell.current_gene in grn.nodes
                                  else names[0])
                        target = names[(mode & 0x0F) % len(names)]
                        weight = (REGULATE_EDGE_WEIGHT
                                  if not (mode & 0x80)
                                  else -REGULATE_EDGE_WEIGHT)
                        existing = next(
                            (e for e in grn.edges
                             if e.source == source and e.target == target),
                            None,
                        )
                        if existing is not None:
                            existing.weight = weight
                        else:
                            grn.add_edge(source, target, weight)
                case Op.OP_BIND:
                    site = self._read_cell_u8(cell)
                    grn = cell.grn
                    if grn is not None and grn.nodes:
                        binder = cell.current_gene
                        tf_kind: int | str | None = None
                        if binder is not None and binder in cell.proteins:
                            tf_kind = binder
                        elif cell.proteins:
                            tf_kind = next(iter(cell.proteins))
                        if tf_kind is not None:
                            avail = cell.proteins.get(tf_kind, 0.0)
                            consumed = min(avail, 1.0)
                            if consumed > 0:
                                cell.proteins[tf_kind] = avail - consumed
                                names = list(grn.nodes)
                                target = names[site % len(names)]
                                grn.set_level(
                                    target,
                                    grn.nodes[target].level + BIND_LEVEL_BOOST,
                                )
                case Op.OP_CALL_GENE:
                    off = self._read_cell_u16(cell)
                    frames.append(cell.vm_ip)
                    cell.current_gene = "<call>"
                    cell.vm_ip = off
                case Op.OP_JUMP:
                    off = self._read_cell_u16(cell)
                    cell.vm_ip += off
                case Op.OP_JUMP_IF_ZERO:
                    off = self._read_cell_u16(cell)
                    top = stack.pop() if stack else 0
                    if not top:
                        cell.vm_ip += off
                case Op.OP_ADD:
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(a + b)
                case Op.OP_SUB:
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(a - b)
                case Op.OP_MUL:
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(a * b)
                case Op.OP_LT:
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(1 if a < b else 0)
                case Op.OP_NOT:
                    if stack:
                        a = stack.pop()
                        stack.append(0 if a else 1)
                case _:
                    nbytes = OP_OPERAND_BYTES.get(op, 0)
                    cell.vm_ip = min(cell.vm_ip + nbytes, len(code))
            quota -= 1

    # -- Spatial mechanics (iDynoMiCS-style exclusion / force) --
    def _apply_mechanics(self, cells: list[PopulationCell]) -> None:
        """Relax overlapping cells after division.

        - ``shoving``: extra cells on a site are shoved to the nearest
          empty neighboring site (Lardon et al. 2011, iDynoMiCS).
        - ``force``: cells are displaced one step toward the least
          crowded neighboring site, a lattice realization of the
          force-balance mechanics in iDynoMiCS 2.0 (Cockx et al. 2024).
        """
        config = self.config
        occ = self._occupancy(cells)
        if config.mechanics == "shoving":
            for c in cells:
                if not c.alive or not self._in_bounds(c.x, c.y):
                    continue
                if occ[c.y][c.x] <= 1:
                    continue
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = c.x + dx, c.y + dy
                    if self._in_bounds(nx, ny) and occ[ny][nx] == 0:
                        occ[c.y][c.x] -= 1
                        occ[ny][nx] = 1
                        c.x, c.y = nx, ny
                        break
        elif config.mechanics == "force":
            for c in cells:
                if not c.alive or not self._in_bounds(c.x, c.y):
                    continue
                if occ[c.y][c.x] <= 1:
                    continue
                best: tuple[int, int, int] | None = None
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (1, 1), (-1, 1), (1, -1)):
                    nx, ny = c.x + dx, c.y + dy
                    if not self._in_bounds(nx, ny):
                        continue
                    score = occ[ny][nx]
                    if best is None or score < best[0]:
                        best = (score, nx, ny)
                if best is not None and best[0] < occ[c.y][c.x]:
                    occ[c.y][c.x] -= 1
                    occ[best[2]][best[1]] += 1
                    c.x, c.y = best[1], best[2]

    # -- T1.5: streaming per-cell trace --
    def _append_trace(self) -> None:
        """Append a per-cell snapshot to ``self.trace`` for this tick."""
        snap: dict[str, object] = {
            "tick": self._generation,
            "cells": [
                {
                    "id": c.id,
                    "x": c.x,
                    "y": c.y,
                    "alive": c.alive,
                    "energy": c.energy,
                    "proteins": dict(c.proteins),
                    "gene_levels": ({n: nd.level for n, nd in c.grn.nodes.items()}
                                    if c.grn is not None else None),
                }
                for c in self.cells
            ],
            "signal_field": self.get_signal_field(),
        }
        self.trace.append(snap)

    def _step_metabolism_python(self) -> tuple[list[PopulationCell], int]:
        """Per-cell metabolism + signal emission + death determination (pure Python)."""
        config = self.config
        if config.dfba_enabled:
            return self._step_dfba_metabolism()
        metabolized: list[PopulationCell] = []
        deaths = 0
        env = config.environment
        for cell in self.cells:
            if not cell.alive:
                continue
            cell.age += 1
            intake = config.energy_intake
            if env is not None and self._in_bounds(cell.x, cell.y):
                # Monod saturation coupling: local glucose scales the
                # flat rich-medium intake (Kovárová-Kovar & Egli 1998),
                # and the cell depletes the field by its demand.
                local_s = env.glucose.get(cell.x, cell.y)
                factor = monod_uptake(
                    1.0, local_s, config.glucose_half_saturation_mm)
                intake = config.energy_intake * factor
                demand = config.max_glucose_uptake_mm * factor
                env.glucose.deplete(cell.x, cell.y, demand)
            cell.energy += intake - config.metabolic_cost
            if config.signaling_enabled:
                cell.signal_emitted += SIGNAL_EMISSION_PER_STEP
                self._emit_signal(cell)
            if cell.energy <= config.death_threshold:
                cell.alive = False
                deaths += 1
                continue
            metabolized.append(cell)
        return metabolized, deaths

    def _step_dfba_metabolism(self) -> tuple[list[PopulationCell], int]:
        """Per-cell dynamic-FBA metabolism (Phase 5, Mahadevan 2002).

        Each alive cell owns a :class:`~helixlang.metabolism.
        DynamicFluxBalance` batch (lazily created at :attr:
        `PopulationCell.dfba`).  Every tick it integrates one
        ``dfba_dt_h``-hour step:

        - the local glucose field is read into the batch
          (:meth:`DynamicFluxBalance.update_from_environment`) and the
          local oxygen field caps the batch's respiratory capacity,
        - the instantaneous FBA LP is solved and the batch is integrated
          (:meth:`DynamicFluxBalance.step`),
        - the glucose and oxygen consumed this tick are depleted from the
          environment fields and the fresh acetate is deposited
          (:meth:`DynamicFluxBalance.apply_to_environment`),
        - the specific growth rate is converted into the cell's energy
          budget (``dfba_energy_scale`` ATP per unit growth rate per
          hour), replacing the flat rich-medium intake + maintenance
          model.

        Each cell gets its own **deep-copied** model (``FluxBalanceAnalysis``
        keeps a shared reference to :data:`ECOLI_CORE_MODEL`), so the
        per-cell respiratory-capacity bounds never leak into other cells
        or standalone FBA solves.  O2-limited respiration redirects the
        LP to fermentative acetate overflow — the mechanism behind the
        colony-core metabolic stratification (Cockx 2024; COSMIC-dFBA
        2024).

        The flat ``METABOLIC_COST_PER_STEP`` is not subtracted here: the
        FBA biomass reaction already carries maintenance (ATPM), so
        subtracting it again would double-count.
        """
        config = self.config
        dt_h = config.dfba_dt_h
        env = config.environment
        metabolized: list[PopulationCell] = []
        deaths = 0
        for cell in self.cells:
            if not cell.alive:
                continue
            cell.age += 1
            if cell.dfba is None:
                cell.dfba = DynamicFluxBalance(
                    model=copy.deepcopy(ECOLI_CORE_MODEL),
                    config=DynamicFBAConfig(
                        dt_h=dt_h,
                        initial_biomass_gdw=config.dfba_initial_biomass_gdw,
                        initial_glucose_mm=0.0,
                    ))
            if env is not None and self._in_bounds(cell.x, cell.y):
                dfba = cell.dfba
                glucose_before = env.substrate_at(cell.x, cell.y, "glucose")
                dfba.update_from_environment(env, cell.x, cell.y)
                self._apply_dfba_oxygen_cap(dfba, env, cell.x, cell.y)
                dfba.step(dt_h)
                consumed = max(0.0, glucose_before - dfba.glucose_mm)
                if consumed > 0.0:
                    env.glucose.deplete(cell.x, cell.y, consumed)
                self._deplete_dfba_oxygen(dfba, env, cell.x, cell.y, dt_h)
                if dfba.byproducts_mm.get("acetate", 0.0) > 0.0:
                    dfba.apply_to_environment(env, cell.x, cell.y)
                    dfba.set_state(acetate_mm=0.0)
                cell.energy += (
                    dfba.growth_rate * dt_h * config.dfba_energy_scale)
            if config.signaling_enabled:
                cell.signal_emitted += SIGNAL_EMISSION_PER_STEP
                self._emit_signal(cell)
            if cell.energy <= config.death_threshold:
                cell.alive = False
                deaths += 1
                continue
            metabolized.append(cell)
        return metabolized, deaths

    def _apply_dfba_oxygen_cap(self,
                               dfba: DynamicFluxBalance,
                               env: Environment,
                               x: int,
                               y: int) -> None:
        """Cap the batch's respiratory capacity from the local O2 field.

        v_o2 = v_max * O2 / (Ks + O2); respiration flux (NADH/FADH2
        equivalents) is capped at ``2 * v_o2`` (2 reducing equivalents
        per O2, the ETC yield).  When O2 is scarce the cap binds and the
        LP must satisfy its ATP demand fermentatively -> acetate
        overflow.
        """
        cfg = self.config
        o2 = env.oxygen.get(x, y)
        v_o2 = monod_uptake(cfg.dfba_oxygen_max_uptake, o2,
                            cfg.dfba_oxygen_half_saturation_mm)
        cap = 2.0 * v_o2
        reactions = dfba.fba.model.reactions
        for rid in ("NADH_OX", "FADH2_OX"):
            if rid in reactions:
                reactions[rid].upper_bound = min(
                    reactions[rid].upper_bound, cap)

    def _deplete_dfba_oxygen(self,
                             dfba: DynamicFluxBalance,
                             env: Environment,
                             x: int,
                             y: int,
                             dt_h: float) -> None:
        """Deplete the local O2 field by the batch's respiratory demand."""
        fluxes = dfba.last_fluxes
        nadh = fluxes.get("NADH_OX", 0.0)
        fadh2 = fluxes.get("FADH2_OX", 0.0)
        o2_used = 0.5 * (nadh + fadh2) * dfba.biomass_gdw * dt_h
        if o2_used > 0.0:
            env.oxygen.deplete(x, y, o2_used)

    def _emit_signal(self, cell: PopulationCell) -> None:
        """Accumulate one tick of AI-2 signal emission at the cell's site.

        Hook so 3D populations (:class:`CellPopulation3D`) can emit into
        a ``[z][y][x]`` volume instead of the 2D ``[y][x]`` grid.
        """
        if self._in_bounds(cell.x, cell.y):
            self.signal_field[cell.y][cell.x] += SIGNAL_EMISSION_PER_STEP

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
            signals += SIGNAL_EMISSION_PER_STEP

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
                    SIGNAL_EMISSION_PER_STEP,
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

    # -- Phase 5: colony observables (per-cell dFBA) --

    def _colony_center(self, cells: list[PopulationCell]
                       ) -> tuple[float, float]:
        """Mean position of the alive cells (colony centre)."""
        if not cells:
            return self.config.grid_width / 2.0, self.config.grid_height / 2.0
        return (sum(c.x for c in cells) / len(cells),
                sum(c.y for c in cells) / len(cells))

    def colony_observables(self) -> dict:
        """Colony-scale observables for the per-cell dFBA population
        (Phase 5, BM3 / iDynoMiCS-style).

        Returns:
            - ``doubling_times_h``: sorted per-cell doubling times
              (``ln(2)/mu``) from each cell's batch growth rate,
            - ``birth_sizes_um3``: sorted newborn volumes (cells with
              age 0; the Phase-2 adder birth-size distribution),
            - ``volume_weighted_growth_h``: ``sum(mu_i * V_i)/sum(V_i)``,
            - ``radial_density``: cell count per radial band around the
              colony centre (BM3-style density profile),
            - ``colony_radius_sites``: farthest alive-cell distance from
              the centre.
        """
        alive = [c for c in self.cells if c.alive]
        cx, cy = self._colony_center(alive)

        def _mu(c: PopulationCell) -> float:
            if c.dfba is not None and c.dfba.history:
                return c.dfba.growth_rate
            return 0.0

        doubling = [math.log(2.0) / m for m in (_mu(c) for c in alive)
                    if m > 1e-12]
        birth = [c.volume_um3 for c in alive if c.age == 0]
        vol = sum(c.volume_um3 for c in alive)
        vw_growth = (sum(_mu(c) * c.volume_um3 for c in alive) / vol
                     if vol > 0 else 0.0)
        radii = [math.hypot(c.x - cx, c.y - cy) for c in alive]
        rmax = max(radii) if radii else 1.0
        bands = 10
        density = [0] * bands
        for r in radii:
            b = min(bands - 1, int(r / rmax * bands)) if rmax > 0 else 0
            density[b] += 1
        return {
            "doubling_times_h": sorted(doubling),
            "birth_sizes_um3": sorted(birth),
            "volume_weighted_growth_h": vw_growth,
            "radial_density": density,
            "colony_radius_sites": rmax,
        }

    def dfba_stratification(self, quantile: float = 0.25,
                            min_cells: int = 4) -> dict:
        """Core-vs-edge metabolic stratification (Phase 5).

        Alive cells are ranked by distance from the colony centre; the
        ``quantile``-innermost form the ``core`` and the
        ``quantile``-outermost the ``edge`` group.  Returns each group's
        median local glucose / oxygen, mean growth rate and mean
        extracellular acetate.

        With dFBA disabled, no environment, or fewer than ``min_cells``
        alive cells, returns a zeroed summary.
        """
        out: dict[str, float | int] = {
            "core_cell_count": 0, "edge_cell_count": 0,
            "core_glucose_mm": 0.0, "edge_glucose_mm": 0.0,
            "core_oxygen_mm": 0.0, "edge_oxygen_mm": 0.0,
            "core_growth_rate_h": 0.0, "edge_growth_rate_h": 0.0,
            "core_acetate_mm": 0.0, "edge_acetate_mm": 0.0,
        }
        alive = [c for c in self.cells if c.alive]
        env = self.config.environment
        if not alive or env is None or len(alive) < min_cells:
            return out
        cx, cy = self._colony_center(alive)
        ranked = sorted(alive,
                        key=lambda c: math.hypot(c.x - cx, c.y - cy))
        n_core = max(1, int(len(ranked) * quantile))
        core = ranked[:n_core]
        edge = ranked[-n_core:]

        def _med(vals: list[float]) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            return s[len(s) // 2]

        def _local(c: PopulationCell, name: str) -> float:
            try:
                return env.substrate_at(c.x, c.y, name)
            except KeyError:
                return 0.0

        def _growth(c: PopulationCell) -> float:
            if c.dfba is not None and c.dfba.history:
                return c.dfba.growth_rate
            return 0.0

        def _acetate(c: PopulationCell) -> float:
            try:
                return env.get_field("acetate").get(c.x, c.y)
            except KeyError:
                return 0.0

        out["core_cell_count"] = len(core)
        out["edge_cell_count"] = len(edge)
        out["core_glucose_mm"] = _med([_local(c, "glucose") for c in core])
        out["edge_glucose_mm"] = _med([_local(c, "glucose") for c in edge])
        out["core_oxygen_mm"] = _med([_local(c, "oxygen") for c in core])
        out["edge_oxygen_mm"] = _med([_local(c, "oxygen") for c in edge])
        out["core_growth_rate_h"] = sum(_growth(c) for c in core) / len(core)
        out["edge_growth_rate_h"] = sum(_growth(c) for c in edge) / len(edge)
        out["core_acetate_mm"] = _med([_acetate(c) for c in core])
        out["edge_acetate_mm"] = _med([_acetate(c) for c in edge])
        return out


# ============================================================================
# 3D population (T2.7, NUFEB-style volume)
# ============================================================================

#: stability ceiling for the explicit 3D 7-point diffusion scheme (< 1/6)
_MAX_SUBSTEP_D_3D = 0.15


def _laplacian_step_3d(
    grid: list[list[list[float]]],
    d_lattice: float,
    w: int,
    h: int,
    depth: int,
) -> list[list[list[float]]]:
    """One explicit 7-point-Laplacian diffusion step in 3D (Neumann).

    up/down (y), left/right (x), front/back (z); zero-flux (reflecting)
    boundaries, matching the 5-point 2D scheme in :func:
    `signal_diffusion_step` but extended to the z axis (NUFEB 2019 3D
    chemical fields).
    """
    if _HAS_NUMPY:
        a = np.asarray(grid, dtype=float)
        padded = np.pad(a, 1, mode="edge")
        lap = (padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
               + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
               + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2]
               - 6.0 * a)
        new = a + d_lattice * lap
        np.clip(new, 0.0, None, out=new)
        return new.tolist()  # type: ignore[no-any-return]
    new_grid: list[list[list[float]]] = []
    for k in range(depth):
        plane = grid[k]
        front = grid[k - 1] if k > 0 else plane
        back = grid[k + 1] if k < depth - 1 else plane
        new_plane: list[list[float]] = []
        for i in range(h):
            row = plane[i]
            new_row: list[float] = []
            for j in range(w):
                cur = row[j]
                up = plane[i - 1][j] if i > 0 else cur
                down = plane[i + 1][j] if i < h - 1 else cur
                left = row[j - 1] if j > 0 else cur
                right = row[j + 1] if j < w - 1 else cur
                lap = (up + down + left + right + front[i][j] + back[i][j]
                       - 6.0 * cur)
                v = cur + d_lattice * lap
                new_row.append(v if v > 0.0 else 0.0)
            new_plane.append(new_row)
        new_grid.append(new_plane)
    return new_grid


def _neighbors_3d(x: int, y: int, z: int,
                  w: int, h: int, depth: int,
                  connectivity: int = 6) -> list[tuple[int, int, int]]:
    """In-bounds 3D neighbor coordinates of (x, y, z).

    ``connectivity=6``: face neighbors (up/down/left/right/front/back).
    ``connectivity=26``: full 3x3x3 neighborhood (z-neighborhoods).
    """
    if connectivity not in (6, 26):
        raise ValueError("connectivity must be 6 or 26")
    out: list[tuple[int, int, int]] = []
    if connectivity == 6:
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                           (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < w and 0 <= ny < h and 0 <= nz < depth:
                out.append((nx, ny, nz))
        return out
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < w and 0 <= ny < h and 0 <= nz < depth:
                    out.append((nx, ny, nz))
    return out


class CellPopulation3D(CellPopulation):
    """3D multicellular population (T2.7).

    Extends :class:`CellPopulation` with a z axis:

    - ``config.grid_depth`` (default 1) sets the lattice depth; cells
      carry a ``z`` coordinate.
    - The signal field is a ``[z][y][x]`` volume diffused with the
      explicit 7-point Laplacian (:func:`_laplacian_step_3d`, Neumann
      boundaries, sub-stepped so ``D_lattice <= 0.15 < 1/6`` stays
      stable — NUFEB 2019 3D chemical-field style).
    - :meth:`neighbors_3d` exposes z-neighborhoods (6- or 26-connectivity).
    - Division offsets along z as well as x/y.
    - :meth:`to_lsystem3d` exports the colony occupancy as an
      :class:`helixlang.morphology_3d.LSystem3D` geometry for
      morphology/rendering reuse.

    The metabolism, program-execution, quorum and death phases are
    inherited from the base class (per-cell state unchanged); only the
    spatial/signal bookkeeping is 3D.
    """

    def __init__(self, initial_cells: list[PopulationCell],
                 config: PopulationConfig = PopulationConfig(),
                 seed: int | None = None) -> None:
        if config.grid_depth < 1:
            raise ValueError("grid_depth must be >= 1")
        super().__init__(initial_cells, config, seed)
        self.signal_field: list[list[list[float]]] = [  # type: ignore[assignment]
            [[0.0] * config.grid_width for _ in range(config.grid_height)]
            for _ in range(config.grid_depth)
        ]

    @property
    def depth(self) -> int:
        return self.config.grid_depth

    def _in_bounds_3d(self, x: int, y: int, z: int) -> bool:
        return (0 <= x < self.config.grid_width
                and 0 <= y < self.config.grid_height
                and 0 <= z < self.config.grid_depth)

    def _emit_signal(self, cell: PopulationCell) -> None:
        """Accumulate AI-2 emission into the 3D ``[z][y][x]`` field."""
        if self._in_bounds_3d(cell.x, cell.y, cell.z):
            self.signal_field[cell.z][cell.y][cell.x] += SIGNAL_EMISSION_PER_STEP

    def _step_vectorized_metabolism(self) -> tuple[list[PopulationCell], int]:
        """Batch metabolism + signal emission + death in 3D (numpy).

        Mirrors the 2D scatter path (``_step_vectorized_metabolism``) but
        scatters signal emission into the ``[z][y][x]`` volume and bounds
        every axis.  Lets large 3D colonies (NUFEB 2019 / iDynoMiCS 2.0
        scale, 10^4..10^5 agents) advance at near-linear cost instead of
        the pure-Python per-cell loop.
        """
        config = self.config
        cells = self.cells
        alive_idx = [i for i, c in enumerate(cells) if c.alive]
        m = len(alive_idx)
        if m == 0:
            return [], 0

        energies = np.fromiter(
            (cells[i].energy for i in alive_idx), dtype=float, count=m)
        ages = np.fromiter(
            (cells[i].age for i in alive_idx), dtype=np.int64, count=m)
        xs = np.fromiter(
            (cells[i].x for i in alive_idx), dtype=np.int64, count=m)
        ys = np.fromiter(
            (cells[i].y for i in alive_idx), dtype=np.int64, count=m)
        zs = np.fromiter(
            (cells[i].z for i in alive_idx), dtype=np.int64, count=m)
        signals = np.fromiter(
            (cells[i].signal_emitted for i in alive_idx), dtype=float, count=m)

        energies += config.energy_intake - config.metabolic_cost
        ages += 1
        if config.signaling_enabled:
            signals += SIGNAL_EMISSION_PER_STEP
            in_bounds = ((xs >= 0) & (xs < config.grid_width)
                         & (ys >= 0) & (ys < config.grid_height)
                         & (zs >= 0) & (zs < config.grid_depth))
            if in_bounds.any():
                sig_field = np.asarray(self.signal_field, dtype=float)
                np.add.at(
                    sig_field,
                    (zs[in_bounds], ys[in_bounds], xs[in_bounds]),
                    SIGNAL_EMISSION_PER_STEP,
                )
                self.signal_field = sig_field.tolist()

        death_mask = energies <= config.death_threshold
        deaths = int(death_mask.sum())

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

    def _occupancy_3d(self, cells: list[PopulationCell]
                      ) -> list[list[list[int]]]:
        """Alive-cell count per ``[z][y][x]`` site (numpy scatter)."""
        w = self.config.grid_width
        h = self.config.grid_height
        d = self.config.grid_depth
        zs: list[int] = []
        ys: list[int] = []
        xs: list[int] = []
        for c in cells:
            if c.alive and self._in_bounds_3d(c.x, c.y, c.z):
                zs.append(c.z)
                ys.append(c.y)
                xs.append(c.x)
        if not zs:
            return [[[0] * w for _ in range(h)] for _ in range(d)]
        if _HAS_NUMPY:
            occ_arr = np.zeros((d, h, w), dtype=np.int64)
            np.add.at(occ_arr, (zs, ys, xs), 1)
            return occ_arr.tolist()
        occ = [[[0] * w for _ in range(h)] for _ in range(d)]
        for z, y, x in zip(zs, ys, xs, strict=True):
            occ[z][y][x] += 1
        return occ

    def _apply_mechanics_3d(self, cells: list[PopulationCell]) -> None:
        """3D relaxation of crowded cells (iDynoMiCS/CROMICS realization).

        ``shoving`` shoves excess cells to the nearest empty face
        neighbor; ``force`` displaces them toward the least-crowded site
        in the full 26-neighborhood (force-balance mechanics, Lardon et
        al. 2011 / Cockx et al. 2024), extended to the z axis.
        """
        config = self.config
        occ = self._occupancy_3d(cells)
        if config.mechanics == "shoving":
            for c in cells:
                if not c.alive or not self._in_bounds_3d(c.x, c.y, c.z):
                    continue
                if occ[c.z][c.y][c.x] <= 1:
                    continue
                for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                                   (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                    nx, ny, nz = c.x + dx, c.y + dy, c.z + dz
                    if (self._in_bounds_3d(nx, ny, nz)
                            and occ[nz][ny][nx] == 0):
                        occ[c.z][c.y][c.x] -= 1
                        occ[nz][ny][nx] = 1
                        c.x, c.y, c.z = nx, ny, nz
                        break
        elif config.mechanics == "force":
            for c in cells:
                if not c.alive or not self._in_bounds_3d(c.x, c.y, c.z):
                    continue
                occ_here = occ[c.z][c.y][c.x]
                if occ_here <= 1:
                    continue
                best: tuple[int, int, int, int] | None = None
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == dy == dz == 0:
                                continue
                            nx, ny, nz = c.x + dx, c.y + dy, c.z + dz
                            if not self._in_bounds_3d(nx, ny, nz):
                                continue
                            score = occ[nz][ny][nx]
                            if best is None or score < best[0]:
                                best = (score, nx, ny, nz)
                if best is not None and best[0] < occ_here:
                    occ[c.z][c.y][c.x] -= 1
                    occ[best[3]][best[2]][best[1]] += 1
                    c.x, c.y, c.z = best[1], best[2], best[3]

    def _apply_mechanics(self, cells: list[PopulationCell]) -> None:
        """3D dispatch for mechanics (shoving/force, 26-neighborhood)."""
        self._apply_mechanics_3d(cells)

    def neighbors_3d(self, x: int, y: int, z: int,
                     connectivity: int = 6) -> list[tuple[int, int, int]]:
        """In-bounds neighbor coordinates including the z axis."""
        return _neighbors_3d(x, y, z, self.config.grid_width,
                             self.config.grid_height, self.config.grid_depth,
                             connectivity=connectivity)

    def _diffuse(self, config: PopulationConfig) -> list[list[list[float]]]:  # type: ignore[override]
        """3D signal diffusion (7-point Laplacian, stable sub-steps).

        With numpy the field stays a single array across the (potentially
        hundreds of) stability-constrained sub-steps, so large volumes
        (biofilm-scale lattices) diffuse without per-sub-step list<->array
        round trips.
        """
        d_lattice = diffusion_to_lattice(
            config.signal_diffusion, DIFFUSION_DT_S, LATTICE_SPACING_UM)
        if d_lattice <= 0.0:
            return self.signal_field
        n = math.ceil(d_lattice / _MAX_SUBSTEP_D_3D)
        dt = d_lattice / n
        w = config.grid_width
        h = config.grid_height
        depth = config.grid_depth
        if _HAS_NUMPY:
            a = np.asarray(self.signal_field, dtype=float)
            for _ in range(n):
                padded = np.pad(a, 1, mode="edge")
                lap = (padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
                       + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
                       + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2]
                       - 6.0 * a)
                a = a + dt * lap
                np.clip(a, 0.0, None, out=a)
            return a.tolist()  # type: ignore[no-any-return]
        field = self.signal_field
        for _ in range(n):
            field = _laplacian_step_3d(field, dt, w, h, depth)
        return field

    def occupancy_3d(self) -> list[list[list[int]]]:
        """Alive-cell count per site, indexed [z][y][x]."""
        return self._occupancy_3d(self.cells)

    def _divide_3d(self, cell: PopulationCell, config: PopulationConfig,
                   rng: random.Random
                   ) -> tuple[PopulationCell, PopulationCell]:
        """Binary fission with a z offset (daughters separate in 3D)."""
        a, b = divide_cell(cell, config, rng)
        if config.grid_depth > 1:
            dz = rng.choice((-1, 0, 1))
            a.z = _clamp(cell.z + dz, 0, config.grid_depth - 1)
            b.z = _clamp(cell.z - dz, 0, config.grid_depth - 1)
        return a, b

    def step(self) -> dict:
        """Advance the population one tick (3D signal/division path)."""
        alive_before = sum(1 for c in self.cells if c.alive)
        self._step_start_alive = alive_before

        if (_HAS_NUMPY and alive_before > 100
                and self.config.environment is None):
            metabolized, deaths = self._step_vectorized_metabolism()
        else:
            metabolized, deaths = self._step_metabolism_python()

        if self.config.program is not None and metabolized:
            metabolized, prog_deaths = self._step_programs(metabolized)
            deaths += prog_deaths

        config = self.config
        if config.signaling_enabled:
            self.signal_field = self._diffuse(config)

        divisions_allowed = max(0, config.max_size - len(metabolized))
        divisions_done = 0
        next_cells: list[PopulationCell] = []
        program_controls = (config.program is not None
                            and config.program_controlled_division)
        for cell in metabolized:
            if (config.signaling_enabled
                    and self._in_bounds_3d(cell.x, cell.y, cell.z)):
                sig = self.signal_field[cell.z][cell.y][cell.x]
                quorum_sensing(cell, sig, config.signal_threshold)
            wants_division = (cell.flag_divide
                              if program_controls
                              else cell.energy >= config.division_threshold)
            if (wants_division
                    and cell.energy >= config.division_threshold
                    and divisions_done < divisions_allowed):
                a, b = self._divide_3d(cell, config, self.rng)
                divisions_done += 1
                next_cells.append(a)
                next_cells.append(b)
            else:
                next_cells.append(cell)

        if config.mechanics is not None and next_cells:
            self._apply_mechanics(next_cells)

        self.cells = next_cells
        self._assign_ids()
        self._last_divisions = divisions_done
        self._last_deaths = deaths
        self._total_deaths += deaths
        self._generation += 1

        if config.environment is not None:
            config.environment.step()
        if config.trace_streaming:
            self._append_trace()
        return self.get_statistics()

    def get_signal_field(self) -> list[list[list[float]]]:  # type: ignore[override]
        """Return the 3D signal field (a copy), indexed [z][y][x]."""
        return [[row[:] for row in plane] for plane in self.signal_field]

    def to_lsystem3d(self, step: float = 1.0,
                     ) -> LSystem3D:
        """Export the colony's 3D occupancy as an LSystem3D morphology.

        Builds an axiom that navigates the 3D turtle to every occupied
        site and drops a short ``F`` marker there, so the derived
        ``Line3D`` geometry is a 3D dot-cloud of the colony at the
        correct lattice coordinates.  Reuses
        :class:`helixlang.morphology_3d.LSystem3D` for drawing/rendering
        (NUFEB-style spatial output).
        """
        from helixlang.morphology_3d import LSystem3D

        occupied = sorted(
            (c.x, c.y, c.z) for c in self.cells
            if c.alive and self._in_bounds_3d(c.x, c.y, c.z))
        if not occupied:
            return LSystem3D(axiom="", rules={}, angle=90.0, step=step)
        # Per site: [ push origin -> yaw to +X, travel x -> restore +Y,
        #            travel +Y -> pitch to +Z, travel z -> restore +Y
        #            -> F marker -> ] pop.  angle=90 makes turns exact.
        parts: list[str] = []
        for x, y, z in occupied:
            parts.append("[")
            if x > 0:
                parts.append("-")          # +Y -> +X (yaw left 90 deg)
                parts.append("f" * x)
                parts.append("+")          # +X -> +Y
            if y > 0:
                parts.append("f" * y)      # travel along +Y
            if z > 0:
                parts.append("^")          # +Y -> +Z (pitch up 90 deg)
                parts.append("f" * z)
                parts.append("&")          # +Z -> +Y
            parts.append("F")
            parts.append("]")
        return LSystem3D(axiom="".join(parts), rules={},
                         angle=90.0, step=step)


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
