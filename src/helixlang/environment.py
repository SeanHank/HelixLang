"""Extracellular environment fields: diffusing substrates (glucose, O2)
with Monod / Michaelis-Menten uptake kinetics.

Biological grounding (physical units from :mod:`helixlang.units`; one
tick = 1 min, lattice site edge = 10 µm -> site volume ~1e-15 L):

- Diffusion follows Fick's law with a physical coefficient in µm^2/s,
  converted to the dimensionless on-lattice form via
  :func:`helixlang.units.diffusion_to_lattice` (same sub-stepped
  explicit 5-point Laplacian as the AI-2 field in
  :mod:`helixlang.population`).
  - Glucose in water: D ~ 6e-6 cm^2/s = 600 µm^2/s (Stewart 2003,
    Adv Microb Ecol; glucose D_water 6.7e-6 cm^2/s, CRC Handbook).
  - Oxygen in water: D ~ 2.5e-5 cm^2/s = 2500 µm^2/s (CRC Handbook).
- Substrate uptake follows saturation kinetics:
  - Monod growth-saturating term ``v_max * S / (Ks + S)`` (Monod
    1949, Annu Rev Microbiol 3:371; Kovárová-Kovar & Egli 1998,
    Microbiol Mol Biol Rev 62:646).  E. coli Ks(glucose) ~ 0.01-0.2 mM,
    default 0.1 mM; Ks(O2) ~ 0.05 mM.
  - The Michaelis-Menten transporter form is identical; provided as a
    named alias for clarity.
- A flowing medium (chemostat / biofilm bulk flow) is approximated by a
  per-tick ``flow_rate``: each tick a fraction of each site's volume is
  replaced by the source (bulk) concentration (Beyenal & Babauta
  2015, *Biofilms in Bioelectrochemical Systems*; iDynoMiCS 2.0 bulk
  medium description, Cockx et al. 2024).

Energy yield: full glucose oxidation yields 38 ATP per glucose
(Alberts, *Molecular Biology of the Cell*; :data:`helixlang.units.
ATP_PER_GLUCOSE`).  A site at concentration S mM holds ``S * N_A *
V_site`` glucose molecules; per-cell ATP intake scales as the consumed
molecule count times 38.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from helixlang.units import (
    ATP_PER_GLUCOSE,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    diffusion_to_lattice,
)

# numpy is optional (pure-Python fallback, matching the rest of the runtime)
try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is an optional extra
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ============================================================================
# Physical constants
# ============================================================================

#: glucose diffusion coefficient in water, µm^2/s (D ~ 6e-6 cm^2/s,
#: Stewart 2003; CRC Handbook of Chemistry and Physics)
GLUCOSE_DIFFUSION_UM2_S = 600.0
#: oxygen diffusion coefficient in water, µm^2/s (D ~ 2.5e-5 cm^2/s,
#: CRC Handbook of Chemistry and Physics)
OXYGEN_DIFFUSION_UM2_S = 2500.0
#: acetate diffusion coefficient in water, µm^2/s (small organic acid,
#: D ~ 1.2e-5 cm^2/s, CRC Handbook of Chemistry and Physics)
ACETATE_DIFFUSION_UM2_S = 1200.0
#: E. coli Monod half-saturation constant for glucose, mM
#: (Kovárová-Kovar & Egli 1998: Ks ~ 0.01-0.2 mM)
GLUCOSE_HALF_SATURATION_MM = 0.1
#: E. coli Monod half-saturation constant for oxygen, mM (typical
#: measured Ks ~ 0.05 mM, Kovárová-Kovar & Egli 1998)
OXYGEN_HALF_SATURATION_MM = 0.05
#: air-saturated water O2 concentration at 25 °C, mM (~0.21 mM)
BULK_OXYGEN_MM = 0.21
#: glucose concentration of a typical rich medium, mM (LB and similar
#: complex media ~1 mM glucose-equivalent, iDynoMiCS 2.0 defaults)
BULK_GLUCOSE_MM = 1.0
#: site volume in litres: (10 µm)^3 = 1000 µm^3 = 1e-12 L
SITE_VOLUME_L = (LATTICE_SPACING_UM ** 3) * 1e-15

#: Avogadro constant (molecules/mol)
_AVOGADRO = 6.022e23


# ============================================================================
# Saturation kinetics
# ============================================================================
def monod_uptake(
    v_max: float,
    substrate_concentration: float,
    half_saturation: float,
) -> float:
    """Monod growth-saturating uptake term ``v_max * S / (Ks + S)``.

    (Monod 1949; Kovárová-Kovar & Egli 1998).  At ``S = Ks`` uptake is
    half of ``v_max``; at ``S >> Ks`` uptake saturates at ``v_max``.
    Concentrations must be non-negative; ``half_saturation`` must be
    positive (real enzymes have Ks > 0).

    Args:
        v_max: maximum uptake rate (any units/time)
        substrate_concentration: substrate concentration S (mM)
        half_saturation: Monod constant Ks (mM)

    Returns:
        the uptake rate in ``v_max`` units
    """
    if v_max < 0.0:
        raise ValueError("v_max must be >= 0")
    if substrate_concentration < 0.0:
        raise ValueError("substrate_concentration must be >= 0")
    if half_saturation <= 0.0:
        raise ValueError("half_saturation must be > 0")
    return v_max * substrate_concentration / (half_saturation + substrate_concentration)


def michaelis_menten_rate(
    v_max: float,
    substrate_concentration: float,
    km: float,
) -> float:
    """Michaelis-Menten transporter rate ``v_max * S / (Km + S)``.

    The standard enzyme-kinetics form (Michaelis & Menten 1913,
    Biochem Z 49:333); identical algebra to :func:`monod_uptake` and
    provided as a named alias so transporter characterizations in the
    literature (Km) map directly onto the code.
    """
    return monod_uptake(v_max, substrate_concentration, km)


def molecules_per_site(concentration_mm: float) -> float:
    """Molecule count of a substrate at ``concentration_mm`` mM in one
    lattice site.

    ``c_mM * 1e-3 * N_A * V_site``; e.g. 1 mM glucose in a (10 µm)^3
    site holds ~6e8 molecules (a 10 µm cube = 1e-12 L).  Used to convert
    field depletions into ATP intake (38 ATP per glucose, Alberts).
    """
    if concentration_mm < 0.0:
        raise ValueError("concentration_mm must be >= 0")
    return concentration_mm * 1e-3 * _AVOGADRO * SITE_VOLUME_L


def atp_yield(glucose_molecules: float) -> float:
    """ATP produced from full aerobic oxidation of ``glucose_molecules``
    glucose (38 ATP/glucose, Alberts)."""
    return glucose_molecules * ATP_PER_GLUCOSE


#: CROMICS critical volume fraction above which crowding measurably
#: alters microbial dynamics (Angeles-Martinez & Hatzimanikatis 2021,
#: PLoS Comput Biol 17:e1009158: "cells occupy more than 14% of the
#: volume fraction").
CROMICS_CRITICAL_VOLUME_FRACTION = 0.14


def crowding_diffusion_factor(volume_fraction: float) -> float:
    """Effective-diffusion reduction factor for a solute in a crowded
    medium, ``D_eff = factor * D0``.

    CROMICS (Angeles-Martinez & Hatzimanikatis 2021, PLoS Comput Biol
    17:e1009140) computes the effective diffusion of a metabolite as
    ``D_eff,met = (1/gamma_met) * D0_met`` where ``gamma_met`` is the
    scaled-particle-theory (SPT) activity coefficient (ratio of total to
    available volume).  For solutes much smaller than the crowding
    cells (glucose, O2 vs. E. coli) the paper's Eq. 3 reduces exactly
    to the free volume fraction not occupied by cells::

        factor(phi) = 1 - phi

    Args:
        volume_fraction: local volume fraction occupied by cells/
            biomass, in [0, 1).

    Returns:
        ``1 - volume_fraction`` clamped to [0, 1] (``0`` at close
        packing, ``1`` in empty medium).
    """
    if volume_fraction < 0.0 or volume_fraction >= 1.0:
        raise ValueError("volume_fraction must be in [0, 1)")
    return max(0.0, 1.0 - volume_fraction)


# ============================================================================
# Fields
# ============================================================================
class ConcentrationField:
    """A 2D substrate concentration field with Fickian diffusion.

    Concentrations are in mM; diffusion uses the physical µm^2/s
    coefficient converted to the stable on-lattice form at the declared
    lattice edge (explicit 5-point Laplacian, zero-flux boundaries,
    sub-stepped so ``D_lattice <= 0.25`` per step).
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        diffusion_um2_s: float,
        initial_concentration: float = 0.0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("field dimensions must be > 0")
        self.name = name
        self.width = width
        self.height = height
        self.diffusion_um2_s = float(diffusion_um2_s)
        self.concentration: list[list[float]] = [
            [float(initial_concentration)] * width for _ in range(height)
        ]

    def _d_lattice(self) -> float:
        return diffusion_to_lattice(
            self.diffusion_um2_s, DIFFUSION_DT_S, LATTICE_SPACING_UM)

    def get(self, x: int, y: int) -> float:
        """Concentration (mM) at lattice position (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        return self.concentration[y][x]

    def set(self, x: int, y: int, value: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.concentration[y][x] = max(0.0, float(value))

    def add(self, x: int, y: int, amount: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.concentration[y][x] = max(
                0.0, self.concentration[y][x] + amount)

    def deplete(self, x: int, y: int, amount: float) -> float:
        """Remove ``amount`` (mM) at (x, y); returns the amount actually
        removed (never more than available, never negative)."""
        if amount <= 0.0 or not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        cur = self.concentration[y][x]
        removed = min(cur, amount)
        self.concentration[y][x] = cur - removed
        return removed

    def snapshot(self) -> list[list[float]]:
        """Copy of the concentration grid, indexed [y][x]."""
        return [row[:] for row in self.concentration]

    def diffuse(self) -> None:
        """Advance diffusion by one tick (sub-stepped stable scheme)."""
        d = self._d_lattice()
        if d <= 0.0:
            return
        steps = max(1, math.ceil(d / 0.25))
        d_sub = d / steps
        grid = self.concentration
        w, h = self.width, self.height
        for _ in range(steps):
            grid = _laplacian_step(grid, d_sub, w, h)
        self.concentration = grid

    def total_mm(self) -> float:
        """Total substrate in the field (mM x sites)."""
        return sum(sum(row) for row in self.concentration)


def _laplacian_step(
    grid: list[list[float]],
    d_lattice: float,
    w: int,
    h: int,
) -> list[list[float]]:
    """One explicit 5-point-Laplacian diffusion step (Neumann boundaries)."""
    if _HAS_NUMPY:
        a = _np.asarray(grid, dtype=float)
        padded = _np.pad(a, 1, mode="edge")
        lap = (padded[:-2, 1:-1] + padded[2:, 1:-1]
               + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * a)
        new = a + d_lattice * lap
        _np.clip(new, 0.0, None, out=new)
        result: list[list[float]] = new.tolist()
        return result
    new_grid: list[list[float]] = []
    for i in range(h):
        row = grid[i]
        new_row: list[float] = []
        for j in range(w):
            cur = row[j]
            up = grid[i - 1][j] if i > 0 else cur
            down = grid[i + 1][j] if i < h - 1 else cur
            left = row[j - 1] if j > 0 else cur
            right = row[j + 1] if j < w - 1 else cur
            lap = up + down + left + right - 4.0 * cur
            v = cur + d_lattice * lap
            new_row.append(v if v > 0.0 else 0.0)
        new_grid.append(new_row)
    return new_grid


#: stability ceiling for the explicit 3D 7-point scheme (must be < 1/6)
_MAX_SUBSTEP_D_3D = 0.15


class ConcentrationField3D:
    """A 3D substrate concentration field (T2.7, NUFEB-style 3D volume).

    Same physics as :class:`ConcentrationField` (mM, Fickian diffusion,
    physical µm^2/s -> on-lattice conversion) but over a
    ``width x height x depth`` box using the explicit 7-point Laplacian
    (up/down/left/right/front/back) with Neumann (reflecting) boundaries.
    Sub-stepped so ``D_lattice <= 0.15 < 1/6`` per step, which keeps the
    explicit 3D scheme stable.  Stored ``[z][y][x]``.

    References:
    - NUFEB 2019 (Kick et al. Commun Comput Phys): 3D individual-based
      microbial simulation with chemical fields
    - Fick's law; explicit finite-difference 3D Laplacian (Press et al.
      Numerical Recipes)
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        depth: int,
        diffusion_um2_s: float,
        initial_concentration: float = 0.0,
    ) -> None:
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError("field dimensions must be > 0")
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        self.diffusion_um2_s = float(diffusion_um2_s)
        self.concentration: list[list[list[float]]] = [
            [[float(initial_concentration)] * width for _ in range(height)]
            for _ in range(depth)
        ]

    def _d_lattice(self) -> float:
        return diffusion_to_lattice(
            self.diffusion_um2_s, DIFFUSION_DT_S, LATTICE_SPACING_UM)

    def get(self, x: int, y: int, z: int) -> float:
        """Concentration (mM) at lattice position (x, y, z)."""
        if not (0 <= x < self.width and 0 <= y < self.height
                and 0 <= z < self.depth):
            return 0.0
        return self.concentration[z][y][x]

    def set(self, x: int, y: int, z: int, value: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height \
                and 0 <= z < self.depth:
            self.concentration[z][y][x] = max(0.0, float(value))

    def add(self, x: int, y: int, z: int, amount: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height \
                and 0 <= z < self.depth:
            self.concentration[z][y][x] = max(
                0.0, self.concentration[z][y][x] + amount)

    def deplete(self, x: int, y: int, z: int, amount: float) -> float:
        """Remove ``amount`` (mM) at (x, y, z); returns what was removed."""
        if amount <= 0.0 or not (0 <= x < self.width and 0 <= y < self.height
                                 and 0 <= z < self.depth):
            return 0.0
        cur = self.concentration[z][y][x]
        removed = min(cur, amount)
        self.concentration[z][y][x] = cur - removed
        return removed

    def snapshot(self) -> list[list[list[float]]]:
        """Copy of the concentration volume, indexed [z][y][x]."""
        return [[row[:] for row in plane] for plane in self.concentration]

    def layer(self, z: int) -> list[list[float]]:
        """A 2D horizontal slice at depth ``z``, indexed [y][x]."""
        if not 0 <= z < self.depth:
            raise ValueError(f"z={z} out of range [0, {self.depth})")
        return [row[:] for row in self.concentration[z]]

    def diffuse(self) -> None:
        """Advance diffusion by one tick (sub-stepped 3D stable scheme)."""
        d = self._d_lattice()
        if d <= 0.0:
            return
        steps = max(1, math.ceil(d / _MAX_SUBSTEP_D_3D))
        d_sub = d / steps
        grid = self.concentration
        w, h, depth = self.width, self.height, self.depth
        for _ in range(steps):
            grid = _laplacian_step_3d(grid, d_sub, w, h, depth)
        self.concentration = grid

    def total_mm(self) -> float:
        """Total substrate in the field (mM x sites)."""
        return sum(sum(sum(row) for row in plane)
                   for plane in self.concentration)


def _laplacian_step_3d(
    grid: list[list[list[float]]],
    d_lattice: float,
    w: int,
    h: int,
    depth: int,
) -> list[list[list[float]]]:
    """One explicit 7-point-Laplacian diffusion step in 3D (Neumann)."""
    if _HAS_NUMPY:
        a = _np.asarray(grid, dtype=float)
        padded = _np.pad(a, 1, mode="edge")
        lap = (padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
               + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
               + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2]
               - 6.0 * a)
        new = a + d_lattice * lap
        _np.clip(new, 0.0, None, out=new)
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


# ============================================================================
# Environment
# ============================================================================
@dataclass(slots=True)
class EnvironmentConfig:
    """Environment configuration.

    Args:
        width, height: lattice dimensions (site edge = 10 µm)
        flow_rate: per-tick fraction of each site's volume replaced by
            bulk medium (chemostat / biofilm flow, 0 = batch closed)
        bulk_glucose_mm: source (bulk) glucose concentration, mM
        bulk_oxygen_mm: source (bulk) oxygen concentration, mM
        glucose_diffusion_um2_s: physical glucose diffusion coefficient
        oxygen_diffusion_um2_s: physical oxygen diffusion coefficient
        glucose_initial_mm: initial glucose concentration, mM
        oxygen_initial_mm: initial oxygen concentration, mM
    """

    width: int = 100
    height: int = 100
    flow_rate: float = 0.0
    bulk_glucose_mm: float = BULK_GLUCOSE_MM
    bulk_oxygen_mm: float = BULK_OXYGEN_MM
    glucose_diffusion_um2_s: float = GLUCOSE_DIFFUSION_UM2_S
    oxygen_diffusion_um2_s: float = OXYGEN_DIFFUSION_UM2_S
    glucose_initial_mm: float = BULK_GLUCOSE_MM
    oxygen_initial_mm: float = BULK_OXYGEN_MM


class Environment:
    """Extracellular medium: diffusing substrate fields + flow.

    Usage::

        env = Environment(EnvironmentConfig(width=50, height=50))
        env.step()                     # diffuse + flow refresh
        v = env.glucose.get(x, y)      # local concentration (mM)
        uptake = monod_uptake(1.0, v, GLUCOSE_HALF_SATURATION_MM)
        env.glucose.deplete(x, y, uptake)
    """

    def __init__(self, config: EnvironmentConfig = EnvironmentConfig()) -> None:
        if config.width <= 0 or config.height <= 0:
            raise ValueError("environment dimensions must be > 0")
        if not 0.0 <= config.flow_rate <= 1.0:
            raise ValueError("flow_rate must be in [0, 1]")
        self.config = config
        self.glucose = ConcentrationField(
            "glucose", config.width, config.height,
            config.glucose_diffusion_um2_s, config.glucose_initial_mm)
        self.oxygen = ConcentrationField(
            "oxygen", config.width, config.height,
            config.oxygen_diffusion_um2_s, config.oxygen_initial_mm)
        self.fields: dict[str, ConcentrationField] = {
            "glucose": self.glucose,
            "oxygen": self.oxygen,
        }
        self.tick = 0

    def add_field(self, name: str, field: ConcentrationField) -> None:
        """Register an additional named substrate field."""
        if field.width != self.config.width or field.height != self.config.height:
            raise ValueError(
                f"field {name!r} dimensions must match the environment")
        self.fields[name] = field

    def get_field(self, name: str) -> ConcentrationField:
        """Look up a substrate field by name."""
        return self.fields[name]

    def step(self) -> None:
        """Advance the environment one tick: diffuse every field, then
        apply the flow refresh (chemostat mixing toward the bulk)."""
        for field in self.fields.values():
            field.diffuse()
        if self.config.flow_rate > 0.0:
            self._replenish()
        self.tick += 1

    def _replenish(self) -> None:
        """Chemostat refresh: replace ``flow_rate`` of every site with
        bulk medium of the matching source concentration."""
        cfg = self.config
        source: dict[str, float] = {
            "glucose": cfg.bulk_glucose_mm,
            "oxygen": cfg.bulk_oxygen_mm,
        }
        for name, field in self.fields.items():
            bulk = source.get(name)
            if bulk is None:
                continue
            for y in range(field.height):
                row = field.concentration[y]
                for x in range(field.width):
                    row[x] = row[x] + cfg.flow_rate * (bulk - row[x])

    def substrate_at(self, x: int, y: int, name: str = "glucose") -> float:
        """Concentration (mM) of ``name`` at (x, y)."""
        return self.fields[name].get(x, y)

    def local_uptake(self, x: int, y: int, name: str = "glucose",
                     half_saturation: float | None = None,
                     v_max: float = 1.0) -> float:
        """Monod uptake rate (in ``v_max`` units) at (x, y)."""
        field = self.fields[name]
        if half_saturation is None:
            if name == "glucose":
                half_saturation = GLUCOSE_HALF_SATURATION_MM
            elif name == "oxygen":
                half_saturation = OXYGEN_HALF_SATURATION_MM
            else:
                half_saturation = GLUCOSE_HALF_SATURATION_MM
        return monod_uptake(v_max, field.get(x, y), half_saturation)


__all__ = [
    "GLUCOSE_DIFFUSION_UM2_S", "OXYGEN_DIFFUSION_UM2_S",
    "ACETATE_DIFFUSION_UM2_S",
    "GLUCOSE_HALF_SATURATION_MM", "OXYGEN_HALF_SATURATION_MM",
    "BULK_OXYGEN_MM", "BULK_GLUCOSE_MM", "SITE_VOLUME_L",
    "monod_uptake", "michaelis_menten_rate",
    "molecules_per_site", "atp_yield",
    "ConcentrationField", "EnvironmentConfig", "Environment",
    "ConcentrationField3D",
]
