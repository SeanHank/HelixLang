"""Spatial dynamic FBA: substrate-gradient colony growth (S6).

Couples dynamic flux balance analysis (dFBA, Mahadevan et al. 2002,
Biophys J 83:1905-1928) to a one-dimensional substrate field.  Each site
along a strip is a well-mixed dFBA batch (Michaelis-Menten glucose
uptake bound + instantaneous FBA LP per step, the "static optimization
approach" of Mahadevan 2002) coupled to its local glucose concentration;
the glucose field diffuses between sites (biofilm-reduced effective
diffusion) and is optionally resupplied at a fixed-concentration inlet,
producing the travelling substrate-depletion waves and spatially graded
growth of a microbial biofilm / plug-flow reactor (Rittmann & McCarty,
Environmental Biotechnology: Principles and Applications, 2020).

Model notes
-----------
- Glucose free-solution diffusion in water is ~600 µm^2/s; inside a
  biofilm the effective coefficient is strongly reduced, so the default
  here is 2.0 µm^2/s (biofilm-effective; the physical conversion via
  :func:`helixlang.units.diffusion_to_lattice` keeps units honest).
- The reduced 37-reaction core model produces a fermentative CO2
  overflow during glucose-limited growth and has no glyoxylate shunt,
  so the classic second (acetate-utilizing) growth phase cannot occur
  (same limitation as :class:`~helixlang.metabolism.DynamicFluxBalance`).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from helixlang.metabolism import (
    ECOLI_CORE_MODEL,
    DynamicFBAConfig,
    DynamicFluxBalance,
)
from helixlang.units import (
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    diffusion_to_lattice,
)


@dataclass(slots=True)
class SpatialDFBAConfig:
    """Configuration of the 1-D spatial dFBA strip.

    Args:
        length: number of sites along the strip.
        glucose_diffusion_um2_s: effective glucose diffusion coefficient
            (µm^2/s); free-solution glucose is ~600, biofilms reduce it
            by orders of magnitude.
        initial_glucose_mm: uniform initial glucose (mM) at every site
            (ignored when ``initial_glucose_profile`` is given).
        initial_glucose_profile: explicit per-site initial glucose (mM);
            lets the caller seed a spatial gradient (a glucose reservoir
            at one end), which diffusion then smooths while growth
            consumes it.
        inlet_glucose_mm: glucose (mM) maintained at site 0 (an open
            reservoir / flow inlet).  None closes the system so the
            initial glucose is conserved (consumption only).
        initial_biomass_gdw: starting biomass (gDW/L) per site.
        max_biomass_gdw: biomass carrying capacity (gDW/L) per site
            (biofilm-thickness / crowding limit); None for unlimited
            batch growth.  With an open inlet the site biomass would
            otherwise grow exponentially forever, hogging the substrate.
        dt_h: dFBA integration step (hours).
        max_glucose_uptake: glucose uptake v_max (mmol/gDW/h); None
            keeps the model default.
        seed: RNG seed (reserved).
    """

    length: int = 32
    glucose_diffusion_um2_s: float = 2.0
    initial_glucose_mm: float = 5.0
    initial_glucose_profile: list[float] | None = None
    inlet_glucose_mm: float | None = 5.0
    initial_biomass_gdw: float = 0.05
    max_biomass_gdw: float | None = None
    dt_h: float = 0.05
    max_glucose_uptake: float | None = None
    seed: int | None = None


@dataclass(slots=True)
class SpatialDFBASite:
    """Integrated state of one strip site."""

    index: int
    biomass_gdw: float
    glucose_mm: float
    acetate_mm: float = 0.0
    glucose_consumed_mm: float = 0.0


class SpatialDFBA:
    """1-D spatial dFBA: dFBA batches coupled by glucose diffusion."""

    def __init__(self, config: SpatialDFBAConfig | None = None) -> None:
        self.config = config or SpatialDFBAConfig()
        if self.config.length <= 0:
            raise ValueError("length must be >= 1")
        self.rng = random.Random(self.config.seed)
        if self.config.initial_glucose_profile is not None:
            if len(self.config.initial_glucose_profile) != self.config.length:
                raise ValueError(
                    "initial_glucose_profile must have one entry per site")
            self.field = [float(v) for v in self.config.initial_glucose_profile]
        else:
            self.field = [
                self.config.initial_glucose_mm] * self.config.length
        self.batches: list[DynamicFluxBalance] = []
        fba_cfg = DynamicFBAConfig(
            dt_h=self.config.dt_h,
            initial_biomass_gdw=self.config.initial_biomass_gdw,
            initial_glucose_mm=self.config.initial_glucose_mm,
        )
        for _ in range(self.config.length):
            batch = DynamicFluxBalance(ECOLI_CORE_MODEL, config=fba_cfg)
            if self.config.max_glucose_uptake is not None:
                batch.fba.set_uptake("GLC", self.config.max_glucose_uptake)
            self.batches.append(batch)
        self.consumed: list[float] = [0.0] * self.config.length
        self.tick = 0
        self.history: list[dict] = []
        self.time_h = 0.0

    # -- internals ----------------------------------------------------------

    def _d_lattice(self) -> float:
        return diffusion_to_lattice(
            self.config.glucose_diffusion_um2_s,
            DIFFUSION_DT_S, LATTICE_SPACING_UM)

    def _diffuse(self) -> None:
        """1-D explicit diffusion with stable sub-steps (Neumann bounds)."""
        d = self._d_lattice()
        if d <= 0.0:
            return
        n = max(1, math.ceil(d / 0.25))
        d_sub = d / n
        c = self.field
        for _ in range(n):
            new = [0.0] * len(c)
            for i in range(len(c)):
                left = c[i - 1] if i > 0 else c[i]
                right = c[i + 1] if i < len(c) - 1 else c[i]
                v = c[i] + d_sub * (left + right - 2.0 * c[i])
                new[i] = v if v > 0.0 else 0.0
            c = new
        self.field = c

    # -- main loop ----------------------------------------------------------

    def step(self) -> dict:
        """Advance one dFBA step; returns the state snapshot."""
        cfg = self.config
        for i, batch in enumerate(self.batches):
            before = self.field[i]
            batch.set_state(glucose_mm=before)
            batch.step(cfg.dt_h)
            if cfg.max_biomass_gdw is not None:
                batch.biomass_gdw = min(batch.biomass_gdw,
                                        cfg.max_biomass_gdw)
            removed = before - batch.glucose_mm
            if removed < 0.0:
                removed = 0.0
            self.consumed[i] += removed
            self.field[i] = batch.glucose_mm
        self._diffuse()
        if cfg.inlet_glucose_mm is not None:
            self.field[0] = cfg.inlet_glucose_mm
        self.tick += 1
        self.time_h += cfg.dt_h
        snapshot = {
            "tick": self.tick,
            "time_h": self.time_h,
            "glucose": list(self.field),
            "biomass": [b.biomass_gdw for b in self.batches],
            "acetate": [b.byproducts_mm.get("acetate", 0.0)
                        for b in self.batches],
            "consumed": list(self.consumed),
        }
        self.history.append(snapshot)
        return snapshot

    def run(self, n_steps: int) -> list[dict]:
        """Run ``n_steps`` dFBA steps; returns :attr:`history`."""
        for _ in range(n_steps):
            self.step()
        return self.history

    # -- queries ------------------------------------------------------------

    def glucose_profile(self) -> list[float]:
        return list(self.field)

    def biomass_profile(self) -> list[float]:
        return [b.biomass_gdw for b in self.batches]

    def acetate_profile(self) -> list[float]:
        return [b.byproducts_mm.get("acetate", 0.0) for b in self.batches]

    def byproduct_profile(self, pool: str) -> list[float]:
        """Per-site accumulated byproduct (e.g. ``"co2"``, ``"acetate"``)."""
        return [b.byproducts_mm.get(pool, 0.0) for b in self.batches]

    def total_byproduct(self, pool: str) -> float:
        """Total accumulated byproduct across the strip."""
        return sum(self.byproduct_profile(pool))

    def consumed_profile(self) -> list[float]:
        return list(self.consumed)

    def total_glucose(self) -> float:
        return sum(self.field)

    def total_biomass(self) -> float:
        return sum(self.biomass_profile())

    def total_acetate(self) -> float:
        return sum(self.acetate_profile())

    def total_consumed(self) -> float:
        return sum(self.consumed)

    def sites(self) -> list[SpatialDFBASite]:
        """Per-site integrated state as dataclasses."""
        out: list[SpatialDFBASite] = []
        for i, batch in enumerate(self.batches):
            out.append(SpatialDFBASite(
                index=i,
                biomass_gdw=batch.biomass_gdw,
                glucose_mm=self.field[i],
                acetate_mm=batch.byproducts_mm.get("acetate", 0.0),
                glucose_consumed_mm=self.consumed[i],
            ))
        return out

    def depletion_front(self, threshold_mm: float = 0.5) -> int:
        """First site from the inlet whose glucose is below ``threshold``.

        In the open-inlet configuration the front of the substrate-
        depletion wave travels downstream as biomass consumes glucose
        (a travelling wave in the reaction-diffusion sense).
        """
        for i, g in enumerate(self.field):
            if g < threshold_mm:
                return i
        return self.config.length

    def colonization_front(self, biomass_threshold_gdw: float = 0.5
                           ) -> int:
        """First site from the inlet whose biomass is below ``threshold``.

        With glucose initially absent downstream, the growing biomass
        colonizes sites progressively as glucose diffuses in -- the
        front of the growth wave advances downstream.
        """
        for i, b in enumerate(self.biomass_profile()):
            if b < biomass_threshold_gdw:
                return i
        return self.config.length


__all__ = [
    "SpatialDFBAConfig", "SpatialDFBASite", "SpatialDFBA",
]
