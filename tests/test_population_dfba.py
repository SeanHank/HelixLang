"""Spatial dFBA tests: diffusion-coupled metabolic growth (S6).

Verification goals:
- In a closed system (no inlet) the total glucose mass is conserved:
  whatever the growing biomass consumes is exactly the initial pool minus
  what remains (consumption is the only glucose sink; diffusion
  conserves mass and Neumann boundaries leak nothing).
- An initially graded glucose field seeds graded growth: the high-glucose
  end of the strip accumulates more biomass than the low-glucose end.
- Raising the effective diffusion coefficient flattens the glucose field
  and therefore reduces that spatial heterogeneity (transport outruns
  growth), while very slow diffusion keeps the profile steep.
- With an open inlet the inlet site is continuously resupplied, so its
  glucose stays at the reservoir level while downstream sites are
  depleted: a substrate-depletion front forms (biofilm / plug-flow
  reactor; Rittmann & McCarty 2020).
- The growth that occurs produces a fermentative CO2 overflow (the
  reduced core model has no glyoxylate shunt, so no acetate overflow).

Physics notes: D_lattice = D*60/100 (unit conversion, see units.py).
With D = 2.0 µm^2/s, D_lattice = 1.2 per dFBA step.
"""
from __future__ import annotations

import pytest

from helixlang.plugins.apps.spatial_dfba import (
    SpatialDFBA,
    SpatialDFBAConfig,
)


def _closed_gradient(diffusion_um2_s: float,
                     n_steps: int = 60) -> SpatialDFBA:
    n = 32
    profile = [10.0 - 9.5 * i / (n - 1) for i in range(n)]
    sim = SpatialDFBA(SpatialDFBAConfig(
        length=n,
        inlet_glucose_mm=None,
        initial_glucose_profile=profile,
        glucose_diffusion_um2_s=diffusion_um2_s,
        initial_biomass_gdw=0.05,
    ))
    sim.run(n_steps)
    return sim


def test_closed_system_conserves_glucose_mass() -> None:
    sim = _closed_gradient(diffusion_um2_s=2.0)
    initial = sum(sim.config.initial_glucose_profile or [])
    remaining = sim.total_glucose()
    consumed = sim.total_consumed()
    # consumption is the only sink: initial = remaining + consumed
    assert remaining == pytest.approx(initial - consumed, abs=1e-6)
    assert consumed > 0.0


def test_glucose_gradient_seeds_graded_biomass() -> None:
    sim = _closed_gradient(diffusion_um2_s=2.0)
    biomass = sim.biomass_profile()
    assert biomass[0] > 2.0 * biomass[-1]


def test_high_diffusion_flattens_spatial_heterogeneity() -> None:
    slow = _closed_gradient(diffusion_um2_s=2.0)
    fast = _closed_gradient(diffusion_um2_s=50.0)
    slow_b = slow.biomass_profile()
    fast_b = fast.biomass_profile()
    slow_ratio = slow_b[0] / slow_b[-1]
    fast_ratio = fast_b[0] / fast_b[-1]
    assert fast_ratio < slow_ratio
    assert slow_ratio > 2.0


def test_open_inlet_sustains_inlet_but_depletes_downstream() -> None:
    sim = SpatialDFBA(SpatialDFBAConfig(
        length=32,
        inlet_glucose_mm=5.0,
        initial_glucose_mm=5.0,
        initial_biomass_gdw=0.05,
        max_biomass_gdw=2.0,
        glucose_diffusion_um2_s=2.0,
    ))
    sim.run(120)
    glucose = sim.glucose_profile()
    biomass = sim.biomass_profile()
    # inlet is continuously resupplied
    assert glucose[0] == pytest.approx(5.0)
    # downstream is depleted below the inlet level
    assert glucose[-1] < 0.1
    assert biomass[0] > 2.0 * biomass[-1]
    # the site closest to the inlet reached its carrying capacity
    assert biomass[0] == pytest.approx(2.0)


def test_open_inlet_forms_depletion_region() -> None:
    """Biofilm / plug-flow steady state: a sharp depletion region.

    With slow effective diffusion the consumption-driven depletion wave
    is transport-limited, so glucose only survives in a thin boundary
    layer at the inlet while the bulk of the strip is exhausted.
    """
    sim = SpatialDFBA(SpatialDFBAConfig(
        length=32,
        inlet_glucose_mm=5.0,
        initial_glucose_mm=5.0,
        initial_biomass_gdw=0.05,
        max_biomass_gdw=2.0,
        glucose_diffusion_um2_s=2.0,
    ))
    sim.run(120)
    glucose = sim.glucose_profile()
    depleted = sum(1 for g in glucose if g < 0.1)
    assert depleted >= int(0.8 * sim.config.length)
    # the substrate front sits close to the inlet
    assert sim.depletion_front() < sim.config.length // 4


def test_growth_produces_co2_overflow() -> None:
    sim = SpatialDFBA(SpatialDFBAConfig(
        length=16,
        inlet_glucose_mm=5.0,
        initial_glucose_mm=5.0,
        initial_biomass_gdw=0.05,
        max_biomass_gdw=2.0,
    ))
    sim.run(60)
    assert sim.total_byproduct("co2") > 0.0
    assert sim.total_biomass() > sim.config.length * 0.05
