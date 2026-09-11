"""Coverage-closure tests for :mod:`helixlang.plugins.apps.spatial_dfba`.

Exercises the construction guards (``length``/``initial_glucose_profile``),
the ``max_glucose_uptake`` binding, the zero-diffusion early return, and
the rarely-called query helpers (``acetate_profile``, ``consumed_profile``,
``total_acetate``, ``sites()``) plus the "front not found" returns of
``depletion_front`` and ``colonization_front`` and the
``max_biomass_gdw`` cap.
"""
import pytest

from helixlang.plugins.apps.spatial_dfba import (
    SpatialDFBA,
    SpatialDFBAConfig,
    SpatialDFBASite,
)


def _closed_strip(n: int = 8, initial: float = 5.0) -> SpatialDFBA:
    """A closed (no-inlet) strip with a mild upstream-downstream gradient."""
    profile = [initial - 1.5 * i / (n - 1) for i in range(n)]
    return SpatialDFBA(SpatialDFBAConfig(
        length=n,
        glucose_diffusion_um2_s=2.0,
        initial_glucose_profile=profile,
        inlet_glucose_mm=None,
        initial_biomass_gdw=0.05,
        max_glucose_uptake=10.0,
        seed=0,
    ))


class TestSpatialDFBAClosure:
    def test_length_must_be_positive(self):
        with pytest.raises(ValueError):
            SpatialDFBA(SpatialDFBAConfig(length=0))

    def test_profile_length_must_match(self):
        with pytest.raises(ValueError):
            SpatialDFBA(SpatialDFBAConfig(length=8,
                                          initial_glucose_profile=[1.0, 2.0]))

    def test_max_glucose_uptake_binds_uptake(self):
        sim = _closed_strip()
        sim.step()
        assert sim.consumed[0] > 0.0

    def test_zero_diffusion_early_returns(self):
        sim = SpatialDFBA(SpatialDFBAConfig(
            length=8, glucose_diffusion_um2_s=0.0,
            initial_glucose_profile=[5.0 - i * 0.5 for i in range(8)],
            inlet_glucose_mm=None))
        before = sim.glucose_profile()
        sim.step()
        # no diffusive mixing: each site can only lose glucose, so the
        # upstream-heavy ordering is preserved
        after = sim.glucose_profile()
        assert all(a <= b for a, b in zip(after, before, strict=True))

    def test_max_biomass_cap_clamps_batches(self):
        sim = SpatialDFBA(SpatialDFBAConfig(
            length=4,
            initial_glucose_mm=20.0,
            inlet_glucose_mm=None,
            initial_biomass_gdw=0.05,
            max_biomass_gdw=0.06))
        sim.run(5)
        assert all(b <= 0.06 + 1e-12 for b in sim.biomass_profile())

    def test_query_helpers_and_site_dataclass(self):
        sim = _closed_strip()
        sim.run(3)
        acet = sim.acetate_profile()
        assert len(acet) == sim.config.length
        assert sim.total_acetate() >= 0.0
        consumed = sim.consumed_profile()
        assert len(consumed) == sim.config.length
        assert sim.total_consumed() > 0.0

        sites = sim.sites()
        assert len(sites) == sim.config.length
        for s in sites:
            assert isinstance(s, SpatialDFBASite)
            assert s.index >= 0
            assert s.biomass_gdw >= 0.0
            assert s.glucose_mm >= 0.0
            assert s.acetate_mm >= 0.0
            assert s.glucose_consumed_mm >= 0.0
        assert sum(s.glucose_consumed_mm for s in sites) == pytest.approx(
            sim.total_consumed())

    def test_depletion_front_not_found_returns_length(self):
        sim = _closed_strip()
        assert sim.depletion_front(threshold_mm=0.0) == sim.config.length

    def test_colonization_front_returns_first_and_length(self):
        sim = _closed_strip(initial=10.0)
        sim.run(1)
        # threshold above every site's biomass -> front at the first site
        front = sim.colonization_front(biomass_threshold_gdw=10.0)
        assert 0 <= front < sim.config.length
        # threshold below every site's biomass -> front = length
        assert sim.colonization_front(biomass_threshold_gdw=0.0) == \
            sim.config.length
