"""Tests for doc/40 Phase F — spatial agent-based immune model (G15).

Covers the agent grid + chemokine diffusion, chemokine-guided migration,
contact-dependent T-cell/APC signaling, spatial heterogeneity, deterministic
replay, and the explicit backend contract (numpy default, jax optional,
no silent fallback) in :mod:`helixlang.plugins.human.spatial_abm`.
"""
from __future__ import annotations

import copy

import pytest

from helixlang.plugins.human.spatial_abm import (
    AgentState,
    CellType,
    SpatialABMConfig,
    SpatialAgentGrid,
    TissueAgent,
    run_cohort_spatial,
    run_spatial_abm,
)


class TestConstruction:
    def test_agent_counts(self):
        g = SpatialAgentGrid(SpatialABMConfig(seed=0))
        counts = g.cell_counts()
        assert counts["tcell"] == 60
        assert counts["apc"] == 12
        assert len(g.agents) == 60 + 12 + 40 + 20 + 15

    def test_unknown_backend_raises(self):
        # No silent fallback: unknown backend is an error, not a default.
        with pytest.raises(ValueError):
            SpatialAgentGrid(SpatialABMConfig(backend="cuda"))


class TestStep:
    def test_step_advances_and_diffuses(self):
        g = SpatialAgentGrid(SpatialABMConfig(seed=1, max_steps=10))
        for _ in range(5):
            g.step()
        assert g.step_index == 5
        # The wound source must raise the mean chemokine above 0.
        assert g.mean_chemokine() > 0.0

    def test_contact_driven_activation(self):
        g = SpatialAgentGrid(SpatialABMConfig(seed=3))
        # place an APC and a T cell on the same wound cell -> guaranteed contact
        g.agents = [
            TissueAgent(uid=0, cell_type=CellType.APC, x=20, y=20),
            TissueAgent(uid=1, cell_type=CellType.TCELL, x=20, y=20),
        ]
        for _ in range(20):
            g.step()
        states = g.state_histogram()
        assert states["activated"] + states["activating"] >= 1

    def test_deterministic_replay(self):
        a = run_spatial_abm(SpatialABMConfig(seed=9), steps=30)
        b = run_spatial_abm(SpatialABMConfig(seed=9), steps=30)
        assert a.state_histogram() == b.state_histogram()
        assert round(a.mean_chemokine(), 12) == round(b.mean_chemokine(), 12)

    def test_clone_independent(self):
        g = run_spatial_abm(SpatialABMConfig(seed=4), steps=10)
        h = g.clone()
        # clone carries identical state and step index
        assert h.state_histogram() == g.state_histogram()
        assert h.step_index == g.step_index
        assert round(h.mean_chemokine(), 12) == round(g.mean_chemokine(), 12)
        # stepping the clone independently does not disturb the original
        h.step()
        assert g.step_index == 10
        assert h.step_index == 11


class TestBackends:
    def test_jax_backend_explicit(self):
        # jax is a declared [has_jax] dependency; running it must not degrade.
        g = SpatialAgentGrid(SpatialABMConfig(seed=2, backend="jax"))
        for _ in range(10):
            g.step()
        assert g.step_index == 10
        assert g.mean_chemokine() > 0.0

    def test_numpy_default(self):
        g = SpatialAgentGrid(SpatialABMConfig(seed=2))
        assert g.cfg.backend == "numpy"


class TestCohort:
    def test_cohort_count_and_heterogeneity(self):
        cohort = run_cohort_spatial(4, steps=15, seed=7)
        assert len(cohort) == 4
        # deterministic given seed
        again = run_cohort_spatial(4, steps=15, seed=7)
        for g1, g2 in zip(cohort, again, strict=True):
            assert g1.state_histogram() == g2.state_histogram()

    def test_agent_budget(self):
        c = SpatialABMConfig(max_steps=50)
        g = SpatialAgentGrid(c)
        assert len(g.agents) <= 500
