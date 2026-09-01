"""Tests for doc/39 performance-optimization cohort paths.

Covers:
  * O2 ``cohort_immune_step`` — cohort-vectorized innate-immune stepping is
    bit-identical to the scalar per-model path, and the non-numpy fallback is
    bit-identical to the scalar path.
  * O3 adaptive PBPK sub-stepping (``PBPKModel.advance`` in
    ``plugins.human.virtual_patient``) — deterministic and monotonic progress
    with a positive observable exposure.
"""
from __future__ import annotations

import copy
import random

from helixlang.plugins.human.drug import get_predefined_drug
from helixlang.plugins.human.immune import (
    cohort_immune_step,
    create_immune_model,
    run_cohort,
    sample_virtual_population,
)
from helixlang.plugins.human.physiology import create_default_physiology
from helixlang.plugins.human.virtual_patient import _DrugPBPK


def _make_model(seed_rng: random.Random):
    m, _ = create_immune_model(
        infection_severity=seed_rng.random(),
        autoimmune_activation=seed_rng.random() * 0.3,
        cortisol_level=12.0 + seed_rng.random() * 20.0,
        immunosuppression=seed_rng.random() * 0.4,
    )
    m.anti_inflammatory = seed_rng.random() * 0.7
    for name in ("tnf_alpha", "il1_beta", "il6", "il10"):
        setattr(m.cytokines, name, seed_rng.random() * 15.0)
    m.cells.neutrophils = 1.0 + seed_rng.random() * 6.0
    m.cells.macrophages = 0.1 + seed_rng.random()
    m.cells.monocytes = 0.05 + seed_rng.random() * 0.5
    m.cells.dendritic_cells = 0.02 + seed_rng.random() * 0.2
    m.cells.t_cells = 0.2 + seed_rng.random() * 2.0
    return m


def _flat(m):
    return (
        m.cytokines.tnf_alpha, m.cytokines.il1_beta, m.cytokines.il6,
        m.cytokines.il10, m.cells.neutrophils, m.cells.macrophages,
        m.cells.monocytes, m.cells.dendritic_cells, m.cells.t_cells,
        m.cells.neutrophil_production, m.cells.monocyte_production,
    )


def test_o2_cohort_step_matches_scalar():
    """Vectorized cohort_immune_step == scalar step, exact to 1e-9."""
    rng = random.Random(7)
    models = [_make_model(rng) for _ in range(20)]
    scalar = copy.deepcopy(models)
    for _ in range(50):
        for m in scalar:
            m.step(1.0)
        cohort_immune_step(models, 1.0)
    sa, sb = [_flat(m) for m in scalar], [_flat(m) for m in models]
    for ra, rb in zip(sa, sb, strict=True):
        for x, y in zip(ra, rb, strict=True):
            assert abs(x - y) <= 1e-9, (x, y)


def test_o2_fallback_matches_scalar():
    """use_numpy=False fallback drives identical per-model states."""
    rng = random.Random(11)
    models = [_make_model(rng) for _ in range(8)]
    scalar = copy.deepcopy(models)
    for _ in range(30):
        for m in scalar:
            m.step(1.0)
        cohort_immune_step(models, 1.0, use_numpy=False)
    sa, sb = [_flat(m) for m in scalar], [_flat(m) for m in models]
    for ra, rb in zip(sa, sb, strict=True):
        for x, y in zip(ra, rb, strict=True):
            assert abs(x - y) <= 1e-12


def test_o3_pbpk_advance_adaptive():
    """Adaptive sub-stepping (doc/39 O3) is deterministic and monotonic."""
    drug = get_predefined_drug("IBUPROFEN")
    phys = create_default_physiology()
    model = _DrugPBPK(drug, phys)
    state0 = model._snapshot_state()
    model.advance(1.0, 0.0)
    state1 = model._snapshot_state()
    # central concentration is non-negative; the first oral dose lands in the
    # depot (so the depot rises from zero to the dose), then is absorbed.
    assert state1[0]["central"] >= 0.0
    assert state1[1] > state0[1]
    # subsequent step without a new dose depletes the depot monotonically
    model.advance(1.0, 1.0)
    state2 = model._snapshot_state()
    assert state2[1] <= state1[1]
    # deterministic: two fresh models taking the same single step agree
    m_a = _DrugPBPK(drug, phys)
    m_b = _DrugPBPK(drug, phys)
    m_a.advance(1.0, 0.0)
    m_b.advance(1.0, 0.0)
    assert abs(m_a._snapshot_state()[0]["central"]
               - m_b._snapshot_state()[0]["central"]) < 1e-12


def test_o3_pbpk_zero_dt_noop():
    """advance(0) is a no-op and leaves the state unchanged."""
    drug = get_predefined_drug("IBUPROFEN")
    phys = create_default_physiology()
    model = _DrugPBPK(drug, phys)
    before = model._snapshot_state()
    model.advance(0.0, 0.0)
    assert model._snapshot_state() == before


def test_o9_run_cohort_matches_scalar():
    """run_cohort with multiprocessing (doc/39 O9) == serial vectorized path."""
    rng = random.Random(13)
    models = [_make_model(rng) for _ in range(10)]
    serial = copy.deepcopy(models)
    parallel = copy.deepcopy(models)
    run_cohort(serial, 40, 1.0, workers=1)
    run_cohort(parallel, 40, 1.0, workers=4)
    state1 = [_flat(m) for m in serial]
    state2 = [_flat(m) for m in parallel]
    for r1, r2 in zip(state1, state2, strict=True):
        for x, y in zip(r1, r2, strict=True):
            assert abs(x - y) <= 1e-9, (x, y)


def test_o9_run_cohort_two_workers_matches():
    """workers=2 slab merge (stride-2 layout) is bit-identical to workers=1."""
    rng = random.Random(17)
    models = [_make_model(rng) for _ in range(5)]
    serial = copy.deepcopy(models)
    parallel = copy.deepcopy(models)
    run_cohort(serial, 25, 1.0, workers=1)
    run_cohort(parallel, 25, 1.0, workers=2)
    for r1, r2 in zip(
        (_flat(m) for m in serial), (_flat(m) for m in parallel), strict=True
    ):
        for x, y in zip(r1, r2, strict=True):
            assert abs(x - y) <= 1e-9, (x, y)


def test_g13_sample_pop_spread():
    """sample_virtual_population produces physiologic variance."""
    pop = sample_virtual_population(200, seed=42)
    anc = [p.cells.neutrophils for p in pop]
    assert min(anc) > 0.5
    assert max(anc) > 4.0
    assert min(anc) < 4.0 < max(anc)
    assert len({round(a, 4) for a in anc}) > 100


def test_g13_sample_pop_deterministic():
    """Same seed → identical population."""
    pop1 = sample_virtual_population(50, seed=99)
    pop2 = sample_virtual_population(50, seed=99)
    for p1, p2 in zip(pop1, pop2, strict=True):
        assert p1.cells.neutrophils == p2.cells.neutrophils
        assert p1.cytokines.il6 == p2.cytokines.il6
        assert p1.ifn.ifn_vmax == p2.ifn.ifn_vmax


def test_g13_sample_pop_different_seeds():
    """Different seeds → different populations."""
    pop1 = sample_virtual_population(50, seed=1)
    pop2 = sample_virtual_population(50, seed=2)
    assert pop1[0].cells.neutrophils != pop2[0].cells.neutrophils


def test_g13_cohort_run():
    """Sampled cohort runs through O2 vectorized kernel without error."""
    pop = sample_virtual_population(20, seed=7)
    for p in pop:
        p.infection_severity = 0.8
    run_cohort(pop, 48, 1.0, workers=1)
    il6s = [p.cytokines.il6 for p in pop]
    assert max(il6s) > 5.0
    assert len({round(x, 3) for x in il6s}) > 5
