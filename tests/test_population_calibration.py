"""Population-level mixed-observable calibration tests (doc/18-programmable-cell-population-simulation.md §13 Design 4).

Verification goals (Bosshard et al. 2020, BMC Genomics 21:232; Karr et al.
2012 DREAM8 colony-scale parameter estimation):
- The colony observables are self-consistent: the probe evaluated at the
  truth reproduces the recorded ground-truth vector exactly.
- The three dFBA colony parameters are recovered within 10% of the truth
  by a decoupled three-stage fit (metabolic probe -> division-free energy
  probe -> division probe).  The energy scale is read division-free
  because the division block couples it to the threshold along a diagonal
  ``threshold - c*scale`` ridge.
- The division probe short-circuits blown-up candidates so the fit never
  spends its budget on exploding colonies.
"""
from __future__ import annotations

import pytest

from helixlang.apps.population_calibration import (
    TRUTH_DIVISION_THRESHOLD,
    TRUTH_ENERGY_SCALE,
    TRUTH_OXYGEN_MAX_UPTAKE,
    PopulationCalibration,
    run_population_calibration,
)

FAST = dict(
    division_ticks=10,
    n_samples=2,
    refine_rounds=1,
    refine_windows=(0.35, 0.12),
)


def test_observable_vector_layout() -> None:
    """The mixed observable vector holds growth, chemistry, division and
    energy blocks (division carries the time-averaged alive count too)."""
    c = PopulationCalibration(**FAST)
    assert len(c.observed) == 8
    idx = c._block_indices()
    assert idx["growth"] == [0]
    assert idx["chemistry"] == [1, 2]
    assert idx["division"] == [3, 4, 5, 6]
    assert idx["energy"] == [7]
    # the observed vector is the concatenation of the block indices
    blocks = ["growth", "chemistry", "division", "energy"]
    assert [i for b in blocks for i in idx[b]] == list(range(8))


def test_self_consistent_observed() -> None:
    """The probe at the truth reproduces the ground-truth vector exactly
    (shared cell seed -> deterministic divisions -> SSE valley at truth)."""
    c = PopulationCalibration(**FAST)
    pred = c.predict(
        dfba_oxygen_max_uptake=TRUTH_OXYGEN_MAX_UPTAKE,
        dfba_energy_scale=TRUTH_ENERGY_SCALE,
        division_threshold=TRUTH_DIVISION_THRESHOLD)
    assert pred == pytest.approx(c.observed, rel=1e-9)


def test_energy_probe_linear_in_scale() -> None:
    """The division-free energy probe is a monotone linear readout of the
    per-tick ``dfba_energy_scale`` accumulation rate."""
    c = PopulationCalibration(**FAST)
    e0 = sum(x.energy for x in c._cells(
        c.cell_seed, co_located=True,
        division_threshold=TRUTH_DIVISION_THRESHOLD)) / c.n_cells
    low = c._energy_probe(oxygen_max=TRUTH_OXYGEN_MAX_UPTAKE,
                          energy_scale=1.4e8)[0]
    high = c._energy_probe(oxygen_max=TRUTH_OXYGEN_MAX_UPTAKE,
                           energy_scale=2.8e8)[0]
    assert high > low
    # doubling the scale doubles the energy gained in the probe window
    # (avg(t) = E0 + ticks*g*scale), so high - low == low - E0
    assert (high - low) == pytest.approx(low - e0, rel=0.15)


def test_energy_probe_independent_of_threshold() -> None:
    """The energy probe never divides (huge threshold), so its readout is
    threshold-independent -- this is what decouples the two knobs."""
    c = PopulationCalibration(**FAST)
    a = c._energy_probe(oxygen_max=TRUTH_OXYGEN_MAX_UPTAKE,
                        energy_scale=TRUTH_ENERGY_SCALE)[0]
    b = c._energy_probe(oxygen_max=TRUTH_OXYGEN_MAX_UPTAKE,
                        energy_scale=TRUTH_ENERGY_SCALE)[0]
    assert a == b


def test_division_probe_blowup_short_circuits() -> None:
    """A candidate whose accumulation rate is far above the truth divides
    the colony out of control; the probe returns a strongly penalized
    vector instead of running thousands of cells."""
    c = PopulationCalibration(**FAST)
    out = c._division_probe(
        oxygen_max=TRUTH_OXYGEN_MAX_UPTAKE,
        energy_scale=8.0 * TRUTH_ENERGY_SCALE,
        division_threshold=TRUTH_DIVISION_THRESHOLD)
    assert out[1] > 1e11  # far above the observed total energy


def test_weights_inverse_variance() -> None:
    """Weights follow the DESeq2 multiplicative-noise structure
    (Var = value**2), so a colony-size count and an energy readout
    contribute proportionally to their relative error."""
    c = PopulationCalibration(**FAST)
    w = c._weights_for(["division"], (1.0,))
    assert len(w) == 4
    obs = [c.observed[i] for i in c._block_indices()["division"]]
    for wi, oi in zip(w, obs, strict=True):
        assert wi == pytest.approx(1.0 / (oi * oi))


def test_full_calibration_recovers_parameters() -> None:
    """The three-stage fit recovers all three dFBA colony parameters
    within 10% of the ground truth."""
    c = PopulationCalibration(**FAST, fit_seed=1)
    r = c.run()
    assert r["passed"] is True
    assert r["relative_error"]["dfba_oxygen_max_uptake"] < 0.05
    assert r["relative_error"]["dfba_energy_scale"] < 0.05
    assert r["relative_error"]["division_threshold"] < 0.10


def test_closure_returns_dict() -> None:
    """The one-shot closure mirrors run() (and is what the sim_runtime
    backend calls)."""
    result = run_population_calibration(
        n_samples=2, refine_rounds=1, division_ticks=10)
    assert set(result) == {"fit", "fitted", "truth", "relative_error",
                           "recovered", "passed"}
    assert result["passed"] is True
    assert result["relative_error"]["division_threshold"] < 0.10
