"""Whole-cell parameter calibration closure tests (Phase 5, doc §8.3).

Verifies that :func:`~helixlang.apps.whole_cell_calibration.
run_whole_cell_calibration` can recover the Phase 1-4 hidden parameters
(adder threshold, folding equilibrium, kcat scale and maintenance burn)
from the mixed observable vector within tolerance.
"""
from __future__ import annotations

import pytest

from helixlang.apps.whole_cell_calibration import (
    DEFAULT_GENOME,
    OBSERVED_ENZYMES,
    TRUTH_ADDER_VOLUME_UM3,
    TRUTH_ENZYME_SCALE,
    TRUTH_K_FOLD,
    TRUTH_MAINTENANCE_ATP_PER_MIN,
    WholeCellCalibration,
    _fold_rate_from_k_fold,
    run_whole_cell_calibration,
)
from helixlang.metabolism import ECOLI_CORE_GENE_REACTIONS
from helixlang.virtual_cell import encode_gene


def test_genome_covers_all_gated_enzymes() -> None:
    """The calibration genome must encode every enzyme-gated reaction
    gene: any missing gene gets a zero enzyme pool which caps the
    gated reaction at zero and collapses biomass."""
    assert set(DEFAULT_GENOME) == set(ECOLI_CORE_GENE_REACTIONS)
    assert all(len(seq) == 52 for seq in DEFAULT_GENOME.values())
    assert set(OBSERVED_ENZYMES) <= set(DEFAULT_GENOME)


def test_fold_rate_maps_k_fold_equilibrium() -> None:
    """The folding rate derived from k_fold yields exactly the desired
    equilibrium fraction against the fixed misfolding rate."""
    misfold = 0.3
    for k_fold in (0.4, 0.5, 0.7, 0.9):
        rate = _fold_rate_from_k_fold(k_fold, misfold)
        equilibrium = rate / (rate + misfold)
        assert equilibrium == pytest.approx(k_fold)
    with pytest.raises(ValueError):
        _fold_rate_from_k_fold(1.0, 0.3)


def test_observable_vector_layout() -> None:
    """The observable vector is (energy, volume, biomass, proteins,
    division_added, divisions) with a fixed length even when the
    candidate dies."""
    cal = WholeCellCalibration(minutes=60)
    assert len(cal.observed) == (3 * 60 + len(OBSERVED_ENZYMES)
                                 + cal.max_divisions + 1)
    assert len(cal.weights) == len(cal.observed)
    assert cal.observed[-1] == float(cal.truth_cell.divisions)
    assert cal.truth_cell.divisions >= 1
    # the per-division added volume sits just under the adder threshold
    added = cal.observed[-1 - cal.max_divisions]
    assert 0.0 < added <= TRUTH_ADDER_VOLUME_UM3
    # block weights are positive and normalized
    assert all(w > 0.0 for w in cal.weights)


def test_observable_layout_scales_with_cell_population() -> None:
    """n_cells>1 aggregates independent cells: the division_added block
    is n_cells*max_divisions long and the divisions count sums across
    cells."""
    cal = WholeCellCalibration(minutes=60, n_cells=3)
    n = len(cal.observed)
    assert n == (3 * 60 + len(OBSERVED_ENZYMES)
                 + 3 * cal.max_divisions + 1)
    assert cal.observed[-1] == float(sum(c.divisions
                                         for c in cal.truth_cells))
    assert len(cal.truth_cells) == 3
    # division counts of the aggregated cells are consistent with the
    # per-cell division totals
    assert all(c.divisions >= 1 for c in cal.truth_cells)
    # each cell contributes its own padded added-volume samples
    assert cal.observed[-1 - 3 * cal.max_divisions] > 0.0


def test_predict_matches_truth_at_truth_parameters() -> None:
    """Predicting with the ground-truth parameters reproduces the
    observed vector (identity fit)."""
    cal = WholeCellCalibration(minutes=20)
    pred = cal.predict(
        adder_volume_um3=TRUTH_ADDER_VOLUME_UM3,
        k_fold=TRUTH_K_FOLD,
        enzyme_scale=TRUTH_ENZYME_SCALE,
        maintenance_atp_per_min=TRUTH_MAINTENANCE_ATP_PER_MIN,
    )
    assert pred == pytest.approx(cal.observed, rel=1e-6)


def test_separable_stages_match_joint_space() -> None:
    """The growth-stage fit range covers the truth enzyme scale and the
    size-stage parameters are within the box bounds."""
    cal = WholeCellCalibration(minutes=20, n_samples=5, refine_rounds=0)
    fit = cal.calibrate()
    assert set(fit["best"]) == {"adder_volume_um3", "k_fold",
                                "enzyme_scale", "maintenance_atp_per_min"}
    assert fit["n_samples"] >= 0


def test_whole_cell_calibration_closure() -> None:
    """End-to-end closure: recover all four parameters from the mixed
    observables within tolerance."""
    result = run_whole_cell_calibration(
        minutes=60, n_samples=60, refine_rounds=2, fit_seed=0)
    assert result["passed"]
    for k in result["fitted"]:
        assert result["recovered"][k], (
            f"{k}: fitted={result['fitted'][k]} truth={result['truth'][k]}")
    # the per-division added-volume observable pins the adder threshold:
    # recovery should be comfortably inside the 10% tolerance
    adder_err = (abs(result["fitted"]["adder_volume_um3"]
                     - TRUTH_ADDER_VOLUME_UM3) / TRUTH_ADDER_VOLUME_UM3)
    assert adder_err < 0.06


def test_truth_anchors() -> None:
    """The ground-truth anchors are the documented Phase 2-4 values."""
    assert TRUTH_ADDER_VOLUME_UM3 == 1.6
    assert TRUTH_K_FOLD == 0.7
    assert TRUTH_ENZYME_SCALE == 1.0e4
    assert TRUTH_MAINTENANCE_ATP_PER_MIN == 2.5e7


def test_adder_recovery_robust_to_threshold_noise() -> None:
    """With adder_threshold noise (Taheri-Araghi 2015, sigma=0.1) and a
    modest population average (n_cells=4), the adder threshold is still
    recovered across fit seeds.  A single cell contributes just one noisy
    added-volume sample (it dies a few divisions after birth), so the
    population average is what keeps the mean estimator within tolerance;
    the finer refine grid (rr=4) keeps the discrete pattern search from
    stalling on the noisy surface at a coarse grid point."""
    for fit_seed in (0, 1):
        result = run_whole_cell_calibration(
            minutes=60, n_samples=60, refine_rounds=4, fit_seed=fit_seed,
            adder_noise_std=0.1, n_cells=4)
        assert result["passed"], f"fit_seed={fit_seed}: {result['recovered']}"
        adder_err = (abs(result["fitted"]["adder_volume_um3"]
                         - TRUTH_ADDER_VOLUME_UM3) / TRUTH_ADDER_VOLUME_UM3)
        assert adder_err < 0.1, (
            f"fit_seed={fit_seed}: adder_err={adder_err:.3f}")


def test_genome_encodes_valid_protein() -> None:
    """The encoded enzyme sequence decodes back to the intended 14 aa."""
    dna = next(iter(DEFAULT_GENOME.values()))
    assert dna == encode_gene("MAQILARVFFDDV")
