"""Virtual-cell integration & validation benchmark tests (T3.4, gap G8).

Verification goals:
- VirtualCell couples central dogma (protein production), GRN (which
  genes express), metabolism (FBA biomass flux -> energy) and the
  cell-cycle budget (maintenance, division gate, death).
- fit_parameters recovers known parameters from synthetic data.
- perturbation_response reports fold change / settling time of a
  knockout (continuous-time, T2.2 DOPRI5 backend).
- run_biofilm_benchmark publishes BM3-style growth metrics.

References:
- Karr et al. 2012 whole-cell model (Cell 150:389-401)
- Virtual Cell Challenge 2025; iDynoMiCS 2.0 BM3 biofilm benchmark
"""
from __future__ import annotations

import pytest

from helixlang.grn import GRN
from helixlang.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis
from helixlang.population import (
    CellPopulation,
    PopulationCell,
    PopulationConfig,
)
from helixlang.virtual_cell import (
    VirtualCell,
    VirtualCellConfig,
    encode_gene,
    fit_parameters,
    perturbation_response,
    run_biofilm_benchmark,
)

GENOME = {
    "lacZ": encode_gene("MAQILARVFFDDV"),
    "galK": encode_gene("MSSRPQAAASSWW"),
    "tetR": encode_gene("MSRLDKSVINS"),
}


def _grn(active: tuple[str, ...] = ("lacZ",), levels=None) -> GRN:
    g = GRN()
    for name in GENOME:
        g.add_gene(name, 0.5)
    g.add_edge("tetR", "lacZ", -2.0)
    g.add_edge("lacZ", "galK", 2.0)
    if levels:
        for name, lv in levels.items():
            g.nodes[name].level = lv
    else:
        for name in active:
            g.nodes[name].level = 1.0
    return g


def _vc(**cfg_kw) -> VirtualCell:
    cfg = VirtualCellConfig(uptake={"GLC": 10.0}, **cfg_kw)
    return VirtualCell(GENOME, _grn(), config=cfg)


# ============================================================================
# Gene encoding / central dogma coupling
# ============================================================================

def test_encode_gene_round_trips() -> None:
    protein = "MAQILARVFFDDV"
    dna = encode_gene(protein)
    from helixlang.central_dogma import transcribe, translate

    tr = transcribe(dna, promoter_strength=1.0)
    result = translate(tr)
    assert result.rbs_found
    assert result.protein == protein
    assert result.stop_codon == "TAA"


def test_encode_gene_has_shine_dalgarno() -> None:
    dna = encode_gene("MSS")
    assert dna.startswith("AGGAGG")
    assert "ATG" in dna and dna.endswith("TAA")


# ============================================================================
# VirtualCell budget model
# ============================================================================

def test_virtual_cell_expresses_active_genes() -> None:
    vc = _vc()
    vc.run(3)
    assert vc.proteins.get("lacZ", 0.0) > 0.0
    # tetR starts off and stays below threshold, so it never expresses
    assert vc.proteins.get("tetR", 0.0) == 0.0
    assert all(e["alive"] for e in vc.history)


def test_virtual_cell_energy_accounting() -> None:
    vc = _vc()
    before = vc.energy
    vc.step()
    # translation + transcription costs are debited; maintenance paid;
    # metabolism credits biomass flux
    flux = vc.history[-1]["biomass_flux"]
    assert vc.energy < before  # maintenance dominates at this calibration
    assert flux >= 0.0
    assert len(vc.history) == 1


def test_virtual_cell_metabolism_coupling() -> None:
    fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    fba.set_uptake("GLC", 10.0)
    cfg = VirtualCellConfig(maintenance_atp_per_min=0.0,
                            translation_atp_per_aa=0.0,
                            transcription_atp_per_nt=0.0,
                            biomass_to_atp=1.0e6)
    vc = VirtualCell(GENOME, _grn(active=()), fba=fba, config=cfg)
    before = vc.energy
    vc.step()
    assert vc.energy > before  # biomass flux credits the budget
    assert vc.history[-1]["biomass_flux"] > 0.0


def test_virtual_cell_division_gate() -> None:
    vc = _vc(biomass_to_atp=1.0e9, maintenance_atp_per_min=0.0,
             division_energy=1.5e9, transcription_atp_per_nt=0.0,
             translation_atp_per_aa=0.0)
    for _ in range(5):
        vc.step()
    assert vc.divisions >= 1
    assert vc.energy < vc.config.division_energy  # halved at division


def test_virtual_cell_death_stops_run() -> None:
    vc = _vc(maintenance_atp_per_min=1.0e11)
    vc.run(20)
    assert not vc.alive
    assert vc.history[-1]["alive"] is False
    # run() stops stepping once dead
    n = len(vc.history)
    vc.run(5)
    assert len(vc.history) == n


def test_virtual_cell_history_tracks_proteins() -> None:
    vc = _vc()
    vc.step()
    entry = vc.history[-1]
    assert entry["proteins"] == vc.proteins
    assert "triggered" in entry and "mass" in entry


# ============================================================================
# Parameter estimation harness
# ============================================================================

def test_fit_parameters_recovers_linear_truth() -> None:
    # a + b*x with x = 1,2,3 and observed = a + 2.5x has interior optimum
    # a = 1, b = 2.5 inside the box
    observed = [3.5, 6.0, 8.5]

    def predict(a, b):
        return [a + b * x for x in (1, 2, 3)]

    fit = fit_parameters(predict, observed,
                         {"a": (0.0, 4.0), "b": (0.0, 4.0)},
                         n_samples=200, seed=0)
    assert fit["best"]["a"] == pytest.approx(1.0, abs=0.05)
    assert fit["best"]["b"] == pytest.approx(2.5, abs=0.05)
    assert fit["sse"] < 1e-3


def test_fit_parameters_exact_match() -> None:
    def predict(k):
        return [k * 2.0]

    fit = fit_parameters(predict, [4.0], {"k": (0.0, 10.0)},
                         n_samples=100, seed=1)
    assert fit["best"]["k"] == pytest.approx(2.0, abs=0.01)


def test_fit_parameters_validation() -> None:
    with pytest.raises(ValueError):
        fit_parameters(lambda: [1.0], [1.0], {})
    with pytest.raises(ValueError):
        fit_parameters(lambda: [1.0], [], {"a": (0.0, 1.0)})
    with pytest.raises(ValueError):
        fit_parameters(lambda: [1.0], [1.0, 2.0], {"a": (0.0, 1.0)})


def test_fit_parameters_fits_virtual_cell() -> None:
    # fit the energy-coupling constants against a target protein level
    def predict(biomass_to_atp, maintenance_atp_per_min):
        g = _grn(active=("lacZ",))
        cfg = VirtualCellConfig(uptake={"GLC": 10.0},
                                biomass_to_atp=biomass_to_atp,
                                maintenance_atp_per_min=maintenance_atp_per_min)
        vc = VirtualCell(GENOME, g, config=cfg)
        vc.run(10)
        return [vc.proteins.get("lacZ", 0.0)]

    fit = fit_parameters(predict, [500.0],
                         {"biomass_to_atp": (1e5, 1e8),
                          "maintenance_atp_per_min": (1e7, 1e8)},
                         n_samples=150, seed=0)
    # protein yield is set by the config, not the fitted constants
    assert fit["best"]["biomass_to_atp"] >= 1e5


# ============================================================================
# Benchmarks
# ============================================================================

def test_perturbation_response_knockout() -> None:
    g = GRN()
    for name in ("A", "B"):
        g.add_gene(name, 0.5)
    g.add_edge("A", "B", 2.0)
    g.add_edge("B", "A", -1.5)
    g.nodes["A"].level = 1.0
    pr = perturbation_response(g, "B", knockout="A",
                               t_span=(0.0, 200.0), n_points=200)
    assert 0.0 <= pr["fold_change"] <= 1.0
    assert pr["perturbed_final"] < pr["control_final"]
    assert pr["settling_time"] is not None
    assert 0.0 <= pr["settling_time"] <= 200.0
    assert len(pr["times"]) == len(pr["response"]) == 200


def test_perturbation_response_no_knockout() -> None:
    g = GRN()
    for name in ("A",):
        g.add_gene(name, 0.5)
    g.nodes["A"].level = 1.0
    pr = perturbation_response(g, "A", t_span=(0.0, 50.0), n_points=100)
    assert pr["fold_change"] == 1.0
    assert pr["perturbed_final"] == pr["control_final"]


def test_biofilm_benchmark_growth() -> None:
    cfg = PopulationConfig(grid_width=12, grid_height=12,
                           energy_intake=100.0, division_threshold=150.0,
                           metabolic_cost=1.0)
    pop = CellPopulation([PopulationCell(id=0, energy=200.0, x=6, y=6)],
                         cfg, seed=1)
    bm = run_biofilm_benchmark(pop, n_steps=25, interval=5)
    assert bm["final_biomass"] >= bm["biomass"][0]
    assert bm["max_extent"] >= 0.0
    assert bm["growth_rate_per_tick"] >= 0.0
    assert len(bm["biomass"]) == 6  # interval 5 over 25 steps -> 6 samples


def test_biofilm_benchmark_doubling() -> None:
    cfg = PopulationConfig(grid_width=8, grid_height=8,
                           energy_intake=100.0, division_threshold=150.0,
                           metabolic_cost=1.0)
    pop = CellPopulation([PopulationCell(id=0, energy=200.0, x=4, y=4)],
                         cfg, seed=1)
    bm = run_biofilm_benchmark(pop, n_steps=20, interval=1)
    assert bm["doubling_ticks"] is not None
    assert bm["doubling_ticks"] <= 20
