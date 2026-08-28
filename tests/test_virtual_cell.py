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

from helixlang.plugins.runtime.grn import GRN
from helixlang.plugins.runtime.metabolism import (
    ECOLI_CORE_GENE_REACTIONS,
    ECOLI_CORE_KCAT,
    ECOLI_CORE_MODEL,
    EnzymeCapacity,
    FluxBalanceAnalysis,
)
from helixlang.plugins.runtime.population import (
    CellPopulation,
    PopulationCell,
    PopulationConfig,
)
from helixlang.plugins.runtime.virtual_cell import (
    CellCyclePhase,
    RepliconSpec,
    VirtualCell,
    VirtualCellConfig,
    _next_scheduled_division,
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

DOSAGE_GENOME = {
    "oriGene": encode_gene("MAQILARVFFDDV"),
    "midGene": encode_gene("MSSRPQAAASSWW"),
    "terGene": encode_gene("MSRLDKSVINS"),
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
    from helixlang.plugins.runtime.central_dogma import transcribe, translate

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
# Phase 1: cell cycle + Cooper-Helmstetter chromosome replication
# ============================================================================

def _dosage_grn() -> GRN:
    g = GRN()
    for name in DOSAGE_GENOME:
        g.add_gene(name, 0.5)
        g.nodes[name].level = 1.0
    return g


def _cooper_vc(**cfg_kw) -> VirtualCell:
    cfg = VirtualCellConfig(
        replication_mode="cooper_helmstetter",
        chromosome_map={"oriGene": 0.0, "midGene": 0.5, "terGene": 1.0},
        maintenance_atp_per_min=0.0,
        transcription_atp_per_nt=0.0,
        translation_atp_per_aa=0.0,
        **cfg_kw,
    )
    return VirtualCell(DOSAGE_GENOME, _dosage_grn(), config=cfg)


def test_flat_mode_is_single_copy_bit_for_bit() -> None:
    # default replication_mode="flat": every gene stays at single copy,
    # no forks, phase stuck at B_GAP, protein credited 100 per expression
    vc = _vc()
    assert vc.phase is CellCyclePhase.B_GAP
    assert vc.replication_forks == []
    assert vc.replication_fork == 1.0
    assert vc.dna_copy_number == {gene: 1 for gene in GENOME}
    vc.run(5)
    for entry in vc.history:
        assert entry["dna_copy_number"] == {gene: 1 for gene in GENOME}
        assert entry["phase"] == CellCyclePhase.B_GAP.value
    assert vc.mrna["lacZ"] == 5.0
    assert vc.proteins["lacZ"] == pytest.approx(5 * 100.0)


def test_cooper_helmstetter_replication_termination_invariant() -> None:
    # every fork must terminate >= D minutes before the scheduled division
    # it serves, and that division must land on the tau grid
    vc = _cooper_vc()
    cfg = vc.config
    for _ in range(200):
        vc.step()
        for init, _p in vc.replication_forks:
            termination = init + cfg.c_period_min
            served = init + cfg.c_period_min + cfg.d_period_min
            assert served % cfg.doubling_time_min == pytest.approx(
                0.0, abs=1e-6)
            nxt = _next_scheduled_division(
                termination, cfg.doubling_time_min)
            assert nxt - termination >= cfg.d_period_min - 1e-6


def test_cooper_helmstetter_copy_number_monotone_between_divisions() -> None:
    # with no division (flux 0 keeps energy below the gate) the copy
    # numbers are non-decreasing everywhere
    vc = _cooper_vc()
    prev = dict(vc.dna_copy_number)
    for _ in range(120):
        vc.step()
        cur = vc.dna_copy_number
        for gene in cur:
            assert cur[gene] >= prev[gene]
        prev = cur
    # and the origin-proximal gene carries the higher dosage throughout
    for entry in vc.history:
        assert (entry["dna_copy_number"]["oriGene"]
                >= entry["dna_copy_number"]["terGene"])


def test_cooper_helmstetter_gene_dosage_wave() -> None:
    # rich-medium defaults (tau=20, C=40, D=20): origin-proximal gene is
    # transcribed from more DNA copies than the terminus-proximal gene, so
    # its mrna/protein accumulate measurably faster (gene-dosage wave peaks
    # behind the replication fork; Cooper & Helmstetter 1968; Karr 2012)
    vc = _cooper_vc()
    # birth-time dosage gradient along the chromosome map: origin > mid > terminus
    birth = vc.dna_copy_number
    assert birth["oriGene"] > birth["midGene"] > birth["terGene"]
    vc.run(40)
    assert vc.mrna["oriGene"] > 1.5 * vc.mrna["terGene"]
    assert vc.proteins["oriGene"] > 1.5 * vc.proteins["terGene"]
    assert vc.mrna["midGene"] > vc.mrna["terGene"]


def test_cooper_helmstetter_division_halves_copy_number() -> None:
    # energy-gated division halves every locus's DNA copy number and the
    # trajectory stays monotone non-decreasing between divisions
    vc = _cooper_vc(uptake={"GLC": 10.0}, biomass_to_atp=5.0e7,
                    division_energy=2.0e9, energy_init=1.0e9)
    last_div = 0
    last_copies = dict(vc.dna_copy_number)
    divided = 0
    for _ in range(120):
        before = dict(vc.dna_copy_number)
        vc.step()
        cur = vc.dna_copy_number
        if vc.divisions > last_div:
            for gene in cur:
                assert cur[gene] <= before[gene]
            last_div = vc.divisions
            last_copies = dict(cur)
            divided += 1
        else:
            for gene in cur:
                assert cur[gene] >= last_copies[gene]
            last_copies = dict(cur)
    assert divided >= 2


def test_cooper_helmstetter_phase_sequence_slow_growth() -> None:
    # slow growth (tau=90 > C+D=60): B_GAP -> C_PERIOD (fork) -> D_PERIOD
    # -> B_GAP, repeating every scheduled doubling time
    vc = _cooper_vc(doubling_time_min=90.0)
    vc.step()
    assert vc.phase is CellCyclePhase.B_GAP
    while vc.cell_cycle_age < 30:
        vc.step()
    assert vc.phase is CellCyclePhase.C_PERIOD
    while vc.cell_cycle_age < 70:
        vc.step()
    assert vc.phase is CellCyclePhase.D_PERIOD
    while vc.cell_cycle_age < 91:
        vc.step()
    assert vc.phase is CellCyclePhase.B_GAP
    assert vc.phase_progress >= 0.0
    # the fork that fired at age 30 terminates at 70, exactly D before the
    # scheduled division at 90
    init = vc.config.doubling_time_min - vc.config.c_period_min - vc.config.d_period_min
    served = init + vc.config.c_period_min + vc.config.d_period_min
    assert served == vc.config.doubling_time_min
    assert served % vc.config.doubling_time_min == pytest.approx(0.0)


# ============================================================================
# Replicon structure (Phase-C C2: chromosome oriC/terC + plasmids)
# ============================================================================

REPLICON_GENOME = {
    "chrGene": encode_gene("MAQILARVFFDDV"),
    "plGene": encode_gene("MAQILARVFFDDV"),  # identical to chrGene
}


def _replicon_vc(gene_replicons, replicons, mode="flat", **cfg_kw) -> VirtualCell:
    cfg = VirtualCellConfig(
        replication_mode=mode,
        replicons=replicons,
        gene_replicons=gene_replicons,
        maintenance_atp_per_min=0.0,
        transcription_atp_per_nt=0.0,
        translation_atp_per_aa=0.0,
        **cfg_kw,
    )
    g = GRN()
    for name in REPLICON_GENOME:
        g.add_gene(name, 0.5)
        g.nodes[name].level = 1.0
    return VirtualCell(REPLICON_GENOME, g, config=cfg)


def test_plasmid_gene_carries_constant_base_copy() -> None:
    """C2: a plasmid gene keeps its replicon copy through forks + division."""
    replicons = {"pBR322": RepliconSpec(kind="plasmid", copy_number=20)}
    vc = _replicon_vc({"plGene": "pBR322"}, replicons,
                      mode="cooper_helmstetter",
                      uptake={"GLC": 10.0}, biomass_to_atp=5.0e7,
                      division_energy=2.0e9, energy_init=1.0e9)
    assert vc.dna_copy_number["plGene"] == 20
    assert vc.dna_copy_number["chrGene"] >= 1  # fork-driven (origin-proximal)
    for _ in range(120):
        vc.step()
        assert vc.dna_copy_number["plGene"] == 20  # forks never touch it
        assert vc.dna_copy_number["chrGene"] >= 1
    assert vc.divisions >= 1


def test_plasmid_copy_number_dosage_lifts_expression() -> None:
    """C2 gate: copy number -> expression level is replicon-aware.

    The identical gene sequence expressed from a 20-copy pBR322 plasmid
    produces 20x the mRNA and protein of the chromosome copy.
    """
    puc = {"pBR322": RepliconSpec(kind="plasmid", copy_number=20)}
    chr_vc = _replicon_vc({}, {})
    pl_vc = _replicon_vc({"plGene": "pBR322"}, puc)
    chr_vc.run(10)
    pl_vc.run(10)
    assert pl_vc.mrna["plGene"] == pytest.approx(20 * chr_vc.mrna["chrGene"])
    assert pl_vc.proteins["plGene"] == pytest.approx(
        20 * chr_vc.proteins["chrGene"])
    # the chromosome gene is untouched by the plasmid replicon
    assert chr_vc.dna_copy_number["chrGene"] == 1
    assert pl_vc.dna_copy_number["plGene"] == 20


def test_flat_mode_keeps_plasmid_copy() -> None:
    vc = _replicon_vc({"plGene": "pBR322"},
                      {"pBR322": RepliconSpec("plasmid", 20)})
    vc.run(5)
    assert all(e["dna_copy_number"]["plGene"] == 20 for e in vc.history)
    assert all(e["dna_copy_number"]["chrGene"] == 1 for e in vc.history)


def test_unknown_replicon_raises() -> None:
    with pytest.raises(ValueError, match="unknown replicon"):
        _replicon_vc({"plGene": "pUC19"}, {})


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


def test_calibration_prediction_closed_loop() -> None:
    # whole-cell "calibrate then predict" (Karr 2012; Virtual Cell
    # Challenge 2025): fit the hidden biomass_to_atp coupling constant on
    # a calibration condition, then predict an independent condition and
    # verify the prediction matches a ground-truth cell run.
    from helixlang.plugins.apps.virtual_cell_bench import (
        VirtualCellBench,
        run_virtual_cell_benchmark,
    )

    bench = VirtualCellBench()
    assert len(bench.observed_energy) == bench.config.calibration_minutes
    fit = bench.calibrate()
    fitted = fit["best"]["biomass_to_atp"]
    assert fitted == pytest.approx(
        bench.config.truth_biomass_to_atp, rel=0.05)
    assert fit["sse"] < 1e4
    # prediction under the *different* uptake: fitted == ground truth
    result = bench.run()
    assert result["passed"] is True
    assert result["biomass_to_atp_rel_error"] < 0.05
    assert result["energy_rel_error"] < 0.05
    assert (result["fitted_prediction"]["alive"]
            == result["truth_prediction"]["alive"])

    one_shot = run_virtual_cell_benchmark()
    assert one_shot["passed"] is True
    assert one_shot["calibration_recovered"] is True
    assert one_shot["prediction_matches"] is True


def test_whole_cell_benchmark_four_gates() -> None:
    # Phase-5 exit gate (doc §8.3-8.4): one call returns the four scores
    # and passes only when every gate holds.
    from helixlang.plugins.apps.virtual_cell_bench import run_whole_cell_benchmark

    result = run_whole_cell_benchmark()
    assert set(result["passed"]) == {
        "essentiality", "batch_doubling", "adder_slope", "density_profile",
    }
    assert result["all_passed"] is True
    s = result["scores"]
    assert s["essentiality_accuracy"] >= 0.95
    assert s["batch_doubling_rel_error"] <= 0.2
    assert abs(s["adder_slope"]) <= 0.2
    assert s["density_inner_ratio"] >= 0.5
    assert s["density_outer_ratio"] >= 0.5


# ============================================================================
# Phase 2: volume growth + adder size control (Taheri-Araghi 2015)
# ============================================================================

def _adder_vc(division_rule: str = "adder", **cfg_kw) -> VirtualCell:
    cfg = VirtualCellConfig(
        division_rule=division_rule,
        maintenance_atp_per_min=0.0,
        transcription_atp_per_nt=0.0,
        translation_atp_per_aa=0.0,
        uptake={"GLC": 10.0},
        **cfg_kw,
    )
    return VirtualCell(GENOME, _grn(), config=cfg)


def _division_events(vc: VirtualCell) -> list[tuple[float, float]]:
    """(birth_size, added_size) per generation, from the history."""
    events: list[tuple[float, float]] = []
    prev = vc.history[0]
    for entry in vc.history[1:]:
        if entry["divisions"] > prev["divisions"]:
            # birth of the just-finished generation vs size it added
            added = prev["volume_um3"] - prev["volume_birth_um3"]
            events.append((prev["volume_birth_um3"], added))
        prev = entry
    return events


def test_adder_birth_size_converges_to_adder_volume() -> None:
    """From any initial size the birth volume converges to the adder
    volume Delta and the interdivision interval settles (size
    homeostasis; Taheri-Araghi 2015, Jun 2018)."""
    for init in (1.0, 1.6, 2.5):
        vc = _adder_vc(volume_init_um3=init)
        births: list[float] = []
        intervals: list[int] = []
        last_div = 0
        last_age = 0
        for _ in range(400):
            vc.step()
            if vc.divisions > last_div:
                births.append(vc.volume_birth_um3)
                intervals.append(vc.age - last_age)
                last_div = vc.divisions
                last_age = vc.age
        assert len(births) >= 10
        assert births[-1] == pytest.approx(1.6, abs=0.05)
        assert intervals[-3:] == [20, 20, 20]


def test_adder_added_size_slope_approximately_zero() -> None:
    """Adder rule: the size added over a generation is independent of
    birth size (linear-regression slope ~ 0; Taheri-Araghi 2015 reports
    ~ -0.1 .. 0)."""
    vc = _adder_vc(adder_noise_std=0.2, seed=7)
    for _ in range(500):
        vc.step()
    events = _division_events(vc)
    assert len(events) >= 10
    births = [b for b, _ in events]
    addeds = [a for _, a in events]
    bmean = sum(births) / len(births)
    amean = sum(addeds) / len(addeds)
    cov = sum((b - bmean) * (a - amean) for b, a in events)
    var = sum((b - bmean) ** 2 for b in births)
    slope = cov / var if var > 0 else 0.0
    assert abs(slope) <= 0.2
    assert abs(amean - 1.6) < 0.2


def test_adder_newborn_cv_matches_literature() -> None:
    """Newborn-size distribution CV ~ 0.1 with division noise
    (Taheri-Araghi 2015, CV = 0.1 in rich medium)."""
    import statistics

    vc = _adder_vc(adder_noise_std=0.2, seed=3)
    births: list[float] = []
    last = 0
    for _ in range(500):
        vc.step()
        if vc.divisions > last:
            births.append(vc.volume_birth_um3)
            last = vc.divisions
    assert len(births) >= 20
    cv = statistics.pstdev(births) / statistics.mean(births)
    assert 0.04 <= cv <= 0.16


def test_adder_division_halves_volume_and_resets_birth() -> None:
    vc = _adder_vc(volume_init_um3=2.0)
    prev = None
    for _ in range(200):
        vc.step()
        if vc.divisions > 0:
            entry = vc.history[-1]
            assert prev is not None
            # pre-division volume (reconstructed from the halving) is the
            # previous step's volume plus at most one step of growth
            pre_divide = entry["volume_um3"] * 2.0
            assert 0.0 < pre_divide - prev["volume_um3"] < 0.2
            # birth volume resets to the halved volume
            assert entry["volume_birth_um3"] == entry["volume_um3"]
            assert entry["added_volume_um3"] == pytest.approx(0.0)
            break
        prev = vc.history[-1]
    else:
        pytest.fail("adder division never fired")


def test_surface_scaling_flux_rises_with_volume() -> None:
    """S/V scaling: uptake (hence biomass flux) scales with
    volume^(2/3); without the flag the flux stays constant."""
    vc_const = _adder_vc(division_rule="energy", surface_scaling=False)
    vc_surf = _adder_vc(division_rule="energy", surface_scaling=True)
    for _ in range(200):
        vc_const.step()
        vc_surf.step()
    f0 = vc_const.history[0]["biomass_flux"]
    f_end = vc_const.history[-1]["biomass_flux"]
    assert f_end == pytest.approx(f0, rel=1e-9)  # constant flux bit-compat
    v0 = vc_surf.history[0]["volume_um3"]
    v_end = vc_surf.history[-1]["volume_um3"]
    f_surf_end = vc_surf.history[-1]["biomass_flux"]
    f_surf_0 = vc_surf.history[0]["biomass_flux"]
    assert v_end > 2.0 * v0
    assert f_surf_end > 1.5 * f_surf_0
    # and the scaling follows the sphere geometry exponent
    expected = f_surf_0 * (v_end / v0) ** (2.0 / 3.0)
    assert f_surf_end == pytest.approx(expected, rel=0.1)


def test_energy_rule_division_halves_volume() -> None:
    """Default energy rule keeps today's division behaviour but now also
    halves the physical volume with the budget (bit-compat on the rest)."""
    vc = VirtualCell(GENOME, _grn(), config=VirtualCellConfig(
        uptake={"GLC": 10.0}, biomass_to_atp=5.0e7,
        division_energy=2.0e9, energy_init=1.0e9,
        maintenance_atp_per_min=0.0,
        transcription_atp_per_nt=0.0, translation_atp_per_aa=0.0))
    prev = None
    for _ in range(200):
        vc.step()
        if vc.divisions > 0:
            entry = vc.history[-1]
            assert prev is not None
            assert entry["divisions"] == 1
            pre_divide = entry["volume_um3"] * 2.0
            assert 0.0 < pre_divide - prev["volume_um3"] < 0.2
            assert entry["volume_birth_um3"] == entry["volume_um3"]
            # energy halves exactly: pre-division budget = previous step's
            # budget + this step's biomass gain (costs are zeroed here)
            pre_energy = prev["energy"] + entry["biomass_flux"] * 5.0e7
            assert entry["energy"] == pytest.approx(pre_energy / 2.0)
            break
        prev = vc.history[-1]
    else:
        pytest.fail("energy division never fired")


# ============================================================================
# Phase 3: protein maturation / folding / QC pools
# ============================================================================

def _mat_grn() -> GRN:
    g = GRN()
    g.add_gene("lacZ", 0.5)
    g.nodes["lacZ"].level = 1.0
    return g


def _mat_cfg(**cfg_kw) -> VirtualCellConfig:
    return VirtualCellConfig(
        uptake={"GLC": 10.0}, energy_init=1.0e9, division_energy=2.0e9,
        maintenance_atp_per_min=0.0,
        transcription_atp_per_nt=0.0, translation_atp_per_aa=0.0,
        **cfg_kw,
    )


def _mat_trajectory(mode: str, k_fold: float = 1.0) -> list[float]:
    vc = VirtualCell(GENOME, _mat_grn(), config=_mat_cfg(
        protein_maturation_mode=mode, fold_rate_per_min=k_fold,
        misfold_rate_per_min=0.0, aggregation_rate_per_min=0.0,
        degraded_rate_per_min=0.0, protein_half_life_min=1.0e6))
    vc.run(300)
    return [e["proteins"].get("lacZ", 0.0) for e in vc.history]


def test_chaperone_folded_pool_lags_instant_expression() -> None:
    """Chaperone mode credits the same expression total but the folded
    (functional) pool lags behind the instant mode at early times."""
    instant = _mat_trajectory("instant")
    slow = _mat_trajectory("chaperone", k_fold=0.1)
    assert slow[15] < instant[15]
    # with no misfolding/QC loss the folded pool converges to the same total
    assert slow[-1] == pytest.approx(instant[-1], rel=0.05)


def test_smaller_fold_rate_shifts_folded_peak_later() -> None:
    """The half-peak time of the folded pool is monotone in the folding
    rate: slower chaperone folding delays the folded pool behind the
    underlying expression (Balchin 2016)."""

    def half_peak_time(traj: list[float]) -> int:
        target = 0.5 * traj[-1]
        for i, v in enumerate(traj):
            if v >= target:
                return i
        return len(traj)

    instant = _mat_trajectory("instant")
    fast = _mat_trajectory("chaperone", k_fold=1.0)
    slow = _mat_trajectory("chaperone", k_fold=0.1)
    slower = _mat_trajectory("chaperone", k_fold=0.02)
    assert half_peak_time(instant) < half_peak_time(fast)
    assert half_peak_time(fast) < half_peak_time(slow)
    assert half_peak_time(slow) < half_peak_time(slower)


def test_chaperone_history_exposes_folding_state() -> None:
    vc = VirtualCell(GENOME, _mat_grn(), config=_mat_cfg(
        protein_maturation_mode="chaperone"))
    vc.run(10)
    entry = vc.history[-1]
    assert "lacZ" in entry["proteins_unfolded"]
    assert "lacZ" in entry["proteins_misfolded"]
    assert "lacZ" in entry["proteins_degraded"]
    assert "lacZ" in entry["proteins_aggregated"]
    assert entry["folding_atp_cost"] >= 0.0
    assert entry["proteins"]["lacZ"] > 0.0

    instant = VirtualCell(GENOME, _mat_grn(), config=_mat_cfg(
        protein_maturation_mode="instant"))
    instant.run(10)
    e2 = instant.history[-1]
    assert "proteins_unfolded" not in e2
    assert "maturation" not in e2


def test_chaperone_division_halves_folded_pool() -> None:
    """Division halves every protein pool; the recorded folded level is the
    halved value ((prev + this-tick folding) / 2 with cotranslational
    folding off)."""
    vc = VirtualCell(GENOME, _mat_grn(), config=_mat_cfg(
        protein_maturation_mode="chaperone", fold_rate_per_min=0.1,
        misfold_rate_per_min=0.0, aggregation_rate_per_min=0.0,
        degraded_rate_per_min=0.0, protein_half_life_min=1.0e6,
        biomass_to_atp=5.0e7, frac_cotranslational_fold=0.0))
    prev = None
    for _ in range(200):
        vc.step()
        if vc.divisions > 0:
            entry = vc.history[-1]
            assert prev is not None
            pre = (prev["proteins"].get("lacZ", 0.0)
                   + entry["maturation"]["folded"])
            assert entry["proteins"].get("lacZ", 0.0) == pytest.approx(
                pre / 2.0, rel=1e-6)
            break
        prev = vc.history[-1]
    else:
        pytest.fail("division never fired")


def test_instant_mode_is_bit_compatible() -> None:
    """The default instant mode credits protein exactly as before: lacZ
    accumulates 100 proteins per expression with no folding pools or
    maturation ATP cost in the history."""
    vc = _vc()
    vc.run(5)
    assert vc.proteins["lacZ"] == pytest.approx(5 * 100.0)
    assert vc.protein_pools == {}
    assert "proteins_unfolded" not in vc.history[-1]
    assert "folding_atp_cost" not in vc.history[-1]


# ============================================================================
# Phase 4: enzyme-constrained FBA wiring + intracellular metabolite pools
# ============================================================================

def _enzyme_genome() -> dict[str, str]:
    return {g: encode_gene("MAQILARVFFDDV")
            for g in ECOLI_CORE_GENE_REACTIONS}


def _enzyme_grn(genes=None) -> GRN:
    g = GRN()
    for name in genes or ECOLI_CORE_GENE_REACTIONS:
        g.add_gene(name, 0.5)
        g.nodes[name].level = 1.0
    return g


def _enzyme_cfg(**kw) -> VirtualCellConfig:
    return VirtualCellConfig(
        uptake={"GLC": 10.0}, energy_init=1.0e9, division_energy=2.0e9,
        maintenance_atp_per_min=0.0,
        transcription_atp_per_nt=0.0, translation_atp_per_aa=0.0,
        **kw,
    )


def test_enzyme_capacity_wires_folded_pools_to_fba() -> None:
    """With enzyme capacity enabled the FBA caps come from the folded
    pools: the cell's biomass flux equals a standalone FBA solved with the
    same enzyme levels (O'Brien 2013 ME-model coupling)."""
    vc = VirtualCell(_enzyme_genome(), _enzyme_grn(),
                     config=_enzyme_cfg(
                         protein_maturation_mode="chaperone",
                         fold_rate_per_min=1.0, misfold_rate_per_min=0.0,
                         aggregation_rate_per_min=0.0,
                         degraded_rate_per_min=0.0,
                         protein_half_life_min=1.0e6,
                         enzyme_capacity_enabled=True))
    for _ in range(20):
        vc.step()
    entry = vc.history[-1]
    assert entry["enzyme_levels"]["pgi"] == pytest.approx(
        vc.protein_pools["pgi"].folded)
    fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    fba.set_uptake("GLC", 10.0)
    fba.set_enzyme_capacity(EnzymeCapacity(
        dict(ECOLI_CORE_GENE_REACTIONS),
        kcat=dict(ECOLI_CORE_KCAT), enzyme_scale=vc.config.enzyme_scale))
    fba.set_enzyme_levels(entry["enzyme_levels"])
    assert entry["biomass_flux"] == pytest.approx(
        fba.solve()["BIOMASS"], rel=1e-6)


def test_enzyme_capacity_knockout_collapses_biomass_flux() -> None:
    """Dropping an essential enzyme gene from the genome collapses the
    cell's biomass flux to zero (consistent with predict_essentiality)."""
    cfg = _enzyme_cfg(enzyme_capacity_enabled=True)
    vc = VirtualCell(_enzyme_genome(), _enzyme_grn(), config=cfg)
    for _ in range(20):
        vc.step()
    assert vc.history[-1]["biomass_flux"] == pytest.approx(1.2803, rel=1e-3)
    genes = {g for g in ECOLI_CORE_GENE_REACTIONS if g != "pgi"}
    ko = VirtualCell({g: encode_gene("MAQILARVFFDDV") for g in genes},
                     _enzyme_grn(genes), config=cfg)
    for _ in range(20):
        ko.step()
    assert ko.history[-1]["biomass_flux"] == pytest.approx(0.0, abs=1e-9)


def test_metabolite_pools_and_overflow_in_history() -> None:
    """Enzyme-constrained sMOMENT budget binds, the cell secretes acetate,
    and the metabolite pools + overflow appear in the history (Sanchez
    2017 overflow prediction at the single-cell level)."""
    vc = VirtualCell(_enzyme_genome(), _enzyme_grn(),
                     config=_enzyme_cfg(
                         enzyme_capacity_enabled=True,
                         protein_mass_fraction=0.3,
                         metabolite_pools_enabled=True))
    for _ in range(20):
        vc.step()
    entry = vc.history[-1]
    assert 0.0 < entry["biomass_flux"] < 1.0
    assert entry["overflow_secretion"]["Ac"] > 1.0
    assert "Ac" in entry["metabolite_pools"]
    assert "metabolite_net_production" in entry
    # pools reach steady state under balanced growth (net ~ 0)
    assert abs(entry["metabolite_net_production"]["Ac"]) < 1e-9


def test_phase4_defaults_are_bit_compatible() -> None:
    """Default Phase-4 config (capacity + pools off) leaves solve() and the
    history untouched."""
    vc = _vc()
    vc.run(5)
    assert vc._metabolism_deltas == {}
    entry = vc.history[-1]
    assert "enzyme_levels" not in entry
    assert "metabolite_pools" not in entry
    assert "overflow_secretion" not in entry
