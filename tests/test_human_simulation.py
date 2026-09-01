"""Tests for helixlang.plugins.human.simulation (doc/27 Stage F engine).

Covers HumanSimulationConfig/Result contracts and end-to-end long-term
runs coupling PBPK + PD + disease-perturbed dFBA: Gaucher on
imiglucerase, PKU biomarker normalization, healthy controls, and type 2
diabetes under metformin.
"""
from __future__ import annotations

from helixlang.plugins.human.disease import DISEASE_PROFILES
from helixlang.plugins.human.drug import get_predefined_drug
from helixlang.plugins.human.pharmacodynamics import PREDEFINED_PD, PDEffect, Pharmacodynamics
from helixlang.plugins.human.simulation import (
    HumanSimulation,
    HumanSimulationConfig,
    HumanSimulationResult,
    run_human_simulation_cohort,
)


def _config(days: float = 2.0, **overrides) -> HumanSimulationConfig:
    """Short-horizon config keeping the dFBA loop fast in tests."""
    base = {
        "total_duration_days": days,
        "track_fluxes": False,
        "output_time_resolution_h": 2.0,
    }
    base.update(overrides)
    return HumanSimulationConfig(**base)


def _pah_activation_pd() -> Pharmacodynamics:
    """PD profile that restores PAH activity (biomarker driver for PKU)."""
    return Pharmacodynamics(
        drug_name="metformin",
        effects=[PDEffect(
            target_reaction="PAH", target_gene="PAH",
            effect_type="activation", ec50_um=1.0, emax=1.0,
            baseline_effect=0.0,
        )],
    )


def test_simulation_config_defaults():
    """Config defaults match the doc/27 section 9 baseline."""
    config = HumanSimulationConfig()
    assert config.total_duration_days == 30.0
    assert config.dfa_dt_h == 1.0
    assert config.pbpk_dt_min == 1.0
    assert config.target_tissue == "liver"
    assert config.drugs == []
    assert config.disease is None


def test_simulation_creation():
    """HumanSimulation builds engines, pools, and a diseased model."""
    config = _config(drugs=[get_predefined_drug("METFORMIN")])
    simulation = HumanSimulation(config)
    assert len(simulation.engines) == 1
    assert simulation.model.reactions


def test_simulation_run_returns_result():
    """run() returns a HumanSimulationResult instance."""
    config = _config(drugs=[get_predefined_drug("METFORMIN")], days=1)
    result = HumanSimulation(config).run()
    assert isinstance(result, HumanSimulationResult)


def test_simulation_result_has_time():
    """The sampling grid records t=0 through the full horizon."""
    config = _config(days=1, output_time_resolution_h=1.0)
    result = HumanSimulation(config).run()
    assert result.time_h
    assert result.time_h[0] == 0.0
    assert result.time_h[-1] > 0.0
    assert len(result.plasma_concentration) == len(result.time_h)


def test_simulation_result_has_drug_levels():
    """Per-drug concentration series are tracked at output resolution."""
    drug = get_predefined_drug("IBUPROFEN")
    config = _config(drugs=[drug], days=1, output_time_resolution_h=1.0)
    result = HumanSimulation(config).run()
    assert set(result.drug_concentrations) == {"ibuprofen"}
    series = result.drug_concentrations["ibuprofen"]
    assert len(series) == len(result.time_h)
    assert max(series) > 0.0


def test_simulation_gaucher_imiglucerase():
    """Gaucher patient on IV imiglucerase completes without error."""
    config = _config(
        drugs=[get_predefined_drug("IMIGLUCERASE")],
        disease=DISEASE_PROFILES["GAUCHER"],
        pharmacodynamics={"imiglucerase": PREDEFINED_PD["imiglucerase_gaucher"]},
    )
    result = HumanSimulation(config).run()
    assert result.time_h[-1] == 48.0
    assert "glucosylceramide" in result.biomarker_history
    assert result.biomarker_history["glucosylceramide"][0] > 0.0


def test_simulation_pku_phe_reduced():
    """Effective therapy relaxes plasma phenylalanine toward normal."""
    config = _config(
        drugs=[get_predefined_drug("METFORMIN")],
        disease=DISEASE_PROFILES["PKU"],
        pharmacodynamics={"metformin": _pah_activation_pd()},
    )
    result = HumanSimulation(config).run()
    phe = result.biomarker_history["phenylalanine"]
    assert phe[0] == 2.4  # pathological seed at full severity
    assert phe[-1] < phe[0]


def test_simulation_result_to_dict():
    """to_dict() serializes every trajectory to JSON-safe types."""
    config = _config(drugs=[get_predefined_drug("METFORMIN")], days=1)
    result = HumanSimulation(config).run()
    data = result.to_dict()
    assert isinstance(data, dict)
    assert isinstance(data["time_h"], list)
    assert isinstance(data["auc_plasma"], float)


def test_simulation_result_summary():
    """summary() renders a human-readable multi-line report."""
    config = _config(drugs=[get_predefined_drug("METFORMIN")], days=1)
    result = HumanSimulation(config).run()
    text = result.summary()
    assert isinstance(text, str)
    assert "Human simulation summary" in text
    assert "plasma Cmax" in text


def test_simulation_no_disease():
    """A healthy virtual patient simulates with no biomarkers tracked."""
    config = _config(drugs=[get_predefined_drug("IBUPROFEN")], days=1)
    result = HumanSimulation(config).run()
    assert result.time_h
    assert not result.biomarker_history
    assert result.auc_plasma >= 0.0


def test_simulation_metformin_diabetes():
    """Diabetic patient on metformin runs with glucose biomarkers."""
    config = _config(
        drugs=[get_predefined_drug("METFORMIN")],
        disease=DISEASE_PROFILES["DIABETES_T2"],
        pharmacodynamics={"metformin": PREDEFINED_PD["metformin_complex1"]},
    )
    result = HumanSimulation(config).run()
    assert "glucose" in result.biomarker_history
    glucose = result.biomarker_history["glucose"]
    assert glucose[0] > 5.0  # seeded above the 5 mM healthy reference
    assert result.time_h[-1] == 48.0


def test_simulation_batch_multi_drug_advance_matches_per_engine():
    """doc/39 O4: batched advance_batch == per-engine solve_ivp (tolerance)."""
    from helixlang.plugins.human.physiology import create_default_physiology
    from helixlang.plugins.human.simulation import _PBPKEngine

    phys = create_default_physiology()
    names = ("IBUPROFEN", "METFORMIN", "OMEPRAZOLE")
    per = [_PBPKEngine(get_predefined_drug(n), phys, 10.0) for n in names]
    bat = [_PBPKEngine(get_predefined_drug(n), phys, 10.0) for n in names]
    worst = 0.0
    for hour in range(12):
        for engine in per:
            engine.apply_due_doses(hour)
            engine.advance(1.0)
        for engine in bat:
            engine.apply_due_doses(hour)
        assert bat[0].advance_batch(1.0, bat[1:])
        for a, b in zip(per, bat, strict=True):
            for key in a.conc_um:
                aa, bb = a.conc_um[key], b.conc_um[key]
                worst = max(worst, abs(aa - bb) / max(1.0, abs(aa), abs(bb)))
    assert worst < 1e-4
    # all engines advanced in lockstep to the same time
    assert all(b.time_h == bat[0].time_h for b in bat)
    assert bat[0].time_h == 12.0


class TestCohortParallelism:
    """doc/42 Phase C PF-3 — cohort-level parallel HumanSimulation runs."""

    def test_single_process_cohort(self):
        cfg = _config(days=1.0)
        results = run_human_simulation_cohort(cfg, n_simulations=2, workers=1)
        assert len(results) == 2
        for r in results:
            assert len(r.time_h) > 0
            assert len(r.plasma_concentration) > 0

    def test_parallel_matches_serial(self):
        cfg = _config(days=1.0)
        serial = run_human_simulation_cohort(cfg, n_simulations=2, workers=1)
        parallel = run_human_simulation_cohort(cfg, n_simulations=2, workers=2)
        assert [x.plasma_concentration for x in serial] == [
            x.plasma_concentration for x in parallel
        ]
        assert [x.biomarker_history for x in serial] == [
            x.biomarker_history for x in parallel
        ]

    def test_cohort_with_drug(self):
        cfg = _config(
            days=1.0,
            drugs=[get_predefined_drug("METFORMIN")],
        )
        results = run_human_simulation_cohort(cfg, n_simulations=2, workers=2)
        assert len(results) == 2
        assert all(len(r.drug_concentrations) > 0 for r in results)
