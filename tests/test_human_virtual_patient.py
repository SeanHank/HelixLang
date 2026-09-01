"""Tests for the Virtual Patient unified facade (doc/28)."""
from __future__ import annotations

from helixlang.plugins.human.drug import Drug, DrugMolecule, get_predefined_drug
from helixlang.plugins.human.genotype import create_default_genotype
from helixlang.plugins.human.phenotype import ExternalTraits
from helixlang.plugins.human.virtual_patient import (
    VirtualPatient,
    VirtualPatientConfig,
    VirtualPatientResult,
    run_virtual_patient_cohort,
)


class TestVirtualPatientConfig:
    def test_defaults(self):
        cfg = VirtualPatientConfig()
        assert cfg.total_duration_days == 30.0
        assert cfg.dfa_dt_h == 1.0
        assert cfg.disease is None

    def test_with_disease_profile_name(self):
        cfg = VirtualPatientConfig(disease_profile_name="PKU")
        assert cfg.disease is not None
        assert "phenyl" in cfg.disease.name.lower()

    def test_auto_pd_assignment(self):
        drug = get_predefined_drug("metformin")
        assert drug is not None
        cfg = VirtualPatientConfig(
            drugs=[drug],
            disease_profile_name="DIABETES_T2",
        )
        key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
        assert key in cfg.pharmacodynamics

    def test_custom_genotype(self):
        gen = create_default_genotype()
        cfg = VirtualPatientConfig(genotype=gen)
        assert cfg.genotype is gen

    def test_custom_traits(self):
        traits = ExternalTraits(age_years=65, sex="female", body_weight_kg=60.0)
        cfg = VirtualPatientConfig(traits=traits)
        assert cfg.traits.age_years == 65.0


class TestVirtualPatientResult:
    def test_to_dict_keys(self):
        r = VirtualPatientResult()
        d = r.to_dict()
        assert "time_h" in d
        assert "vitals" in d
        assert "labs" in d
        assert "drug_concentrations" in d
        assert "disease" in d
        assert "summary" in d

    def test_summary_string(self):
        r = VirtualPatientResult()
        s = r.summary()
        assert "Virtual Patient Simulation Summary" in s


class TestVirtualPatientRun:
    def test_healthy_patient_1day(self):
        cfg = VirtualPatientConfig(total_duration_days=1.0)
        vp = VirtualPatient(cfg)
        result = vp.run()
        assert len(result.time_h) > 0
        assert len(result.systolic_bp) > 0
        assert len(result.alt) > 0

    def test_healthy_vitals_in_range(self):
        cfg = VirtualPatientConfig(total_duration_days=1.0)
        result = VirtualPatient(cfg).run()
        assert all(60.0 <= bp <= 200.0 for bp in result.systolic_bp)
        assert all(30.0 <= hr <= 180.0 for hr in result.heart_rate)

    def test_diabetes_patient_glucose(self):
        cfg = VirtualPatientConfig(
            disease_profile_name="DIABETES_T2",
            total_duration_days=2.0,
        )
        result = VirtualPatient(cfg).run()
        assert result.glucose[0] > 80.0

    def test_cisplatin_alt_rises(self):
        drug = get_predefined_drug("cisplatin")
        assert drug is not None
        cfg = VirtualPatientConfig(drugs=[drug], total_duration_days=3.0)
        result = VirtualPatient(cfg).run()
        assert result.max_alt >= result.alt[0]

    def test_drug_concentrations_tracked(self):
        drug = get_predefined_drug("metformin")
        assert drug is not None
        cfg = VirtualPatientConfig(drugs=[drug], total_duration_days=1.0)
        result = VirtualPatient(cfg).run()
        assert len(result.drug_concentrations) > 0

    def test_result_disease_stage(self):
        cfg = VirtualPatientConfig(
            disease_profile_name="DIABETES_T2",
            total_duration_days=2.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.disease_stage) > 0
        assert any(s != "healthy" for s in result.disease_stage)


def _simple_drug(name: str) -> Drug:
    return Drug(
        molecule=DrugMolecule(name=name),
        dose_mg=10.0,
        dosing_interval_h=24.0,
        duration_days=3.0,
        hepatic_extraction_ratio=0.8,
    )


def _peak(drugs, name):
    cfg = VirtualPatientConfig(drugs=drugs, total_duration_days=5.0)
    result = VirtualPatient(cfg).run()
    return max(result.drug_concentrations.get(name, [0.0]))


class TestMultiDrugMechanisticDDI:
    def test_multi_drug_run_completes(self):
        cfg = VirtualPatientConfig(
            drugs=[_simple_drug("amiodarone"), _simple_drug("warfarin")],
            total_duration_days=2.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.time_h) > 0
        assert len(result.alt) == len(result.time_h)

    def test_ddi_alert_raises_victim_exposure(self):
        alone = _peak([_simple_drug("warfarin")], "warfarin")
        combo = _peak(
            [_simple_drug("amiodarone"), _simple_drug("warfarin")], "warfarin"
        )
        assert combo > alone * 1.1

    def test_three_drug_run_completes(self):
        cfg = VirtualPatientConfig(
            drugs=[
                _simple_drug("amiodarone"),
                _simple_drug("warfarin"),
                _simple_drug("metformin"),
            ],
            total_duration_days=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.time_h) > 0


class TestStochasticAndDenoising:
    def test_default_run_is_deterministic(self):
        r1 = VirtualPatient(VirtualPatientConfig(total_duration_days=2.0)).run()
        r2 = VirtualPatient(VirtualPatientConfig(total_duration_days=2.0)).run()
        assert r1.alt == r2.alt

    def test_stochastic_same_seed_reproducible(self):
        runs = []
        for _ in range(2):
            vp = VirtualPatient(VirtualPatientConfig(total_duration_days=3.0))
            vp.enable_stochastic(seed=42)
            runs.append(vp.run())
        assert runs[0].alt == runs[1].alt
        assert runs[0].creatinine == runs[1].creatinine

    def test_stochastic_changes_trajectory(self):
        base = VirtualPatient(VirtualPatientConfig(total_duration_days=3.0)).run()
        vp = VirtualPatient(VirtualPatientConfig(total_duration_days=3.0))
        vp.enable_stochastic(seed=7)
        noisy = vp.run()
        assert any(a != b for a, b in zip(base.alt, noisy.alt, strict=True))

    def test_disable_stochastic_restores_determinism(self):
        vp = VirtualPatient(VirtualPatientConfig(total_duration_days=2.0))
        vp.enable_stochastic(seed=3)
        vp.disable_stochastic()
        r1 = vp.run()
        r2 = VirtualPatient(VirtualPatientConfig(total_duration_days=2.0)).run()
        assert r1.alt == r2.alt

    def test_denoising_smooths_and_preserves_raw(self):
        vp = VirtualPatient(VirtualPatientConfig(total_duration_days=3.0))
        vp.enable_stochastic(seed=11)
        vp.enable_denoising()
        res = vp.run()
        assert len(res.raw_alt) == len(res.alt)
        assert res.alt != res.raw_alt
        raw_jitter = sum(
            abs(b - a) for a, b in zip(res.raw_alt, res.raw_alt[1:], strict=False)
        )
        smooth_jitter = sum(abs(b - a) for a, b in zip(res.alt, res.alt[1:], strict=False))
        assert smooth_jitter < raw_jitter


class TestCohortParallelism:
    """doc/42 Phase C PF-3 — cohort-level parallel VirtualPatient runs."""

    def test_single_process_cohort(self):
        cfg = VirtualPatientConfig(total_duration_days=1.0)
        results = run_virtual_patient_cohort(
            cfg, n_patients=2, workers=1, base_seed=7, stochastic=True
        )
        assert len(results) == 2
        for r in results:
            assert len(r.time_h) > 0
            assert len(r.alt) > 0

    def test_seed_contract_varies_patients(self):
        cfg = VirtualPatientConfig(total_duration_days=1.0)
        results = run_virtual_patient_cohort(
            cfg, n_patients=2, workers=1, base_seed=7, stochastic=True
        )
        assert results[0].alt != results[1].alt

    def test_same_base_seed_reproduces(self):
        cfg = VirtualPatientConfig(total_duration_days=1.0)
        r1 = run_virtual_patient_cohort(
            cfg, n_patients=2, workers=1, base_seed=7, stochastic=True
        )
        r2 = run_virtual_patient_cohort(
            cfg, n_patients=2, workers=1, base_seed=7, stochastic=True
        )
        assert [x.alt for x in r1] == [x.alt for x in r2]
        assert [x.creatinine for x in r1] == [x.creatinine for x in r2]

    def test_parallel_matches_serial(self):
        cfg = VirtualPatientConfig(total_duration_days=1.0)
        serial = run_virtual_patient_cohort(
            cfg, n_patients=2, workers=1, base_seed=7, stochastic=True
        )
        parallel = run_virtual_patient_cohort(
            cfg, n_patients=2, workers=2, base_seed=7, stochastic=True
        )
        assert [x.alt for x in serial] == [x.alt for x in parallel]
        assert [x.heart_rate for x in serial] == [x.heart_rate for x in parallel]

    def test_default_deterministic_cohort(self):
        cfg = VirtualPatientConfig(total_duration_days=1.0)
        results = run_virtual_patient_cohort(
            cfg, n_patients=2, workers=1, base_seed=7, stochastic=False
        )
        assert results[0].alt == results[1].alt


class TestRl5ClearanceCoupling:
    """doc/42 Phase B RL-5: organ function -> organ clearance coupling."""

    def test_default_modifiers_are_one(self):
        drug = get_predefined_drug("metformin")
        cfg = VirtualPatientConfig(drugs=[drug], physiological_core=True)
        assert cfg.physiological_core is True
        vp = VirtualPatient(cfg)
        vp._init_drug_engines()
        key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
        engine = vp._drug_engine[key]
        assert engine.renal_clearance_modifier == 1.0
        assert engine.hepatic_clearance_modifier == 1.0

    def test_falling_egfr_reduces_renal_modifier(self):
        drug = get_predefined_drug("metformin")
        cfg = VirtualPatientConfig(drugs=[drug], physiological_core=True)
        vp = VirtualPatient(cfg)
        vp._init_drug_engines()
        key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
        engine = vp._drug_engine[key]
        before = engine.renal_clearance_modifier
        labs = type("L", (), {"egfr_ml_per_min": 30.0})()
        vp._apply_rl5_clearance_coupling(labs)
        assert engine.renal_clearance_modifier < before

    def test_severe_disease_reduces_hepatic_modifier(self):
        drug = get_predefined_drug("metformin")
        cfg = VirtualPatientConfig(
            drugs=[drug],
            physiological_core=True,
        )
        vp = VirtualPatient(cfg)
        # assign organ-disease severity after construction (the virtual
        # patient reads a full DiseaseState during __init__ only)
        vp.config.disease = type("D", (), {"name": "liver_failure", "severity": 1.0})()
        vp._init_drug_engines()
        key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
        engine = vp._drug_engine[key]
        before = engine.hepatic_clearance_modifier
        vp._apply_rl5_clearance_coupling(type("L", (), {"egfr_ml_per_min": 100.0})())
        assert engine.hepatic_clearance_modifier < before
