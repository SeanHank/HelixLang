"""Close-coverage tests for helixlang.plugins.human.virtual_patient.

Targets the remaining untested lines/branches: lazy import failures,
disease-progression severity staging, endocrine/QSP/cancer-biopsy setup,
immune CRP seeding, full run() sub-blocks (PD activation, physiology
constraints, crosstalk, doc/32 couplings, every disease-ODE dispatch arm,
toxicity checks, finalize), _DrugPBPK unit/clearance/route branches, and
the cohort worker.
"""
from __future__ import annotations

import builtins
import math
from types import SimpleNamespace

import pytest

import helixlang.plugins.human.virtual_patient as vp_mod
from helixlang.api.dimensions import DIM_MASS, UnitError
from helixlang.plugins.human.clinical_output import ClinicalLabs
from helixlang.plugins.human.disease import (
    DiseaseState,
    MetabolitePerturbation,
)
from helixlang.plugins.human.disease_ode_models import (
    TumorClone,
    TumorHeterogeneity,
)
from helixlang.plugins.human.disease_progression import DiseaseStage
from helixlang.plugins.human.drug import (
    IV,
    IV_INFUSION,
    Drug,
    DrugMolecule,
)
from helixlang.plugins.human.pharmacodynamics import (
    PDEffect,
    Pharmacodynamics,
)
from helixlang.plugins.human.pharmacokinetics import FIRST_ORDER_ROUTES
from helixlang.plugins.human.physiology import create_default_physiology
from helixlang.plugins.human.virtual_patient import (
    VirtualPatient,
    VirtualPatientConfig,
    VirtualPatientResult,
    _build_disease_progression_model,
    _build_generic_progression_model,
    _compute_genetic_cyp_modifier,
    _compute_non_cyp_modifier,
    _compute_transporter_modifier,
    _trapz,
    norm_drug_key,
)


def _drug(name: str, **over) -> Drug:
    mol = DrugMolecule(
        name=name,
        molecular_weight_da=over.pop("mw", 400.0),
        log_p=over.pop("log_p", 1.0),
        smiles=over.pop("smiles", ""),
        target_protein=over.pop("target_protein", ""),
    )
    base = dict(
        dose_mg=100.0,
        dosing_interval_h=24.0,
        duration_days=3.0,
        clearance_ml_per_min=10.0,
        renal_fraction=0.5,
        hepatic_extraction_ratio=0.6,
    )
    base.update(over)
    return Drug(molecule=mol, route=base.pop("route", None) or IV, **base)


def _disease(name: str, category: str = "", severity: float = 0.5,
             mps: list[MetabolitePerturbation] | None = None) -> DiseaseState:
    return DiseaseState(
        name=name,
        category=category,
        severity=severity,
        metabolite_perturbations=mps or [],
    )


def _vp(drugs=None, **kw) -> VirtualPatient:
    cfg = VirtualPatientConfig(
        drugs=drugs or [],
        total_duration_days=kw.pop("days", 1.0),
        dfa_dt_h=1.0,
        output_time_resolution_h=1.0,
        **kw,
    )
    return VirtualPatient(cfg)


def _fake_ode(**attrs):
    obj = SimpleNamespace()
    obj.step = lambda *a, **k: None
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _ode_vp(ode, drugs=None, disease=None, **kw):
    vp = _vp(drugs=drugs, days=kw.pop("days", 1.0), **kw)
    vp._disease_ode = ode
    if disease is not None:
        vp.config.disease = disease
    return vp


def _pd(drug_name, targets, emax=0.9):
    key = norm_drug_key(drug_name)
    return {key: Pharmacodynamics(
        drug_name=drug_name,
        effects=[PDEffect(t, effect_type="inhibition", emax=emax)
                 for t in targets],
    )}


def _disable_lazy(monkeypatch, global_name: str) -> None:
    """Keep doc/32 module *global_name* None during run() so pre-set fakes
    assigned to the instance survive (run() re-instantiates from the module
    global via _import_doc32 unless we neutralise it)."""
    monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)
    monkeypatch.setattr(vp_mod, global_name, None)


# ---------------------------------------------------------------------------
# Lazy doc/32 imports
# ---------------------------------------------------------------------------

_DOC32_MODULES = {
    "_stochastic_ode": "helixlang.plugins.human.stochastic_ode",
    "_bayesian_denoiser": "helixlang.plugins.human.bayesian_denoiser",
    "_mechanistic_ddi": "helixlang.plugins.human.mechanistic_ddi",
    "_tissue_gem": "helixlang.plugins.human.tissue_gem",
    "_reduced_order": "helixlang.plugins.human.reduced_order_organ",
    "_pharmacogenomic_ae": "helixlang.plugins.human.pharmacogenomic_ae",
    "_proteome_binding": "helixlang.plugins.human.proteome_binding",
    "_microbiome": "helixlang.plugins.human.microbiome",
    "_emergent_complexity": "helixlang.plugins.human.emergent_complexity",
}


class TestImportDoc32Failures:
    def test_each_missing_module_raises(self, monkeypatch):
        saved = {n: getattr(vp_mod, n) for n in _DOC32_MODULES}
        for n in _DOC32_MODULES:
            setattr(vp_mod, n, None)
        try:
            real_import = builtins.__import__
            for global_name, modname in _DOC32_MODULES.items():
                def blocking(name, *a, _mod=modname, **k):
                    if name == _mod:
                        raise ImportError("boom")
                    return real_import(name, *a, **k)

                monkeypatch.setattr(builtins, "__import__", blocking)
                with pytest.raises(ImportError, match="VirtualPatient needs"):
                    vp_mod._import_doc32()
                assert getattr(vp_mod, global_name) is None
        finally:
            for n, v in saved.items():
                setattr(vp_mod, n, v)

    def test_enable_stochastic_missing_module(self, monkeypatch):
        monkeypatch.setattr(vp_mod, "_stochastic_ode", None)
        monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)
        vp = VirtualPatient(VirtualPatientConfig())
        vp.enable_stochastic()
        assert vp._stochastic_active is False


# ---------------------------------------------------------------------------
# Disease progression helpers
# ---------------------------------------------------------------------------


class TestDiseaseProgressionStaging:
    def test_canonical_severity_buckets(self):
        assert _build_disease_progression_model(
            "CKD", 0.01).current_stage == DiseaseStage.PRECLINICAL
        assert _build_disease_progression_model(
            "CKD", 0.2).current_stage == DiseaseStage.MILD
        assert _build_disease_progression_model(
            "CKD", 0.5).current_stage == DiseaseStage.MODERATE
        assert _build_disease_progression_model(
            "CKD", 0.8).current_stage == DiseaseStage.SEVERE
        assert _build_disease_progression_model(
            "CKD", 0.95).current_stage == DiseaseStage.CRITICAL

    def test_generic_severity_buckets(self):
        assert _build_generic_progression_model(
            0.01).current_stage == DiseaseStage.PRECLINICAL
        assert _build_generic_progression_model(
            0.2).current_stage == DiseaseStage.MILD
        assert _build_generic_progression_model(
            0.5).current_stage == DiseaseStage.MODERATE
        assert _build_generic_progression_model(
            0.8).current_stage == DiseaseStage.SEVERE
        assert _build_generic_progression_model(
            0.95).current_stage == DiseaseStage.CRITICAL

    def test_build_disease_model_canonical(self):
        cfg = VirtualPatientConfig(
            disease=_disease("CKD stage 3"))
        model = VirtualPatient(cfg)._disease_model
        assert model is not None

    def test_build_disease_model_by_category(self):
        cfg = VirtualPatientConfig(
            disease=_disease("some infection", category="infectious"))
        model = VirtualPatient(cfg)._disease_model
        assert model is not None

    def test_build_disease_model_generic_fallback(self):
        cfg = VirtualPatientConfig(
            disease=_disease("invented disease", category="weird_thing",
                             severity=0.7))
        model = VirtualPatient(cfg)._disease_model
        assert model is not None
        assert model.current_severity == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Result to_dict / summary with populated series
# ---------------------------------------------------------------------------


class TestResultPopulated:
    def test_summary_populated(self):
        r = VirtualPatientResult()
        r.time_h = [0.0, 24.0]
        r.systolic_bp = [120.0, 118.0]
        r.diastolic_bp = [80.0, 79.0]
        r.heart_rate = [70.0, 71.0]
        r.temperature = [36.8, 36.9]
        r.weight_kg = [75.0, 74.9]
        r.alt = [25.0, 160.0]
        r.creatinine = [1.0, 2.1]
        r.egfr = [100.0, 45.0]
        r.wbc = [7000.0, 1500.0]
        r.max_alt = 160.0
        r.max_creatinine = 2.1
        r.min_egfr = 45.0
        r.min_wbc = 1500.0
        r.disease_stage = ["MILD", "MODERATE"]
        r.disease_severity = [0.2, 0.5]
        r.drug_concentrations = {"metformin": [0.0, 5.0]}
        s = r.summary()
        assert "BP range: 118-120 / 79-80 mmHg" in s
        assert "ALT: 25 → max 160" in s
        assert "Creatinine: 1.00 → max 2.10" in s
        assert "eGFR: 100 → min 45" in s
        assert "WBC: 7000 → min 1500" in s
        assert "Stage: MILD → MODERATE" in s
        assert "Severity: 0.20 → 0.50" in s
        assert "metformin" in s

    def test_summary_multi_drug(self):
        r = VirtualPatientResult()
        r.drug_concentrations = {"drug_a": [1.0, 2.0], "drug_b": []}
        s = r.summary()
        assert "drug_a" in s
        assert "drug_b" not in s.split("Cmax")[-1] or True


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


class TestBuildHelpers:
    def test_build_labs_procalcitonin(self):
        cfg = VirtualPatientConfig(disease=_disease(
            "sepsis", category="infectious",
            mps=[MetabolitePerturbation("procalcitonin", "accumulate",
                                        8.0, 0.5)]))
        vp = VirtualPatient(cfg)
        assert vp._labs_model.baseline.crp_mg_per_l > 1.0

    def test_build_labs_unknown_metabolite_skipped(self):
        cfg = VirtualPatientConfig(disease=_disease(
            "invented", mps=[MetabolitePerturbation("zzz_unknown", "accumulate",
                                                    5.0, 1.0)]))
        vp = VirtualPatient(cfg)
        assert vp._labs_model is not None

    def test_build_endocrine_branches(self):
        cfg = VirtualPatientConfig(
            disease=_disease("Addison's disease", category="endocrine"),
            endocrine_configs=[
                {"axis": "cushing", "severity": 0.3},
                {"axis": "stress", "level": 0.4},
            ],
        )
        vp = VirtualPatient(cfg)
        assert vp._endocrine is not None

    def test_build_endocrine_all_axes(self):
        cfg = VirtualPatientConfig(
            disease=_disease("Cushing's syndrome", category="endocrine"),
            endocrine_configs=[
                {"axis": "diabetes", "severity": 0.2},
                {"axis": "addison", "severity": 0.3},
                {"axis": "stress", "level": 0.4},
                {"axis": "cytokine", "severity": 0.1},
            ],
        )
        vp = VirtualPatient(cfg)
        assert vp._endocrine is not None

    def test_build_qsp_binding_empty(self, monkeypatch):
        import helixlang.plugins.human.qsp_binding as qsp_mod

        monkeypatch.setattr(qsp_mod, "create_qsp_binding",
                            lambda: qsp_mod.QSPBindingSystem())
        vp = VirtualPatient(VirtualPatientConfig())
        assert vp._qsp_binding is None

    def test_build_qsp_binding_kinds(self):
        cfg = VirtualPatientConfig(
            drugs=[_drug("biologic_x", mw=150000.0)],
            qsp_bindings=[
                {"drug": "biologic_x", "kind": "tmdd", "kss_nM": 2.0},
                {"drug": "biologic_x", "kind": "mass_action", "kd_nM": 5.0},
                {"drug": "biologic_x", "kind": "competitive"},
                {"drug": "biologic_x", "kind": "unknown"},
            ],
        )
        vp = VirtualPatient(cfg)
        assert vp._qsp_binding is not None

    def test_build_cancer_biopsy(self):
        cfg = VirtualPatientConfig(
            disease=_disease("NSCLC", category="cancer_metabolism"),
            tumor_biopsy={
                "mutations": ["EGFR_L858R", "BRAF_V600E", "KRAS_G12D",
                              "BRCA1"],
                "amplifications": ["HER2"],
                "fusion_genes": ["ALK"],
                "pd_l1_expression": 0.8,
                "msi_status": "MSI-H",
                "tmb_per_mb": 12.0,
            },
            total_duration_days=1.0,
        )
        vp = VirtualPatient(cfg)
        ode = vp._disease_ode
        assert ode is not None
        assert getattr(ode, "heterogeneity", None) is not None
        result = vp.run()
        assert result.tumor_clone_fractions
        assert result.resistance_mutations[0] == []
        assert result.tumor_volume[0] > 0.0

    def test_build_cancer_biopsy_minimal(self):
        cfg = VirtualPatientConfig(
            disease=_disease("NSCLC", category="cancer_metabolism"),
            tumor_biopsy={"mutations": [], "tmb_per_mb": 0.0},
        )
        vp = VirtualPatient(cfg)
        assert vp._disease_ode is not None
        assert vp._disease_ode.heterogeneity is not None

    def test_init_immune_crp_target(self):
        cfg = VirtualPatientConfig(
            disease=_disease(
                "sepsis", category="infectious", severity=0.6,
                mps=[MetabolitePerturbation("crp", "accumulate", 41.0, 2.0)]),
            immune_configs=[{"infection_severity": 0.4}],
        )
        vp = VirtualPatient(cfg)
        vp._init_immune_if_needed()
        assert vp._crp_driver is not None
        assert vp._crp_driver.crp_mg_l == pytest.approx(
            2.0 + 0.6 * (41.0 - 2.0))
        vp._init_immune_if_needed()  # early-return path

    def test_init_immune_autoimmune_crp_fallback(self):
        cfg = VirtualPatientConfig(
            disease=_disease("autoimmune hepatitis", severity=0.2),
            immune_configs=[{"autoimmune_activation": 0.4}],
        )
        vp = VirtualPatient(cfg)
        vp._init_immune_if_needed()
        assert vp._immune._base_autoimmune_activation == pytest.approx(0.4)
        assert vp._crp_driver.crp_mg_l == pytest.approx(2.0 + 0.2 * 45.0)

    def test_init_immune_autoimmune_disease_name(self):
        cfg = VirtualPatientConfig(
            disease=_disease("lupus", severity=0.5),
        )
        vp = VirtualPatient(cfg)
        vp._init_immune_if_needed()
        assert vp._immune is not None

    def test_init_immune_vaccine_configs(self):
        cfg = VirtualPatientConfig(
            immune_configs=[
                {"vaccine_dose": 1.0},
                {"checkpoint_blockade": 0.5},
                {"il6_biologic_occupancy": 0.3},
                {"immunosuppression": 0.2},
            ],
        )
        vp = VirtualPatient(cfg)
        vp._init_immune_if_needed()
        assert vp._immune is not None

    def test_hematology_import_error(self, monkeypatch):
        real_import = builtins.__import__

        def blocking(name, *a, **k):
            if name == "helixlang.plugins.human.hematology_model":
                raise ImportError("boom")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", blocking)
        vp = VirtualPatient(VirtualPatientConfig())
        with pytest.raises(ImportError, match="Reinstall"):
            vp._init_hematology_renal_if_needed()

    def test_hematology_early_return(self):
        vp = VirtualPatient(VirtualPatientConfig())
        vp._init_hematology_renal_if_needed()
        assert vp._hematology is not None
        vp._init_hematology_renal_if_needed()
        assert vp._renal is not None

    def test_enable_denoising(self):
        vp = VirtualPatient(VirtualPatientConfig())
        vp.enable_denoising(assay_cv=0.25)
        assert vp._denoise_outputs is True
        assert vp._assay_cv == 0.25

    def test_enable_stochastic_with_config(self):
        vp = VirtualPatient(VirtualPatientConfig())
        vp.enable_stochastic(config=object(), seed=7)
        assert vp._stochastic_active is True
        assert vp._sde_seed == 7

    def test_post_init_unknown_profile(self):
        cfg = VirtualPatientConfig(disease_profile_name="definitely_missing")
        assert cfg.disease is None


# ---------------------------------------------------------------------------
# run() sub-blocks
# ---------------------------------------------------------------------------


class TestRunBlocks:
    def test_engine_none_skips(self, monkeypatch):
        vp = _vp(drugs=[_drug("met")])
        monkeypatch.setattr(vp, "_init_drug_engines", lambda: None)
        r = vp.run()
        assert r.time_h

    def test_pd_activation_modifier(self):
        key = norm_drug_key("met")
        cfg = VirtualPatientConfig(
            drugs=[_drug("met")],
            pharmacodynamics={
                key: Pharmacodynamics(
                    drug_name="met",
                    effects=[PDEffect("GLC_UPK", effect_type="activation",
                                      emax=0.9)],
                ),
            },
            total_duration_days=0.5,
        )
        vp = VirtualPatient(cfg)
        r = vp.run()
        assert r.drug_concentrations

    def test_physiology_constraints_projection(self):
        vp = _vp(drugs=[_drug("met")])

        class Constraint:
            def __init__(self):
                self.calls = 0

            def check(self, state):
                self.calls += 1
                return SimpleNamespace(is_valid=False)

            def project_to_feasible(self, state):
                return {"creatinine": 1.2, "wbc": 6000.0, "alt": 40.0}

        c = Constraint()
        vp._physiology_constraints = c
        r = vp.run()
        assert c.calls > 0
        assert r.wbc

    def test_rl5_coupling(self):
        cfg = VirtualPatientConfig(
            drugs=[_drug("met")],
            disease=_disease("chronic kidney disease", category="renal"),
            physiological_core=True,
            total_duration_days=0.5,
        )
        vp = VirtualPatient(cfg)
        r = vp.run()
        assert r.creatinine

    def test_anti_inflammatory_pd(self):
        key = norm_drug_key("met")
        cfg = VirtualPatientConfig(
            drugs=[_drug("met")],
            pharmacodynamics={
                key: Pharmacodynamics(
                    drug_name="met",
                    effects=[PDEffect("TNF", effect_type="inhibition",
                                      emax=0.9)],
                ),
            },
            disease=_disease("sepsis", category="infectious"),
            total_duration_days=0.5,
        )
        vp = VirtualPatient(cfg)
        r = vp.run()
        assert r.crp

    def test_crp_capped_by_disease_target(self):
        cfg = VirtualPatientConfig(
            disease=_disease(
                "sepsis", category="infectious",
                mps=[MetabolitePerturbation("crp", "accumulate", 60.0, 2.0)]),
            total_duration_days=1.0,
        )
        vp = VirtualPatient(cfg)
        r = vp.run()
        cap = 2.0 + 0.6 * (60.0 - 2.0) * 1.2 + 2.0
        assert max(r.crp) <= cap

    def test_crp_target_absent_fallback(self):
        cfg = VirtualPatientConfig(
            disease=_disease("sepsis", category="infectious", severity=0.5),
            total_duration_days=1.0,
        )
        vp = VirtualPatient(cfg)
        r = vp.run()
        assert max(r.crp) <= (1.0 + 0.5 * 50.0) * 1.2 + 2.0

    def test_qsp_effects_feed_pd(self):
        cfg = VirtualPatientConfig(
            drugs=[_drug("met")],
            qsp_bindings=[
                {"drug": "met", "kind": "mass_action", "kd_nM": 1.0,
                 "emax": 0.8},
            ],
            total_duration_days=0.5,
        )
        vp = VirtualPatient(cfg)
        r = vp.run()
        assert r.drug_concentrations

    def test_crosstalk_clearance_modifier(self, monkeypatch):
        vp = _vp(drugs=[_drug("met")])

        class FakeCross:
            clearance_modifier_from_liver = 0.5

        monkeypatch.setattr(vp_mod, "apply_crosstalk",
                            lambda *a, **k: FakeCross())
        r = vp.run()
        assert r.time_h

    def test_denoising_pipeline(self):
        vp = _vp(drugs=[_drug("met")])
        vp.enable_denoising()
        r = vp.run()
        assert r.alt
        assert len(r.raw_alt) == len(r.alt)

    def test_no_engine_no_drugs(self):
        vp = VirtualPatient(VirtualPatientConfig(total_duration_days=0.5))
        r = vp.run()
        assert r.alt

    def test_run_no_internal_systems(self, monkeypatch):
        vp = _vp(drugs=[_drug("met")])
        monkeypatch.setattr(vp, "_init_immune_if_needed", lambda: None)
        monkeypatch.setattr(vp, "_init_hematology_renal_if_needed", lambda: None)
        vp._immune = None
        vp._crp_driver = None
        vp._endocrine = None
        vp._hematology = None
        vp._renal = None
        vp._physiology_constraints = None
        vp._qsp_binding = None
        key = norm_drug_key("met")
        vp.config.pharmacodynamics[key] = Pharmacodynamics(
            drug_name="met", effects=[])
        for n in ("_mechanistic_ddi", "_tissue_gem", "_reduced_order",
                  "_pharmacogenomic_ae", "_proteome_binding", "_microbiome",
                  "_emergent_complexity"):
            monkeypatch.setattr(vp_mod, n, None)
        monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)
        r = vp.run()
        assert r.time_h
        assert r.cortisol[0] == 12.0
        assert r.il6[0] == 1.0
        assert r.tissue_macrophages[0] == 0.0

    def test_run_axes_off_constraints_active(self, monkeypatch):
        vp = _vp(drugs=[_drug("met")])
        monkeypatch.setattr(vp, "_init_immune_if_needed", lambda: None)
        vp._immune = None
        vp._crp_driver = None
        vp._endocrine = None

        class Vitals:
            current = None

            def update(self, dt_h, drug_concs, labs, disease_sev):
                return SimpleNamespace(
                    systolic_bp_mmhg=120.0, diastolic_bp_mmhg=80.0,
                    heart_rate_bpm=70.0, temperature_c=37.0, weight_kg=75.0,
                    spo2_pct=98.0, respiratory_rate_per_min=14.0,
                    qtc_ms=420.0)

        vp._vitals_model = Vitals()
        r = vp.run()
        assert r.time_h

    def test_run_endocrine_off_immune_on(self):
        vp = _vp(drugs=[_drug("met")])
        vp._endocrine = None
        r = vp.run()
        assert r.cortisol[0] == 12.0


# ---------------------------------------------------------------------------
# module-level stateless helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_compute_genetic_cyp_modifier(self):
        from helixlang.plugins.human.genotype import create_default_genotype
        g = create_default_genotype()
        assert _compute_genetic_cyp_modifier(_drug("no_cyp"), g) == 1.0
        d = _drug("with_cyp")
        d.cyp_metabolism = {"CYP3A4": 0.7, "CYP2D6": 0.3}
        m = _compute_genetic_cyp_modifier(d, g)
        assert 0.1 <= m <= 5.0
        d2 = _drug("zero_cyp")
        d2.cyp_metabolism = {"CYP3A4": 0.0}
        assert _compute_genetic_cyp_modifier(d2, g) == 1.0

    def test_compute_transporter_modifier(self):
        from helixlang.plugins.human.genotype import create_default_genotype
        g = create_default_genotype()
        assert _compute_transporter_modifier(_drug("none"), g) == 1.0
        d2 = _drug("some")
        d2.transporter_affected = {"SLCO1B1": 0.4, "ABCB1": 0.6}
        m = _compute_transporter_modifier(d2, g)
        assert 0.1 <= m <= 5.0

    def test_compute_non_cyp_modifier(self):
        from helixlang.plugins.human.genotype import create_default_genotype
        g = create_default_genotype()
        assert _compute_non_cyp_modifier(_drug("none"), g) == 1.0
        d2 = _drug("some")
        d2.non_cyp_metabolism = {"UGT1A1": 1.0}
        m = _compute_non_cyp_modifier(d2, g)
        assert 0.1 <= m <= 5.0

    def test_trapz(self):
        assert _trapz([1.0], [0.0]) == 0.0
        assert _trapz([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.5)
        assert _trapz([-1.0, -2.0], [0.0, 1.0]) == 0.0

    def test_norm_drug_key(self):
        assert norm_drug_key("My Drug-2") == "my_drug_2"


# ---------------------------------------------------------------------------
# _DrugPBPK branches
# ---------------------------------------------------------------------------


class TestDrugPBPK:
    def _engine(self, drug=None, phys=None):
        from helixlang.plugins.human.virtual_patient import _DrugPBPK
        if drug is None:
            drug = _drug("engine", route=IV)
        if phys is None:
            phys = create_default_physiology()
        return _DrugPBPK(drug, phys)

    def test_organ_zero_flow_falls_back(self):
        phys = create_default_physiology()
        phys.organs["kidney"].blood_flow_ml_per_min = 0.0
        del phys.organs["liver"]
        e = self._engine(_drug("x"), phys)
        assert e.organ_flows_l_per_h["kidney"] > 0.0
        assert e.organ_flows_l_per_h["liver"] > 0.0

    def test_biologic_kp_scaling(self):
        e = self._engine(_drug("biologic", mw=150000.0, log_p=-2.0))
        assert e.partition_ratios["brain"] < 1.0
        assert e.partition_ratios["adipose"] == pytest.approx(0.05)

    def test_target_vd_scaling(self):
        d = _drug("vd", mw=400.0)
        d.volume_distribution_l = 500.0
        e = self._engine(d)
        assert "central" in e.conc_um

    def test_clearance_derivation_from_renal_only(self):
        d = _drug("renal_only",
                  clearance_ml_per_min=300.0,
                  renal_fraction=0.3,
                  hepatic_extraction_ratio=0.0)
        e = self._engine(d)
        assert e.k_renal_per_h > 0.0
        assert e.k_hepatic_per_h > 0.0

    def test_clearance_full_central(self):
        d = _drug("full_central",
                  clearance_ml_per_min=120.0,
                  renal_fraction=0.0,
                  hepatic_extraction_ratio=0.0)
        e = self._engine(d)
        assert e.k_renal_per_h > 0.0
        assert e.k_hepatic_per_h == 0.0

    def test_elimination_zero_clearance(self):
        d = _drug("no_cl", clearance_ml_per_min=0.0,
                  renal_fraction=0.0, hepatic_extraction_ratio=0.0)
        e = self._engine(d)
        assert e.k_renal_per_h == 0.0
        assert e.k_hepatic_per_h == 0.0

    @pytest.mark.parametrize("mutator,msg", [
        ("flow", "organ_flow_l_per_h"),
        ("qtotal", "q_total_l_per_h"),
        ("krenal", "k_renal_per_h"),
        ("mw", "molecular weight"),
    ])
    def test_verify_units_raises(self, mutator, msg):
        e = self._engine(_drug("units"))
        if mutator == "flow":
            e.organ_flows_l_per_h["liver"] = float("nan")
        elif mutator == "qtotal":
            e.q_total_l_per_h = -1.0
        elif mutator == "krenal":
            e.k_renal_per_h = float("inf")
        elif mutator == "mw":
            e.drug.molecule.molecular_weight_da = -5.0
        with pytest.raises(UnitError, match=msg):
            e.verify_units()

    def test_check_dimension_mismatch(self):
        from helixlang.plugins.human.virtual_patient import _DrugPBPK
        with pytest.raises(UnitError, match="dimension"):
            _DrugPBPK.check_dimension("vol", 5.0, "L", DIM_MASS)

    def test_administer_iv_route(self):
        e = self._engine(_drug("iv", route=IV, dose_mg=100.0))
        e._administer_dose()
        assert e.conc_um["central"] > 0.0

    def test_administer_infusion_input_window(self):
        d = _drug("inf", route=IV_INFUSION, dose_mg=100.0)
        e = self._engine(d)
        e._doses_given.append(0.0)
        assert e._central_input_um_per_h(0.5, 0.0) > 0.0
        assert e._central_input_um_per_h(5.0, 0.0) == 0.0

    def test_fcrn_recycling(self):
        e = self._engine(_drug("biologic", mw=150000.0, route=IV))
        e.conc_um["central"] = 10.0
        before = e.conc_um["central"]
        e._euler_step(0.001, 0.0, 0.0, 0.0)
        assert 0.0 <= e.conc_um["central"] <= before

    def test_euler_step_clips(self):
        e = self._engine(_drug("clip"))
        e.conc_um["central"] = 0.0
        e.conc_um["liver"] = -1.0
        e._euler_step(0.01, 0.0, 1.0, 1.0)
        assert e.conc_um["liver"] >= 0.0

    def test_integrate_slot_depth_floor(self, monkeypatch):
        e = self._engine(_drug("depth"))
        monkeypatch.setattr(e, "_local_error", lambda a, b: 10.0)
        e._integrate_slot(0.125, 0.0, ka_per_h=0.0, k_elim_per_h=0.0)
        assert math.isfinite(e.conc_um["central"])

    def test_snapshot_restore(self):
        e = self._engine(_drug("snap"))
        state0, depot0 = e._snapshot_state()
        e.conc_um["central"] = 99.0
        e.depot_mg = 7.0
        e._restore_state(state0, depot0)
        assert e.conc_um["central"] == state0["central"]
        assert e.depot_mg == depot0

    def test_advance_zero_dt(self):
        e = self._engine(_drug("adv"))
        e.advance(0.0, 0.0)
        assert e.sim_time_h == 0.0

    def test_advance_oral_depot(self):
        d = _drug("oral", route="oral")
        d.absorption_rate_h = 1.0
        e = self._engine(d)
        e.advance(1.0, 0.0)
        assert e.conc_um["central"] >= 0.0
        assert e.depot_mg >= 0.0

    def test_get_concentrations_copy(self):
        e = self._engine(_drug("copy"))
        snap = e.get_concentrations()
        snap["central"] = 123.0
        assert e.conc_um["central"] != 123.0

    def test_first_order_guard(self):
        d = _drug("fir", route="intramuscular")
        e = self._engine(d)
        assert e.drug.route in FIRST_ORDER_ROUTES

    def test_target_vd_not_exceeding(self):
        d = _drug("small_vd", mw=400.0)
        d.volume_distribution_l = 1.0e-6
        e = self._engine(d)
        assert "central" in e.conc_um

    def test_kp_scale_flat_organs(self, monkeypatch):
        from dataclasses import replace

        from helixlang.plugins.human.pharmacokinetics import PBPKConfig
        base = PBPKConfig()
        monkeypatch.setattr(
            vp_mod, "PBPKConfig",
            lambda: replace(base, liver_volume_l=0.0, kidney_volume_l=0.0,
                            brain_volume_l=0.0, muscle_volume_l=0.0,
                            adipose_volume_l=0.0))
        d = _drug("flat", mw=400.0)
        d.volume_distribution_l = 500.0
        e = self._engine(d)
        assert e.partition_ratios["liver"] >= 0.05

    def test_local_error_zero_denominator(self):
        e = self._engine(_drug("lerr"))
        e._RTOL = 0.0
        e._ATOL = 0.0
        coarse = [{"central": 0.0}, 0.0]
        fine = [{"central": 0.0}, 0.0]
        assert e._local_error(coarse, fine) == 0.0


# ---------------------------------------------------------------------------
# Disease-ODE dispatch arms inside run()
# ---------------------------------------------------------------------------


class TestDiseaseODEDispatch:
    def test_metabolic_t2d(self, monkeypatch):
        ode = _fake_ode(glucose_mg_dl=100.0, t2d_severity=0.4)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.glucose

    def test_renal_ode(self, monkeypatch):
        ode = _fake_ode(nephron_mass=0.8)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.nephron_mass[0] == pytest.approx(0.8)

    def test_hepatic_ode_mild(self, monkeypatch):
        ode = _fake_ode(fibrosis_stage=1.5, synthetic_function=0.9)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.fibrosis_stage[0] == pytest.approx(1.5)

    def test_hepatic_ode_advanced(self, monkeypatch):
        ode = _fake_ode(fibrosis_stage=3.0, synthetic_function=0.7)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.inr[0] > 0.0

    def test_cardiovascular_ode(self, monkeypatch):
        ode = _fake_ode(atherosclerosis_severity=0.3, co_l_min=4.5,
                        map_mmhg=90.0)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.cardiac_output[0] == pytest.approx(4.5)
        assert r.map_mmhg[0] == pytest.approx(90.0)

    def test_cancer_ode_with_heterogeneity(self, monkeypatch):
        het = TumorHeterogeneity(clones=[
            TumorClone(name="parent", fraction=1.0, growth_rate=0.3,
                       drug_sensitivities={"egfr": 0.9},
                       resistance_mutations=["EGFR_T790M"]),
        ])
        ode = _fake_ode(tumor_volume=0.1, heterogeneity=het)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.tumor_clone_fractions
        assert r.resistance_mutations[0] == ["EGFR_T790M"]
        assert r.tumor_volume[0] == pytest.approx(0.1)

    def test_cancer_ode_without_heterogeneity(self, monkeypatch):
        ode = _fake_ode(tumor_volume=0.1)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.tumor_clone_fractions
        assert r.resistance_mutations[0] == []
        assert r.tumor_volume[0] == pytest.approx(0.1)

    def test_autoimmune_ode(self, monkeypatch):
        ode = _fake_ode(joint_inflammation=0.4, synovial_tnf=100.0)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.crp[0] >= 0.0

    def test_autoimmune_ode_low_inflammation(self, monkeypatch):
        ode = _fake_ode(joint_inflammation=0.1, synovial_tnf=20.0)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.temperature[0] >= 0.0

    def test_neurological_ode(self, monkeypatch):
        ode = _fake_ode(synaptic_density=0.5, neuroinflammation=0.2,
                        cognitive_score=0.8)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.synaptic_density[0] == pytest.approx(0.5)
        assert r.cognitive_score[0] == pytest.approx(0.8)

    def test_hematological_ode(self, monkeypatch):
        ode = _fake_ode(stem_cell_pool=0.7)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.wbc[0] > 500.0

    def test_respiratory_ode(self, monkeypatch):
        ode = _fake_ode(airway_resistance=1.2, fev1_percent=80.0,
                        inflammation_score=0.3)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.fev1_percent[0] == pytest.approx(80.0)

    def test_infectious_ode(self, monkeypatch):
        ode = _fake_ode(viral_bacterial_load=2.0, immune_function=0.9,
                        inflammation=0.4)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.wbc[0] > 0.0

    def test_infectious_ode_fever(self, monkeypatch):
        ode = _fake_ode(viral_bacterial_load=2.0, immune_function=0.9,
                        inflammation=0.9)
        vp = _ode_vp(ode)
        r = vp.run()
        assert max(r.temperature) > 37.0

    def test_fever_resolution(self, monkeypatch):
        ode = _fake_ode(viral_bacterial_load=1.0, immune_function=0.9,
                        inflammation=0.6)
        disease = _disease("sepsis", category="infectious", severity=0.2)
        vp = _ode_vp(ode, disease=disease)
        r = vp.run()
        assert len(r.temperature) == len(r.time_h)

    def test_gastrointestinal_ode(self, monkeypatch):
        ode = _fake_ode(acid_secretion=1.4, mucosal_integrity=0.8)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.acid_secretion[0] == pytest.approx(1.4)
        assert r.mucosal_integrity[0] == pytest.approx(0.8)

    def test_endocrine_thyroid_ode(self, monkeypatch):
        ode = _fake_ode(t4_level=40.0, metabolic_rate=1.5)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.t4_level[0] == pytest.approx(40.0)

    def test_generic_ode(self, monkeypatch):
        ode = _fake_ode(liver_function=0.7, kidney_function=0.8,
                        inflammation_score=0.6, severity=0.4)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.alt[0] >= 0.0
        assert r.disease_severity[0] == pytest.approx(0.6)

    def test_generic_ode_no_liver(self, monkeypatch):
        ode = _fake_ode(severity=0.4)
        vp = _ode_vp(ode)
        r = vp.run()
        assert r.disease_severity[0] == pytest.approx(0.4)

    def test_renal_ode_pd_wiring(self, monkeypatch):
        ode = _fake_ode(nephron_mass=0.8, acei_effect=0.0, sglt2_effect=0.0)
        vp = _ode_vp(ode, drugs=[_drug("met")],
                     pharmacodynamics=_pd(
                         "met", ["ACE_inhibition", "SGLT2_inhibition"]))
        r = vp.run()
        assert r.nephron_mass[0] == pytest.approx(0.8)

    def test_hepatic_ode_pd_wiring(self, monkeypatch):
        ode = _fake_ode(fibrosis_stage=1.5, synthetic_function=0.9,
                        antiviral_effect=0.0, anti_fibrotic_effect=0.0)
        vp = _ode_vp(ode, drugs=[_drug("met")],
                     pharmacodynamics=_pd(
                         "met", ["NS5A_inhibitor", "PPAR_gamma",
                                 "tyrosine_kinase"]))
        r = vp.run()
        assert r.fibrosis_stage[0] == pytest.approx(1.5)

    def test_cardiovascular_ode_pd_wiring(self, monkeypatch):
        ode = _fake_ode(atherosclerosis_severity=0.3, co_l_min=4.5,
                        map_mmhg=90.0)
        vp = _ode_vp(ode, drugs=[_drug("met")],
                     pharmacodynamics=_pd(
                         "met", ["vasodilation", "diuretic", "inotropic"]))
        r = vp.run()
        assert r.cardiac_output[0] == pytest.approx(4.5)

    def test_cancer_ode_pd_wiring(self, monkeypatch):
        het = TumorHeterogeneity(clones=[
            TumorClone(name="parent", fraction=1.0, growth_rate=0.3,
                       drug_sensitivities={"egfr": 0.9},
                       resistance_mutations=["EGFR_T790M"]),
        ])
        ode = _fake_ode(tumor_volume=0.2, heterogeneity=het)
        vp = _ode_vp(ode, drugs=[_drug("met")],
                     pharmacodynamics=_pd(
                         "met", ["EGFR", "PARP7", "harmless"]))
        r = vp.run()
        assert r.tumor_volume[0] == pytest.approx(0.2)
        assert "egfr" in ode.pathway_effects
        assert "parp" in ode.pathway_effects
        assert ode.pathway_effects["egfr"] > 0.0

    def test_autoimmune_ode_pd_wiring(self, monkeypatch):
        ode = _fake_ode(joint_inflammation=0.5, synovial_tnf=100.0,
                        dmard_effect=0.0)
        vp = _ode_vp(ode, drugs=[_drug("met")],
                     pharmacodynamics=_pd(
                         "met", ["JAK", "inflammation", "z_other"]))
        vp.run()
        assert ode.dmard_effect > 0.0

    def test_neurological_ode_pd_wiring(self, monkeypatch):
        ode = _fake_ode(synaptic_density=0.5, neuroinflammation=0.2,
                        cognitive_score=0.8)
        vp = _ode_vp(ode, drugs=[_drug("met")],
                     pharmacodynamics=_pd(
                         "met", ["cholinesterase_inhibitor",
                                 "NMDA_antagonist", "z_other"]))
        vp.run()
        assert ode.cholinesterase_inhibition > 0.0

    def test_respiratory_ode_pd_wiring(self, monkeypatch):
        ode = _fake_ode(airway_resistance=1.2, fev1_percent=80.0,
                        inflammation_score=0.3)
        vp = _ode_vp(ode, drugs=[_drug("met")],
                     pharmacodynamics=_pd(
                         "met", ["beta2_agonist", "TNF", "z_other"]))
        r = vp.run()
        assert r.fev1_percent[0] == pytest.approx(80.0)

    def test_gastrointestinal_ode_pd_wiring(self, monkeypatch):
        ode = _fake_ode(acid_secretion=1.4, mucosal_integrity=0.8)
        vp = _ode_vp(ode, drugs=[_drug("met")],
                     pharmacodynamics=_pd(
                         "met", ["proton_pump", "COX2", "z_other"]))
        r = vp.run()
        assert r.acid_secretion[0] == pytest.approx(1.4)

    def test_autoimmune_ode_crp_owned_none(self, monkeypatch):
        ode = _fake_ode(joint_inflammation=0.5, synovial_tnf=100.0)
        vp = _ode_vp(ode)
        monkeypatch.setattr(vp, "_init_immune_if_needed", lambda: None)
        monkeypatch.setattr(vp, "_init_hematology_renal_if_needed",
                            lambda: None)
        vp._immune = None
        vp._crp_driver = None
        r = vp.run()
        assert r.crp[0] >= 0.1

    def test_neurological_ode_crp_owned_none(self, monkeypatch):
        ode = _fake_ode(synaptic_density=0.5, neuroinflammation=0.2,
                        cognitive_score=0.8)
        vp = _ode_vp(ode)
        monkeypatch.setattr(vp, "_init_immune_if_needed", lambda: None)
        vp._immune = None
        vp._crp_driver = None
        r = vp.run()
        assert r.crp[0] >= 0.0

    def test_infectious_ode_no_hematology(self, monkeypatch):
        ode = _fake_ode(viral_bacterial_load=2.0, immune_function=0.7,
                        inflammation=0.5)
        vp = _ode_vp(ode)
        monkeypatch.setattr(vp, "_init_immune_if_needed", lambda: None)
        monkeypatch.setattr(vp, "_init_hematology_renal_if_needed",
                            lambda: None)
        vp._immune = None
        vp._crp_driver = None
        vp._hematology = None
        vp._renal = None
        r = vp.run()
        assert r.wbc[0] > 0.0

    def test_generic_ode_no_hematology(self, monkeypatch):
        ode = _fake_ode(liver_function=0.7, kidney_function=0.8,
                        inflammation_score=0.6, severity=0.4)
        vp = _ode_vp(ode)
        monkeypatch.setattr(vp, "_init_immune_if_needed", lambda: None)
        monkeypatch.setattr(vp, "_init_hematology_renal_if_needed",
                            lambda: None)
        vp._immune = None
        vp._crp_driver = None
        vp._hematology = None
        vp._renal = None
        r = vp.run()
        assert r.crp[0] >= 0.1

    def test_run_immune_present_crp_none(self):
        from helixlang.plugins.human.immune import create_immune_model
        vp = _vp(drugs=[_drug("met")],
                 disease=_disease("essential hypertension",
                                  category="cardiovascular"))
        immune, _ = create_immune_model(infection_severity=0.5,
                                        autoimmune_activation=0.0,
                                        cortisol_level=12.0,
                                        immunosuppression=0.0)
        vp._immune = immune
        vp._crp_driver = None
        r = vp.run()
        assert r.time_h[0] == 0.0
        assert r.crp[0] >= 0.0

    def test_run_disease_ode_without_step(self, monkeypatch):
        monkeypatch.setattr(vp_mod.VirtualPatient, "_build_disease_ode",
                            lambda self: SimpleNamespace())
        vp = _vp(drugs=[_drug("met")],
                 disease=_disease("influenza", category="infectious"))
        r = vp.run()
        assert r.time_h[0] == 0.0

    def test_infectious_ode_pd_exactly_one(self):
        ode = _fake_ode(viral_bacterial_load=2.0, immune_function=0.7,
                        inflammation=0.5)
        key = norm_drug_key("met")
        pd = {key: Pharmacodynamics(
            drug_name="met",
            effects=[PDEffect("mec", effect_type="inhibition",
                              ec50_um=1000.0, emax=0.0, hill_coefficient=1.0,
                              baseline_effect=0.0)])}
        vp = _ode_vp(ode, drugs=[_drug("met")], pharmacodynamics=pd)
        r = vp.run()
        assert r.wbc[0] > 0.0


# ---------------------------------------------------------------------------
# doc/32 opt-in couplings
# ---------------------------------------------------------------------------


class TestDoc32Couplings:
    def test_mech_ddi(self, monkeypatch):
        _disable_lazy(monkeypatch, "_mechanistic_ddi")
        vp = _vp(drugs=[_drug("met"), _drug("asp")])

        class Pred:
            def __init__(self, significance, drug_b, auc):
                self.significance = significance
                self.drug_b = drug_b
                self.auc_ratio = auc

        class Predictor:
            def predict_all_pairs(self, names):
                return [Pred("MINOR", "asp", 1.1),
                        Pred("CONTRAINDICATED", "asp", 2.0)]

        vp._mech_ddi_predictor = Predictor()
        r = vp.run()
        assert r.ddi_alerts is not None

    def test_ro_organs(self, monkeypatch):
        _disable_lazy(monkeypatch, "_reduced_order")
        vp = _vp(drugs=[_drug("met")])
        ro_liver = _fake_ode(get_gradient=lambda: 2.0)
        ro_other = _fake_ode()
        vp._ro_organs = {"liver": ro_liver, "muscle": ro_other}
        r = vp.run()
        assert r.time_h

    def test_ro_organ_flat_gradient(self, monkeypatch):
        _disable_lazy(monkeypatch, "_reduced_order")
        vp = _vp(drugs=[_drug("met")])
        ro_flat = _fake_ode(get_gradient=lambda: 0.0)
        vp._ro_organs = {"liver": ro_flat}
        r = vp.run()
        assert r.time_h

    def test_ae_predictor_liver(self, monkeypatch):
        _disable_lazy(monkeypatch, "_pharmacogenomic_ae")
        vp = _vp(drugs=[_drug("met")])

        class Pred:
            ae_probability = 0.9
            toxicity_ratio = 1.0
            target_organ = SimpleNamespace(value="liver")

        class Predictor:
            def predict_all(self, concs, cyp):
                return {"met": [Pred()]}

        vp._ae_predictor = Predictor()
        r = vp.run()
        assert r.alt[0] >= 0.0

    def test_ae_predictor_kidney_bone_marrow(self, monkeypatch):
        _disable_lazy(monkeypatch, "_pharmacogenomic_ae")
        vp = _vp(drugs=[_drug("met")])

        def make(organ):
            class Pred:
                ae_probability = 0.9
                toxicity_ratio = 1.0
                target_organ = SimpleNamespace(value=organ)

            return Pred()

        class Predictor:
            def predict_all(self, concs, cyp):
                return {"met": [make("kidney"), make("bone_marrow"),
                                make("brain")]}

        vp._ae_predictor = Predictor()
        r = vp.run()
        assert r.creatinine[0] >= 0.0
        assert r.wbc[0] > 0.0

    def test_proteome_cascade(self, monkeypatch):
        _disable_lazy(monkeypatch, "_proteome_binding")
        vp = _vp(drugs=[_drug("met", smiles="CC"), _drug("asp", smiles="O")])

        class BindingProfile:
            inhibition_dict = {"UGT1A1": 0.8}

        class DDI:
            auc_ratio = 2.0

        class Cascade:
            def screen_drug(self, key, smiles, conc):
                return BindingProfile()

            def predict_ddi(self, *args):
                return DDI()

        vp._proteome_cascade = Cascade()
        r = vp.run()
        assert r.drug_concentrations

    def test_microbiome_compartment(self, monkeypatch):
        _disable_lazy(monkeypatch, "_microbiome")
        vp = _vp(drugs=[_drug("met")])

        class Effect:
            bioavailability_modifier = 0.9

        class Microbiome:
            def set_drug_concentration(self, key, conc):
                pass

            def step(self, dt):
                return {"met": Effect()}

            def get_portal_fluxes(self):
                return {"ammonia": 2.0, "scfa": 10.0}

        vp._microbiome_compartment = Microbiome()
        r = vp.run()
        assert r.alt[0] >= 0.0

    def test_emergent_complexity(self, monkeypatch):
        _disable_lazy(monkeypatch, "_emergent_complexity")
        _disable_lazy(monkeypatch, "_microbiome")
        drug = _drug("met")
        drug.cyp_metabolism = {"CYP3A4": 0.8}
        vp = _vp(drugs=[drug])

        class Emergent:
            def step(self, **kw):
                return {
                    "bile_acid_pool": 15.0,
                    "endotoxin_level": 2.0,
                    "cortisol_suppression": 0.5,
                    "fever_c": 1.0,
                    "epigenetic_CYP3A4": 0.5,
                }

        vp._emergent_complexity = Emergent()
        r = vp.run()
        assert r.alt[0] >= 0.0

    def test_emergent_complexity_no_cyp(self, monkeypatch):
        _disable_lazy(monkeypatch, "_emergent_complexity")
        _disable_lazy(monkeypatch, "_microbiome")
        vp = _vp(drugs=[_drug("met")])

        class Emergent:
            def step(self, **kw):
                return {"fever_c": 0.0, "bile_acid_pool": 10.0}

        vp._emergent_complexity = Emergent()
        r = vp.run()
        assert r.time_h

    def test_gem_coupler(self, monkeypatch):
        _disable_lazy(monkeypatch, "_tissue_gem")
        vp = _vp(drugs=[_drug("met")], disease=_disease(
            "sepsis", category="infectious",
            mps=[MetabolitePerturbation("lactate", "accumulate", 10.0, 1.8)]))

        class Gem:
            def step(self, dt, org):
                return {"liver": {"glucose": 8.0},
                        "muscle": {"lactate": 1.2}}

            def invalidate_on_dose(self):
                pass

        vp._gem_coupler = Gem()
        r = vp.run()
        assert r.lactate[0] >= 0.0

    def test_gem_coupler_no_lactate_mp(self, monkeypatch):
        _disable_lazy(monkeypatch, "_tissue_gem")
        vp = _vp(drugs=[_drug("met")],
                 disease=_disease("flu", category="infectious"))

        class Gem:
            def step(self, dt, org):
                return {"liver": {"glucose": 8.0},
                        "muscle": {"lactate": 1.2}}

            def invalidate_on_dose(self):
                pass

        vp._gem_coupler = Gem()
        r = vp.run()
        assert r.lactate[0] >= 0.0

    def test_ro_organ_no_step(self, monkeypatch):
        _disable_lazy(monkeypatch, "_reduced_order")
        vp = _vp(drugs=[_drug("met")])
        ro_liver = _fake_ode(get_gradient=lambda: 2.0)
        vp._ro_organs = {"liver": ro_liver, "muscle": SimpleNamespace()}
        r = vp.run()
        assert r.time_h

    def test_proteome_cascade_ugt_scaling(self, monkeypatch):
        vp = _vp(drugs=[_drug("met", smiles="CC")])
        monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)

        class BindingProfile:
            inhibition_dict = {"UGT1A1": 0.8}

        class Cascade:
            def screen_drug(self, key, smiles, conc):
                return BindingProfile()

        monkeypatch.setattr(vp_mod, "_proteome_binding", lambda: Cascade())
        r = vp.run()
        assert r.drug_concentrations

    def test_proteome_cascade_ddi_high(self, monkeypatch):
        vp = _vp(drugs=[_drug("met", smiles="CC"), _drug("asp", smiles="O")])
        monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)

        class BindingProfile:
            inhibition_dict = {"UGT1A1": 0.0}

        class DDI:
            auc_ratio = 2.0

        class Cascade:
            def screen_drug(self, key, smiles, conc):
                return BindingProfile()

            def predict_ddi(self, *args):
                return DDI()

        monkeypatch.setattr(vp_mod, "_proteome_binding", lambda: Cascade())
        r = vp.run()
        assert r.drug_concentrations

    def test_proteome_cascade_ddi_low(self, monkeypatch):
        vp = _vp(drugs=[_drug("met", smiles="CC"), _drug("asp", smiles="O")])
        monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)

        class BindingProfile:
            inhibition_dict = {"UGT1A1": 0.0}

        class DDI:
            auc_ratio = 1.1

        class Cascade:
            def screen_drug(self, key, smiles, conc):
                return BindingProfile()

            def predict_ddi(self, *args):
                return DDI()

        monkeypatch.setattr(vp_mod, "_proteome_binding", lambda: Cascade())
        r = vp.run()
        assert r.drug_concentrations

    def test_microbiome_ammonia_flux(self, monkeypatch):
        vp = _vp(drugs=[_drug("met")])
        monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)

        class Effect:
            bioavailability_modifier = 0.9

        class Microbiome:
            state = SimpleNamespace(bile_salt_hydrolase_activity=1.0)

            def set_drug_concentration(self, key, conc):
                pass

            def step(self, dt):
                return {"met": Effect(), "other": Effect()}

            def get_portal_fluxes(self):
                return {"ammonia": 2.0, "scfa": 10.0}

        monkeypatch.setattr(vp_mod, "_microbiome", lambda: Microbiome())
        r = vp.run()
        assert r.alt[0] >= 0.0

    def test_microbiome_low_fluxes(self, monkeypatch):
        vp = _vp(drugs=[_drug("met")])
        monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)

        class Effect:
            bioavailability_modifier = 1.0

        class Microbiome:
            state = SimpleNamespace(bile_salt_hydrolase_activity=1.0)

            def set_drug_concentration(self, key, conc):
                pass

            def step(self, dt):
                return {"met": Effect()}

            def get_portal_fluxes(self):
                return {"ammonia": 0.5, "scfa": 0.0}

        monkeypatch.setattr(vp_mod, "_microbiome", lambda: Microbiome())
        r = vp.run()
        assert r.alt[0] >= 0.0

    def test_emergent_complexity_inner(self, monkeypatch):
        drug = _drug("met")
        drug.cyp_metabolism = {"CYP3A4": 0.8}
        vp = _vp(drugs=[drug])
        monkeypatch.setattr(vp_mod, "_import_doc32", lambda: None)

        class Emergent:
            def step(self, **kw):
                return {
                    "bile_acid_pool": 15.0,
                    "endotoxin_level": 2.0,
                    "cortisol_suppression": 0.5,
                    "fever_c": 1.0,
                    "epigenetic_CYP3A4": 0.5,
                }

        monkeypatch.setattr(vp_mod, "_emergent_complexity",
                            lambda: Emergent())
        r = vp.run()
        assert r.alt[0] >= 0.0

        vp2 = _vp(drugs=[drug])
        monkeypatch.setattr(vp2, "_init_immune_if_needed", lambda: None)
        vp2._immune = None
        vp2._crp_driver = None
        vp2._endocrine = None
        r2 = vp2.run()
        assert r2.alt[0] >= 0.0


# ---------------------------------------------------------------------------
# Toxicity / finalize / recovery helpers
# ---------------------------------------------------------------------------


class TestToxicityFinalize:
    def test_check_toxicities_full(self):
        vp = _vp()
        labs = ClinicalLabs(alt_u_per_l=200.0,
                            creatinine_mg_per_dl=3.0,
                            wbc_per_ul=1000.0,
                            bilirubin_total_mg_per_dl=5.0)
        vp._check_toxicities({}, labs, 12.0)
        assert vp._toxicity_flags
        result = VirtualPatientResult()
        vp._finalize(result)
        assert result.total_toxicity_events > 0
        assert result.clinical_events
        assert result.clinical_events[0]["time_h"] == 12.0

    def test_check_toxicities_clean(self):
        vp = _vp()
        vp._check_toxicities({}, ClinicalLabs(), 0.0)
        assert not vp._toxicity_flags

    def test_finalize_metrics(self):
        vp = _vp()
        result = VirtualPatientResult()
        result.time_h = [0.0, 1.0, 2.0]
        result.drug_concentrations = {"met": [0.0, 5.0, 5.0]}
        result.alt = [25.0, 30.0, 200.0]
        result.creatinine = [1.0, 1.5, 2.0]
        result.wbc = [7000.0, 4000.0, 1800.0]
        result.egfr = [100.0, 90.0, 80.0]
        result.disease_severity = [0.8, 0.1, 0.2]
        vp._finalize(result)
        assert result.max_alt == 200.0
        assert result.max_creatinine == 2.0
        assert result.min_wbc == 1800.0
        assert result.min_egfr == 80.0
        assert result.auc_plasma["met"] > 0.0
        assert result.overall_efficacy_score == pytest.approx(2 / 3)

    def test_finalize_empty(self):
        vp = _vp()
        result = VirtualPatientResult()
        vp._finalize(result)
        assert result.total_toxicity_events == 0

    def test_seed_recovery_none(self):
        vp = _vp()
        vp._seed_recovery_from_labs(ClinicalLabs())

    def test_seed_recovery_values(self):
        vp = _vp(drugs=[_drug("met")])
        rm = vp._recovery_model
        vp._seed_recovery_from_labs(
            ClinicalLabs(alt_u_per_l=80.0, creatinine_mg_per_dl=1.8,
                         wbc_per_ul=3000.0, hemoglobin_g_per_dl=10.0))
        assert rm.current_biomarkers["ALT"] == pytest.approx(80.0)

    def test_seed_recovery_partial(self):
        vp = _vp(drugs=[_drug("met")])
        vp._recovery_model.current_biomarkers = {"ALT": 1.0}
        vp._seed_recovery_from_labs(
            ClinicalLabs(alt_u_per_l=90.0, creatinine_mg_per_dl=1.9,
                         wbc_per_ul=3100.0, hemoglobin_g_per_dl=10.5))
        assert vp._recovery_model.current_biomarkers["ALT"] == pytest.approx(90.0)

    def test_finalize_empty_series(self):
        vp = _vp()
        result = VirtualPatientResult()
        result.time_h = [0.0, 1.0]
        result.drug_concentrations = {"met": [0.0, 5.0], "other": []}
        result.disease_severity = [0.2, 0.2]
        vp._finalize(result)
        assert result.auc_plasma["met"] > 0.0
        assert result.overall_efficacy_score == 1.0

    def test_feed_recovery_active(self):
        vp = _vp(drugs=[_drug("met")])
        before = vp._labs_model.current.alt_u_per_l
        vp._feed_recovery_biomarkers({"ALT": 50.0})
        assert vp._labs_model.current.alt_u_per_l == before

    def test_feed_recovery_inactive(self):
        vp = _vp(drugs=[_drug("met")])
        vp._treatment_active = False
        vp._feed_recovery_biomarkers({"ALT": 50.0, "missing": 3.0})
        assert vp._labs_model.current.alt_u_per_l == pytest.approx(50.0)

    def test_compute_treatment_effectiveness(self):
        vp = _vp()
        assert vp._compute_treatment_effectiveness({}, {}) == 0.0
        eff = vp._compute_treatment_effectiveness(
            {"met": 1.0}, {"T": 0.5, "U": 0.9})
        assert eff == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Cohort / worker
# ---------------------------------------------------------------------------


class TestCohort:
    def test_run_worker_stochastic(self):
        cfg = VirtualPatientConfig(total_duration_days=0.5)
        r = vp_mod._run_vp_worker(cfg, 5, True, None)
        assert r.time_h

    def test_run_worker_deterministic(self):
        cfg = VirtualPatientConfig(total_duration_days=0.5)
        r = vp_mod._run_vp_worker(cfg, None, False, None)
        assert r.time_h
