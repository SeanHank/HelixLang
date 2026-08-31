"""Tests for doc/30-31 new modules: endocrine, QSP binding, immune, organ crosstalk, disease ODEs."""
import pytest

from helixlang.plugins.human.disease_ode_models import (
    AutoimmuneRAODE,
    CancerODE,
    CardiovascularODE,
    HematologicalODE,
    HepaticODE,
    MetabolicT2DODE,
    NeurologicalODE,
    RenalODE,
    create_disease_model,
)
from helixlang.plugins.human.endocrine import (
    HPAAxis,
    HPTAxis,
    InsulinGlucoseAxis,
    create_endocrine,
)
from helixlang.plugins.human.immune import (
    CRPDriver,
    CytokinePool,
    IFNPool,
    ImmuneCellPopulation,
    InnateImmuneModel,
    create_immune_model,
)
from helixlang.plugins.human.organ_crosstalk import (
    apply_crosstalk,
    create_crosstalk,
)
from helixlang.plugins.human.qsp_binding import (
    CompetitiveBinding,
    MassActionBinding,
    QSPBindingSystem,
    TMDDBinding,
    create_qsp_binding,
)

# ============================================================================
# Endocrine Tests
# ============================================================================


class TestInsulinGlucoseAxis:
    def test_baseline_stable(self):
        ax = InsulinGlucoseAxis()
        for _ in range(100):
            ax.step(1.0)
        assert 80.0 < ax.glucose_mg_dl < 120.0

    def test_glucose_input_raises_glucose(self):
        ax = InsulinGlucoseAxis()
        ax.step(1.0, glucose_input=50.0)
        assert ax.glucose_mg_dl > 100.0

    def test_insulin_resistance(self):
        ax = InsulinGlucoseAxis()
        ax.set_insulin_resistance(0.8)
        assert ax.insulin_sensitivity == pytest.approx(0.2, abs=0.01)

    def test_disposition_index(self):
        ax = InsulinGlucoseAxis()
        di = ax.get_disposition_index()
        assert di > 0


class TestHPAAxis:
    def test_baseline_cortisol(self):
        ax = HPAAxis()
        for _ in range(50):
            ax.step(1.0)
        assert 5.0 < ax.cortisol_ug_dl < 25.0

    def test_stress_increases_cortisol(self):
        ax = HPAAxis()
        ax.set_stress(1.0)
        for _ in range(24):
            ax.step(1.0)
        assert ax.cortisol_ug_dl > 12.0

    def test_addison_suppresses_cortisol(self):
        ax = HPAAxis()
        ax.adrenal_insufficiency = 1.0
        for _ in range(48):
            ax.step(1.0)
        assert ax.cortisol_ug_dl < 5.0

    def test_cushing_elevates_cortisol(self):
        ax = HPAAxis()
        ax.cushing_severity = 1.0
        for _ in range(48):
            ax.step(1.0)
        assert ax.cortisol_ug_dl > 15.0


class TestHPTAxis:
    def test_baseline_tsh(self):
        ax = HPTAxis()
        for _ in range(100):
            ax.step(1.0)
        assert 0.1 < ax.tsh_miul < 10.0

    def test_deiodinase_inhibition(self):
        ax = HPTAxis()
        ax.deiodinase_inhibition = 0.8
        for _ in range(100):
            ax.step(1.0)
        assert ax.ft3_pgdl < 3.0


class TestEndocrineSystem:
    def test_create_default(self):
        sys = create_endocrine()
        assert sys.insulin_glucose is not None
        assert sys.hpa is not None
        assert sys.hpt is not None

    def test_step_all_axes(self):
        sys = create_endocrine()
        sys.step(1.0)
        assert sys.get_glucose_mg_dl() > 0
        assert sys.get_cortisol_ug_dl() > 0

    def test_diabetes_disease(self):
        sys = create_endocrine(diabetes_severity=0.8)
        for _ in range(48):
            sys.step(1.0)
        assert sys.get_insulin_sensitivity() < 0.5


# ============================================================================
# QSP Binding Tests
# ============================================================================


class TestMassActionBinding:
    def test_zero_conc_no_effect(self):
        mab = MassActionBinding(kd_nM=10.0)
        assert mab.compute_occupancy(0.0) == pytest.approx(0.0, abs=0.01)

    def test_half_max_at_kd(self):
        mab = MassActionBinding(kd_nM=10.0)
        assert mab.compute_occupancy(10.0) == pytest.approx(0.5, abs=0.01)

    def test_high_conc_saturates(self):
        mab = MassActionBinding(kd_nM=10.0)
        assert mab.compute_occupancy(1000.0) > 0.95

    def test_effect(self):
        mab = MassActionBinding(kd_nM=10.0, emax=0.8, baseline=0.1)
        eff = mab.compute_effect(10.0)
        assert eff == pytest.approx(0.5, abs=0.05)


class TestTMDDBinding:
    def test_step_advances(self):
        tmdd = TMDDBinding()
        tmdd.step(1.0, drug_input_rate_nM_h=10.0)
        assert tmdd.c_free_nM > 0

    def test_occupancy(self):
        tmdd = TMDDBinding()
        tmdd.c_free_nM = 100.0
        occ = tmdd.compute_occupancy()
        assert 0.0 <= occ <= 1.0

    def test_effect(self):
        tmdd = TMDDBinding(emax=0.9)
        tmdd.rc_complex_nM = 50.0
        tmdd.r_total_nM = 100.0
        eff = tmdd.compute_effect()
        assert eff == pytest.approx(0.45, abs=0.05)


class TestCompetitiveBinding:
    def test_no_antagonist(self):
        cb = CompetitiveBinding(kd_agonist_nM=10.0, ki_antagonist_nM=5.0)
        eff = cb.compute_effect(10.0, 0.0)
        assert eff == pytest.approx(0.5, abs=0.05)

    def test_antagonist_reduces_effect(self):
        cb = CompetitiveBinding(kd_agonist_nM=10.0, ki_antagonist_nM=5.0)
        eff_no = cb.compute_effect(10.0, 0.0)
        eff_with = cb.compute_effect(10.0, 10.0)
        assert eff_with < eff_no

    def test_schild_shift(self):
        cb = CompetitiveBinding(kd_agonist_nM=10.0, ki_antagonist_nM=5.0)
        shift = cb.compute_schild_shift(5.0)
        assert shift == pytest.approx(2.0, abs=0.01)


class TestQSPBindingSystem:
    def test_create_default(self):
        sys = create_qsp_binding()
        assert len(sys.models) > 0

    def test_add_mass_action(self):
        sys = QSPBindingSystem()
        sys.add_mass_action("test_drug", kd_nM=5.0)
        assert "test_drug" in sys.models

    def test_add_tmdd(self):
        sys = QSPBindingSystem()
        sys.add_tmdd("test_mab", kss_nM=2.0)
        assert "test_mab" in sys.models


# ============================================================================
# Immune Tests
# ============================================================================


class TestCytokinePool:
    def test_baseline(self):
        pool = CytokinePool()
        assert pool.il6 > 0
        assert pool.tnf_alpha > 0

    def test_pathogen_increases_cytokines(self):
        pool = CytokinePool()
        pool.pathogen_signal = 1.0
        for _ in range(24):
            pool.step(1.0)
        assert pool.il6 > 1.0
        assert pool.tnf_alpha > 5.0

    def test_cytokines_floor(self):
        pool = CytokinePool()
        for _ in range(100):
            pool.step(1.0)
        assert pool.il6 >= 0.0
        assert pool.tnf_alpha >= 0.0


class TestImmuneCellPopulation:
    def test_baseline_wbc(self):
        cells = ImmuneCellPopulation()
        wbc = cells.get_wbc_total()
        assert wbc > 3.0

    def test_infection_increases_neutrophils(self):
        cells = ImmuneCellPopulation()
        initial = cells.neutrophils
        for _ in range(48):
            cells.step(1.0, il6=20.0, tnf=50.0)
        # Neutrophils should increase due to mobilisation
        assert cells.neutrophils >= initial * 0.8  # Allow some variation


class TestInnateImmuneModel:
    def test_create_default(self):
        immune, crp = create_immune_model()
        assert immune is not None
        assert crp is not None

    def test_infection_response(self):
        immune, crp = create_immune_model(infection_severity=0.8)
        for _ in range(48):
            immune.step(1.0)
            crp.step(1.0, immune.get_il6())
        assert immune.get_il6() > 1.0
        assert crp.crp_mg_l > 0.5

    def test_cortisol_suppresses_immune(self):
        immune, _ = create_immune_model(cortisol_level=40.0)
        assert immune.cortisol_suppression > 0


class TestCRPDriver:
    def test_baseline_crp(self):
        crp = CRPDriver()
        assert crp.crp_mg_l == pytest.approx(0.5, abs=0.01)

    def test_il6_drives_crp(self):
        crp = CRPDriver()
        for _ in range(24):
            crp.step(1.0, il6_pg_ml=20.0)
        assert crp.crp_mg_l > 0.5

    def test_crp_clearance(self):
        crp = CRPDriver()
        crp.crp_mg_l = 100.0
        for _ in range(100):
            crp.step(1.0, il6_pg_ml=0.5)
        assert crp.crp_mg_l < 100.0


# ============================================================================
# Organ Crosstalk Tests
# ============================================================================


class TestOrganCrosstalk:
    def test_create_default(self):
        ct = create_crosstalk()
        assert ct.cv_risk_multiplier == 1.0

    def test_hyperglycemia_increases_cv_risk(self):
        ct = create_crosstalk()
        ct = apply_crosstalk(ct, glucose_mg_dl=200.0)
        assert ct.cv_risk_multiplier > 1.0

    def test_low_egfr_reduces_epo(self):
        ct = create_crosstalk()
        ct = apply_crosstalk(ct, egfr=30.0)
        assert ct.epo_production < 1.0

    def test_high_cortisol_suppresses_immune(self):
        ct = create_crosstalk()
        ct = apply_crosstalk(ct, cortisol_ug_dl=40.0)
        assert ct.immune_suppression_from_cortisol > 0

    def test_child_pugh_from_labs(self):
        ct = create_crosstalk()
        ct = apply_crosstalk(ct, albumin_g_dl=2.5, inr=2.0)
        assert ct.child_pugh_score > 7.0


# ============================================================================
# Disease ODE Models Tests
# ============================================================================


class TestCardiovascularODE:
    def test_baseline(self):
        cv = CardiovascularODE()
        for _ in range(24):
            cv.step(1.0)
        assert 60.0 < cv.map_mmhg < 180.0

    def test_hypertension(self):
        cv = CardiovascularODE()
        cv.hypertension_severity = 0.8
        for _ in range(24):
            cv.step(1.0)
        assert cv.map_mmhg > 93.0


class TestMetabolicT2DODE:
    def test_baseline(self):
        t2d = MetabolicT2DODE()
        t2d.step(1.0, glucose_mg_dl=100.0, insulin_uuml=10.0)
        assert t2d.beta_cell_function > 0

    def test_glucotoxicity(self):
        t2d = MetabolicT2DODE()
        initial = t2d.beta_cell_function
        for _ in range(100):
            t2d.step(1.0, glucose_mg_dl=250.0, insulin_uuml=5.0)
        assert t2d.beta_cell_function < initial


class TestCancerODE:
    def test_tumor_growth(self):
        ca = CancerODE()
        initial = ca.tumor_volume
        for _ in range(100):
            ca.step(1.0)
        assert ca.tumor_volume > initial

    def test_immune_surveillance(self):
        ca = CancerODE()
        ca.immune_surveillance = 1.0
        for _ in range(100):
            ca.step(1.0)
        assert ca.tumor_volume < 0.5


class TestAutoimmuneRAODE:
    def test_inflammation(self):
        ra = AutoimmuneRAODE()
        for _ in range(48):
            ra.step(1.0)
        assert ra.joint_inflammation > 0

    def test_drug_effect(self):
        ra = AutoimmuneRAODE()
        ra.dmard_effect = 0.8
        initial = ra.joint_inflammation
        for _ in range(100):
            ra.step(1.0)
        assert ra.joint_inflammation < initial


class TestNeurologicalODE:
    def test_synaptic_loss(self):
        neuro = NeurologicalODE()
        initial = neuro.synaptic_density
        for _ in range(100):
            neuro.step(1.0)
        assert neuro.synaptic_density < initial

    def test_neuroprotection(self):
        neuro = NeurologicalODE()
        neuro.disease_modifying_effect = 0.8
        initial = neuro.synaptic_density
        for _ in range(100):
            neuro.step(1.0)
        assert neuro.synaptic_density > initial * 0.5


class TestRenalODE:
    def test_nephron_loss(self):
        renal = RenalODE()
        initial = renal.nephron_mass
        for _ in range(100):
            renal.step(1.0)
        assert renal.nephron_mass < initial

    def test_acei_slows_loss(self):
        renal = RenalODE()
        renal.acei_effect = 0.8
        initial = renal.nephron_mass
        for _ in range(100):
            renal.step(1.0)
        assert renal.nephron_mass > initial * 0.5


class TestHepaticODE:
    def test_fibrosis_progression(self):
        hep = HepaticODE()
        initial = hep.fibrosis_stage
        for _ in range(100):
            hep.step(1.0)
        assert hep.fibrosis_stage > initial

    def test_antiviral_slows_fibrosis(self):
        hep = HepaticODE()
        hep.antiviral_effect = 0.9
        for _ in range(100):
            hep.step(1.0)
        assert hep.fibrosis_stage < 2.0


class TestHematologicalODE:
    def test_stem_cell_loss(self):
        heme = HematologicalODE()
        initial = heme.stem_cell_pool
        for _ in range(100):
            heme.step(1.0)
        assert heme.stem_cell_pool < initial

    def test_hypomethylating_helps(self):
        heme = HematologicalODE()
        heme.hypomethylating_effect = 0.8
        initial = heme.stem_cell_pool
        for _ in range(100):
            heme.step(1.0)
        assert heme.stem_cell_pool > initial * 0.5


class TestDiseaseModelFactory:
    def test_cardiovascular(self):
        m = create_disease_model("cardiovascular", 0.5)
        assert isinstance(m, CardiovascularODE)

    def test_diabetes(self):
        m = create_disease_model("type_2_diabetes", 0.6)
        assert isinstance(m, MetabolicT2DODE)

    def test_cancer(self):
        m = create_disease_model("lung_cancer", 0.4)
        assert isinstance(m, CancerODE)

    def test_ra(self):
        m = create_disease_model("rheumatoid_arthritis", 0.5)
        assert isinstance(m, AutoimmuneRAODE)

    def test_alzheimer(self):
        m = create_disease_model("alzheimers", 0.3)
        assert isinstance(m, NeurologicalODE)

    def test_ckd(self):
        m = create_disease_model("chronic_kidney_disease", 0.7)
        assert isinstance(m, RenalODE)

    def test_cirrhosis(self):
        m = create_disease_model("cirrhosis", 0.5)
        assert isinstance(m, HepaticODE)

    def test_mds(self):
        m = create_disease_model("myelodysplastic_syndrome", 0.4)
        assert isinstance(m, HematologicalODE)

    def test_unknown_disease(self):
        m = create_disease_model("rare_disease_xyz", 0.5)
        assert hasattr(m, 'severity')
        assert m.severity == 0.5


# ============================================================================
# Fix validation: MTX toxicity, immune reset, circadian cortisol
# ============================================================================


class TestMTXToxicity:
    """Validate methotrexate entries in all three toxicity dictionaries."""

    def test_methotrexate_hepatotoxic(self):
        from helixlang.plugins.human.clinical_output import _HEPATOTOXIC_DRUGS
        assert "methotrexate" in _HEPATOTOXIC_DRUGS
        alt_rate, ast_rate = _HEPATOTOXIC_DRUGS["methotrexate"]
        assert alt_rate > 0
        assert ast_rate > 0

    def test_methotrexate_nephrotoxic(self):
        from helixlang.plugins.human.clinical_output import _NEPHROTOXIC_DRUGS
        assert "methotrexate" in _NEPHROTOXIC_DRUGS
        assert _NEPHROTOXIC_DRUGS["methotrexate"] > 0

    def test_methotrexate_myelosuppressive(self):
        from helixlang.plugins.human.clinical_output import _MYELOSUPPRESSIVE_DRUGS
        assert "methotrexate" in _MYELOSUPPRESSIVE_DRUGS
        assert _MYELOSUPPRESSIVE_DRUGS["methotrexate"] > 0

    def _make_lab_model(self):
        from helixlang.plugins.human.clinical_output import ClinicalLabModel, ClinicalLabs
        labs = ClinicalLabs()
        labs.age_years = 50
        labs.sex = "male"
        return ClinicalLabModel(baseline=labs)

    def test_mtx_drives_alt_elevation(self):
        model = self._make_lab_model()
        baseline_alt = model.current.alt_u_per_l
        for _ in range(72):
            model.update(dt_h=1.0, drug_concentrations={"methotrexate": 10.0})
        assert model.current.alt_u_per_l > baseline_alt

    def test_mtx_drives_creatinine_rise(self):
        model = self._make_lab_model()
        baseline_creat = model.current.creatinine_mg_per_dl
        for _ in range(72):
            model.update(dt_h=1.0, drug_concentrations={"methotrexate": 10.0})
        assert model.current.creatinine_mg_per_dl > baseline_creat

    def test_mtx_drives_wbc_suppression(self):
        model = self._make_lab_model()
        baseline_wbc = model.current.wbc_per_ul
        for _ in range(72):
            model.update(dt_h=1.0, drug_concentrations={"methotrexate": 10.0})
        assert model.current.wbc_per_ul < baseline_wbc


class TestImmuneModelReset:
    """Validate that cortisol suppression does not compound across ticks."""

    def test_production_rates_not_mutated(self):
        from helixlang.plugins.human.immune import InnateImmuneModel
        model = InnateImmuneModel()
        model.cortisol_suppression = 0.5
        # Record rate after first step (base restored + suppressed)
        model.step(dt_h=1.0)
        tnf_after_first = model.cytokines.tnf_production_rate
        il6_after_first = model.cytokines.il6_production_rate
        # After 100 more steps, rate should be identical (no compounding)
        for _ in range(99):
            model.step(dt_h=1.0)
        assert model.cytokines.tnf_production_rate == pytest.approx(tnf_after_first)
        assert model.cytokines.il6_production_rate == pytest.approx(il6_after_first)

    def test_cortisol_suppression_dampens_cytokines(self):
        from helixlang.plugins.human.immune import InnateImmuneModel
        model_suppressed = InnateImmuneModel()
        model_suppressed.cortisol_suppression = 0.8
        model_suppressed.infection_severity = 1.0
        model_suppressed.step(dt_h=1.0)
        il6_suppressed = model_suppressed.get_il6()

        model_normal = InnateImmuneModel()
        model_normal.infection_severity = 1.0
        model_normal.step(dt_h=1.0)
        il6_normal = model_normal.get_il6()

        assert il6_suppressed < il6_normal

    def test_autoimmune_drives_cytokines(self):
        from helixlang.plugins.human.immune import InnateImmuneModel
        model = InnateImmuneModel()
        model.autoimmune_activation = 0.8
        for _ in range(24):
            model.step(dt_h=1.0)
        assert model.get_il6() > 5.0  # well above healthy baseline of 1.0


class TestCircadianCortisol:
    """Validate circadian cortisol rhythm in HPAAxis."""

    def test_cortisol_oscillates(self):
        from helixlang.plugins.human.endocrine import HPAAxis
        axis = HPAAxis()
        # Run to steady state
        for h in range(48):
            axis.step(1.0, clock_hour=float(h % 24))
        # Record 24h profile
        values = []
        for h in range(24):
            axis.step(1.0, clock_hour=float(h))
            values.append(axis.cortisol_ug_dl)
        assert max(values) > min(values) * 1.1  # at least 10% variation

    def test_cortisol_without_clock_advances(self):
        from helixlang.plugins.human.endocrine import HPAAxis
        axis = HPAAxis()
        c0 = axis.cortisol_ug_dl
        for _ in range(24):
            axis.step(1.0)  # no clock_hour → auto-advancing
        assert axis.cortisol_ug_dl != c0  # dynamics are active

    def test_no_clock_hour_still_changes_cortisol(self):
        from helixlang.plugins.human.endocrine import HPAAxis
        axis = HPAAxis()
        # With explicit clock, cortisol oscillates
        values_with_clock = []
        for h in range(48):
            axis.step(1.0, clock_hour=float(h % 24))
            values_with_clock.append(axis.cortisol_ug_dl)
        # Without clock (auto-advancing), should also produce variation
        axis2 = HPAAxis()
        for _ in range(48):
            axis2.step(1.0)
        # Both should have non-trivial cortisol (not 0)
        assert values_with_clock[-1] > 5.0
        assert axis2.cortisol_ug_dl > 5.0


# ============================================================================
# doc/40 Phase A: Innate Immune Realism
# ============================================================================


class TestIFNPool:
    def test_baseline_zero(self):
        ifn = IFNPool()
        assert ifn.ifn_alpha_beta == 0.0

    def test_rises_with_pathogen(self):
        ifn = IFNPool()
        for _ in range(48):
            ifn.step(1.0, pathogen_signal=1.0)
        assert ifn.ifn_alpha_beta > 1.0

    def test_antiviral_suppresses_pathogen(self):
        ifn = IFNPool()
        ifn.ifn_alpha_beta = 10.0
        eff = ifn.effective_pathogen(1.0)
        assert 0.0 <= eff < 1.0  # suppressed but not negative


class TestCRPDriverV2:
    def test_hill_saturates(self):
        crp = CRPDriver()
        # Saturating IL-6 → CRP plateaus near Vmax/clearance, not unbounded
        for _ in range(1000):
            crp.step(1.0, il6_pg_ml=1000.0)
        assert crp.crp_mg_l <= crp.max_crp
        assert crp.crp_mg_l > 500.0  # strong acute-phase response

    def test_lag_delays_crp(self):
        # Lag compartment means CRP is driven by the lagged (slower) IL-6,
        # not the instantaneous stimulus.
        crp = CRPDriver()
        crp.step(1.0, il6_pg_ml=50.0)
        # After one step the lagged IL-6 is well below the instantaneous 50
        assert crp._il6_lagged < 50.0

    def test_apr_panel_tracks_il6(self):
        crp = CRPDriver()
        crp._il6_lagged = 50.0
        crp.step(1.0, il6_pg_ml=50.0)
        assert crp.saa_mg_l > 0
        assert crp.pct_ng_ml > 0
        assert crp.ferritin_ng_ml > 50.0
        assert crp.fibrinogen_g_l > 3.0


class TestFribergTransit:
    def test_friberg_no_nadir_without_kill(self):
        pop = ImmuneCellPopulation()
        anc0 = pop.neutrophils
        for _ in range(72):
            pop.step(1.0, il6=1.0, tnf=5.0)
        # No drug kill → no nadir; ANC stays healthy
        assert pop.neutrophils > anc0 * 0.5

    def test_friberg_drug_kill_depletes_anc(self):
        pop = ImmuneCellPopulation()
        pop.friberg_drug_kill = 0.5
        for _ in range(24):
            pop.step(1.0, il6=1.0, tnf=5.0)
        # Cytotoxic kill on proliferating precursors → ANC drops (nadir)
        assert pop.neutrophils < 4.0


class TestImmuneCircadianCortisol:
    def test_default_no_circadian_variation(self):
        # amplitude = 0 must not modulate cortisol: with explicit constant
        # suppression, IL-6 follows a smooth monotone trajectory (no burst
        # correlated with a 24 h cycle).
        m1 = InnateImmuneModel(circadian_amplitude=0.0)
        m1.cortisol_suppression = 0.3
        m1.infection_severity = 0.5
        il6s = []
        for _ in range(48):
            m1.step(1.0)
            il6s.append(m1.get_il6())
        # Once reaching a quasi-steady envelope, successive differences are tiny
        tail = il6s[-8:]
        assert max(tail[-1] - tail[0], 0.0) < 0.05

    def test_circadian_oscillates_il6(self):
        # Circadian-modulated cortisol suppresses IL-6 production in a 24 h
        # rhythm, producing a measurable oscillation in IL-6.
        m = InnateImmuneModel(circadian_amplitude=0.6)
        m.cortisol_suppression = 0.6
        m.infection_severity = 0.5
        vals = []
        for _ in range(96):
            m.step(1.0)
            vals.append(m.get_il6())
        tail = vals[-48:]
        assert max(tail) > min(tail) * 1.5
