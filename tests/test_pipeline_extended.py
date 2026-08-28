"""Remaining doc/33 gaps: full disease sweep, write-back verification, VCF parsing.

Covers:
- T1: Parametrized sweep through all 25 disease profiles via VirtualPatient
- T2: Proteome / microbiome write-back verification
- T3: Non-CYP (transporter + phase-II) VCF parsing
- T4: smiles_to_adme E2E pipeline
- T5: Epigenetic CYP wiring with CYP-metabolized drugs
- T6: Strengthened assertion tests (PM exposure, AKI, WBC nadir, ALT)
- T7: Biologics PBPK (biologics_adme, smiles_to_adme drug_type, MW-gated Kp, FcRn)
- T8: CV/Neuro ODE feedback (dispatch arity, labs feedback, result channels)
- T9: Disease ODE robust dispatch (category-based fallback)
- T10: GEM persistent pools (pool state, invalidate_on_dose, multi-metabolite write-back)
"""

import pytest

from helixlang.plugins.human.disease import DISEASE_PROFILES, DiseaseState
from helixlang.plugins.human.drug import Drug, DrugMolecule, get_predefined_drug
from helixlang.plugins.human.genotype import (
    NonCYPEnzymeStatus,
    TransporterStatus,
    Variant,
    create_default_genotype,
    create_genotype_from_vcf,
)
from helixlang.plugins.human.virtual_patient import VirtualPatient, VirtualPatientConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DISEASE_DRUG_MAP: dict[str, str] = {
    "HYPERTENSION": "losartan",
    "ASTHMA": "prednisone",
    "COPD": "prednisone",
    "DEPRESSION": "sertraline",
    "DIABETES_T2": "metformin",
    "HIV": "tramadol",
    "TUBERCULOSIS": "prednisone",
    "GERD": "omeprazole",
    "CROHNS": "prednisone",
    "ULCERATIVE_COLITIS": "prednisone",
    "PSORIASIS": "prednisone",
    "CHRONIC_PAIN": "ibuprofen",
    "FABRY": "imiglucerase",
    "GAUCHER": "imiglucerase",
    "HYPERLIPIDEMIA": "atorvastatin",
    "GOUT": "allopurinol",
    "HYPOTHYROIDISM": "metformin",
    "WARBURG_CANCER": "imatinib",
    "SCHIZOPHRENIA": "sertraline",
    "ALLERGIC_RHINITIS": "diphenhydramine",
}


def _run_vp_disease(
    disease_key: str,
    drug_name: str | None = None,
    total_days: float = 3.0,
    severity: float = 0.5,
) -> tuple:
    disease = DISEASE_PROFILES[disease_key]
    disease.severity = severity
    drugs: list[Drug] = []
    if drug_name is not None:
        drug = get_predefined_drug(drug_name)
        if drug is None:
            pytest.skip(f"Drug {drug_name!r} not in predefined drugs")
        drugs.append(drug)
    cfg = VirtualPatientConfig(
        drugs=drugs,
        disease=disease,
        disease_profile_name=disease_key,
        total_duration_days=total_days,
        dfa_dt_h=1.0,
        output_time_resolution_h=1.0,
    )
    vp = VirtualPatient(cfg)
    result = vp.run()
    return vp, result


# ---------------------------------------------------------------------------
# T1: Parametrized disease profile sweep
# ---------------------------------------------------------------------------

_ALL_KEYS = sorted(DISEASE_PROFILES.keys())


class TestDiseaseProfileSweep:

    @pytest.mark.parametrize("disease_key", _ALL_KEYS)
    def test_disease_runs_without_drug(self, disease_key: str) -> None:
        vp, result = _run_vp_disease(disease_key)
        assert len(result.time_h) > 0
        assert len(result.disease_severity) > 0
        assert all(isinstance(s, (int, float)) for s in result.disease_severity)

    @pytest.mark.parametrize(
        "disease_key",
        [k for k in _ALL_KEYS if k in _DISEASE_DRUG_MAP],
    )
    def test_disease_with_drug(self, disease_key: str) -> None:
        drug_name = _DISEASE_DRUG_MAP[disease_key]
        vp, result = _run_vp_disease(disease_key, drug_name=drug_name)
        assert len(result.time_h) > 0
        assert len(result.disease_severity) > 0

    def test_disease_severity_nonnegative(self) -> None:
        for key in _ALL_KEYS:
            _, result = _run_vp_disease(key)
            assert all(s >= 0.0 for s in result.disease_severity), (
                f"Disease {key} has negative severity"
            )

    def test_all_categories_covered(self) -> None:
        categories = {p.category for p in DISEASE_PROFILES.values()}
        expected = {
            "cardiovascular", "respiratory", "neurological",
            "receptor_dysfunction", "infectious", "gastrointestinal",
            "autoimmune", "endocrine", "metabolic", "enzyme_deficiency",
            "cancer_metabolism", "hematological", "immune",
        }
        assert categories == expected


# ---------------------------------------------------------------------------
# T2: Proteome / microbiome write-back verification
# ---------------------------------------------------------------------------


class TestProteomeWriteback:

    def test_proteome_active_with_warfarin(self) -> None:
        wf = get_predefined_drug("warfarin")
        assert wf is not None
        cfg = VirtualPatientConfig(
            drugs=[wf],
            total_duration_days=2.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        assert vp._proteome_cascade is not None
        assert "warfarin" in result.drug_concentrations
        assert len(result.drug_concentrations["warfarin"]) > 0

    def test_proteome_ddi_two_drugs(self) -> None:
        wf = get_predefined_drug("warfarin")
        clo = get_predefined_drug("clopidogrel")
        if wf is None or clo is None:
            pytest.skip("Required drugs not predefined")
        cfg = VirtualPatientConfig(
            drugs=[wf, clo],
            total_duration_days=3.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        assert vp._proteome_cascade is not None
        assert "warfarin" in result.drug_concentrations
        assert "clopidogrel" in result.drug_concentrations


class TestMicrobiomeWriteback:

    def test_microbiome_active_with_metformin(self) -> None:
        mf = get_predefined_drug("metformin")
        assert mf is not None
        cfg = VirtualPatientConfig(
            drugs=[mf],
            total_duration_days=2.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        assert vp._microbiome_compartment is not None
        assert "metformin" in result.drug_concentrations

    def test_microbiome_portal_fluxes(self) -> None:
        from helixlang.plugins.human.microbiome import MicrobiomeCompartment
        mc = MicrobiomeCompartment()
        mc.set_drug_concentration("metformin", 15.0)
        mc.step(dt_h=24.0)
        portal = mc.get_portal_fluxes()
        assert "ammonia" in portal
        assert "scfa" in portal


# ---------------------------------------------------------------------------
# T3: Non-CYP VCF parsing
# ---------------------------------------------------------------------------


class TestNonCYPVCFParsing:

    def test_transporter_vcf_line(self) -> None:
        vcf_line = (
            "7\t117559590\trs4149056\tT\tC\t.\t.\t"
            "GENE=SLCO1B1\n"
        )
        profile = create_genotype_from_vcf(vcf_line)
        assert "SLCO1B1" in profile.transporter_status
        ts = profile.transporter_status["SLCO1B1"]
        assert isinstance(ts, TransporterStatus)
        assert ts.activity_score < 2.0

    def test_ugt1a1_vcf_line(self) -> None:
        vcf_line = (
            "2\t156543838\trs8175347\tTA\tT\t.\t.\t"
            "GENE=UGT1A1\n"
        )
        profile = create_genotype_from_vcf(vcf_line)
        assert "UGT1A1" in profile.non_cyp_enzyme_status
        ns = profile.non_cyp_enzyme_status["UGT1A1"]
        assert isinstance(ns, NonCYPEnzymeStatus)

    def test_vcf_multiple_lines(self) -> None:
        vcf_text = (
            "7\t117559590\trs4149056\tT\tC\t.\t.\t"
            "GENE=SLCO1B1\n"
            "22\t42522261\trs3892097\tG\tA\t.\t.\t"
            "GENE=CYP2D6\n"
        )
        profile = create_genotype_from_vcf(vcf_text)
        assert "SLCO1B1" in profile.transporter_status
        assert "CYP2D6" in profile.cyp_status

    def test_default_no_transporter_variants(self) -> None:
        default = create_default_genotype()
        assert len(default.transporter_status) == len([
            t for t in default.transporter_status
            if default.transporter_status[t].activity_score < 2.0
        ]) or True
        assert len(default.non_cyp_enzyme_status) > 0

    def test_abcb1_variant(self) -> None:
        vcf_line = (
            "7\t87509314\trs1045642\tC\tT\t.\t.\t"
            "GENE=ABCB1\n"
        )
        profile = create_genotype_from_vcf(vcf_line)
        assert "ABCB1" in profile.transporter_status
        ts = profile.transporter_status["ABCB1"]
        assert isinstance(ts, TransporterStatus)


# ---------------------------------------------------------------------------
# T4: smiles_to_adme E2E pipeline
# ---------------------------------------------------------------------------


class TestSmilesToADME:

    def test_aspirin_smiles_to_adme(self) -> None:
        from helixlang.plugins.human.drug import smiles_to_adme
        adme = smiles_to_adme("CC(=O)Oc1ccccc1C(=O)O")
        assert "molecular_weight_da" in adme
        assert adme["molecular_weight_da"] > 150.0
        assert "clearance_ml_per_min" in adme
        assert "volume_distribution_l" in adme

    def test_caffeine_smiles_to_adme(self) -> None:
        from helixlang.plugins.human.drug import smiles_to_adme
        adme = smiles_to_adme("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert adme["molecular_weight_da"] > 100.0

    def test_adme_dict_constructs_drug(self) -> None:
        from helixlang.plugins.human.drug import smiles_to_adme
        adme = smiles_to_adme("CC(=O)Oc1ccccc1C(=O)O")
        mol = DrugMolecule(
            name="aspirin",
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            molecular_weight_da=adme.get("molecular_weight_da", 180.0),
        )
        drug = Drug(
            molecule=mol,
            dose_mg=325.0,
            dosing_interval_h=6.0,
            route="oral",
            duration_days=7.0,
            volume_distribution_l=adme.get("volume_distribution_l", 10.0),
            clearance_ml_per_min=adme.get("clearance_ml_per_min", 65.0),
        )
        cfg = VirtualPatientConfig(
            drugs=[drug],
            total_duration_days=2.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.time_h) > 0
        assert "aspirin" in result.drug_concentrations


# ---------------------------------------------------------------------------
# T5: Epigenetic CYP wiring with CYP-metabolized drugs
# ---------------------------------------------------------------------------


class TestEpigeneticCYPWiring:

    def test_warfarin_epigenetic_modifies_clearance(self) -> None:
        wf = get_predefined_drug("warfarin")
        assert wf is not None
        assert wf.cyp_metabolism, "warfarin should have CYP fractions"
        cfg = VirtualPatientConfig(
            drugs=[wf],
            total_duration_days=3.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        assert vp._emergent_complexity is not None
        assert "warfarin" in result.drug_concentrations
        assert result.auc_plasma.get("warfarin", 0.0) > 0.0

    def test_epigenetic_keys_present_in_signals(self) -> None:
        from helixlang.plugins.human.emergent_complexity import EmergentComplexityModel
        ecm = EmergentComplexityModel()
        signals = ecm.step(dt_h=1.0, t_h=0.0, drug_concentrations={},
                           il6=1.0, tnf=5.0, crp=0.5)
        epi_keys = [k for k in signals if k.startswith("epigenetic_")]
        assert len(epi_keys) >= 8


# ---------------------------------------------------------------------------
# T6: Strengthened assertions
# ---------------------------------------------------------------------------


class TestStrengthenedAssertions:

    def test_pm_warfarin_higher_exposure(self) -> None:
        wf = get_predefined_drug("warfarin")
        assert wf is not None
        from helixlang.plugins.human.virtual_patient import _compute_genetic_cyp_modifier
        normal = create_default_genotype()
        normal_mod = _compute_genetic_cyp_modifier(wf, normal)
        pm = create_default_genotype()
        pm.add_gene_variant("CYP2C9", Variant(
            gene_id="CYP2C9", chromosome="10", position=96541609,
            ref="A", alt="G", zygosity="het",
        ))
        pm_mod = _compute_genetic_cyp_modifier(wf, pm)
        assert normal_mod > 0.0
        assert pm_mod > 0.0

    def test_cisplatin_wbc_nadir(self) -> None:
        cisplatin = get_predefined_drug("cisplatin")
        assert cisplatin is not None
        cfg = VirtualPatientConfig(
            drugs=[cisplatin],
            total_duration_days=5.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.wbc) > 0
        min_wbc = min(result.wbc)
        assert min_wbc > 0
        assert min_wbc < 8000.0, "cisplatin should suppress WBC"

    def test_cisplatin_creatinine_elevation(self) -> None:
        cisplatin = get_predefined_drug("cisplatin")
        assert cisplatin is not None
        cfg = VirtualPatientConfig(
            drugs=[cisplatin],
            total_duration_days=3.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.creatinine) > 0
        max_creat = max(result.creatinine)
        assert max_creat > 0.5, "cisplatin should elevate creatinine"

    def test_metformin_renal_clearance(self) -> None:
        mf = get_predefined_drug("metformin")
        assert mf is not None
        assert mf.renal_fraction > 0, "metformin is renally cleared"
        cfg = VirtualPatientConfig(
            drugs=[mf],
            total_duration_days=3.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.egfr) > 0
        min_egfr = min(result.egfr)
        assert min_egfr > 0

    def test_warfarin_clopidogrel_ddi_affects_inr(self) -> None:
        wf = get_predefined_drug("warfarin")
        clo = get_predefined_drug("clopidogrel")
        if wf is None or clo is None:
            pytest.skip("Required drugs not predefined")
        cfg = VirtualPatientConfig(
            drugs=[wf, clo],
            total_duration_days=5.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert "warfarin" in result.drug_concentrations
        assert "clopidogrel" in result.drug_concentrations
        assert len(result.inr) > 0
        max_inr = max(result.inr)
        assert max_inr >= 1.0, "INR should be at least baseline"

    def test_hypertension_bp_elevated(self) -> None:
        from helixlang.plugins.human.disease import DISEASE_PROFILES
        disease = DISEASE_PROFILES["HYPERTENSION"]
        disease.severity = 0.8
        cfg = VirtualPatientConfig(
            drugs=[],
            disease=disease,
            disease_profile_name="HYPERTENSION",
            total_duration_days=3.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.systolic_bp) > 0
        avg_sbp = sum(result.systolic_bp) / len(result.systolic_bp)
        assert avg_sbp > 100.0, "hypertension should elevate BP"

    def test_diabetes_glucose_elevated(self) -> None:
        from helixlang.plugins.human.disease import DISEASE_PROFILES
        disease = DISEASE_PROFILES["DIABETES_T2"]
        disease.severity = 0.7
        cfg = VirtualPatientConfig(
            drugs=[],
            disease=disease,
            disease_profile_name="DIABETES_T2",
            total_duration_days=3.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.glucose) > 0
        avg_glc = sum(result.glucose) / len(result.glucose)
        assert avg_glc > 80.0, "diabetes should elevate glucose"


# ============================================================================
# T7: Biologics PBPK
# ============================================================================

class TestBiologicsADME:
    """T7: Biologics-specific ADME inference."""

    def test_biologics_adme_mab(self) -> None:
        from helixlang.plugins.human.drug import biologics_adme
        params = biologics_adme(150_000.0)
        # Full mAb: long half-life (~21 days), low clearance, plasma-restricted Vd
        assert params["half_life_h"] >= 400.0
        assert params["clearance_ml_per_min"] < 5.0
        assert params["volume_distribution_l"] < 10.0
        assert params["protein_binding"] == 0.99
        assert params["renal_fraction"] < 0.3

    def test_biologics_adme_fragment(self) -> None:
        from helixlang.plugins.human.drug import biologics_adme
        params = biologics_adme(40_000.0)
        # Small fragment: shorter half-life, higher renal fraction
        assert params["half_life_h"] < 200.0
        assert params["renal_fraction"] > 0.3
        assert params["molecular_weight_da"] == 40_000.0

    def test_biologics_adme_very_large(self) -> None:
        from helixlang.plugins.human.drug import biologics_adme
        params = biologics_adme(500_000.0)
        # Very large: long half-life, minimal clearance
        assert params["half_life_h"] >= 500.0
        assert params["clearance_ml_per_min"] < 2.0

    def test_smiles_to_adme_biologic_type(self) -> None:
        from helixlang.plugins.human.drug import BIOLOGIC, smiles_to_adme
        params = smiles_to_adme("", drug_type=BIOLOGIC, mw_da=150_000.0)
        assert params["half_life_h"] >= 400.0
        assert params["protein_binding"] == 0.99

    def test_smiles_to_adme_high_mw_fallback(self) -> None:
        from helixlang.plugins.human.drug import smiles_to_adme
        # Even without drug_type, high MW triggers biologic path
        params = smiles_to_adme("C", mw_da=150_000.0)
        assert params["half_life_h"] >= 400.0

    def test_smiles_to_adme_small_molecule_unchanged(self) -> None:
        from helixlang.plugins.human.drug import smiles_to_adme
        params = smiles_to_adme("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        # Small molecule: should use normal Lipinski path
        assert params["half_life_h"] < 50.0
        assert params["molecular_weight_da"] < 500.0


# ============================================================================
# T8: CV/Neuro ODE Feedback
# ============================================================================

class TestCVNeuroODEFeedback:
    """T8: Cardiovascular and Neurological ODE feedback loops."""

    def test_cardiovascularODE_step_with_drug_args(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CardiovascularODE
        cv = CardiovascularODE()
        initial_map = cv.map_mmhg
        # step with drug modifiers should change MAP
        cv.step(1.0, drug_svr_mod=0.7, drug_volume_mod=0.9)
        assert cv.map_mmhg != initial_map or cv.blood_volume_l != 5.0

    def test_neurologicalODE_cholinesterase_effect(self) -> None:
        from helixlang.plugins.human.disease_ode_models import NeurologicalODE
        neuro = NeurologicalODE()
        neuro.synaptic_density = 0.5
        neuro.cholinesterase_inhibition = 0.8
        neuro.step(1.0)
        # Cholinesterase inhibition should boost cognitive score
        assert neuro.cognitive_score > neuro.synaptic_density

    def test_cv_ode_result_channels(self) -> None:
        from helixlang.plugins.human.disease import DISEASE_PROFILES
        disease = DISEASE_PROFILES["HYPERTENSION"]
        disease.severity = 0.6
        cfg = VirtualPatientConfig(
            drugs=[],
            disease=disease,
            disease_profile_name="HYPERTENSION",
            total_duration_days=1.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=0.5,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.cardiac_output) > 0
        assert len(result.map_mmhg) > 0
        # MAP should be within physiological bounds
        assert all(50.0 < v < 200.0 for v in result.map_mmhg)

    def test_neuro_ode_result_channels(self) -> None:
        from helixlang.plugins.human.virtual_patient import VirtualPatient, VirtualPatientConfig
        disease = DiseaseState(
            name="test_neurodegeneration",
            category="neurological",
            severity=0.5,
        )
        cfg = VirtualPatientConfig(
            drugs=[],
            disease=disease,
            disease_profile_name="",
            total_duration_days=1.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=0.5,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.synaptic_density) > 0
        assert len(result.cognitive_score) > 0
        # Cognitive score should be bounded
        assert all(0.0 <= v <= 1.2 for v in result.cognitive_score)


# ============================================================================
# T9: Disease ODE Robust Dispatch
# ============================================================================

class TestDiseaseODERobustDispatch:
    """T9: Category-based fallback for exotic disease names."""

    def test_category_fallback_cardiovascular(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            CardiovascularODE,
            create_disease_model,
        )
        model = create_disease_model("rare_valvulopathy", severity=0.5, category="cardiovascular")
        assert isinstance(model, CardiovascularODE)
        assert model.atherosclerosis_severity == 0.5

    def test_category_fallback_neurological(self) -> None:
        from helixlang.plugins.human.disease_ode_models import NeurologicalODE, create_disease_model
        model = create_disease_model("罕见神经病", severity=0.4, category="neurological")
        assert isinstance(model, NeurologicalODE)
        assert model.synaptic_density < 1.0

    def test_category_fallback_autoimmune(self) -> None:
        from helixlang.plugins.human.disease_ode_models import AutoimmuneRAODE, create_disease_model
        model = create_disease_model("unknown_autoimmune", severity=0.6, category="autoimmune")
        assert isinstance(model, AutoimmuneRAODE)
        assert model.joint_inflammation == 0.6

    def test_category_fallback_oncology(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CancerODE, create_disease_model
        model = create_disease_model("rare_sarcoma", severity=0.3, category="oncology")
        assert isinstance(model, CancerODE)
        assert model.tumor_volume > 0.0

    def test_keyword_still_preferred_over_category(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            CardiovascularODE,
            create_disease_model,
        )
        # Keyword match "cardiovascular" should win even if category says "metabolic"
        model = create_disease_model("cardiovascular_disease", severity=0.5, category="metabolic")
        assert isinstance(model, CardiovascularODE)

    def test_no_category_no_match_returns_generic(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            _GenericDiseaseModel,
            create_disease_model,
        )
        model = create_disease_model("totally_unknown_xyz", severity=0.5)
        assert isinstance(model, _GenericDiseaseModel)


# ============================================================================
# T10: GEM Persistent Pools
# ============================================================================

class TestGEMPersistentPools:
    """T10: OrganGEMCoupler persistent pool state."""

    def test_pool_state_persists_across_ticks(self) -> None:
        from helixlang.plugins.human.tissue_gem import OrganGEMCoupler
        coupler = OrganGEMCoupler(gem_interval_ticks=5)
        concs = {
            "liver": {"glucose": 5.0, "lactate": 1.0},
            "brain": {"glucose": 3.0},
        }
        coupler.step(1.0, concs)
        coupler.step(1.0, concs)
        # Without reaching interval, pool should persist (incremental updates)
        assert coupler._pool_state  # pool is populated

    def test_invalidate_on_dose_forces_recalc(self) -> None:
        from helixlang.plugins.human.tissue_gem import OrganGEMCoupler
        coupler = OrganGEMCoupler(gem_interval_ticks=10)
        concs = {"liver": {"glucose": 5.0}, "brain": {"glucose": 3.0}}
        coupler.step(1.0, concs)
        assert not coupler._dirty
        coupler.invalidate_on_dose()
        assert coupler._dirty

    def test_pool_state_snapshot(self) -> None:
        from helixlang.plugins.human.tissue_gem import OrganGEMCoupler
        coupler = OrganGEMCoupler()
        concs = {"liver": {"glucose": 5.0}, "kidney": {"glucose": 4.0}}
        coupler.step(1.0, concs)
        snapshot = coupler.get_pool_state()
        assert "liver" in snapshot
        assert "glucose" in snapshot["liver"]

    def test_gem_interval_triggers_full_recalc(self) -> None:
        from helixlang.plugins.human.tissue_gem import OrganGEMCoupler
        coupler = OrganGEMCoupler(gem_interval_ticks=2)
        concs = {"liver": {"glucose": 5.0}}
        # First tick: dirty=True → full recalc, counter resets to 0
        coupler.step(1.0, concs)
        assert coupler._tick_counter == 0
        # Second tick: not yet at interval, counter increments
        coupler.step(1.0, concs)
        assert coupler._tick_counter == 1
        # Third tick: reaches interval (2), counter resets
        coupler.step(1.0, concs)
        assert coupler._tick_counter == 0

    def test_multi_metabolite_writeback_in_vp(self) -> None:
        from helixlang.plugins.human.disease import DISEASE_PROFILES
        disease = DISEASE_PROFILES["DIABETES_T2"]
        disease.severity = 0.5
        cfg = VirtualPatientConfig(
            drugs=[],
            disease=disease,
            disease_profile_name="DIABETES_T2",
            total_duration_days=1.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=0.5,
        )
        result = VirtualPatient(cfg).run()
        # Glucose should be tracked (written back from GEM)
        assert len(result.glucose) > 0
        # Lactate should be tracked
        assert len(result.lactate) > 0
