"""Tests for the helix human-simulation annotations and VirtualPatient backend.

Covers three layers of the doc/28+29 virtual-patient pipeline:

1. Parser: #person / #trait / #disease / #disease_gene / #disease_metabolite /
   #drug / #pd_effect annotations are merged into ``Program.sim_extensions``.
2. sim_runtime annotation builders: the ``_build_*_from_helix`` helpers turn
   those raw string dicts into typed backend config objects.
3. End-to-end: a programmatically built :class:`VirtualPatientConfig` runs the
   full PBPK → PD → labs → vitals integration loop and yields time-series
   output (clinical labs, vital signs, drug concentrations).
"""
from __future__ import annotations

import pytest

from helixlang.human.disease import DiseaseState, GenePerturbation, MetabolitePerturbation
from helixlang.human.drug import Drug, DrugMolecule
from helixlang.human.genotype import GenotypeProfile, Variant, create_default_genotype
from helixlang.human.pharmacodynamics import PDEffect, Pharmacodynamics
from helixlang.human.phenotype import ExternalTraits
from helixlang.human.virtual_patient import (
    VirtualPatient,
    VirtualPatientConfig,
    VirtualPatientResult,
)
from helixlang.parser import parse_source
from helixlang.sim_runtime import (
    _build_disease_from_helix,
    _build_drugs_from_helix,
    _build_genotype_from_helix,
    _build_pd_from_helix,
    _build_traits_from_helix,
)

# ============================================================================
# Shared fixtures / helpers
# ============================================================================

#: Full annotated program mirroring examples/56_human_patient_simulation.helix.
PATIENT_SOURCE = """
#person name=John age=55 sex=male weight=82 height=175 ethnicity=european

#gene name=CYP2D6 allele=*4 zygosity=het

#trait smoking=former pack_years=10 alcohol=5 exercise=moderate

#disease name="type 2 diabetes" category=metabolic_overload severity=0.7 \
onset_age=45 description="chronic metabolic disease"

#disease_gene gene=INSR type=downregulate activity=0.3
#disease_gene gene=IRS1 type=downregulate activity=0.5

#disease_metabolite id=glucose type=accumulate concentration=7.8 normal=5.5
#disease_metabolite id=triglycerides type=accumulate concentration=2.5 \
normal=1.7

#drug name=metformin smiles=CN(C)C(=N)NC(=N)N formula=C4H11N5 mw=129.16 \
dose=500 route=oral interval=12 duration=30 bioavailability=0.55 vd=654 \
cl=510 half_life=6.0 renal_fraction=1.0

#pd_effect drug=metformin target=BIOMASSReaction type=inhibition ec50=5.0 \
emax=0.4 hill=1.5
#pd_effect drug=metformin target=glucose_uptake type=activation ec50=10.0 \
emax=0.2 hill=1.2
"""


@pytest.fixture(scope="module")
def patient_program():
    """Parsed PATIENT_SOURCE shared by all parser-annotation tests."""
    return parse_source(PATIENT_SOURCE)


@pytest.fixture(scope="module")
def patient_ext(patient_program):
    """sim_extensions dict of the parsed patient program."""
    return patient_program.sim_extensions


# ============================================================================
# 1. Parser annotation tests
# ============================================================================


class TestParserPersonAnnotation:
    def test_person_fields_stored_with_prefix(self, patient_ext):
        """Each #person key=value lands in sim_extensions under person_."""
        assert patient_ext["person_name"] == "John"
        assert patient_ext["person_age"] == "55"
        assert patient_ext["person_sex"] == "male"
        assert patient_ext["person_weight"] == "82"
        assert patient_ext["person_height"] == "175"
        assert patient_ext["person_ethnicity"] == "european"

    def test_person_values_stored_verbatim_as_strings(self):
        """Values stay strings; quoted values have their quotes stripped."""
        prog = parse_source('#person name="Jane Doe" age=42')
        ext = prog.sim_extensions
        assert ext["person_name"] == "Jane Doe"
        assert isinstance(ext["person_age"], str)
        assert ext["person_age"] == "42"


class TestParserTraitAnnotation:
    def test_trait_fields_stored_with_prefix(self, patient_ext):
        """Each #trait key=value lands in sim_extensions under trait_."""
        assert patient_ext["trait_smoking"] == "former"
        assert patient_ext["trait_pack_years"] == "10"
        assert patient_ext["trait_alcohol"] == "5"
        assert patient_ext["trait_exercise"] == "moderate"


class TestParserDiseaseAnnotation:
    def test_disease_fields_stored(self, patient_ext):
        """Scalar disease fields land in sim_extensions under disease_."""
        assert patient_ext["disease_name"] == "type 2 diabetes"
        assert patient_ext["disease_category"] == "metabolic_overload"
        assert patient_ext["disease_severity"] == "0.7"
        assert patient_ext["disease_onset_age"] == "45"

    def test_disease_genes_accumulated_as_list(self, patient_ext):
        """Repeated #disease_gene annotations append to one list."""
        genes = patient_ext["disease_genes"]
        assert isinstance(genes, list)
        assert len(genes) == 2
        assert genes[0] == {
            "gene": "INSR", "type": "downregulate", "activity": "0.3",
        }
        assert genes[1]["gene"] == "IRS1"
        assert genes[1]["activity"] == "0.5"

    def test_disease_metabolites_accumulated_as_list(self, patient_ext):
        """Repeated #disease_metabolite annotations append to one list."""
        mets = patient_ext["disease_metabolites"]
        assert isinstance(mets, list)
        assert len(mets) == 2
        assert mets[0]["id"] == "glucose"
        assert mets[0]["type"] == "accumulate"
        assert float(mets[0]["concentration"]) == pytest.approx(7.8)
        assert float(mets[1]["normal"]) == pytest.approx(1.7)


class TestParserDrugAnnotation:
    def test_drug_entries_stored_as_list(self, patient_ext):
        """#drug appends one entry to sim_extensions['drugs']."""
        drugs = patient_ext["drugs"]
        assert isinstance(drugs, list)
        assert len(drugs) == 1
        entry = drugs[0]
        assert entry["name"] == "metformin"
        assert entry["formula"] == "C4H11N5"
        assert entry["route"] == "oral"
        assert float(entry["dose"]) == pytest.approx(500.0)
        assert float(entry["mw"]) == pytest.approx(129.16)

    def test_multiple_drug_annotations_accumulate(self):
        """Two #drug annotations produce two entries in order."""
        prog = parse_source(
            "#drug name=a dose=10\n"
            "#drug name=b dose=20\n"
        )
        names = [d["name"] for d in prog.sim_extensions["drugs"]]
        assert names == ["a", "b"]


class TestParserPDEffectAnnotation:
    def test_pd_effects_stored_as_list(self, patient_ext):
        """#pd_effect appends one entry to sim_extensions['pd_effects']."""
        effects = patient_ext["pd_effects"]
        assert isinstance(effects, list)
        assert len(effects) == 2
        first = effects[0]
        assert first["drug"] == "metformin"
        assert first["target"] == "BIOMASSReaction"
        assert first["type"] == "inhibition"
        assert float(first["ec50"]) == pytest.approx(5.0)

    def test_annotation_required_fields_enforced(self):
        """#drug without name= and #pd_effect without drug= raise ParseError."""
        from helixlang.errors import ParseError

        with pytest.raises(ParseError):
            parse_source("#drug dose=10")
        with pytest.raises(ParseError):
            parse_source("#pd_effect target=x")


# ============================================================================
# 2. Annotation → config builder tests (sim_runtime backend)
# ============================================================================


class TestGenotypeBuilder:
    def test_builds_gene_variants(self, patient_ext):
        """#gene entries route pharmacogenes to cyp_status."""
        genotype = create_default_genotype()
        _build_genotype_from_helix(genotype, patient_ext)

        # CYP2D6 is a core CYP enzyme → routed to cyp_status, not gene_variants
        assert "CYP2D6" in genotype.cyp_status
        status = genotype.cyp_status["CYP2D6"]
        assert status.phenotype in ("PM", "NM")
        # *4/*1 het → activity ~0.5 → PM or NM
        assert 0.0 <= status.activity_score <= 2.0

    def test_ignores_missing_or_non_list_genes(self):
        """Empty extensions and malformed 'genes' values leave it inert."""
        genotype = GenotypeProfile()
        _build_genotype_from_helix(genotype, {})
        _build_genotype_from_helix(genotype, {"genes": "not-a-list"})
        _build_genotype_from_helix(genotype, {"genes": [{"name": ""}]})
        assert genotype.gene_variants == {}


class TestTraitsBuilder:
    def test_builds_external_traits(self, patient_ext):
        """#person + #trait values populate an ExternalTraits instance."""
        traits = _build_traits_from_helix(patient_ext)
        assert isinstance(traits, ExternalTraits)
        assert traits.age_years == 55.0
        assert traits.sex == "male"
        assert traits.body_weight_kg == 82.0
        assert traits.height_cm == 175.0
        assert traits.ethnicity == "european"
        assert traits.smoking_status == "former"
        assert traits.pack_years == 10.0
        assert traits.alcohol_drinks_per_week == 5.0
        assert traits.exercise_level == "moderate"
        assert traits.pregnant is False

    def test_defaults_when_no_annotations(self):
        """Missing person/trait keys fall back to healthy-adult defaults."""
        traits = _build_traits_from_helix({})
        assert traits.age_years == 30.0
        assert traits.body_weight_kg == 70.0
        assert traits.height_cm == 170.0
        assert traits.smoking_status == "never"


class TestDiseaseBuilder:
    def test_builds_disease_state_with_perturbations(self, patient_ext):
        """#disease + genes + metabolites build a full DiseaseState."""
        disease = _build_disease_from_helix(patient_ext)
        assert isinstance(disease, DiseaseState)
        assert disease.name == "type 2 diabetes"
        assert disease.category == "metabolic_overload"
        assert disease.severity == pytest.approx(0.7)
        assert disease.onset_age_years == pytest.approx(45.0)
        assert disease.description == "chronic metabolic disease"

        assert len(disease.gene_perturbations) == 2
        gp = disease.gene_perturbations[0]
        assert isinstance(gp, GenePerturbation)
        assert gp.gene_id == "INSR"
        assert gp.perturbation_type == "downregulate"
        assert gp.activity_fraction == pytest.approx(0.3)

        assert len(disease.metabolite_perturbations) == 2
        mp = disease.metabolite_perturbations[0]
        assert isinstance(mp, MetabolitePerturbation)
        assert mp.metabolite_id == "glucose"
        assert mp.initial_concentration_mm == pytest.approx(7.8)
        assert mp.normal_concentration_mm == pytest.approx(5.5)

    def test_returns_none_without_disease_name(self):
        """No #disease annotation means no disease state."""
        assert _build_disease_from_helix({}) is None


class TestDrugBuilder:
    def test_builds_drug_with_parameters(self, patient_ext):
        """#drug entries build Drug + DrugMolecule objects."""
        drugs = _build_drugs_from_helix(patient_ext)
        assert len(drugs) == 1
        drug = drugs[0]
        assert isinstance(drug, Drug)
        mol = drug.molecule
        assert isinstance(mol, DrugMolecule)
        assert mol.name == "metformin"
        assert mol.smiles == "CN(C)C(=N)NC(=N)N"
        # SMILES-derived MW via RDKit (parse_drug_smiles), allow small variance
        assert mol.molecular_weight_da == pytest.approx(129.17, abs=0.5)
        assert mol.formula == "C4H11N5"

        assert drug.dose_mg == pytest.approx(500.0)
        assert drug.dosing_interval_h == pytest.approx(12.0)
        assert drug.route == "oral"
        assert drug.duration_days == pytest.approx(30.0)
        assert drug.bioavailability == pytest.approx(0.55)
        assert drug.volume_distribution_l == pytest.approx(654.0)
        assert drug.clearance_ml_per_min == pytest.approx(510.0)
        assert drug.half_life_h == pytest.approx(6.0)
        assert drug.renal_fraction == pytest.approx(1.0)

    def test_empty_extensions_yield_no_drugs(self):
        """No #drug annotations produce an empty regimen."""
        assert _build_drugs_from_helix({}) == []


class TestPDBuilder:
    def test_builds_pharmacodynamics_dict(self, patient_ext):
        """#pd_effect entries group into per-drug Pharmacodynamics models."""
        pd_map = _build_pd_from_helix(patient_ext)
        assert set(pd_map) == {"metformin"}
        pd = pd_map["metformin"]
        assert isinstance(pd, Pharmacodynamics)
        assert pd.drug_name == "metformin"
        assert len(pd.effects) == 2

        eff0 = pd.effects[0]
        assert isinstance(eff0, PDEffect)
        assert eff0.target_reaction == "BIOMASSReaction"
        assert eff0.effect_type == "inhibition"
        assert eff0.ec50_um == pytest.approx(5.0)
        assert eff0.emax == pytest.approx(0.4)
        assert eff0.hill_coefficient == pytest.approx(1.5)

        eff1 = pd.effects[1]
        assert eff1.target_reaction == "glucose_uptake"
        assert eff1.effect_type == "activation"

    def test_no_effects_yields_empty_dict(self):
        """No #pd_effect annotations produce no PD models."""
        assert _build_pd_from_helix({}) == {}


# ============================================================================
# 3. End-to-end VirtualPatient run (programmatic config, no parser)
# ============================================================================


def _make_metformin() -> Drug:
    """Metformin regimen matching the helix example's #drug annotation."""
    molecule = DrugMolecule(
        name="metformin",
        smiles="CN(C)C(=N)NC(=N)N",
        molecular_weight_da=129.16,
        formula="C4H11N5",
    )
    return Drug(
        molecule=molecule,
        dose_mg=500.0,
        dosing_interval_h=12.0,
        route="oral",
        duration_days=30.0,
        bioavailability=0.55,
        volume_distribution_l=654.0,
        clearance_ml_per_min=510.0,
        half_life_h=6.0,
        renal_fraction=1.0,
    )


def _make_diabetes_config(**overrides) -> VirtualPatientConfig:
    """Short T2DM patient config keeping the integration loop fast in tests."""
    genotype = create_default_genotype()
    genotype.add_gene_variant("CYP2D6", Variant(
        gene_id="CYP2D6", chromosome="22", position=42526693,
        ref="G", alt="A", zygosity="het",
    ))
    traits = ExternalTraits(
        age_years=55.0, sex="male", body_weight_kg=82.0,
        height_cm=175.0, ethnicity="european",
        smoking_status="former", pack_years=10.0,
        alcohol_drinks_per_week=5.0, exercise_level="moderate",
    )
    disease = DiseaseState(
        name="type 2 diabetes",
        category="metabolic_overload",
        gene_perturbations=[GenePerturbation(
            gene_id="INSR", perturbation_type="downregulate",
            activity_fraction=0.3,
        )],
        metabolite_perturbations=[MetabolitePerturbation(
            metabolite_id="glucose", perturbation_type="accumulate",
            initial_concentration_mm=7.8, normal_concentration_mm=5.5,
        )],
        severity=0.7,
        onset_age_years=45.0,
        description="chronic metabolic disease",
    )
    metformin_pd = Pharmacodynamics(drug_name="metformin", effects=[
        PDEffect(target_reaction="BIOMASSReaction",
                 effect_type="inhibition", ec50_um=5.0,
                 emax=0.6, hill_coefficient=1.5),
    ])
    base: dict = dict(
        genotype=genotype,
        traits=traits,
        disease=disease,
        drugs=[_make_metformin()],
        pharmacodynamics={"metformin": metformin_pd},
        total_duration_days=1.0,
        dfa_dt_h=1.0,
        output_time_resolution_h=1.0,
    )
    base.update(overrides)
    return VirtualPatientConfig(**base)


class TestVirtualPatientEndToEnd:
    @pytest.fixture(scope="class")
    def result(self) -> VirtualPatientResult:
        """One shared short-horizon run for all end-to-end assertions."""
        return VirtualPatient(_make_diabetes_config()).run()

    def test_run_returns_result_with_time_series(self, result):
        """run() produces hourly records covering the full horizon."""
        assert isinstance(result, VirtualPatientResult)
        assert len(result.time_h) == 25  # t=0..24 h inclusive at 1 h resolution
        assert result.time_h[0] == pytest.approx(0.0)
        assert result.time_h[-1] == pytest.approx(24.0)
        for series in (
            result.systolic_bp, result.diastolic_bp, result.heart_rate,
            result.temperature, result.weight_kg,
        ):
            assert len(series) == len(result.time_h)

    def test_result_has_clinical_lab_channels(self, result):
        """All lab channels recorded and aligned with the time grid."""
        lab_channels = (
            result.alt, result.ast, result.creatinine, result.egfr,
            result.wbc, result.hemoglobin, result.platelets,
            result.glucose, result.hba1c, result.crp,
            result.bilirubin, result.albumin, result.inr,
            result.sodium, result.potassium, result.lactate,
            result.calcium, result.phosphate, result.chloride,
            result.bicarbonate, result.ldl, result.hdl,
            result.triglycerides,
        )
        for series in lab_channels:
            assert len(series) == len(result.time_h)
        # Physiologically plausible spot checks
        assert result.glucose[0] > 80.0  # diabetic baseline above euglycemia
        assert 15.0 < result.alt[0] < 200.0
        assert 0.2 < result.creatinine[0] < 5.0
        assert 90.0 < result.sodium[0] < 150.0
        assert 60.0 < result.heart_rate[0] < 180.0

    def test_result_tracks_drug_concentrations(self, result):
        """Per-drug µM series start dosing immediately and accumulate AUC."""
        assert set(result.drug_concentrations) == {"metformin"}
        concs = result.drug_concentrations["metformin"]
        assert len(concs) == len(result.time_h)
        assert concs[0] > 0.0  # first dose absorbed by t=0 record
        assert max(concs) > 0.0
        assert result.auc_plasma["metformin"] > 0.0

    def test_result_tracks_disease_progression(self, result):
        """Disease severity/stage series align with the time grid."""
        assert len(result.disease_severity) == len(result.time_h)
        assert len(result.disease_stage) == len(result.time_h)
        assert any(s != "healthy" for s in result.disease_stage)

    def test_result_to_dict_is_json_safe(self, result):
        """to_dict() exposes vitals/labs/drugs/disease/summary sections."""
        data = result.to_dict()
        assert isinstance(data, dict)
        for key in ("time_h", "vitals", "labs", "drug_concentrations",
                    "disease", "summary"):
            assert key in data
        labs = data["labs"]
        for channel in ("alt_u_per_l", "creatinine_mg_per_dl",
                        "glucose_mg_per_dl"):
            assert channel in labs
            assert len(labs[channel]) == len(data["time_h"])
        assert "systolic_bp_mmhg" in data["vitals"]
        assert "metformin" in data["drug_concentrations"]

    def test_result_summary_renders_report(self, result):
        """summary() renders a human-readable multi-line report."""
        text = result.summary()
        assert isinstance(text, str)
        assert "Virtual Patient Simulation Summary" in text
        assert "--- Vitals ---" in text
        assert "--- Labs ---" in text
        assert "metformin" in text

    def test_summary_metrics_finalized(self, result):
        """Finalize pass computes toxicity counts and exposure metrics."""
        assert result.total_toxicity_events >= 0
        assert 0.0 <= result.overall_efficacy_score <= 1.0
        assert result.max_alt > 0.0
        assert result.max_creatinine > 0.0
        assert result.min_egfr > 0.0


class TestVirtualPatientHealthyBaseline:
    def test_no_disease_runs_and_records_healthy_stage(self):
        """A drug-free healthy patient still produces a full time-series."""
        config = _make_diabetes_config(
            disease=None, drugs=[], pharmacodynamics={},
            total_duration_days=0.5,
        )
        result = VirtualPatient(config).run()
        assert len(result.time_h) == 13  # t=0..12 h inclusive
        assert all(stage == "healthy" for stage in result.disease_stage)
        assert result.drug_concentrations == {}
        assert result.auc_plasma == {}

    def test_builder_pipeline_matches_programmatic_run(self):
        """Builders applied to parsed annotations yield a runnable config."""
        ext = parse_source(PATIENT_SOURCE).sim_extensions

        genotype = create_default_genotype()
        _build_genotype_from_helix(genotype, ext)
        config = VirtualPatientConfig(
            genotype=genotype,
            traits=_build_traits_from_helix(ext),
            disease=_build_disease_from_helix(ext),
            drugs=_build_drugs_from_helix(ext),
            pharmacodynamics=_build_pd_from_helix(ext),
            total_duration_days=0.5,
            output_time_resolution_h=1.0,
        )
        assert config.disease is not None
        assert config.disease.name == "type 2 diabetes"
        assert [d.molecule.name for d in config.drugs] == ["metformin"]
        assert "metformin" in config.pharmacodynamics

        result = VirtualPatient(config).run()
        assert len(result.time_h) == 13
        assert "metformin" in result.drug_concentrations
        assert result.glucose[0] > 80.0


# ============================================================================
# doc/30-31 new DSL annotations tests
# ============================================================================


QSP_SOURCE = """
#person name=Alice age=45 sex=female weight=65 height=165

#drug name=trastuzumab smiles="[BIOLOGIC]" formula="monoclonal_antibody" \
dose=8 dose_unit=mg_per_kg interval=7 interval_unit=days duration=84 \
route=iv vd=3.0 cl=0.2 half_life=336

#qsp_binding drug=trastuzumab kind=tmdd kss_nM=2.0 emax=0.9

#sim kind=human duration_days=14
"""


class TestQSPBindingDSL:
    def test_parser_qsp_binding(self):
        prog = parse_source(QSP_SOURCE)
        ext = prog.sim_extensions
        assert "qsp_bindings" in ext
        assert len(ext["qsp_bindings"]) == 1
        entry = ext["qsp_bindings"][0]
        assert entry["drug"] == "trastuzumab"
        assert entry["kind"] == "tmdd"
        assert float(entry["kss_nM"]) == 2.0

    def test_builder_qsp_bindings(self):
        from helixlang.sim_runtime import _build_qsp_bindings_from_helix
        prog = parse_source(QSP_SOURCE)
        ext = prog.sim_extensions
        result = _build_qsp_bindings_from_helix(ext)
        assert "qsp_bindings" in result
        assert len(result["qsp_bindings"]) == 1
        assert result["qsp_bindings"][0]["kind"] == "tmdd"


ENDOCRINE_SOURCE = """
#person name=Bob age=60 sex=male weight=90 height=178

#endocrine_config axis=diabetes severity=0.6

#sim kind=human duration_days=7
"""


class TestEndocrineConfigDSL:
    def test_parser_endocrine_config(self):
        prog = parse_source(ENDOCRINE_SOURCE)
        ext = prog.sim_extensions
        assert "endocrine_configs" in ext
        assert len(ext["endocrine_configs"]) == 1
        entry = ext["endocrine_configs"][0]
        assert entry["axis"] == "diabetes"
        assert float(entry["severity"]) == 0.6

    def test_builder_endocrine_config(self):
        from helixlang.sim_runtime import _build_endocrine_config_from_helix
        prog = parse_source(ENDOCRINE_SOURCE)
        ext = prog.sim_extensions
        result = _build_endocrine_config_from_helix(ext)
        assert "endocrine_configs" in result
        assert result["endocrine_configs"][0]["axis"] == "diabetes"


IMMUNE_SOURCE = """
#person name=Carol age=50 sex=female weight=70 height=160

#immune_config infection_severity=0.5 immunosuppression=0.2

#sim kind=human duration_days=7
"""


class TestImmuneConfigDSL:
    def test_parser_immune_config(self):
        prog = parse_source(IMMUNE_SOURCE)
        ext = prog.sim_extensions
        assert "immune_configs" in ext
        assert len(ext["immune_configs"]) == 1
        entry = ext["immune_configs"][0]
        assert float(entry["infection_severity"]) == 0.5
        assert float(entry["immunosuppression"]) == 0.2

    def test_builder_immune_config(self):
        from helixlang.sim_runtime import _build_immune_config_from_helix
        prog = parse_source(IMMUNE_SOURCE)
        ext = prog.sim_extensions
        result = _build_immune_config_from_helix(ext)
        assert "immune_configs" in result
        assert result["immune_configs"][0]["infection_severity"] == 0.5


class TestNewVirtualPatientChannels:
    """Verify new doc/30-31 output channels exist in VirtualPatientResult."""

    def test_endocrine_channels_populated(self):
        config = _make_diabetes_config(
            disease=DiseaseState(
                name="type 2 diabetes",
                category="metabolic_overload",
                severity=0.7,
            ),
            drugs=[],
            pharmacodynamics={},
            total_duration_days=1.0,
        )
        result = VirtualPatient(config).run()
        assert len(result.cortisol) == len(result.time_h)
        assert len(result.insulin) == len(result.time_h)
        assert len(result.glucose_endocrine) == len(result.time_h)
        assert len(result.tsh) == len(result.time_h)
        assert len(result.ft4) == len(result.time_h)
        assert all(c > 0 for c in result.cortisol)
        assert all(i > 0 for i in result.insulin)

    def test_immune_channels_populated(self):
        config = _make_diabetes_config(
            disease=None, drugs=[], pharmacodynamics={},
            total_duration_days=1.0,
        )
        result = VirtualPatient(config).run()
        assert len(result.il6) == len(result.time_h)
        assert len(result.tnf_alpha) == len(result.time_h)
        assert len(result.neutrophils) == len(result.time_h)
        assert all(v >= 0 for v in result.il6)
        assert all(v >= 0 for v in result.tnf_alpha)
        assert all(v > 0 for v in result.neutrophils)

    def test_disease_ode_channels_populated(self):
        config = _make_diabetes_config(
            disease=DiseaseState(
                name="type 2 diabetes",
                category="metabolic_overload",
                severity=0.7,
            ),
            drugs=[], pharmacodynamics={},
            total_duration_days=1.0,
        )
        result = VirtualPatient(config).run()
        assert len(result.beta_cell_function) == len(result.time_h)
        assert all(0 < b <= 1.0 for b in result.beta_cell_function)

    def test_to_dict_includes_new_sections(self):
        config = _make_diabetes_config(
            disease=None, drugs=[], pharmacodynamics={},
            total_duration_days=0.5,
        )
        result = VirtualPatient(config).run()
        d = result.to_dict()
        assert "endocrine" in d
        assert "immune" in d
        assert "disease_ode" in d
        assert "cortisol_ug_dl" in d["endocrine"]
        assert "il6_pg_ml" in d["immune"]
        assert "tumor_volume" in d["disease_ode"]
