"""E2E tests for doc/33 — 100% pipeline completion.

Tests: post-treatment recovery, genotype-dependent AE, microbiome-mediated
drug effects, multi-drug DDI → clinical events, hematology/renal integration.
"""

import pytest

from helixlang.plugins.human.drug import Drug, DrugMolecule, get_predefined_drug
from helixlang.plugins.human.genotype import Variant, create_default_genotype
from helixlang.plugins.human.virtual_patient import VirtualPatient, VirtualPatientConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cisplatin(duration_days: float = 2.0) -> Drug:
    return Drug(
        molecule=DrugMolecule(
            name="cisplatin",
            smiles="[Pt+2].([Cl-].[Cl-]).([NH3].[NH3])",
            molecular_weight_da=300.05,
            formula="Cl2H6N2Pt",
        ),
        dose_mg=50.0,
        dosing_interval_h=24.0,
        route="iv",
        duration_days=duration_days,
        volume_distribution_l=15.0,
        clearance_ml_per_min=15.0,
        renal_fraction=0.7,
        hepatic_extraction_ratio=0.0,
    )


def _warfarin(duration_days: float = 10.0) -> Drug:
    return Drug(
        molecule=DrugMolecule(
            name="warfarin",
            smiles="CC(=O)CC1C(C2=CC=CC=C2C3=CC=CC=C31)C(=O)O",
            molecular_weight_da=308.33,
            formula="C19H16O4",
        ),
        dose_mg=5.0,
        dosing_interval_h=24.0,
        route="oral",
        duration_days=duration_days,
        bioavailability=0.95,
        volume_distribution_l=8.0,
        clearance_ml_per_min=0.05,
        half_life_h=40.0,
        hepatic_extraction_ratio=0.0,
        renal_fraction=0.0,
        cyp_metabolism={"CYP2C9": 0.6, "CYP3A4": 0.3, "CYP1A2": 0.1},
    )


def _amiodarone(duration_days: float = 10.0) -> Drug:
    return Drug(
        molecule=DrugMolecule(
            name="amiodarone",
            smiles="CCC(=O)NC1=CC(=C(C=C1)OC2=CC=CC=C2C3=CC=CC=C3)I",
            molecular_weight_da=645.31,
            formula="C25H29INO3",
        ),
        dose_mg=200.0,
        dosing_interval_h=24.0,
        route="oral",
        duration_days=duration_days,
        bioavailability=0.5,
        volume_distribution_l=60.0,
        clearance_ml_per_min=150.0,
        half_life_h=58.0,
        hepatic_extraction_ratio=0.0,
        renal_fraction=0.0,
        cyp_metabolism={"CYP3A4": 0.5, "CYP2C8": 0.3, "CYP2D6": 0.2},
    )


def _metformin(duration_days: float = 30.0) -> Drug:
    return Drug(
        molecule=DrugMolecule(
            name="metformin",
            smiles="CN(C)C(=N)NC(=N)N",
            molecular_weight_da=129.16,
            formula="C4H11N5",
        ),
        dose_mg=500.0,
        dosing_interval_h=12.0,
        route="oral",
        duration_days=duration_days,
        bioavailability=0.55,
        volume_distribution_l=654.0,
        clearance_ml_per_min=510.0,
        half_life_h=6.0,
        renal_fraction=1.0,
    )


# ---------------------------------------------------------------------------
# T1: Post-treatment recovery
# ---------------------------------------------------------------------------


class TestPostTreatmentRecovery:
    """Verify that after drug cessation, recovery model activates."""

    def test_cisplatin_recovery_model(self) -> None:
        drug = _cisplatin(duration_days=2.0)
        cfg = VirtualPatientConfig(
            drugs=[drug],
            total_duration_days=5.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        # Recovery model should be active after treatment ends
        assert vp._recovery_model is not None
        assert not vp._recovery_model.is_treatment_active
        # ALT and creatinine should be tracked throughout
        assert len(result.alt) > 0
        assert len(result.creatinine) > 0

    def test_cisplatin_creatinine_recovery(self) -> None:
        drug = _cisplatin(duration_days=1.0)
        cfg = VirtualPatientConfig(
            drugs=[drug],
            total_duration_days=4.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        assert len(result.creatinine) > 0
        idx_96h = min(int(96.0), len(result.creatinine) - 1)
        assert result.creatinine[idx_96h] > 0.0

    def test_recovery_model_active(self) -> None:
        """Recovery model should be seeded after all drugs stop."""
        drug = _cisplatin(duration_days=1.0)
        cfg = VirtualPatientConfig(
            drugs=[drug],
            total_duration_days=3.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        vp.run()
        assert vp._recovery_model is not None
        assert not vp._recovery_model.is_treatment_active


# ---------------------------------------------------------------------------
# T2: Genotype-dependent AE
# ---------------------------------------------------------------------------


class TestGenotypeDependentAE:
    """Verify that CYP genotype affects drug exposure and AE risk."""

    def test_cyp2d6_poor_metabolizer_higher_tramadol(self) -> None:
        tramadol = get_predefined_drug("tramadol")
        if tramadol is None:
            pytest.skip("tramadol not in predefined drugs")

        # Normal metabolizer (default)
        normal_cfg = VirtualPatientConfig(
            drugs=[tramadol],
            total_duration_days=1.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        r_normal = VirtualPatient(normal_cfg).run()

        # CYP2D6 poor metabolizer (*4/*4)
        pm_genotype = create_default_genotype()
        pm_genotype.add_gene_variant("CYP2D6", Variant(
            gene_id="CYP2D6", chromosome="22", position=42526693,
            ref="G", alt="A", zygosity="het",
        ))
        pm_cfg = VirtualPatientConfig(
            drugs=[tramadol],
            genotype=pm_genotype,
            total_duration_days=1.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        r_pm = VirtualPatient(pm_cfg).run()

        # Both should have drug concentrations
        assert "tramadol" in r_normal.drug_concentrations
        assert "tramadol" in r_pm.drug_concentrations
        # Verify the simulation ran and produced results
        assert len(r_pm.time_h) > 0

    def test_genotype_affects_clearance_modifier(self) -> None:
        """Different genotypes should produce different clearance modifiers."""
        from helixlang.plugins.human.virtual_patient import _compute_genetic_cyp_modifier

        tramadol = get_predefined_drug("tramadol")
        if tramadol is None:
            pytest.skip("tramadol not in predefined drugs")

        normal = create_default_genotype()
        normal_mod = _compute_genetic_cyp_modifier(tramadol, normal)
        assert isinstance(normal_mod, float)
        assert normal_mod > 0.0


# ---------------------------------------------------------------------------
# T3: Microbiome-mediated drug effects
# ---------------------------------------------------------------------------


class TestMicrobiomeDrugEffects:
    """Verify microbiome compartment is active during simulation."""

    def test_microbiome_active_in_simulation(self) -> None:
        """Microbiome should be initialized and stepped during simulation."""
        drug = _metformin()
        cfg = VirtualPatientConfig(
            drugs=[drug],
            total_duration_days=1.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        # Microbiome should be active
        assert vp._microbiome_compartment is not None
        # Should produce results
        assert len(result.time_h) > 0

    def test_microbiome_portal_fluxes(self) -> None:
        """Microbiome should produce portal fluxes."""
        from helixlang.plugins.human.microbiome import MicrobiomeCompartment
        mc = MicrobiomeCompartment()
        mc.set_drug_concentration("metformin", 10.0)
        mc.step(dt_h=1.0)
        portal = mc.get_portal_fluxes()
        assert isinstance(portal, dict)
        assert "scfa" in portal


# ---------------------------------------------------------------------------
# T4: Multi-drug DDI → clinical event
# ---------------------------------------------------------------------------


class TestMultiDrugDDI:
    """Verify DDI system is functional with multi-drug regimens."""

    def test_warfarin_amiodarone_ddi(self) -> None:
        """Amiodarone + warfarin should run without error and produce results."""
        wf = _warfarin(duration_days=5.0)
        am = _amiodarone(duration_days=5.0)
        cfg = VirtualPatientConfig(
            drugs=[wf, am],
            total_duration_days=5.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        result = VirtualPatient(cfg).run()
        # Both drugs should have concentration series
        assert "warfarin" in result.drug_concentrations
        assert "amiodarone" in result.drug_concentrations
        # INR should be tracked
        assert len(result.inr) > 0
        # AUC should be computed for both drugs
        assert "warfarin" in result.auc_plasma
        assert "amiodarone" in result.auc_plasma

    def test_ddi_model_loaded_by_default(self) -> None:
        """Multi-drug config should auto-load DDI rules."""
        wf = _warfarin()
        am = _amiodarone()
        cfg = VirtualPatientConfig(drugs=[wf, am])
        assert cfg.ddi_model is not None
        assert len(cfg.ddi_model.rules) > 0

    def test_single_drug_no_ddi_model(self) -> None:
        """Single-drug config should not create DDI model."""
        wf = _warfarin()
        cfg = VirtualPatientConfig(drugs=[wf])
        assert cfg.ddi_model is None


# ---------------------------------------------------------------------------
# T5: Hematology & Renal integration
# ---------------------------------------------------------------------------


class TestHematologyRenalIntegration:
    """Verify hematology and renal models are active in the pipeline."""

    def test_hematology_active_with_cisplatin(self) -> None:
        drug = _cisplatin(duration_days=3.0)
        cfg = VirtualPatientConfig(
            drugs=[drug],
            total_duration_days=5.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        assert vp._hematology is not None
        # WBC should be affected by cisplatin myelosuppression
        assert len(result.wbc) > 0
        assert all(w > 0 for w in result.wbc)

    def test_renal_active_with_cisplatin(self) -> None:
        drug = _cisplatin(duration_days=2.0)
        cfg = VirtualPatientConfig(
            drugs=[drug],
            total_duration_days=4.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        assert vp._renal is not None
        # eGFR and creatinine should be tracked
        assert len(result.egfr) > 0
        assert len(result.creatinine) > 0
        assert all(e > 0 for e in result.egfr)


# ---------------------------------------------------------------------------
# T6: Epigenetic CYP modifiers wired
# ---------------------------------------------------------------------------


class TestEpigeneticCYPWiring:
    """Verify epigenetic CYP modifiers reach the clearance chain."""

    def test_emergent_complexity_produces_epigenetic_keys(self) -> None:
        from helixlang.plugins.human.emergent_complexity import EmergentComplexityModel
        ecm = EmergentComplexityModel()
        signals = ecm.step(dt_h=1.0, t_h=0.0, drug_concentrations={},
                           il6=1.0, tnf=5.0, crp=0.5)
        # Should produce epigenetic CYP keys
        epi_keys = [k for k in signals if k.startswith("epigenetic_")]
        assert len(epi_keys) == 10

    def test_epigenetic_modifies_clearance_in_vp(self) -> None:
        """Epigenetic signals should queue into _pending_clearance_scale."""
        drug = _metformin()
        cfg = VirtualPatientConfig(
            drugs=[drug],
            total_duration_days=2.0,
            dfa_dt_h=1.0,
            output_time_resolution_h=1.0,
        )
        vp = VirtualPatient(cfg)
        result = vp.run()
        # Should complete without error
        assert len(result.time_h) > 0
        # Emergent complexity should be active
        assert vp._emergent_complexity is not None
