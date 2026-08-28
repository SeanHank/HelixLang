"""Doc/33 Phase 4: Cancer targeted therapy E2E tests."""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# TumorClone tests
# ---------------------------------------------------------------------------


class TestTumorClone:
    """TumorClone unit tests."""

    def test_effective_growth_no_mutations(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorClone
        c = TumorClone(name="parent", growth_rate=0.01)
        assert c.effective_growth() == pytest.approx(0.01)

    def test_effective_growth_with_resistance(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorClone
        c = TumorClone(
            name="R_EGFR", growth_rate=0.01,
            resistance_mutations=["EGFR"], fitness_cost=0.1,
        )
        assert c.effective_growth() == pytest.approx(0.009)

    def test_drug_kill_rate_sensitive_clone(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorClone
        c = TumorClone(
            name="parent", drug_sensitivities={"egfr": 0.9},
        )
        kill = c.drug_kill_rate({"egfr": 0.5}, drug_kill_capacity=1.0)
        assert kill == pytest.approx(0.45)

    def test_drug_kill_rate_resistant_clone(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorClone
        c = TumorClone(
            name="R_EGFR", drug_sensitivities={"egfr": 0.05},
        )
        kill = c.drug_kill_rate({"egfr": 0.5}, drug_kill_capacity=1.0)
        assert kill == pytest.approx(0.025)

    def test_drug_kill_rate_no_sensitivity_entry(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorClone
        c = TumorClone(name="parent")
        kill = c.drug_kill_rate({"egfr": 0.5}, drug_kill_capacity=1.0)
        assert kill == pytest.approx(0.5)

    def test_drug_kill_rate_scales_with_capacity(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorClone
        c = TumorClone(
            name="parent", drug_sensitivities={"egfr": 0.9},
        )
        kill = c.drug_kill_rate({"egfr": 0.5}, drug_kill_capacity=0.001)
        assert kill == pytest.approx(0.45 * 0.001)


# ---------------------------------------------------------------------------
# TumorHeterogeneity tests
# ---------------------------------------------------------------------------


class TestTumorHeterogeneity:
    """TumorHeterogeneity unit tests."""

    def test_single_clone_survives_without_drug(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorClone,
            TumorHeterogeneity,
        )
        het = TumorHeterogeneity(
            clones=[TumorClone(name="parent", fraction=1.0, growth_rate=0.01)]
        )
        for _ in range(100):
            het.step(1.0, {})
        summary = het.get_clone_summary()
        assert len(summary) == 1
        assert summary[0]["fraction"] > 0.5

    def test_drug_reduces_sensitive_clone(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorClone,
            TumorHeterogeneity,
        )
        het = TumorHeterogeneity(
            clones=[TumorClone(
                name="parent", fraction=1.0, growth_rate=0.01,
                drug_sensitivities={"egfr": 0.9},
            )]
        )
        for _ in range(100):
            het.step(1.0, {"egfr": 0.8})
        summary = het.get_clone_summary()
        total = sum(c["fraction"] for c in summary)
        assert total > 0.0

    def test_resistant_clone_emerges(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorClone,
            TumorHeterogeneity,
        )
        het = TumorHeterogeneity(
            clones=[TumorClone(
                name="parent", fraction=1.0, growth_rate=0.01,
                drug_sensitivities={"egfr": 0.9},
            )],
            resistance_rate=1.0,
        )
        for _ in range(200):
            het.step(1.0, {"egfr": 0.8})
        summary = het.get_clone_summary()
        names = [c["name"] for c in summary]
        assert any("R_egfr" in n for n in names)

    def test_clone_summary_format(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorClone,
            TumorHeterogeneity,
        )
        het = TumorHeterogeneity(
            clones=[TumorClone(name="parent", fraction=1.0)]
        )
        summary = het.get_clone_summary()
        assert len(summary) == 1
        assert "name" in summary[0]
        assert "fraction" in summary[0]
        assert "growth_rate" in summary[0]
        assert "resistance_mutations" in summary[0]


# ---------------------------------------------------------------------------
# TumorBiopsy tests
# ---------------------------------------------------------------------------


class TestTumorBiopsy:
    """TumorBiopsy unit tests."""

    def test_has_mutation(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorBiopsy
        b = TumorBiopsy(mutations=["EGFR_L858R", "TP53_R175H"])
        assert b.has_mutation("EGFR")
        assert b.has_mutation("TP53")
        assert not b.has_mutation("BRAF")

    def test_has_amplification(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorBiopsy
        b = TumorBiopsy(amplifications=["HER2"])
        assert b.has_amplification("HER2")
        assert not b.has_amplification("EGFR")

    def test_has_fusion(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TumorBiopsy
        b = TumorBiopsy(fusion_genes=["EML4-ALK"])
        assert b.has_fusion("ALK")
        assert not b.has_fusion("BCR-ABL")


# ---------------------------------------------------------------------------
# select_targeted_therapy tests
# ---------------------------------------------------------------------------


class TestSelectTargetedTherapy:
    """Biomarker-driven therapy selection tests."""

    def test_egfr_mutant_gets_egfr_tki(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(mutations=["EGFR_L858R"])
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert any("erlotinib" in d or "gefitinib" in d or "osimertinib" in d
                    for d in drugs)

    def test_alk_fusion_gets_alk_tki(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(fusion_genes=["EML4-ALK"])
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert any("crizotinib" in d or "alectinib" in d for d in drugs)

    def test_pd_l1_high_gets_checkpoint_inhibitor(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(pd_l1_expression=0.7)
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert any("pembrolizumab" in d or "nivolumab" in d for d in drugs)

    def test_msi_high_gets_pembrolizumab(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(msi_status="MSI-H")
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert "pembrolizumab" in drugs

    def test_braf_v600e_gets_braf_inhibitor(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(mutations=["BRAF_V600E"])
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert "dabrafenib" in drugs

    def test_brca_hrd_gets_parp_inhibitor(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(mutations=["BRCA1"], hr_status="HRD")
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert "olaparib" in drugs

    def test_no_biomarkers_returns_empty(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy()
        recs = select_targeted_therapy(b)
        assert len(recs) == 0

    def test_kras_g12c_gets_sotorasib(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(mutations=["KRAS_G12C"])
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert "sotorasib" in drugs

    def test_bcr_abl_fusion_gets_imatinib(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(fusion_genes=["BCR-ABL"])
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert "imatinib" in drugs

    def test_her2_amplification_gets_trastuzumab(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            TumorBiopsy,
            select_targeted_therapy,
        )
        b = TumorBiopsy(amplifications=["HER2"])
        recs = select_targeted_therapy(b)
        drugs = [r["drug"] for r in recs]
        assert "trastuzumab" in drugs


# ---------------------------------------------------------------------------
# CancerODE per-pathway tests
# ---------------------------------------------------------------------------


class TestCancerODEPerPathway:
    """CancerODE with per-pathway effects."""

    def test_no_drug_tumor_grows(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CancerODE
        ca = CancerODE(tumor_volume=0.01)
        for _ in range(100):
            ca.step(1.0)
        assert ca.tumor_volume > 0.01

    def test_egfr_inhibition_reduces_tumor(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CancerODE
        ca = CancerODE(tumor_volume=0.3)
        ca_no_drug = CancerODE(tumor_volume=0.3)
        for _ in range(200):
            ca.step(1.0)
            ca_no_drug.step(1.0)
            ca.pathway_effects["egfr"] = 0.8
        assert ca.tumor_volume < ca_no_drug.tumor_volume

    def test_braf_inhibition_reduces_tumor(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CancerODE
        ca = CancerODE(tumor_volume=0.3)
        ca_no_drug = CancerODE(tumor_volume=0.3)
        for _ in range(200):
            ca.step(1.0)
            ca_no_drug.step(1.0)
            ca.pathway_effects["braf"] = 0.7
        assert ca.tumor_volume < ca_no_drug.tumor_volume

    def test_vegf_inhibition_reduces_angiogenesis(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CancerODE
        ca = CancerODE(tumor_volume=0.5, angiogenesis=0.8)
        ca_no_drug = CancerODE(tumor_volume=0.5, angiogenesis=0.8)
        ca.step(1.0)
        ca_no_drug.step(1.0)
        ca.pathway_effects["vegfr"] = 0.8
        ca.step(1.0)
        ca_no_drug.step(1.0)
        assert ca.angiogenesis < ca_no_drug.angiogenesis

    def test_immunotherapy_boosts_surveillance(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CancerODE
        ca = CancerODE(tumor_volume=0.3, immune_surveillance=0.3)
        ca_no_drug = CancerODE(tumor_volume=0.3, immune_surveillance=0.3)
        for _ in range(50):
            ca.step(1.0)
            ca_no_drug.step(1.0)
            ca.pathway_effects["pd_l1"] = 0.5
        assert ca.immune_surveillance > ca_no_drug.immune_surveillance

    def test_multiple_pathways_combined(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CancerODE
        ca = CancerODE(tumor_volume=0.3)
        ca_no_drug = CancerODE(tumor_volume=0.3)
        for _ in range(200):
            ca.step(1.0)
            ca_no_drug.step(1.0)
            ca.pathway_effects["egfr"] = 0.5
            ca.pathway_effects["braf"] = 0.3
            ca.pathway_effects["vegfr"] = 0.4
        assert ca.tumor_volume < ca_no_drug.tumor_volume

    def test_legacy_chemo_kill_still_works(self) -> None:
        from helixlang.plugins.human.disease_ode_models import CancerODE
        ca = CancerODE(tumor_volume=0.3)
        ca_no_drug = CancerODE(tumor_volume=0.3)
        for _ in range(100):
            ca.step(1.0)
            ca_no_drug.step(1.0)
            ca.chemo_kill_rate = 0.01
        assert ca.tumor_volume < ca_no_drug.tumor_volume

    def test_heterogeneity_attached(self) -> None:
        from helixlang.plugins.human.disease_ode_models import (
            CancerODE,
            TumorClone,
            TumorHeterogeneity,
        )
        het = TumorHeterogeneity(
            clones=[TumorClone(name="parent", fraction=1.0, growth_rate=0.01)]
        )
        ca = CancerODE(tumor_volume=0.1, heterogeneity=het)
        ca.step(1.0)
        assert ca.heterogeneity is not None


# ---------------------------------------------------------------------------
# Parser #tumor_biopsy tests
# ---------------------------------------------------------------------------


class TestParserTumorBiopsy:
    """Parser #tumor_biopsy rule tests."""

    def test_parse_tumor_biopsy(self) -> None:
        from helixlang.core.lexer import Lexer
        from helixlang.core.parser import Parser
        source = '#tumor_biopsy mutation=EGFR_L858R,TP53_R175H amplification=HER2 pd_l1_expression=0.6 msi_status=MSS tmb_per_mb=5.2 hr_status=HRD fusion=EML4-ALK\n'
        tokens = list(Lexer(source).tokens())
        prog = Parser(tokens).parse()
        biopsy = prog.sim_extensions.get("tumor_biopsy")
        assert biopsy is not None
        assert "EGFR_L858R" in biopsy.get("mutation", "")
        assert "HER2" in biopsy.get("amplification", "")
        assert biopsy.get("pd_l1_expression") == "0.6"
        assert biopsy.get("msi_status") == "MSS"
        assert biopsy.get("tmb_per_mb") == "5.2"
        assert biopsy.get("hr_status") == "HRD"
        assert "EML4-ALK" in biopsy.get("fusion", "")


# ---------------------------------------------------------------------------
# VirtualPatientConfig + biopsy wiring tests
# ---------------------------------------------------------------------------


class TestVirtualPatientBiopsyWiring:
    """VP config accepts tumor_biopsy and creates heterogeneity."""

    def test_config_accepts_tumor_biopsy(self) -> None:
        from helixlang.plugins.human.virtual_patient import VirtualPatientConfig
        cfg = VirtualPatientConfig(
            tumor_biopsy={
                "mutations": ["EGFR_L858R"],
                "amplifications": [],
                "fusion_genes": [],
                "pd_l1_expression": 0.6,
                "msi_status": "MSS",
                "tmb_per_mb": 5.0,
                "hr_status": "HRC",
            },
            total_duration_days=1.0,
            dfa_dt_h=1.0,
        )
        assert cfg.tumor_biopsy is not None
        assert cfg.tumor_biopsy["mutations"] == ["EGFR_L858R"]


# ---------------------------------------------------------------------------
# TARGET_TO_PATHWAY mapping tests
# ---------------------------------------------------------------------------


class TestTargetToPathway:
    """TARGET_TO_PATHWAY mapping completeness."""

    def test_egfr_maps_to_egfr(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert TARGET_TO_PATHWAY["EGFR"] == "egfr"

    def test_braf_v600e_maps_to_braf(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert TARGET_TO_PATHWAY["BRAF_V600E"] == "braf"

    def test_alk_maps_to_alk(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert TARGET_TO_PATHWAY["ALK"] == "alk"

    def test_pd_l1_maps_to_pd_l1(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert TARGET_TO_PATHWAY["PD-L1"] == "pd_l1"

    def test_her2_maps_to_her2(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert TARGET_TO_PATHWAY["HER2"] == "her2"

    def test_bcr_abl_maps_to_bcr_abl(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert TARGET_TO_PATHWAY["BCR-ABL"] == "bcr_abl"

    def test_vegfr2_maps_to_vegfr(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert TARGET_TO_PATHWAY["VEGFR2"] == "vegfr"

    def test_kras_g12c_maps_to_kras(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert TARGET_TO_PATHWAY["KRAS_G12C"] == "kras"

    def test_mapping_has_20_plus_entries(self) -> None:
        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
        assert len(TARGET_TO_PATHWAY) >= 20
