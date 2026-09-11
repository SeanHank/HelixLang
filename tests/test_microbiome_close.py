"""Branch-completion tests for helixlang.plugins.human.microbiome."""
from __future__ import annotations

import pytest

from helixlang.plugins.human.microbiome import MicrobiomeCompartment


class TestAntibiotics:
    def test_ampicillin(self):
        micro = MicrobiomeCompartment()
        before = micro._species["Lactobacillus_sp."].abundance
        micro.apply_antibiotic("ampicillin", dose_mg=500.0)
        assert micro._species["Lactobacillus_sp."].abundance < before

    def test_ampicillin_missing_species(self):
        micro = MicrobiomeCompartment()
        del micro._species["Bifidobacterium_sp."]
        micro.apply_antibiotic("ampicillin")
        assert "Bifidobacterium_sp." not in micro._species

    def test_fluoroquinolone(self):
        micro = MicrobiomeCompartment()
        before = micro._species["E._coli"].abundance
        micro.apply_antibiotic("ciprofloxacin", dose_mg=250.0)
        assert micro._species["E._coli"].abundance < before

    def test_fluoroquinolone_missing_species(self):
        micro = MicrobiomeCompartment()
        del micro._species["Pseudomonas_sp."]
        micro.apply_antibiotic("moxifloxacin")
        assert "Pseudomonas_sp." not in micro._species

    def test_metronidazole(self):
        micro = MicrobiomeCompartment()
        before = micro._species["Clostridium_sp."].abundance
        micro.apply_antibiotic("metronidazole", dose_mg=250.0)
        assert micro._species["Clostridium_sp."].abundance < before

    def test_metronidazole_missing_species(self):
        micro = MicrobiomeCompartment()
        del micro._species["Bacteroides_sp."]
        micro.apply_antibiotic("metronidazole")
        assert "Bacteroides_sp." not in micro._species

    def test_vancomycin(self):
        micro = MicrobiomeCompartment()
        before = micro._species["Enterococcus_faecium"].abundance
        micro.apply_antibiotic("vancomycin", dose_mg=1000.0)
        assert micro._species["Enterococcus_faecium"].abundance < before

    def test_vancomycin_missing_species(self):
        micro = MicrobiomeCompartment()
        del micro._species["Lactobacillus_sp."]
        micro.apply_antibiotic("vancomycin")
        assert "Lactobacillus_sp." not in micro._species

    def test_generic_antibiotic(self):
        micro = MicrobiomeCompartment()
        before = micro._species["E._coli"].abundance
        micro.apply_antibiotic("clindamycin", dose_mg=300.0)
        assert micro._species["E._coli"].abundance < before


class TestDysbiosisAndRestore:
    def test_dysbiosis_present(self):
        micro = MicrobiomeCompartment()
        micro.induce_dysbiosis(severity=0.8)
        assert micro.state.diversity_index < 1.0

    def test_dysbiosis_missing_species(self):
        micro = MicrobiomeCompartment()
        for name in ("Lactobacillus_sp.", "E._coli"):
            del micro._species[name]
        micro.induce_dysbiosis(severity=0.5)
        assert micro.state.inflammation_score == pytest.approx(0.15)

    def test_restore_lacto(self):
        micro = MicrobiomeCompartment()
        before = micro._species["Lactobacillus_sp."].abundance
        micro.restore_microbiome("Lactobacillus")
        assert micro._species["Lactobacillus_sp."].abundance > before

    def test_restore_lacto_missing_species(self):
        micro = MicrobiomeCompartment()
        del micro._species["Lactobacillus_sp."]
        micro.restore_microbiome("Lactobacillus")
        assert "Lactobacillus_sp." not in micro._species

    def test_restore_bifido(self):
        micro = MicrobiomeCompartment()
        before = micro._species["Bifidobacterium_sp."].abundance
        micro.restore_microbiome("Bifidobacterium")
        assert micro._species["Bifidobacterium_sp."].abundance > before

    def test_restore_bifido_missing_species(self):
        micro = MicrobiomeCompartment()
        del micro._species["Bifidobacterium_sp."]
        micro.restore_microbiome("Bifidobacterium")
        assert "Bifidobacterium_sp." not in micro._species

    def test_restore_sacc(self):
        micro = MicrobiomeCompartment()
        micro.restore_microbiome("Saccharomyces")
        assert micro.state.diversity_index == pytest.approx(1.0, rel=1e-3)

    def test_restore_unknown(self):
        micro = MicrobiomeCompartment()
        micro.restore_microbiome("Streptococcus")
        assert micro.state.diversity_index == pytest.approx(1.0, rel=1e-3)

    def test_dysbiotic_initialization(self):
        micro = MicrobiomeCompartment(healthy_composition=False)
        assert micro._species["E._coli"].abundance == pytest.approx(0.35)
        assert micro.state.diversity_index == pytest.approx(0.6)

    def test_portal_fluxes(self):
        micro = MicrobiomeCompartment()
        fluxes = micro.get_portal_fluxes()
        assert set(fluxes) == {
            "scfa", "ammonia", "tma", "bile_acids", "lactate",
            "gut_permeability", "inflammation",
        }

    def test_overall_drug_effect_unknown(self):
        micro = MicrobiomeCompartment()
        effect = micro.get_overall_drug_effect("unknown_drug")
        assert effect.drug_name == "unknown_drug"
        assert effect.bioavailability_modifier == 1.0


class TestStep:
    def test_zero_concentration_skipped(self):
        micro = MicrobiomeCompartment()
        micro.set_drug_concentration("irinotecan", 0.0)
        effects = micro.step(1.0)
        assert effects == {}

    def test_activation_reaction(self):
        micro = MicrobiomeCompartment()
        micro.set_drug_concentration("codeine", 10.0)
        effects = micro.step(1.0)
        effect = effects["codeine"]
        assert effect.amount_activated_umol > 0.0
        assert effect.active_metabolite_generated == "morphine"

    def test_deactivation_raises_toxicity(self):
        micro = MicrobiomeCompartment()
        micro.set_drug_concentration("sn_38_glucuronide", 5.0)
        effects = micro.step(1.0)
        assert effects["sn_38_glucuronide"].toxicity_modifier > 1.0

    def test_no_microbial_reaction(self):
        micro = MicrobiomeCompartment()
        micro.set_drug_concentration("nonexistent_drug", 10.0)
        effects = micro.step(1.0)
        assert "nonexistent_drug" in effects
        assert effects["nonexistent_drug"].bioavailability_modifier == 1.0

    def test_drug_that_is_product_only(self):
        micro = MicrobiomeCompartment()
        micro.set_drug_concentration("morphine", 5.0)
        effects = micro.step(1.0)
        assert "morphine" in effects
        assert effects["morphine"].toxicity_modifier == 1.0
