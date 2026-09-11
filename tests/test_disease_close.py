"""Branch-completion tests for helixlang.plugins.human.disease."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from helixlang.plugins.human.disease import (
    DiseaseState,
    GenePerturbation,
    MetabolitePerturbation,
    _gpr_blocked,
    _resolve_affected_reactions,
    apply_disease_state,
)
from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction


def _toy_model() -> MetabolicModel:
    model = MetabolicModel()
    model.add_reaction(Reaction(
        id="PAHG", name="Phe hydroxylase", stoichiometry={"phe": -1.0},
        lower_bound=0.0, upper_bound=100.0, subsystem="amino_acids",
        gene_reaction_rule="PAH",
    ))
    model.add_reaction(Reaction(
        id="PAHR", name="reversible PAH", stoichiometry={"phe_r": -1.0},
        lower_bound=-100.0, upper_bound=100.0, subsystem="amino_acids",
        gene_reaction_rule="PAH",
    ))
    model.add_reaction(Reaction(
        id="ISO", name="isozyme", stoichiometry={"x": -1.0},
        lower_bound=0.0, upper_bound=100.0, subsystem="amino_acids",
        gene_reaction_rule="OTHER",
    ))
    model.add_reaction(Reaction(
        id="EX_phe", name="phe export",
        stoichiometry={"phe": -1.0}, lower_bound=-100.0, upper_bound=100.0,
        subsystem="exchange",
    ))
    model.add_reaction(Reaction(
        id="EX_tyr", name="tyr export",
        stoichiometry={"tyr": -1.0}, lower_bound=-100.0, upper_bound=100.0,
        subsystem="transport",
    ))
    return model


class TestDiseaseValidations:
    def test_gene_bad_type(self):
        with pytest.raises(ValueError, match="perturbation_type"):
            GenePerturbation("PAH", "silence", 0.0)

    def test_gene_bad_fraction(self):
        with pytest.raises(ValueError, match="activity_fraction"):
            GenePerturbation("PAH", "knockout", 1.5)

    def test_metabolite_bad_type(self):
        with pytest.raises(ValueError, match="perturbation_type"):
            MetabolitePerturbation("phe", "zap")

    def test_metabolite_bad_restriction(self):
        with pytest.raises(ValueError, match="transport_restriction"):
            MetabolitePerturbation("phe", "accumulate",
                                   transport_restriction=2.0)


class TestResolveAffectedReactions:
    def test_gene_rules_dict(self):
        model = _toy_model()
        model.genes = {"g1": SimpleNamespace(protein_reaction_rules=["RX1"])}
        assert _resolve_affected_reactions(model, "g1") == ["RX1"]

    def test_gene_rules_dict_empty_entry(self):
        model = _toy_model()
        model.genes = {"g1": SimpleNamespace(protein_reaction_rules=[])}
        assert _resolve_affected_reactions(model, "g1") == []

    def test_gene_rules_dict_missing_entry(self):
        model = _toy_model()
        model.genes = {"g1": SimpleNamespace(protein_reaction_rules=["RX1"])}
        assert _resolve_affected_reactions(model, "g9") == []

    def test_no_genes_attribute(self):
        model = _toy_model()
        model.genes = None
        model.gene_reactions = {"direct": ["GX"]}
        assert _resolve_affected_reactions(model, "GX") == ["direct"]

    def test_raw_map_direct_list(self):
        model = _toy_model()
        model.gene_reactions = {"bad": ["RXA"]}
        assert _resolve_affected_reactions(model, "bad") == ["RXA"]

    def test_raw_map_scanned(self):
        model = _toy_model()
        model.gene_reactions = {"surrogate": ["scan_me"], "empty": None}
        out = _resolve_affected_reactions(model, "scan_me")
        assert "surrogate" in out

    def test_gpr_scan_hit(self):
        model = _toy_model()
        out = _resolve_affected_reactions(model, "PAH")
        assert {"PAHG", "PAHR"} <= set(out)

    def test_ecoli_fallback(self):
        model = _toy_model()
        out = _resolve_affected_reactions(model, "ptsG")
        assert out == ["GLCpts"]


class TestGprBlocked:
    def test_empty_rule(self):
        assert _gpr_blocked("  ", {"PAH"}) is True

    def test_isozyme_compensates(self):
        assert _gpr_blocked("PAH or OTHER", {"PAH"}) is False

    def test_all_blocked(self):
        assert _gpr_blocked("PAH and PRS", {"PAH", "PRS"}) is True

    def test_quoted_genes(self):
        assert _gpr_blocked("'PAH' and 'PRS'", {"PAH"}) is True


class TestApplyDiseaseStateSides:
    def test_missing_reaction_skipped(self):
        gp = GenePerturbation(
            "NONE", "knockout", 0.0, affected_reactions=["NOPE"])
        out = apply_disease_state(_toy_model(), DiseaseState(
            name="x", category="enzyme_deficiency", gene_perturbations=[gp]))
        assert out.reactions["PAHG"].upper_bound == pytest.approx(100.0)

    def test_overexpress_bounds(self):
        gp = GenePerturbation(
            "PAH", "overexpress", 0.5, affected_reactions=["PAHG"])
        out = apply_disease_state(_toy_model(), DiseaseState(
            name="x", category="cancer_metabolism", gene_perturbations=[gp],
            severity=1.0))
        assert out.reactions["PAHG"].upper_bound == pytest.approx(200.0)

    def test_isozyme_keeps_reaction_open(self):
        gp = GenePerturbation(
            "PAH", "knockout", 0.0, affected_reactions=["ISO"])
        out = apply_disease_state(_toy_model(), DiseaseState(
            name="x", category="enzyme_deficiency", gene_perturbations=[gp],
            severity=1.0))
        assert out.reactions["ISO"].upper_bound == pytest.approx(100.0)

    def test_reversible_knockout_zeroes_both_bounds(self):
        gp = GenePerturbation(
            "PAH", "knockout", 0.0, affected_reactions=["PAHR"])
        out = apply_disease_state(_toy_model(), DiseaseState(
            name="x", category="enzyme_deficiency", gene_perturbations=[gp],
            severity=1.0))
        assert out.reactions["PAHR"].upper_bound == pytest.approx(0.0)
        assert out.reactions["PAHR"].lower_bound == pytest.approx(0.0)

    def test_block_export_scales_exchange(self):
        mp = MetabolitePerturbation("phe", "block_export")
        out = apply_disease_state(_toy_model(), DiseaseState(
            name="x", category="enzyme_deficiency",
            metabolite_perturbations=[mp]))
        assert out.reactions["EX_phe"].upper_bound == pytest.approx(0.0)
        assert out.reactions["EX_phe"].lower_bound == pytest.approx(0.0)
        assert out.reactions["EX_tyr"].upper_bound == pytest.approx(100.0)

    def test_partial_restriction_scales_exchange(self):
        mp = MetabolitePerturbation(
            "phe", "deplete", transport_restriction=0.5)
        out = apply_disease_state(_toy_model(), DiseaseState(
            name="x", category="enzyme_deficiency",
            metabolite_perturbations=[mp]))
        assert out.reactions["EX_phe"].upper_bound == pytest.approx(50.0)
        assert out.reactions["EX_phe"].lower_bound == pytest.approx(-50.0)
