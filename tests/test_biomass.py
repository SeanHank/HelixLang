"""Taxonomy-aware biomass reaction construction tests (doc/20 §6.2).

Coverage goals:
- BiomassComponent.to_string formatting for reactants and products.
- _classify_organism across taxonomy buckets (archaea / yeast / gram+).
- _resolve_organism_key exact/partial/unknown matching.
- get_biomass_composition fallback and copy semantics.
- list_available_templates.
- build_biomass_reaction with and without custom composition.
"""
from __future__ import annotations

from helixlang.plugins.gem.biomass import (
    ECOLI_BIOMASS_COMPONENTS,
    BiomassComponent,
    BiomassReaction,
    _classify_organism,
    _resolve_organism_key,
    build_biomass_reaction,
    get_biomass_composition,
    list_available_templates,
)


class TestToString:
    def test_single_reactant_product(self) -> None:
        br = BiomassReaction(name="bm", equation="", components=[
            BiomassComponent("ATP_c", -1.0, "cofactor"),
            BiomassComponent("BIOMASS_c", 1.0, "cell_wall"),
        ])
        assert "1.0000 ATP_c" in br.to_string()
        assert "1.0000 BIOMASS_c" in br.to_string()
        assert "->" in br.to_string()

    def test_fractional_coefficients(self) -> None:
        br = BiomassReaction(name="bm", equation="", components=[
            BiomassComponent("ALA_c", -0.045, "amino_acid"),
        ])
        assert "0.0450 ALA_c" in br.to_string()


class TestClassifyOrganism:
    def test_gram_negative_default(self) -> None:
        assert _classify_organism("e_coli") == "gram_negative"
        assert _classify_organism("unknown_organism") == "gram_negative"

    def test_archaea(self) -> None:
        assert _classify_organism("s_solfataricus") == "archaea"

    def test_yeast(self) -> None:
        assert _classify_organism("s_cerevisiae") == "yeast"

    def test_gram_positive(self) -> None:
        assert _classify_organism("b_subtilis") == "gram_positive"


class TestResolveOrganismKey:
    def test_exact(self) -> None:
        assert _resolve_organism_key("e_coli_k12") == "e_coli_k12"

    def test_partial(self) -> None:
        assert _resolve_organism_key("k12") is not None

    def test_unknown(self) -> None:
        assert _resolve_organism_key("no_such_organism_xyz") is None


class TestGetBiomassComposition:
    def test_known(self) -> None:
        comps = get_biomass_composition("e_coli_k12")
        assert all(isinstance(c, BiomassComponent) for c in comps)
        assert len(comps) > 0
        assert not any(c.metabolite_id == "" for c in comps)

    def test_unknown_falls_back_ecoli(self) -> None:
        comps = get_biomass_composition("zzz_unknown")
        assert list(comps) == list(ECOLI_BIOMASS_COMPONENTS)


class TestListTemplates:
    def test_returns_mapping(self) -> None:
        tmpl = list_available_templates()
        assert isinstance(tmpl, dict)
        assert "e_coli" in tmpl


class TestBuildBiomassReaction:
    def test_default(self) -> None:
        rxn = build_biomass_reaction("e_coli_k12")
        assert isinstance(rxn, BiomassReaction)
        assert rxn.equation
        assert len(rxn.components) == len(ECOLI_BIOMASS_COMPONENTS)

    def test_custom_composition(self) -> None:
        custom = [BiomassComponent("XYZ_c", -1.0, "cofactor")]
        rxn = build_biomass_reaction("whatever", custom_composition=custom)
        assert rxn.components == custom
