"""Tests for pure-Python SBML Level 3 export (doc/24, plugins/gem/sbml_export.py).

Covers the XML helpers, compartment inference, reaction serialisation, objective
and GPR emission, plus the top-level string/file API — all against real
``MetabolicModel`` objects, with no cobra/libsbml dependency.
"""
from __future__ import annotations

from helixlang.plugins.gem.sbml_export import (
    _compartment_list,
    _esc,
    _extract_compartments,
    _gpr_annotations,
    _objective_element,
    _parameter_list,
    _reaction_element,
    _reaction_list,
    _species_list,
    export_sbml,
    model_to_sbml_string,
)
from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction


def _model():
    m = MetabolicModel()
    m.add_reaction(Reaction(
        id="R1", name="G6P -> F6P",
        stoichiometry={"G6P_c": -1.0, "F6P_c": 1.0},
        lower_bound=-1000.0, upper_bound=1000.0, subsystem="glycolysis",
    ))
    m.add_reaction(Reaction(
        id="EX_glc", name="Glucose exchange",
        stoichiometry={"glc_D_e": 1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="exchange",
    ))
    m.set_biomass("R1")
    return m


# ── escaping / header helpers ─────────────────────────────────────────────────

def test_esc_escapes_xml():
    assert _esc('a < b & "quoted" and \'single\'>') == (
        "a &lt; b &amp; &quot;quoted&quot; and &#x27;single&#x27;&gt;"
    )


def test_extract_compartments_from_suffixes():
    comps = _extract_compartments({"glc_D_e", "G6P_c", "plain"})
    assert comps["e"] == "compartment_e"
    assert comps["c"] == "compartment_c"
    # 'plain' has no valid suffix -> not a compartment, but c already present
    assert "c" in comps


def test_extract_compartments_default_cytosol():
    # a single bare id with no valid <3-char alnum suffix -> cytosol fallback
    assert _extract_compartments({"plain"}) == {"c": "cytosol"}


def test_compartment_list_sorted():
    out = _compartment_list({"e": "compartment_e", "c": "compartment_c"})
    assert out.count("<compartment") == 2
    # sorted: 'c' before 'e'
    assert out.index('id="c"') < out.index('id="e"')


# ── species list ───────────────────────────────────────────────────────────────

def test_species_list_assigns_compartment():
    mets = {"G6P_c", "glc_D_e", "plain"}
    comps = {"c": "compartment_c", "e": "compartment_e"}
    out = _species_list(mets, comps)
    assert 'species id="G6P_c"' in out
    assert 'compartment="c"' in out
    # 'plain' falls back to the first compartment in comps ('c')
    assert 'species id="plain"' in out and 'compartment="c"' in out


# ── parameter / reaction elements ──────────────────────────────────────────────

def test_parameter_list_has_three_params():
    out = _parameter_list(_model())
    assert out.count("<parameter") == 3


def test_reaction_element_reversible_reactants_products():
    rxn = _model().reactions["R1"]  # G6P_c -1, F6P_c +1, lower -1000
    out = _reaction_element(_model(), rxn)
    assert 'reversible="true"' in out
    assert 'species="G6P_c" stoichiometry="1.0"' in out
    assert 'species="F6P_c" stoichiometry="1.0"' in out
    assert 'id="lower_bound" value="-1000.0"' in out
    assert 'id="upper_bound" value="1000.0"' in out


def test_reaction_element_irreversible_single_product():
    m = MetabolicModel()
    m.add_reaction(Reaction(
        id="EX_glc", name="X", stoichiometry={"glc_D_e": 1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="exchange",
    ))
    out = _reaction_element(m, m.reactions["EX_glc"])
    assert 'reversible="false"' in out
    assert '<listOfReactants>' in out
    # single product -> no reactant speciesReference
    assert 'species="glc_D_e" stoichiometry="1.0"' in out
    assert out.count("<speciesReference") == 1


def test_reaction_list_sorted_by_id():
    out = _reaction_list(_model())
    # EX_glc sorts before R1
    assert out.index('reaction id="EX_glc"') < out.index('reaction id="R1"')
    assert out.startswith('    <listOfReactions>')
    assert out.endswith('    </listOfReactions>')


# ── annotations / objective ────────────────────────────────────────────────────

def test_gpr_annotations_emits_subsystem_per_reaction():
    out = _gpr_annotations(_model())
    assert "<rxn:subsystem>glycolysis</rxn:subsystem>" in out
    assert "<rxn:subsystem>exchange</rxn:subsystem>" in out
    assert out.count("<reaction") == 2


def test_objective_element_uses_biomass():
    out = _objective_element(_model())
    assert 'reaction reference="R1"' in out


def test_objective_element_falls_back_to_first_reaction():
    m = MetabolicModel()
    m.add_reaction(Reaction(id="RK", name="x", stoichiometry={"A": 1.0}))
    out = _objective_element(m)
    assert 'reaction reference="RK"' in out


def test_objective_element_empty_model_defaults_biomass():
    out = _objective_element(MetabolicModel())
    assert 'reaction reference="biomass"' in out


# ── top-level API ──────────────────────────────────────────────────────────────

def test_model_to_sbml_string_contains_all_sections():
    out = model_to_sbml_string(_model(), model_id="test_gem", organism_name="E coli")
    for marker in (
        'xmlns="http://www.sbml.org/sbml/level3/version1/core"',
        'id="test_gem"',
        'name="E coli"',
        "<listOfUnitDefinitions>",
        "<listOfCompartments>",
        "<listOfSpecies>",
        "<listOfParameters>",
        "<listOfReactions>",
        "<listOfObjectives>",
        "</sbml>",
    ):
        assert marker in out


def test_model_to_sbml_string_no_organism_name():
    out = model_to_sbml_string(_model(), model_id="m")
    # no name attribute when organism_name empty
    assert 'name="E coli"' not in out


def test_model_to_sbml_string_validish_xml():
    out = model_to_sbml_string(_model())
    # two listOfReactions blocks (data + GPR annotations), each balanced
    assert out.count("<listOfReactions>") == 2
    assert out.count("<reaction id=") == out.count("</reaction>")
    # species are emitted once each as self-closing elements
    assert out.count("<species id=") == 3


def test_export_sbml_writes_file(tmp_path):
    model = _model()
    path = tmp_path / "model.xml"
    export_sbml(model, str(path), model_id="exported", organism_name="Org")
    text = path.read_text(encoding="utf-8")
    assert "exported" in text
    assert "Org" in text
    assert text.startswith("<?xml")


def test_export_roundtrip_has_reaction_bounds():
    m = _model()
    text = model_to_sbml_string(m)
    assert 'value="-1000.0"' in text
    assert "Generated by HelixLang GEM pipeline" in text
