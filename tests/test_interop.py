"""SBML import + SBOL3 export/import interoperability tests (T3.5, gap G8).

Verification goals:
- Minimal SBML L3V1 core documents parse into a MetabolicModel:
  species, reaction stoichiometry (reactants/products, coefficients),
  reversible/irreversible bounds, biomass objective.
- Malformed/non-SBML documents raise BioError.
- SBOL3 RDF/XML export produces a valid document (displayIds, nested
  ComponentDefinitions, Sequences with IUPAC encoding) and round-trips
  through sbol3_loads preserving display ids, roles, and sequences.
- The design-automation payload (:mod:`helixlang.plugins.apps.synbio_automation`)
  exports and round-trips cleanly.

References:
- Hucka M et al. Bioinformatics 2003 19(4):524-531 (SBML)
- Hucka M et al. Systems Biol 2004 1(1):41-53 (SBML Level 2)
- McLaughlin JA et al. ACS Synth Biol 2020 9(4):957-960 (SBOL3)
- LOICA 2022 ACS Synth Biol 11:4049 (SBOL3 + genetic circuit assembly)
"""
from __future__ import annotations

import pytest

from helixlang.core.errors import BioError
from helixlang.interop import (
    SBOL_ENCODING_IUPAC,
    SBOL_ROLE_GENE,
    SBOL_ROLE_PROMOTER,
    SBOL_ROLE_RBS,
    SBOL_ROLE_TERMINATOR,
    load_sbml,
    sbml_to_model,
    sbol3_dumps,
    sbol3_loads,
)

# ============================================================================
# SBML import
# ============================================================================

#: Minimal reversible 2-species, 1-reaction SBML L3V1 core document
_SBML_MINIMAL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="tiny">
    <listOfSpecies>
      <species id="A" name="substrate"/>
      <species id="B" name="product"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="R1" reversible="true">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="2"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1"/>
        </listOfProducts>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

#: Irreversible reactions with a bound annotation and a biomass objective
_SBML_WITH_BOUNDS = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="biomass_model">
    <listOfSpecies>
      <species id="GLC"/>
      <species id="G6P"/>
      <species id="BIOMASS"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="EX_glc" reversible="false">
        <listOfProducts>
          <speciesReference species="G6P" stoichiometry="1"/>
        </listOfProducts>
        <bound upper="10.0"/>
      </reaction>
      <reaction id="BIOMASS" reversible="false">
        <listOfReactants>
          <speciesReference species="G6P" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="BIOMASS" stoichiometry="1"/>
        </listOfProducts>
      </reaction>
      <reaction id="EX_biomass" reversible="false">
        <listOfReactants>
          <speciesReference species="BIOMASS" stoichiometry="1"/>
        </listOfReactants>
        <bound upper="1000.0"/>
      </reaction>
    </listOfReactions>
    <listOfObjectives>
      <objective id="obj1" type="maximize">
        <listOfFluxObjectives>
          <fluxObjective reaction="BIOMASS" coefficient="1.0"/>
        </listOfFluxObjectives>
      </objective>
    </listOfObjectives>
  </model>
</sbml>
"""


def test_sbml_minimal_import() -> None:
    m = sbml_to_model(_SBML_MINIMAL)
    assert set(m.metabolites) == {"A", "B"}
    assert list(m.reactions) == ["R1"]
    rxn = m.reactions["R1"]
    assert rxn.stoichiometry == {"A": -2.0, "B": 1.0}
    # reversible reactions get a negative lower bound
    assert rxn.lower_bound < 0
    assert rxn.upper_bound > 0


def test_sbml_import_with_bounds_and_objective() -> None:
    m = sbml_to_model(_SBML_WITH_BOUNDS)
    glc = m.reactions["EX_glc"]
    assert glc.upper_bound == 10.0
    assert glc.lower_bound == 0.0
    assert m.biomass_reaction == "BIOMASS"


def test_sbml_import_solves() -> None:
    from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis

    m = sbml_to_model(_SBML_WITH_BOUNDS)
    fluxes = FluxBalanceAnalysis(m).solve()
    assert fluxes["BIOMASS"] > 0


def test_sbml_malformed_xml_raises() -> None:
    with pytest.raises(BioError):
        sbml_to_model("<sbml><model>")  # unclosed element


def test_sbml_wrong_root_raises() -> None:
    with pytest.raises(BioError):
        sbml_to_model("<html><body/></html>")


def test_sbml_no_reactions_raises() -> None:
    with pytest.raises(BioError):
        sbml_to_model(
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" '
            'level="3" version="1"><model id="empty"><listOfSpecies>'
            '<species id="A"/></listOfSpecies></model></sbml>')


def test_load_sbml_from_file(tmp_path) -> None:
    path = tmp_path / "model.sbml"
    path.write_text(_SBML_MINIMAL, encoding="utf-8")
    m = load_sbml(str(path))
    assert list(m.reactions) == ["R1"]
    assert set(m.metabolites) == {"A", "B"}


def test_load_sbml_missing_file_raises(tmp_path) -> None:
    with pytest.raises(BioError):
        load_sbml(str(tmp_path / "nope.xml"))


# ============================================================================
# SBOL3 export / import
# ============================================================================

_FEATURES = [
    {"display_id": "pLac", "role": SBOL_ROLE_PROMOTER,
     "sequence": "TTGACATTATGCTCAGAGC"},
    {"display_id": "rbs01", "role": SBOL_ROLE_RBS, "sequence": "AGGAGG"},
    {"display_id": "lacI", "role": SBOL_ROLE_GENE,
     "sequence": "ATGGTAAATCCAGTTACCCTTTATGATGTA"},
    {"display_id": "t01", "role": SBOL_ROLE_TERMINATOR,
     "sequence": "AAAGCCCGAAAGGAACTGAG"},
]

_CDS = [{
    "display_id": "TestCircuit",
    "name": "circuit_y",
    "role": "http://sbols.org/v3#EngineeredRegion",
    "description": "a NOT gate",
    "components": _FEATURES,
}]


def test_sbol3_export_structure() -> None:
    xml = sbol3_dumps(_CDS)
    assert "http://sbols.org/v3#" in xml
    assert "ComponentDefinition" in xml
    assert "Sequence" in xml
    assert "http://www.chem.qmul.ac.uk/iubmb/misc/naseq.html" in xml
    assert "TTGACATTATGCTCAGAGC" in xml
    assert "TestCircuit" in xml
    assert 'rdf:about' in xml or "http://www.w3.org/1999/02/22-rdf-syntax-ns#" in xml


def test_sbol3_roundtrip() -> None:
    xml = sbol3_dumps(_CDS)
    parsed = sbol3_loads(xml)
    assert len(parsed) == 1
    top = parsed[0]
    assert top["display_id"] == "TestCircuit"
    assert top["name"] == "circuit_y"
    assert top["role"] == "http://sbols.org/v3#EngineeredRegion"
    assert len(top["components"]) == len(_FEATURES)
    got = {f["display_id"]: f for f in top["components"]}
    for feat in _FEATURES:
        assert got[feat["display_id"]]["role"] == feat["role"]
        assert got[feat["display_id"]]["sequence"] == feat["sequence"]


def test_sbol3_empty_raises() -> None:
    with pytest.raises(ValueError):
        sbol3_dumps([])


def test_sbol3_bad_display_id_raises() -> None:
    with pytest.raises(ValueError):
        sbol3_dumps([{"display_id": "1bad", "components": [{
            "display_id": "x", "sequence": "ACGT"}]}])


def test_sbol3_bad_sequence_raises() -> None:
    with pytest.raises(ValueError):
        sbol3_dumps([{"display_id": "circuit", "components": [{
            "display_id": "x", "sequence": "ACGTX-W"}]}])


def test_sbol3_loads_wrong_root_raises() -> None:
    with pytest.raises(BioError):
        sbol3_loads("<foo><bar/></foo>")


def test_sbol3_loads_malformed_raises() -> None:
    with pytest.raises(BioError):
        sbol3_loads("<rdf:RDF>")


def test_sbol3_loads_empty_document_raises() -> None:
    xml = sbol3_dumps(_CDS)
    # strip everything but the wrapper to simulate a document with no
    # ComponentDefinitions
    without = xml[:xml.index("ComponentDefinition")]
    without += "</rdf:RDF>"
    with pytest.raises(BioError):
        sbol3_loads(without)


# ============================================================================
# Design-automation interop (T2.3 + T3.5 integration)
# ============================================================================

def test_design_payload_roundtrip() -> None:
    from helixlang.plugins.apps.synbio_automation import not_gate

    design = not_gate()
    parsed = sbol3_loads(design.sbol3_xml)
    assert len(parsed) == 1
    top = parsed[0]
    assert top["display_id"] == "BooleanCircuit"
    seqs = [f["sequence"] for f in top["components"]]
    assert any(seq for seq in seqs)          # at least one non-empty
    assert any("ACGT" in seq.upper() for seq in seqs)
    assert any(f["role"] == SBOL_ROLE_GENE for f in top["components"])


def test_sbol3_encoding_constant() -> None:
    assert SBOL_ENCODING_IUPAC.endswith("naseq.html")


def test_interop_no_cobrapy_required() -> None:
    # The whole SBML path must work without cobrapy (pure stdlib+scipy).
    from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis

    m = sbml_to_model(_SBML_WITH_BOUNDS)
    assert FluxBalanceAnalysis(m).solve()["BIOMASS"] > 0
