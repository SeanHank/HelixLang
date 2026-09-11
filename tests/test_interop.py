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
    SubstrateField,
    VirtualCell,
    VirtualTissue,
    cellml_to_model,
    cells_from_csv,
    cells_to_csv,
    dict_to_tissue,
    load_cellml,
    load_sbml,
    sbml_to_model,
    sbol3_dumps,
    sbol3_loads,
    tissue_dumps,
    tissue_loads,
    tissue_to_dict,
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


# ============================================================================
# CellML interop (doc/42 Phase E, gap RT-6)
# ============================================================================

_CELLML_ODE = """<?xml version="1.0" encoding="UTF-8"?>
<model name="two_compartment" xmlns="http://www.cellml.org/cellml/1.1">
  <component name="main">
    <variable name="A" initial_value="100" units="mol"/>
    <variable name="B" initial_value="0" units="mol"/>
    <variable name="k1" units="per_second" value="0.1"/>
    <variable name="k2" units="per_second" value="0.05"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/>
        <apply><diff/><bvar><ci>A</ci></bvar><ci>A</ci></apply>
        <apply><minus/>
          <apply><times/><ci>k2</ci><ci>B</ci></apply>
          <apply><times/><ci>k1</ci><ci>A</ci></apply>
        </apply>
      </apply>
      <apply><eq/>
        <apply><diff/><bvar><ci>B</ci></bvar><ci>B</ci></apply>
        <apply><minus/>
          <apply><times/><ci>k1</ci><ci>A</ci></apply>
          <apply><times/><ci>k2</ci><ci>B</ci></apply>
        </apply>
      </apply>
    </math>
  </component>
</model>
"""


def test_cellml_parses_species_and_params() -> None:
    m = cellml_to_model(_CELLML_ODE)
    assert m["model_id"] == "two_compartment"
    assert set(m["species"]) == {"A", "B"}
    assert m["species"]["A"]["initial"] == 100.0
    assert m["species"]["B"]["initial"] == 0.0
    assert set(m["parameters"]) == {"k1", "k2"}
    assert m["parameters"]["k1"] == 0.1


def test_cellml_rate_expressions_translate() -> None:
    m = cellml_to_model(_CELLML_ODE)
    assert "k2" in m["rates"]["A"] and "k1" in m["rates"]["A"]
    assert "k1" in m["rates"]["B"] and "k2" in m["rates"]["B"]
    # infix string must be evaluable against the parameter namespace
    ns = dict(m["parameters"])
    ns["A"], ns["B"] = 100.0, 0.0
    assert abs(eval(m["rates"]["A"], {"__builtins__": {}}, ns) - (-10.0)) < 1e-9


def test_cellml_load_from_file(tmp_path) -> None:
    path = tmp_path / "model.cellml"
    path.write_text(_CELLML_ODE, encoding="utf-8")
    m = load_cellml(str(path))
    assert m["model_id"] == "two_compartment"


def test_cellml_malformed_raises() -> None:
    with pytest.raises(BioError):
        cellml_to_model("<model><component>")


def test_cellml_no_species_raises() -> None:
    with pytest.raises(BioError):
        cellml_to_model('<model xmlns="http://www.cellml.org/cellml/1.1">'
                        '<component name="c"><variable name="k" value="1"/>'
                        '</component></model>')


# ============================================================================
# Virtual-tissue interop (doc/42 Phase E, gap RT-6)
# ============================================================================

def _make_tissue() -> VirtualTissue:
    return VirtualTissue(
        cells=[
            VirtualCell(x=1.0, y=2.0, z=0.0, cell_type=1,
                        cycle_phase="M", volume=2.5,
                        custom={"oxygen": 5.0}),
            VirtualCell(x=10.0, y=20.0, z=0.0, cell_type=2, volume=1.0),
        ],
        substrates=[SubstrateField(
            name="glucose", nx=2, ny=2, nz=1, dx=15.0, dy=15.0, dz=15.0,
            data=[1.0, 2.0, 3.0, 4.0],
        )],
        meta={"simulation": "test"},
    )


def test_tissue_to_dict_roundtrip() -> None:
    t = _make_tissue()
    d = tissue_to_dict(t)
    t2 = dict_to_tissue(d)
    assert len(t2.cells) == 2
    c0 = t2.cells[0]
    assert (c0.x, c0.y, c0.z) == (1.0, 2.0, 0.0)
    assert c0.cell_type == 1 and c0.cycle_phase == "M"
    assert c0.custom == {"oxygen": 5.0}
    assert t2.substrates[0].name == "glucose"
    assert t2.substrates[0].data == [1.0, 2.0, 3.0, 4.0]
    assert t2.meta == {"simulation": "test"}


def test_tissue_json_roundtrip() -> None:
    t = _make_tissue()
    text = tissue_dumps(t)
    restored = tissue_loads(text)
    assert len(restored.cells) == 2
    assert restored.cells[1].cell_type == 2
    assert restored.substrates[0].data[3] == 4.0


def test_tissue_malformed_json_raises() -> None:
    with pytest.raises(BioError):
        tissue_loads("{not json")


def test_cells_csv_roundtrip() -> None:
    t = _make_tissue()
    text = cells_to_csv(t)
    assert "position_x" in text.splitlines()[0]
    assert "oxygen" in text.splitlines()[0]
    parsed = cells_from_csv(text)
    assert len(parsed) == 2
    assert parsed[0].custom["oxygen"] == 5.0
    assert parsed[1].cell_type == 2


def test_cells_csv_empty_fieldnames_raises() -> None:
    with pytest.raises(BioError):
        cells_from_csv("")


def test_cells_from_csv_non_numeric_custom() -> None:
    csv_text = (
        "position_x,position_y,position_z,cell_type,volume,note\n"
        "1,2,0,1,2.5,hello\n"
    )
    cells = cells_from_csv(csv_text)
    assert len(cells) == 1
    assert cells[0].custom["note"] == 0.0  # non-numeric falls back to default


def test_tissue_dumps_unserializable_raises() -> None:
    t = _make_tissue()
    t.cells[0].custom["bad"] = {"not": "json serializable", "set": {1, 2, 3}}
    with pytest.raises(BioError):
        tissue_dumps(t)


def test_cells_csv_empty_raises() -> None:
    with pytest.raises(BioError):
        cells_from_csv("position_x,position_y,position_z\n")


# ============================================================================
# CellML extra coverage (MathML operator tokens + model handling)
# ============================================================================

_MATHML_OPS = """<?xml version="1.0" encoding="UTF-8"?>
<model name="mathops" xmlns="http://www.cellml.org/cellml/1.1">
  <component name="main">
    <variable name="A" initial_value="1"/>
    <variable name="B" initial_value="1"/>
    <variable name="C" initial_value="1"/>
    <variable name="D" initial_value="1"/>
    <variable name="E" initial_value="1"/>
    <variable name="F" initial_value="1"/>
    <variable name="G" initial_value="1"/>
    <variable name="H" initial_value="1"/>
    <variable name="constvar" initial_value="42"/>
    <variable name="paramvar" value="7"/>
    <variable name="nothing"/>
    <variable value="5"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><ci>A</ci></apply><apply><minus/><ci>k1</ci></apply></apply>
      <apply><eq/><apply><diff/><ci>B</ci></apply><apply><power/><ci>A</ci><cn>2</cn></apply></apply>
      <apply><eq/><apply><diff/><ci>C</ci></apply><apply><eq/><ci>A</ci><ci>B</ci></apply></apply>
      <apply><eq/><apply><diff/><ci>D</ci></apply><apply><diff/><ci>x</ci><ci>y</ci></apply></apply>
      <apply><eq/><apply><diff/><ci>E</ci></apply><apply><cos/><ci>A</ci></apply></apply>
      <apply><eq/><apply><diff/><ci>F</ci></apply><apply><plus/></apply></apply>
      <apply><eq/><apply><diff/><ci>G</ci></apply><apply/></apply>
      <apply><eq/><apply><diff/><ci>H</ci></apply></apply>
      <apply><eq/><ci>a</ci><ci>b</ci></apply>
      <apply><eq/><apply><plus/><ci>a</ci><ci>b</ci></apply><ci>x</ci></apply>
    </math>
  </component>
</model>
"""


def test_cellml_mathml_operator_tokens() -> None:
    m = cellml_to_model(_MATHML_OPS)
    assert m["model_id"] == "mathops"
    # all seven ODE targets become species
    assert m["rates"]["A"] == "-(k1)"          # unary minus
    assert m["rates"]["B"] == "pow(A, 2)"      # power + cn
    assert m["rates"]["C"] == "A"              # eq falls through to lhs
    assert m["rates"]["D"] == "y"              # nested diff names arg[1]
    assert m["rates"]["E"] == "A"              # unhandled op, single arg
    assert m["rates"]["F"] == "0"              # binary op with no args
    assert m["rates"]["G"] == "0"              # empty apply
    assert "H" not in m["rates"]               # no RHS -> no rate    # classification: constant, parameter, and ignored declaration
    assert m["constants"]["constvar"] == 42.0
    assert m["parameters"]["paramvar"] == 7.0
    assert "nothing" not in m["species"]
    assert "nothing" not in m["parameters"]
    assert "nothing" not in m["constants"]


def test_cellml_model_wrapped_in_cellml_root() -> None:
    doc = ('<cellml xmlns="http://www.cellml.org/cellml/1.1">'
           '<model name="wrapped"><component name="c">'
           '<variable name="A" initial_value="1"/>'
           '<math xmlns="http://www.w3.org/1998/Math/MathML">'
           '<apply><eq/><apply><diff/><ci>A</ci></apply><ci>k</ci></apply>'
           '</math></component></model></cellml>')
    m = cellml_to_model(doc)
    assert m["model_id"] == "wrapped"


def test_cellml_model_wrapped_in_plain_root() -> None:
    doc = ('<wrapper><model name="oddroot"><component name="c">'
           '<variable name="A" initial_value="1"/>'
           '<math xmlns="http://www.w3.org/1998/Math/MathML">'
           '<apply><eq/><apply><diff/><ci>A</ci></apply><ci>k</ci></apply>'
           '</math></component></model></wrapper>')
    m = cellml_to_model(doc)
    assert m["model_id"] == "model"
    assert "A" in m["rates"]


def test_cellml_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(BioError):
        load_cellml(str(tmp_path / "nope.cellml"))


# ============================================================================
# SBML + SBOL edge-case coverage
# ============================================================================

_SBML_EDGE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="edge">
    <listOfSpecies>
      <species id="A"/>
      <species/>
      <species id="B"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction reversible="true">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="bad"/>
          <speciesReference/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="2"/>
        </listOfProducts>
      </reaction>
    </listOfReactions>
    <listOfObjectives>
      <objective id="o2"><something/></objective>
      <objective id="o3">
        <listOfFluxObjectives><fluxObjective reaction="NOPE"/></listOfFluxObjectives>
      </objective>
    </listOfObjectives>
  </model>
</sbml>
"""


def test_sbml_edge_cases() -> None:
    m = sbml_to_model(_SBML_EDGE)
    # reaction without an id gets an auto-generated one
    rxn = m.reactions["rxn_0"]
    # species without an id is ignored, A/B present
    assert {"A", "B"} <= m.metabolites
    # bad stoichiometry falls back to 1.0, missing species reference ignored
    assert rxn.stoichiometry.get("A") == -1.0
    assert rxn.stoichiometry.get("B") == 2.0
    assert m.biomass_reaction is None


def test_sbml_no_model_raises() -> None:
    with pytest.raises(BioError):
        sbml_to_model('<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"/>')


def test_sbml_no_species_with_reactions() -> None:
    doc = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core">
  <model id="nospecies">
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfReactants><speciesReference species="X" stoichiometry="1"/></listOfReactants>
        <listOfProducts><speciesReference species="Y" stoichiometry="1"/></listOfProducts>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
    m = sbml_to_model(doc)
    assert "X" in m.metabolites and "Y" in m.metabolites
    assert "R1" in m.reactions


def test_sbol3_dumps_empty_sequence_raises() -> None:
    with pytest.raises(ValueError):
        sbol3_dumps([{"display_id": "circuit", "components": [{
            "display_id": "x", "sequence": ""}]}])


def test_sbol3_dumps_without_name_and_role() -> None:
    # cd with no name and no per-feature role (defaults applied)
    xml = sbol3_dumps([{
        "display_id": "Plain",
        "components": [{"display_id": "f1", "sequence": "ACGT"}],
    }])
    assert "Plain" in xml


_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_SBOL = "http://sbols.org/v3#"

_SBOL_LOADS_EDGE = f"""<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="{_RDF}" xmlns:sbol="{_SBOL}">
  <foo/>
  <sbol:ComponentDefinition><sbol:displayId/></sbol:ComponentDefinition>
  <sbol:ComponentDefinition>
    <sbol:displayId>Top</sbol:displayId>
    <sbol:component>
      <sbol:Component><sbol:displayId>c_nodef</sbol:displayId></sbol:Component>
    </sbol:component>
    <sbol:component/>
    <sbol:component><sbol:Component/></sbol:component>
    <sbol:component>
      <sbol:Component><sbol:displayId>c_part2</sbol:displayId>
        <sbol:definition rdf:resource="#part2"/></sbol:Component>
    </sbol:component>
    <sbol:component>
      <sbol:Component><sbol:displayId>c_part3</sbol:displayId>
        <sbol:definition rdf:resource="#part3"/></sbol:Component>
    </sbol:component>
    <sbol:component>
      <sbol:Component><sbol:displayId>c_part4</sbol:displayId>
        <sbol:definition rdf:resource="#part4"/></sbol:Component>
    </sbol:component>
  </sbol:ComponentDefinition>
  <sbol:ComponentDefinition rdf:about="#part2">
    <sbol:displayId>part2</sbol:displayId>
  </sbol:ComponentDefinition>
  <sbol:ComponentDefinition rdf:about="#part3">
    <sbol:displayId>part3</sbol:displayId><sbol:role/>
  </sbol:ComponentDefinition>
  <sbol:ComponentDefinition rdf:about="#part4">
    <sbol:displayId>part4</sbol:displayId><sbol:role/>
    <sbol:sequence><sbol:x/></sbol:sequence>
  </sbol:ComponentDefinition>
</rdf:RDF>
"""


def test_sbol3_loads_edge_shapes() -> None:
    parsed = sbol3_loads(_SBOL_LOADS_EDGE)
    assert len(parsed) >= 1
    assert parsed[0]["display_id"] == "Top"
    got = {c["display_id"]: c for c in parsed[0]["components"]}
    # c_nodef had no definition -> empty sequence
    assert got["c_nodef"]["sequence"] == ""
    assert got["c_part2"]["sequence"] == ""
    assert got["c_part3"]["sequence"] == ""
    assert got["c_part4"]["sequence"] == ""


_SBOL_LOADS_EMPTY = f"""<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="{_RDF}" xmlns:sbol="{_SBOL}">
  <sbol:ComponentDefinition><sbol:displayId/></sbol:ComponentDefinition>
  <sbol:ComponentDefinition>
    <sbol:displayId>NoComponents</sbol:displayId>
  </sbol:ComponentDefinition>
</rdf:RDF>
"""


def test_sbol3_loads_no_definitions_raises() -> None:
    with pytest.raises(BioError):
        sbol3_loads(_SBOL_LOADS_EMPTY)

