"""SBML/SBOL interoperability (T3.5, gap G8).

Exchange standards for the design-automation and modeling ecosystem:

- **SBML import** (Hucka et al. 2003, sbml.org): parses a minimal
  Level 3 Version 1 core document (species, reactions with
  stoichiometry and bounds, optional objective) into a
  :class:`helixlang.metabolism.MetabolicModel` so models from
  BioModels-compatible sources can be validated/solved without
  cobrapy.  Network-level constructs only (no math annotations,
  compartments are merged).
- **SBOL3 export/import** (SBOL 3.0, McLaughlin et al. 2020, sbols.org;
  LOICA 2022): serializes a synthetic design — ComponentDefinition(s),
  nested Component features, Sequence elements, roles — to SBOL3
  RDF/XML and reads it back, giving the design-automation workflow a
  Cello/SynBioHub-compatible interchange format.

Stdlib-only (``xml.etree``); see :mod:`helixlang.apps.synbio_automation`
for the design workflow that produces the SBOL3 payload.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from helixlang.errors import BioError
from helixlang.metabolism import (
    DEFAULT_UPPER_BOUND,
    MetabolicModel,
    Reaction,
)

#: SBML Level 3 Version 1 core namespace
SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"
#: SBOL 3 namespace (McLaughlin et al. 2020)
SBOL_NS = "http://sbols.org/v3#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
#: SBOL3 Sequence encoding for nucleotide sequence (IUPAC)
SBOL_ENCODING_IUPAC = (
    "http://www.chem.qmul.ac.uk/iubmb/misc/naseq.html")
#: SBOL3 role for a gene / CDS feature
SBOL_ROLE_GENE = "http://identifiers.org/so/SO:0000110"
SBOL_ROLE_PROMOTER = "http://identifiers.org/so/SO:0000167"
SBOL_ROLE_TERMINATOR = "http://identifiers.org/so/SO:0000141"
SBOL_ROLE_RBS = "http://identifiers.org/so/SO:0000139"
#: SBOL3 component types
SBOL_TYPE_DNA = "http://www.identifiers.org/so/SO:0000252"
SBOL_ROLE_ENGINEERED = "http://sbols.org/v3#EngineeredRegion"

_CDS_SEQ_PATTERN = re.compile(r"^[ACGT]+$", re.IGNORECASE)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node: ET.Element, name: str) -> ET.Element | None:
    for child in node:
        if _localname(child.tag) == name:
            return child
    return None


def _children(node: ET.Element | None, name: str) -> list[ET.Element]:
    if node is None:
        return []
    return [c for c in node if _localname(c.tag) == name]


def _text(node: ET.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


# ============================================================================
# SBML import
# ============================================================================

def sbml_to_model(xml_text: str) -> MetabolicModel:
    """Parse a minimal SBML L3V1 core document into a MetabolicModel.

    Reads ``<listOfSpecies>`` (metabolite ids), ``<listOfReactions>``
    with ``<listOfReactants>``/``<listOfProducts>`` stoichiometry and
    optional ``<listOfFluxBounds>``-style ``<bound>`` annotations, and an
    optional ``<listOfObjectives>`` biomass objective.  Compartments are
    merged into plain metabolite ids; missing bounds default to
    irreversible with the standard 1000 upper bound.

    Args:
        xml_text: SBML document as a string.

    Returns:
        a :class:`MetabolicModel`.
    """
    root = _parse_sbml_root(xml_text)
    model_el = _child(root, "model")
    if model_el is None:
        raise BioError("SBML document has no <model> element")

    m = MetabolicModel()
    species_ids = _collect_species(model_el)
    if species_ids:
        m.metabolites.update(species_ids)

    seen = 0
    for rxn_el in _children(_child(model_el, "listOfReactions"), "reaction"):
        rid = _text(_child(rxn_el, "id")) or rxn_el.get("id")
        if not rid:
            rid = f"rxn_{seen}"
        stoich: dict[str, float] = {}
        for ref_el in _children(_child(rxn_el, "listOfReactants"),
                                "speciesReference"):
            _add_reference(stoich, ref_el, -1.0)
        for ref_el in _children(_child(rxn_el, "listOfProducts"),
                                "speciesReference"):
            _add_reference(stoich, ref_el, +1.0)
        lower = DEFAULT_UPPER_BOUND * -1.0 \
            if rxn_el.get("reversible") == "true" else 0.0
        upper = DEFAULT_UPPER_BOUND
        bound_el = _child(rxn_el, "bound")
        if bound_el is not None:
            upper = float(bound_el.get("upper", upper))
            lower = float(bound_el.get("lower", lower))
        m.add_reaction(Reaction(
            id=rid,
            name=_text(_child(rxn_el, "name")) or rid,
            stoichiometry=stoich,
            lower_bound=lower,
            upper_bound=upper,
            subsystem=_text(_child(rxn_el, "subsystem")) or "other",
        ))
        seen += 1
    if seen == 0:
        raise BioError("SBML document contains no reactions")

    _set_objective(m, model_el)
    return m


def _parse_sbml_root(xml_text: str) -> ET.Element:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise BioError(f"malformed SBML XML: {exc}") from exc
    if _localname(root.tag) != "sbml":
        raise BioError(f"not an SBML document (root <{_localname(root.tag)}>)")
    return root


def _collect_species(model_el: ET.Element) -> list[str]:
    ids: list[str] = []
    for sp in _children(_child(model_el, "listOfSpecies"), "species"):
        sid = sp.get("id") or _text(_child(sp, "id"))
        if sid:
            ids.append(sid)
    return ids


def _add_reference(stoich: dict[str, float], ref_el: ET.Element,
                   sign: float) -> None:
    sid = ref_el.get("species") or _text(_child(ref_el, "species"))
    if not sid:
        return
    try:
        coef = float(ref_el.get("stoichiometry", "1"))
    except ValueError:
        coef = 1.0
    stoich[sid] = stoich.get(sid, 0.0) + sign * coef


def _set_objective(m: MetabolicModel, model_el: ET.Element) -> None:
    obj_el = _child(model_el, "listOfObjectives")
    if obj_el is None:
        return
    for objective in _children(obj_el, "objective"):
        flux_el = _child(objective, "listOfFluxObjectives")
        if flux_el is None:
            continue
        for fo in _children(flux_el, "fluxObjective"):
            rxn_id = fo.get("reaction")
            if rxn_id and rxn_id in m.reactions:
                m.set_biomass(rxn_id)
                return


def load_sbml(path: str) -> MetabolicModel:
    """Load a metabolic model from an SBML file on disk."""
    try:
        with open(path, encoding="utf-8") as fh:
            return sbml_to_model(fh.read())
    except OSError as exc:
        raise BioError(f"could not read SBML file {path!r}: {exc}") from exc


# ============================================================================
# SBOL3 export / import
# ============================================================================

def sbol3_dumps(component_definitions: list[dict[str, Any]]) -> str:
    """Serialize a list of SBOL3 ComponentDefinitions to RDF/XML.

    Each ``component_definitions`` entry is a dict with keys:

    - ``display_id`` (required): stable local identifier
    - ``name``: human-readable name
    - ``role`` (optional): SBOL3 role URI (defaults to EngineeredRegion)
    - ``description`` (optional)
    - ``components``: list of feature dicts with ``display_id``,
      ``role`` (SO role URI), and ``sequence`` (nucleotide string)

    Nested ``Component`` objects reference ``<display_id>Definition``
    ComponentDefinitions that each own a ``Sequence``.  The top-level
    ``<display_id>`` ComponentDefinition is the assembly; each feature
    is a DNA component with a role (gene/promoter/terminator/RBS).

    Returns:
        SBOL3 RDF/XML document as a string.
    """
    if not component_definitions:
        raise ValueError("at least one ComponentDefinition is required")
    _validate_cds(component_definitions)
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("sbol", SBOL_NS)
    rdf = ET.Element(f"{{{RDF_NS}}}RDF")
    base = "http://helixlang.local/sbol"
    for cd in component_definitions:
        did = cd["display_id"]
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", did):
            raise ValueError(f"invalid SBOL displayId {did!r}")
        top_uri = f"{base}/{did}"
        top = ET.SubElement(rdf, f"{{{SBOL_NS}}}ComponentDefinition")
        top.set(f"{{{RDF_NS}}}about", top_uri)
        ET.SubElement(top, f"{{{SBOL_NS}}}displayId").text = did
        if cd.get("name"):
            ET.SubElement(top, f"{{{SBOL_NS}}}name").text = cd["name"]
        if cd.get("description"):
            ET.SubElement(top, f"{{{SBOL_NS}}}description").text = cd["description"]
        ET.SubElement(top, f"{{{SBOL_NS}}}role").set(
            f"{{{RDF_NS}}}resource",
            cd.get("role", SBOL_ROLE_ENGINEERED))
        for i, feature in enumerate(cd.get("components", ())):
            fid = feature["display_id"]
            comp = ET.SubElement(top, f"{{{SBOL_NS}}}component")
            comp_uri = f"{base}/{did}#{fid}"
            comp_obj = ET.SubElement(comp, f"{{{SBOL_NS}}}Component")
            comp_obj.set(f"{{{RDF_NS}}}about", comp_uri)
            ET.SubElement(comp_obj, f"{{{SBOL_NS}}}displayId").text = fid
            ET.SubElement(comp_obj, f"{{{SBOL_NS}}}definition").set(
                f"{{{RDF_NS}}}resource", f"#component_definition_{did}_{i}")
            # nested ComponentDefinition holding the feature
            sub = ET.SubElement(rdf, f"{{{SBOL_NS}}}ComponentDefinition")
            sub.set(f"{{{RDF_NS}}}about", f"{base}/{did}/part_{i}")
            ET.SubElement(sub, f"{{{SBOL_NS}}}displayId").text = (
                f"component_definition_{did}_{i}")
            ET.SubElement(sub, f"{{{SBOL_NS}}}type").set(
                f"{{{RDF_NS}}}resource", SBOL_TYPE_DNA)
            ET.SubElement(sub, f"{{{SBOL_NS}}}role").set(
                f"{{{RDF_NS}}}resource", feature.get("role", SBOL_ROLE_GENE))
            seq = ET.SubElement(sub, f"{{{SBOL_NS}}}sequence")
            seq_obj = ET.SubElement(seq, f"{{{SBOL_NS}}}Sequence")
            seq_obj.set(f"{{{RDF_NS}}}about", f"{base}/{did}/seq_{i}")
            ET.SubElement(seq_obj, f"{{{SBOL_NS}}}displayId").text = (
                f"sequence_{did}_{i}")
            ET.SubElement(seq_obj, f"{{{SBOL_NS}}}encoding").set(
                f"{{{RDF_NS}}}resource", SBOL_ENCODING_IUPAC)
            ET.SubElement(seq_obj, f"{{{SBOL_NS}}}elements").text = (
                feature["sequence"])
    return ET.tostring(rdf, encoding="unicode")


def _validate_cds(component_definitions: list[dict[str, Any]]) -> None:
    for cd in component_definitions:
        for feature in cd.get("components", ()):
            seq = feature.get("sequence", "")
            if not seq:
                raise ValueError(f"feature {feature['display_id']!r} "
                                 "has no sequence")
            if not _CDS_SEQ_PATTERN.match(seq):
                raise ValueError(f"feature {feature['display_id']!r} "
                                 "sequence is not valid IUPAC DNA: "
                                 f"{seq[:40]!r}...")


def sbol3_loads(xml_text: str) -> list[dict[str, Any]]:
    """Parse SBOL3 RDF/XML back into ComponentDefinition dicts.

    The inverse of :func:`sbol3_dumps`: returns the same structured
    representation (top-level ComponentDefinitions with nested
    ``components`` features carrying their role and DNA sequence).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise BioError(f"malformed SBOL3 XML: {exc}") from exc
    if _localname(root.tag) != "RDF":
        raise BioError("not an SBOL RDF document")
    definitions: list[dict[str, Any]] = []
    # find all ComponentDefinition elements with a components list
    for cd_el in root:
        if _localname(cd_el.tag) != "ComponentDefinition":
            continue
        did = _text(_child(cd_el, "displayId"))
        if did is None or not did:
            continue
        comp_children = _children(cd_el, "component")
        if not comp_children:
            continue  # nested part definitions (no direct components)
        entry: dict[str, Any] = {
            "display_id": did,
            "name": _text(_child(cd_el, "name")) or "",
            "role": _child(cd_el, "role").get(f"{{{RDF_NS}}}resource")
            if _child(cd_el, "role") is not None else SBOL_ROLE_ENGINEERED,
            "components": [],
        }
        for comp in comp_children:
            comp_obj = _child(comp, "Component")
            fid = _text(_child(comp_obj, "displayId"))
            if not fid:
                continue
            # find the referenced part definition + its sequence
            seq = ""
            role = SBOL_ROLE_GENE
            ref = _child(comp_obj, "definition")
            target = None
            if ref is not None:
                target = ref.get(f"{{{RDF_NS}}}resource")
            expected = target.lstrip("#") if target else None
            for part_el in root:
                if _localname(part_el.tag) != "ComponentDefinition":
                    continue
                if _text(_child(part_el, "displayId")) != expected:
                    continue
                role_el = _child(part_el, "role")
                if role_el is not None:
                    role = role_el.get(f"{{{RDF_NS}}}resource", role)
                seq_el = _child(part_el, "sequence")
                if seq_el is not None:
                    seq_obj = _child(seq_el, "Sequence")
                    if seq_obj is not None:
                        seq = _text(_child(seq_obj, "elements"))
                break
            entry["components"].append({
                "display_id": fid,
                "role": role,
                "sequence": seq,
            })
        definitions.append(entry)
    if not definitions:
        raise BioError("SBOL3 document contains no ComponentDefinitions")
    return definitions


__all__ = [
    "SBOL_NS", "RDF_NS", "SBOL_ENCODING_IUPAC", "SBOL_TYPE_DNA",
    "SBOL_ROLE_ENGINEERED", "SBOL_ROLE_GENE", "SBOL_ROLE_PROMOTER",
    "SBOL_ROLE_TERMINATOR", "SBOL_ROLE_RBS",
    "sbml_to_model", "load_sbml",
    "sbol3_dumps", "sbol3_loads",
]
