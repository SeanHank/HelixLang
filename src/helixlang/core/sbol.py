"""SBOL 3 identifiers and namespace constants (doc/38 §6.2 ``core.sbol``).

Single source for the SBOL 3 / SBML exchange-format URIs used by the design
toolchain.  ``helixlang.interop`` (full SBML/SBOL translation, which needs
``plugins.runtime.metabolism``) and ``helixlang.api.sbol`` (the plugin-facing
frozen surface) both re-export from here so the URIs never drift.
"""
from __future__ import annotations

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

__all__ = [
    "SBML_NS", "SBOL_NS", "RDF_NS", "SBOL_ENCODING_IUPAC", "SBOL_TYPE_DNA",
    "SBOL_ROLE_ENGINEERED", "SBOL_ROLE_GENE", "SBOL_ROLE_PROMOTER",
    "SBOL_ROLE_TERMINATOR", "SBOL_ROLE_RBS",
]
