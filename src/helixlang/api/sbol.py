"""Public SBOL 3 identifiers (doc/38 §6.2 ``api.sbol``).

Re-exports the SBOL 3 exchange-format URIs (role / component-type constants)
from the single source ``helixlang.core.sbol`` so plugins can tag design
payloads (e.g. SBOL-over-VCF) without importing ``helixlang.interop`` or the
private translation machinery.  Full SBML/SBOL translation stays core-side in
``helixlang.interop``.
"""
from __future__ import annotations

from typing import Any

from helixlang.core.sbol import (  # noqa: F401
    RDF_NS,
    SBML_NS,
    SBOL_ENCODING_IUPAC,
    SBOL_NS,
    SBOL_ROLE_ENGINEERED,
    SBOL_ROLE_GENE,
    SBOL_ROLE_PROMOTER,
    SBOL_ROLE_RBS,
    SBOL_ROLE_TERMINATOR,
    SBOL_TYPE_DNA,
)


def sbol3_dumps(design: Any) -> str:
    """Serialize a component-definition payload to SBOL 3 XML.

    The full translator lives in ``helixlang.interop`` (which needs the
    metabolism aggregate model) and is imported lazily, so importing this
    module never pulls plugin machinery.
    """
    from helixlang.interop import sbol3_dumps as _sbol3_dumps
    return _sbol3_dumps(design)


__all__ = [
    "SBML_NS", "SBOL_NS", "RDF_NS", "SBOL_ENCODING_IUPAC", "SBOL_TYPE_DNA",
    "SBOL_ROLE_ENGINEERED", "SBOL_ROLE_GENE", "SBOL_ROLE_PROMOTER",
    "SBOL_ROLE_TERMINATOR", "SBOL_ROLE_RBS", "sbol3_dumps",
]
