"""HelixLang version (doc/36 §2.2) and artifact semantic manifest (doc/38 §2.4).

The single source of truth for every version-bearing string is
``pyproject.toml``; ``release.py`` rewrites ``__version__`` here and
``server/app.py`` in lockstep and refuses to run on drift (``--check-versions``).
"""
from __future__ import annotations

__version__ = "2026.9.0"

# ── .helixc semantic surface versions (doc/38 §2.4) ─────────────────────────
# All are monotone integers.  The loader policy in core/hxbc.py is:
#   * artifact surface NEWER than this build  → hard SemanticVersionError
#   * artifact surface OLDER than this build  → explicit warning
#   * REFERENCE_DATA mismatch (either way)     → warning, never an error
#
# Bump the right one when the corresponding surface changes incompatibly:
#   LANGUAGE_SPEC_VERSION          grammar/lexing surface (#config, codon table)
#   AST_SCHEMA_VERSION             PROG-section AST encoding (.helixc layout)
#   SIMULATION_SEMANTICS_VERSION   meaning of simulation directives/backends
#   REFERENCE_DATA_VERSION         bundled reference/validation data sets
LANGUAGE_SPEC_VERSION = 2
AST_SCHEMA_VERSION = 1
SIMULATION_SEMANTICS_VERSION = 1
REFERENCE_DATA_VERSION = 1

#: Public, stable tuple of the four surface names (import order dependency-safe).
SEMANTIC_SURFACES: tuple[str, ...] = (
    "LANGUAGE_SPEC_VERSION",
    "AST_SCHEMA_VERSION",
    "SIMULATION_SEMANTICS_VERSION",
    "REFERENCE_DATA_VERSION",
)
