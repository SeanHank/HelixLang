"""HelixLang: DNA codons → bytecode → biological simulation.

Architecture (three layers):

Layer 1 — Helix Language (the minimal, dependency-free core)
    lexer, parser, AST, semantic analysis, compiler, bytecode, HXBC container,
    disassembler, unit system, provenance, error hierarchy, plugin registry and
    the ``use`` statement model.
    Location: :mod:`helixlang.core` (zero external dependencies).

Layer 2 — Biological Runtime (lazy plugins)
    gene expression, regulation, transcription, translation, metabolism, cell
    state, environment, population dynamics.  Location: :mod:`helixlang.plugins.runtime`.

Layer 3 — Scientific Applications (lazy plugins)
    domain-specific simulators and pipelines built on Layers 1+2:
    ``plugins.human`` (virtual patient / pharmacology / disease),
    ``plugins.gem`` (GEM reconstruction / FBA / SBML), ``plugins.apps``
    (pipelines / synbio / consortium / ecosystem / whole-cell),
    ``plugins.kinetics`` (enzyme kinetics), ``plugins.omics`` (expression
    inference).

This top-level package only re-exports the minimal always-installed core API.
Importing ``helixlang`` does **not** import any scientific plugin, so
:mod:`numpy`/:mod:`scipy`/etc. are never pulled in at package import time.
Scientific symbols are reached via their canonical submodule paths
(``helixlang.plugins.*``) or on demand through the plugin registry
(:func:`helixlang.core.plugin_registry.get_registry`).
"""

from helixlang.core.codon_table import (
    CILIATE_TABLE,
    MITO_VERTEBRATE_TABLE,
    OP_OPERAND_BYTES,
    STANDARD_TABLE,
    TABLES,
    Op,
    get_table,
    wobble,
)
from helixlang.core.errors import (
    ABIVersionError,
    CompileError,
    HelixError,
    LexError,
    ModelMissingError,
    NativeBackendError,
    ParseError,
    PluginConflictError,
    PluginDependencyError,
    PluginError,
    PluginMissingError,
    RegulationError,
    RuntimeHelixError,
    SemanticError,
    StackUnderflowError,
    UnknownKeywordError,
    UnknownNodeError,
)
from helixlang.core.plugin_registry import (
    NativeBackend,
    PluginProvider,
    Registry,
    get_registry,
)
from helixlang.core.units import (
    AI2_DIFFUSION_UM2_S,
    ATP_PER_GLUCOSE,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    TIME_TICK_MIN,
    TIME_TICK_S,
    decay_from_half_life_ticks,
    decay_to_half_life_ticks,
    diffusion_lattice_to_dx,
    diffusion_to_lattice,
    ticks_to_min,
)
from helixlang.core.use_stmt import (
    KNOWN_FLAGS,
    UseDirective,
    UseError,
    parse_use_line,
)
from helixlang.core.version import __version__

__all__ = [
    "__version__",
    # language errors
    "HelixError", "LexError", "ParseError", "SemanticError",
    "CompileError", "RegulationError", "RuntimeHelixError",
    # plugin / model / native errors
    "ABIVersionError", "ModelMissingError", "NativeBackendError",
    "PluginConflictError", "PluginDependencyError", "PluginError",
    "PluginMissingError", "StackUnderflowError", "UnknownKeywordError",
    "UnknownNodeError",
    # codon table
    "Op", "STANDARD_TABLE", "MITO_VERTEBRATE_TABLE", "CILIATE_TABLE",
    "TABLES", "get_table", "wobble", "OP_OPERAND_BYTES",
    # units
    "TIME_TICK_MIN", "TIME_TICK_S", "LATTICE_SPACING_UM",
    "AI2_DIFFUSION_UM2_S", "DIFFUSION_DT_S", "ATP_PER_GLUCOSE",
    "PROTEIN_HALF_LIFE_MEDIAN_TICKS",
    "ticks_to_min", "diffusion_to_lattice", "diffusion_lattice_to_dx",
    "decay_from_half_life_ticks", "decay_to_half_life_ticks",
    # plugin registry + use statement
    "NativeBackend", "PluginProvider", "Registry", "get_registry",
    "KNOWN_FLAGS", "UseDirective", "UseError", "parse_use_line",
]
