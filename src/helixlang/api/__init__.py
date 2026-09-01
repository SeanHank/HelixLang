"""Public plugin surface (doc/38 §6.2).

Plugins import **only** from here (plus ``helixlang.core.errors``), never from
``helixlang.core.ast_nodes`` / ``helixlang.core.parser`` /
``helixlang.sim_runtime._engine``.  This package is the frozen, versioned
contract; anything added here is public API and requires a manifest
``abi_version`` bump to change.

Submodule index (doc/38 §6.2):

- :mod:`~helixlang.api.registry`  — ``PluginProvider``, ``NativeBackend``, ``Registry``
- :mod:`~helixlang.api.grammar`   — ``AnnotationGrammar`` (§5 registry entry type)
- :mod:`~helixlang.api.ast`       — ``ASTExtension``, ``ProgramView``, ``ProgramBuilder``
- :mod:`~helixlang.api.ir`        — ``IRExtension``, ``OperandSchema``
- :mod:`~helixlang.api.backend`   — ``Backend``, ``RunRequest``, ``BackendRegistry``
- :mod:`~helixlang.api.capabilities` — ``Capability``
- :mod:`~helixlang.api.language`  — ``LanguageConfig`` + public codon constants
- :mod:`~helixlang.api.units`     — public physical / simulation constants
- :mod:`~helixlang.api.compiler`  — ``Lexer`` → ``Parser`` → ``SemanticAnalyzer`` → ``Compiler``
- :mod:`~helixlang.api.bytecode`  — ``Chunk`` / ``Op`` / opcode tuning constants
- :mod:`~helixlang.api.sbol`      — SBOL 3 identifiers (single source)
- :mod:`~helixlang.api.gem`       — GEM↔sim FBA-medium adapters (lazy)
- :mod:`~helixlang.api.accel`     — accelerated-kernel adapter (lazy)
- :mod:`~helixlang.api.errors`    — the typed error family (re-export)
"""

from helixlang.api import (
           accel,  # noqa: F401
           ast,  # noqa: F401
           backend,  # noqa: F401
           bytecode,  # noqa: F401
           capabilities,  # noqa: F401
           compiler,  # noqa: F401
           dimensions,  # noqa: F401
           errors,  # noqa: F401
           gem,  # noqa: F401
           grammar,  # noqa: F401
           language,  # noqa: F401
           registry,  # noqa: F401
           sbol,  # noqa: F401
           units,  # noqa: F401
)

__all__ = ["accel", "ast", "backend", "bytecode", "capabilities", "compiler",
           "dimensions", "errors", "gem", "grammar", "language", "registry",
           "sbol", "units"]
