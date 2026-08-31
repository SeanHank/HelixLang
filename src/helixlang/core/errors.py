"""HelixLang core error hierarchy (doc/36 §2.2, §3ξ.2).

This is the single error module for the language core: the base language
hierarchy plus the explicit plugin/native/model taxonomy introduced by the
restructure.

Base hierarchy (language):

- ``HelixError``        — root of all HelixLang exceptions.
- ``LexError``          — lexical error (e.g., DNA length not a multiple of 3).
- ``ParseError``        — syntax error (annotation block unclosed, ORF missing terminator).
- ``SemanticError``     — duplicate symbols, references to undefined genes.
- ``CompileError``      — compile-time error (unknown codon, const-pool OOB).
- ``RegulationError``   — regulation graph references an undefined gene.
- ``RuntimeHelixError`` — runtime error (stack underflow, unknown opcode).
- ``BioError``          — biology-module error (invalid input / solver failure).
- ``SimConfigError``    — simulation-backend configuration error.

Independent failure classes added during the restructure:

- ``PluginError``         — plugin discovery/loading failures.
- ``ModelMissingError``   — required scientific model data absent.
- ``ABIVersionError``     — bytecode ABI mismatch (never a wrong-result run).
- ``NativeBackendError``  — a compiled accelerator failed to load or consumed
  bad data; the system never silently degrades to a lower-fidelity backend.
"""
from __future__ import annotations


class HelixError(Exception):
    """Base class for all HelixLang exceptions."""

    def __init__(self, msg: str, *, line: int = 0, col: int = 0, codon_index: int = -1):
        super().__init__(msg)
        self.msg = msg
        self.line = line
        self.col = col
        self.codon_index = codon_index

    def __str__(self) -> str:
        loc = f"line {self.line}" if self.line else "<unknown>"
        if self.codon_index >= 0:
            loc += f" codon #{self.codon_index}"
        return f"[{type(self).__name__} @ {loc}] {self.msg}"


class LexError(HelixError):
    """Lexical error (e.g., DNA length is not a multiple of 3)."""


class ParseError(HelixError):
    """Syntax error (e.g., annotation block unclosed, ORF missing terminator)."""


class SemanticError(HelixError):
    """Semantic error (e.g., duplicate symbols, references to undefined genes)."""


class DimensionError(SemanticError):
    """Compile-time dimensional mismatch (doc/41 Item 5, Ring 1).

    Raised when a program composes quantities of incompatible physical
    dimensions (e.g. ``Float<µM> + Float<µm3>``); distinct from the runtime
    :class:`~helixlang.core.dimensions.UnitError`.
    """


class CompileError(HelixError):
    """Compile-time error (e.g., unknown codon, constant pool out of bounds)."""


class RegulationError(HelixError):
    """Regulation graph error (e.g., regulation references an undefined gene)."""


class RuntimeHelixError(HelixError):
    """Runtime error (stack underflow, unknown opcode). Deliberately avoids Python's built-in RuntimeError."""


class BioError(HelixError):
    """Biology module error (invalid input or solver failure in biological computation layers).

    Used to replace bare ``raise ValueError`` in the bio module, so biology-side
    errors join the unified :class:`HelixError` hierarchy, letting the server
    layer's ``@errorhandler`` handle them by severity.
    """


class SimConfigError(HelixError):
    """Simulation-backend configuration error (bad ``#config sim`` value, unknown
    enum, malformed float/int/dict coercion) raised by :mod:`helixlang.sim_runtime`."""


class UnknownKeywordError(SemanticError):
    """A ``#keyword`` is not recognized by any registered plugin (doc/36 F7).

    The semantic analyzer must know every keyword at compile time — an unknown
    one is never silently dropped.
    """


class StackUnderflowError(RuntimeHelixError):
    """The operand stack underflowed (doc/36 F11).

    The prototype previously ignored stack underflow to preserve robustness; the
    restructure turns it into an explicit, deterministic error.
    """


class UnknownNodeError(RuntimeHelixError):
    """A GRN/network node was referenced but does not exist (doc/36 F12).

    Never a silent default or a bare ``KeyError``.
    """


# ── Restructure-specific explicit errors (doc/36 §3ξ.2) ──────────────────────


class PluginError(HelixError):
    """Base class for all plugin lifecycle errors."""


class PluginMissingError(PluginError):
    """A plugin referenced by ``use`` / a ``#keyword`` is not installed or not
    discoverable.  Message names the plugin plus the pip extra to install.

    Args:
        name: plugin name requested.
        extra: the pip extra that provides it, e.g. ``grn``.
    """

    def __init__(self, name: str, extra: str, *, line: int = 0, col: int = 0):
        msg = (
            f"plugin '{name}' is not available. Install it with: "
            f"`pip install helixlang[{extra}]`, and add `use {name}` to your "
            "helix source (or rely on its #keyword auto-detection)."
        )
        super().__init__(msg, line=line, col=col)
        self.name = name
        self.extra = extra


class PluginDependencyError(PluginError):
    """A plugin is installed but one of its optional dependencies is absent
    (doc/36 F2/F4/F5).  The run fails explicitly rather than computing at a
    lower fidelity; a reduced-fidelity mode requires an explicit opt-in flag."""

    def __init__(self, name: str, dep: str, extra: str, *, line: int = 0,
                 col: int = 0):
        msg = (
            f"plugin '{name}' requires '{dep}', which is not installed. Install "
            f"it with: `pip install helixlang[{extra}]`. To intentionally run at "
            "reduced fidelity, declare the corresponding explicit capability "
            "flag (e.g. `use {name} --low-fidelity`)."
        )
        super().__init__(msg, line=line, col=col)
        self.name = name
        self.dep = dep
        self.extra = extra


class PluginConflictError(PluginError):
    """Two plugins claim the same backend name or #keyword."""

    def __init__(self, key: str, first: str, second: str, *, line: int = 0,
                 col: int = 0):
        msg = (
            f"plugin conflict on '{key}': both '{first}' and '{second}' claim it. "
            "Registering plugins with contradictory claims is not allowed."
        )
        super().__init__(msg, line=line, col=col)
        self.key = key
        self.first = first
        self.second = second


class ModelMissingError(HelixError):
    """A required scientific model (FBA/GEM data) is absent (doc/36 F3).

    Unlike the old code which silently fell back to the built-in core model,
    this raises an explicit, actionable error.
    """

    def __init__(self, model: str, extra: str, *, line: int = 0, col: int = 0,
                 detail: str = ""):
        msg = (
            f"required model '{model}' is not available. Provide it via "
            f"`pip install helixlang[{extra}]` or point the model loader at the "
            "data. " + (detail or "")
        )
        super().__init__(msg, line=line, col=col)
        self.model = model
        self.extra = extra


class ABIVersionError(HelixError):
    """Bytecode ABI mismatch (doc/36 F9).

    Loading a program whose ``OPCODE_VERSION`` differs from the running VM is a
    hard error — never an incompatible silent run.
    """

    def __init__(self, expected: int, got: int, *, line: int = 0, col: int = 0):
        msg = (
            f"bytecode ABI mismatch: running OPCODE_VERSION {expected} but the "
            f"program was compiled with {got}. Recompile the helix source with "
            "the current compiler."
        )
        super().__init__(msg, line=line, col=col)
        self.expected = expected
        self.got = got


class SemanticVersionError(HelixError):
    """Artifact semantic-surface mismatch (doc/38 §2.4).

    Loading an artifact whose ``LANGUAGE_SPEC`` / ``AST_SCHEMA`` /
    ``SIMULATION_SEMANTICS`` surface is *newer* than this build is a hard
    error — never a silent wrong-result run, mirroring :class:`ABIVersionError`.
    Unknown/older surfaces warn; reference-data drift never raises.
    """

    def __init__(self, surface: str, expected: int, got: int, *,
                 line: int = 0, col: int = 0):
        msg = (
            f"{surface} mismatch: this build supports {surface} {expected} but "
            f"the artifact declares {got}. Recompile the helix source with the "
            "current compiler."
        )
        super().__init__(msg, line=line, col=col)
        self.surface = surface
        self.expected = expected
        self.got = got


class NativeBackendError(HelixError):
    """A native accelerator failed to load or consumed invalid data (doc/36).

    The system NEVER silently degrades to a lower-fidelity implementation.  A
    missing ``*.so`` raises this (with the rebuild command) unless the program
    explicitly declared ``--pure-python``.
    """

    def __init__(self, msg: str, *, rebuild: str = "", line: int = 0,
                 col: int = 0):
        if rebuild:
            msg += (
                " Rebuild it with: `" + rebuild +
                "`, or declare the explicit `--pure-python` capability flag."
            )
        super().__init__(msg, line=line, col=col)
        self.rebuild = rebuild
