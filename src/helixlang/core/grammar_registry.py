"""Registry-driven annotation grammar dispatch (doc/38 §5).

``Parser.parse`` dispatches ``#keyword`` annotations through this registry
instead of a literal dict, and ``hxbc.decompile`` walks the *same* registry to
write annotations back — so a plugin grammar is automatically round-trippable
through ``.helixc`` if it declares the ``sim_extensions`` keys it owns.

A plugin registers a grammar with :func:`register_grammar`; a keyword
collision raises :class:`~helixlang.core.errors.PluginConflictError`,
mirroring the plugin backend registry (doc/36 F7).

The full grammar table for core annotations is defined in
``helixlang.core.parser.register_core_grammars``; :func:`ensure_core_grammars`
imports it lazily (the parser imports this module, never the reverse).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from helixlang.core.errors import PluginConflictError

if TYPE_CHECKING:
    from helixlang.core.ast_nodes import Program

# (parser, program) -> None
ParserMethod = Callable[[Any, "Program"], None]
# (analyzer, program) -> None  — runs in the semantic phase (doc/38 §5)
Validator = Callable[[Any, "Program"], None]
# (program) -> annotation source lines (trailing newline omitted)
Decompiler = Callable[["Program"], list[str]]

# Decompiler ordering relative to the generic ``#sim`` fallback in
# ``hxbc.decompile`` (keeps the canonical .helix layout; test_helixc R2).
BEFORE_SIM = "before_sim"
AFTER_SIM = "after_sim"


def fmt_float(x: float) -> str:
    s = repr(x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fmt_str(s: str) -> str:
    if any(c in s for c in " \t=#"):  # noqa: SIM300
        return f'"{s}"'
    return s


def fmt_fields(d: dict[str, str]) -> str:
    parts = []
    for k, v in sorted(d.items()):
        already_quoted = (len(v) >= 2 and v.startswith('"') and v.endswith('"'))
        if (" " in v or "\t" in v) and not already_quoted:
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def sim_entry_decompile(key: str, keyword: str) -> Decompiler:
    """A ``#keyword`` written back from a list of dict entries.

    Each entry in ``sim_extensions[key]`` becomes one annotation line, matching
    the shape its ``parse`` hook produced.  The key is owned only while its
    value *is* a list — a scalar ``#sim key=value`` spelling falls through to
    the generic fallback (the historic behavior).
    """

    def _decompile(program: Program) -> list[str]:
        value = program.sim_extensions.get(key)
        if not isinstance(value, list):
            return []
        lines: list[str] = []
        for entry in value:
            if isinstance(entry, dict):
                lines.append(f"#{keyword} " + fmt_fields(entry))
        return lines

    return _decompile


def dict_entry_decompile(key: str, keyword: str) -> Decompiler:
    """A ``#keyword`` written back from a single dict-valued extension key."""

    def _decompile(program: Program) -> list[str]:
        value = program.sim_extensions.get(key)
        if isinstance(value, dict) and value:
            return [f"#{keyword} " + fmt_fields(value)]
        return []

    return _decompile


def prefix_decompile(prefix: str, keyword: str) -> Decompiler:
    """A ``#keyword`` written back from ``sim_extensions`` keys with a prefix.

    Keys are stripped of the prefix and re-joined into one annotation line
    (e.g. ``person_age=40`` + ``person_sex=M`` -> ``#person age=40 sex=M``).
    """

    def _decompile(program: Program) -> list[str]:
        keys = sorted(
            k for k in program.sim_extensions if k.startswith(prefix))
        fields = {k[len(prefix):]: str(program.sim_extensions[k])
                  for k in keys
                  if not isinstance(program.sim_extensions[k], list)}
        if not fields:
            return []
        return [f"#{keyword} " + fmt_fields(fields)]

    return _decompile


def gem_inline_decompile() -> Decompiler:
    """Re-emit a ``#gem`` annotation whose genome was an inline DNA block.

    Reconstructs the ``#gem`` parameter line from ``gem_*`` keys and writes the
    stored gene/sequence pairs back as ``#<gene_id>`` + 78-column codon lines,
    terminated by ``#end``.
    """

    def _decompile(program: Program) -> list[str]:
        inline = program.sim_extensions.get("gem_inline_genes")
        if not (isinstance(inline, list) and inline):
            return []
        params: list[str] = []
        for k in ("gem_organism", "gem_medium", "gem_use_database",
                  "gem_include_spontaneous", "gem_gapfill",
                  "gem_target_organism", "gem_dynamic", "gem_duration",
                  "gem_dt", "gem_expression"):
            v = program.sim_extensions.get(k, "")
            if v:
                params.append(f"{k[4:]}={v}")
        lines = ["#gem " + " ".join(params)] if params else \
            ["#gem organism=e_coli_k12"]
        for entry in inline:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                gene_id, seq = entry
                lines.append(f"#{gene_id}")
                seq_upper = str(seq).upper()
                codons = [seq_upper[i:i + 3] for i in range(0, len(seq_upper), 3)]
                current_line = ""
                for codon in codons:
                    test = (current_line + " " + codon) if current_line else codon
                    if len(test) > 78:
                        lines.append(current_line)
                        current_line = codon
                    else:
                        current_line = test
                if current_line:
                    lines.append(current_line)
        lines.append("#end")
        return lines

    return _decompile


@dataclass(frozen=True)
class AnnotationGrammar:
    """One ``#keyword`` annotation grammar (doc/38 §5).

    ``parse`` is invoked by ``Parser.parse``; ``validate`` runs in the semantic
    phase; ``decompile`` lets ``hxbc`` write the annotation back.  Extension
    ownership (``extension_keys`` / ``extension_prefixes``) marks which
    ``sim_extensions`` keys this grammar round-trips so the generic ``#sim``
    fallback does not double-emit them.
    """

    keyword: str
    parse: ParserMethod | None = None
    validate: Validator | None = None
    decompile: Decompiler | None = None
    #: Exact ``sim_extensions`` keys this grammar always round-trips.
    extension_keys: frozenset[str] = frozenset()
    #: ``sim_extensions`` keys round-tripped by this grammar while the value
    #: is a list (a scalar ``#sim key=value`` falls through to the fallback).
    list_valued_keys: frozenset[str] = frozenset()
    #: ``sim_extensions`` keys round-tripped by this grammar by key prefix.
    extension_prefixes: frozenset[str] = frozenset()
    sim_section: str = AFTER_SIM
    core: bool = True
    owner: str | None = None

    @property
    def owner_name(self) -> str:
        return self.owner or ("core" if self.core else "plugin")

    def owns_key(self, key: str, value: Any) -> bool:
        if key in self.list_valued_keys:
            return isinstance(value, list)
        if key in self.extension_keys:
            return True
        return any(key.startswith(p) for p in self.extension_prefixes)

    def has_data(self, program: Program) -> bool:
        ext = program.sim_extensions
        if any(k in ext for k in self.extension_keys):
            return True
        if any((k in ext and isinstance(ext[k], list))
               for k in self.list_valued_keys):
            return True
        return any(
            any(k.startswith(p) for k in ext)
            for p in self.extension_prefixes)


class GrammarRegistry:
    """Keyword-keyed store of :class:`AnnotationGrammar` (doc/38 §5).

    Core grammars are registered on parser import; plugins call
    :meth:`register`.  A keyword collision is a hard
    :class:`PluginConflictError` — never a silent override.
    """

    def __init__(self) -> None:
        self._grammars: dict[str, AnnotationGrammar] = {}

    def register(self, grammar: AnnotationGrammar) -> None:
        existing = self._grammars.get(grammar.keyword)
        if existing is grammar:
            return
        if existing is not None:
            raise PluginConflictError(
                f"#{grammar.keyword}", existing.owner_name, grammar.owner_name)
        self._grammars[grammar.keyword] = grammar

    def register_all(self, grammars: list[AnnotationGrammar]) -> None:
        for grammar in grammars:
            self.register(grammar)

    def get(self, keyword: str) -> AnnotationGrammar | None:
        return self._grammars.get(keyword)

    def contains(self, keyword: str) -> bool:
        return keyword in self._grammars

    def grammars(self) -> list[AnnotationGrammar]:
        return list(self._grammars.values())

    def keywords(self) -> list[str]:
        return sorted(self._grammars)

    def sim_grammars(self, section: str) -> list[AnnotationGrammar]:
        return [g for g in self._grammars.values() if g.sim_section == section]

    def owns_key(self, key: str, value: Any) -> bool:
        return any(
            g.owns_key(key, value) for g in self._grammars.values())


#: Process-wide singleton: Parser dispatch and hxbc decompile share it.
grammar_registry = GrammarRegistry()

_CORE_READY = False


def register_grammar(grammar: AnnotationGrammar) -> None:
    """Register a plugin (or over-riding) annotation grammar (doc/38 §5)."""
    grammar_registry.register(grammar)


def ensure_core_grammars() -> None:
    """Register the built-in core grammars once (idempotent).

    The import is deferred so ``grammar_registry`` never imports ``parser`` at
    module scope (``parser`` imports this module).
    """
    global _CORE_READY
    if _CORE_READY:
        return
    from helixlang.core import parser as _parser_mod

    _parser_mod.register_core_grammars()
    _CORE_READY = True
