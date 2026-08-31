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
from typing import TYPE_CHECKING, Any, Literal

from helixlang.core.errors import ParseError, PluginConflictError, SemanticError

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

#: Declared field value types (doc/41 §4.2).  ``list``/``dict`` accept the
#: comma-separated ``k=v[,k=v...]`` spellings already lexed as field values.
FieldType = Literal["str", "float", "int", "bool", "list", "dict"]


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


# ---------------------------------------------------------------------------
# GrammarDescriptor + FieldSpec (doc/41 §4.2): declarative plugin grammars
# ---------------------------------------------------------------------------
def _coerce_field(spec: FieldSpec, raw: str, *, keyword: str) -> str:
    """Validate+coerce one raw field value; returns its canonical string form.

    The canonical form keeps ``sim_extensions`` all-``str`` (the same invariant
    the generic ``#sim`` fallback and every downstream reader expect), which
    lets the co-erced entry round-trip the source byte-for-byte.
    """
    value: Any = raw
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if spec.type == "float":
        try:
            n = float(value)
        except (TypeError, ValueError) as e:
            raise ParseError(
                f"#{keyword} {spec.key}= expects a float, got {raw!r}: {e}") from None
        return fmt_float(n)
    if spec.type == "int":
        try:
            n = int(value)
        except (TypeError, ValueError) as e:
            raise ParseError(
                f"#{keyword} {spec.key}= expects an int, got {raw!r}: {e}") from None
        return str(n)
    if spec.type == "bool":
        lowered = str(value).strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return "true"
        if lowered in ("false", "0", "no", "off"):
            return "false"
        raise ParseError(
            f"#{keyword} {spec.key}= expects a bool (true/false), got {raw!r}")
    if spec.type == "list":
        return ",".join(p.strip() for p in str(value).split(",") if p.strip())
    if spec.type == "dict":
        pairs: list[str] = []
        for chunk in str(value).split(","):
            if chunk.strip():
                k, sep, v = chunk.partition("=")
                if not sep:
                    raise ParseError(
                        f"#{keyword} {spec.key}= expects k=v pairs, got {chunk!r}")
                pairs.append(f"{k.strip()}={v.strip()}")
        return ",".join(pairs)
    return str(value)


@dataclass(frozen=True)
class FieldSpec:
    """One declared ``key=value`` field of a descriptor grammar (doc/41 §4.2).

    ``type`` drives the generic parse hook's validation + canonical coercion.
    ``unit`` is the physical dimension (mol, s, ...) consumed by Item 5's
    static unit checks — the hook stores it for the semantic phase.
    """

    key: str
    type: FieldType = "str"
    required: bool = False
    default: str | None = None
    unit: str | None = None


@dataclass(frozen=True)
class GrammarDescriptor:
    """Declarative ``#keyword`` grammar (doc/41 §4.2).

    Replaces hand-written ``Parser._parse_*`` for data annotations: a
    ``fields``-body descriptor compiles to a generic parse hook (field
    collection + type coercion + required checks) served by the registry, so a
    plugin author declares new ``#keyword`` syntax entirely from their plugin
    module — zero edits to ``parser.py`` or ``lexer.py``.

    ``body`` mirrors the structural bodies the parser still owns:
    ``"fields"`` splits ``k=v`` pairs; ``"raw"`` keeps a plugin-supplied
    ``parse`` callable that uses ``parser.token_hooks``.

    ``target`` declares the AST destination: ``"sim_extensions"`` (default)
    stores a list of field dicts under ``program.sim_extensions[keyword]``;
    ``"section"`` stores into a ``Program`` list-of-dicts section.
    """

    keyword: str
    fields: tuple[FieldSpec, ...] = ()
    allow_extra: bool = True
    body: Literal["fields", "raw"] = "fields"
    target: Literal["section", "sim_extensions"] = "sim_extensions"
    parse: ParserMethod | None = None
    validate: Validator | None = None
    decompile: Decompiler | None = None
    extension_key: str | None = None
    owner: str | None = None

    @property
    def key(self) -> str:
        return self.extension_key or self.keyword


def _parse_spec_keyword(desc: GrammarDescriptor) -> ParserMethod:
    """A generic ``fields``-body parse hook backed by :class:`FieldSpec`."""

    by_key = {f.key: f for f in desc.fields}

    def _parse(parser: Any, program: Program) -> None:
        t = parser._advance()  # ANNOT_START
        raw = parser._collect_fields_until_block_end(allow_no_end=True)
        entry: dict[str, str] = {}
        if not desc.allow_extra:
            unknown = sorted(k for k in raw if k not in by_key)
            if unknown:
                raise ParseError(
                    f"#{desc.keyword} has unknown field(s) {', '.join(unknown)}",
                    line=t.line)
        for spec in desc.fields:
            if spec.key in raw:
                entry[spec.key] = _coerce_field(
                    spec, raw[spec.key], keyword=desc.keyword)
            elif spec.required:
                raise ParseError(
                    f"#{desc.keyword} requires {spec.key}= field", line=t.line)
            elif spec.default is not None:
                entry[spec.key] = spec.default
        for key, value in raw.items():
            if key not in by_key:
                entry[key] = value
        store = getattr(program, "sim_extensions", None)
        if desc.target == "section":
            current = getattr(program, desc.key, None)
            if current is None or not isinstance(current, list):
                raise ParseError(
                    f"#{desc.keyword} target section {desc.key!r} is not a "
                    "list-of-dicts program section", line=t.line)
            current.append(dict(entry))
            return
        if store is not None:
            existing = store.get(desc.key)
            if existing is None:
                existing = []
                store[desc.key] = existing
            existing.append(dict(entry))

    return _parse


def _default_descriptor_decompile(desc: GrammarDescriptor) -> Decompiler:
    if desc.target == "section":
        return lambda program: []  # structural sections own their own emission
    return sim_entry_decompile(desc.key, desc.keyword)


def _parse_quantity_stmt(parser: Any, program: Program) -> None:
    """Parse ``#quantity name=X expr=A+B`` (doc/41 §6, Ring 1).

    The body is a two-atom physical composition ``A+B`` / ``A-B`` where each
    atom is a ``#type``-annotated symbol or a bare number.  The statement is
    stored verbatim in ``sim_extensions["quantity"]``; dimension checking runs
    in the semantic phase (:class:`~helixlang.core.dim_inferencer.DimInferencer`).

    Accepted forms (both round-trip to the canonical two-field spelling):
    ``#quantity total=g+v``  or  ``#quantity name=total expr=g+v``.
    """
    from helixlang.core.dim_inferencer import parse_quantity_expr

    t = parser._advance()  # ANNOT_START
    raw = parser._collect_fields_until_block_end(allow_no_end=True)
    if "name" in raw and "expr" in raw:
        name, expr = raw["name"], raw["expr"]
    elif len(raw) == 1:
        name, expr = next(iter(raw.items()))
    else:
        raise ParseError(
            "#quantity requires `name=TOTAL expr=A+B` "
            "(or the compact `#quantity TOTAL=A+B`)",
            line=t.line)
    if not name or not expr:
        raise ParseError("#quantity requires a non-empty name and expr",
                         line=t.line)
    try:
        parse_quantity_expr(expr)  # reject malformed bodies at parse time
    except (SemanticError, ValueError) as exc:
        raise ParseError(f"#quantity {name}=: {exc}", line=t.line) from None
    program.sim_extensions.setdefault("quantity", []).append(
        {"name": name, "expr": expr})


def _quantity_decompile(program: Program) -> list[str]:
    """Re-emit ``#quantity name=… expr=…`` lines (round-trip R1/R2)."""
    out: list[str] = []
    for entry in program.sim_extensions.get("quantity", []):
        if isinstance(entry, dict) and "name" in entry and "expr" in entry:
            out.append(f"#quantity name={entry['name']} "
                       f"expr={entry['expr']}")
    return out


def compile_descriptor(desc: GrammarDescriptor) -> AnnotationGrammar:
    """Compile a :class:`GrammarDescriptor` into an :class:`AnnotationGrammar`.

    The returned grammar's parse hook is the generic field collector for
    ``body="fields"`` (or ``desc.parse`` for ``body="raw"``); its decompile is
    ``desc.decompile`` or the ``#keyword fields...`` re-emitter by default.
    The ``owner`` propagates so keyword-vs-keyword collisions raise the same
    :class:`PluginConflictError` as backend dispatch.
    """
    parse = desc.parse if (desc.parse is not None or desc.body == "raw") \
        else _parse_spec_keyword(desc)
    decompile = desc.decompile or _default_descriptor_decompile(desc)
    keys = frozenset({desc.key}) if desc.target == "sim_extensions" \
        else frozenset()
    list_keys = frozenset({desc.key}) if desc.target == "sim_extensions" \
        else frozenset()
    return AnnotationGrammar(
        keyword=desc.keyword,
        parse=parse,
        validate=desc.validate,
        decompile=decompile,
        extension_keys=keys if not list_keys else frozenset(),
        list_valued_keys=list_keys if keys else frozenset(),
        core=False,
        owner=desc.owner,
    )


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
    #: When set, the grammar is *inert* until the program declares
    #: ``#use <requires_use>`` (doc/41 §4.2: ``#use`` activates a grammar set).
    requires_use: str | None = None

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
        if existing is not None and existing.owner_name != grammar.owner_name:
            raise PluginConflictError(
                f"#{grammar.keyword}", existing.owner_name, grammar.owner_name)
        self._grammars[grammar.keyword] = grammar

    def register_all(self, grammars: list[AnnotationGrammar]) -> None:
        for grammar in grammars:
            self.register(grammar)

    def register_descriptor(self, descriptor: GrammarDescriptor) -> AnnotationGrammar:
        """Compile and register a :class:`GrammarDescriptor` grammar."""
        grammar = compile_descriptor(descriptor)
        self.register(grammar)
        return grammar

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


def register_descriptor(descriptor: GrammarDescriptor) -> AnnotationGrammar:
    """Compile and register a :class:`GrammarDescriptor` grammar (doc/41 §4.2).

    Public convenience wrapper over :meth:`GrammarRegistry.register_descriptor`,
    exported through the frozen ``helixlang.api.grammar`` surface.
    """
    return grammar_registry.register_descriptor(descriptor)


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
