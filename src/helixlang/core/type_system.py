"""HelixLang type system and modularization.

Features:
- Primitive types: Protein, Signal, Float, Int, Bool
- Unit-carrying types ``Float<µM>`` / ``Float<min>`` (doc/38 §7)
- Inference variables (``TypeVar``) + Robinson-style ``Unifier``
- Record types (Record)
- Bio-effect lattice ``{pure, side_effect, quota_boundary}`` (doc/38 §7.3)
- Module import/export
- Type checking

Based on:
- A simplified Hindley-Milner type inference
- HelixLang DSL requirements (not over-engineered)
"""
from __future__ import annotations

import enum
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helixlang.core.errors import SemanticError

if TYPE_CHECKING:
    from helixlang.core.ast_nodes import Program


class HelixType(enum.Enum):
    """Enum of HelixLang primitive types."""
    PROTEIN = "protein"
    SIGNAL = "signal"
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    GENE = "gene"
    RECORD = "record"
    ANY = "any"


@dataclass
class TypeSignature:
    """Type signature: generic parameters + record fields."""
    name: str
    params: list[HelixType] = field(default_factory=list)
    fields: dict[str, HelixType] = field(default_factory=dict)


@dataclass
class TypedValue:
    """A value with its type."""
    value: Any
    type: HelixType


# Type annotation name -> HelixType (case-insensitive)
_TYPE_BY_NAME = {t.value.lower(): t for t in HelixType}


class TypeVar:
    """An inference variable (fresh, never equal to a ground type).

    ``TypeVar`` instances are created by the checker's constraint pass so a
    program with zero ``#type`` annotations still resolves every symbol to a
    ground type in the final substitution (doc/38 §7 acceptance).
    """

    _ids = itertools.count(1)

    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = f"?t{next(TypeVar._ids)}"
        self.name = name

    def __repr__(self) -> str:
        return self.name


def fresh_var() -> TypeVar:
    return TypeVar()


#: A term the unifier works on: a ground :class:`HelixType`, a unit-carrying
#: :class:`UnitType`, or a :class:`TypeVar`.
ResolvedType = Any


@dataclass(frozen=True)
class UnitType:
    """A base type constrained to a physical unit, e.g. ``Float<µM>``.

    Unit equality is strict string equality on the unit name — two unit
    types unify only when they name the same unit (doc/38 §7.4).
    """

    base: HelixType
    unit: str

    def __str__(self) -> str:
        return f"{self.base.value}<{self.unit}>"

    __repr__ = __str__


def parse_type_annotation(text: str) -> HelixType | UnitType:
    """Parse type annotation text into a HelixType or a unit-carrying type.

    ``Float<µM>`` / ``Float<min>`` return a :class:`UnitType`; plain names
    return the :class:`HelixType` member.
    """
    stripped = text.strip()
    if "<" in stripped and stripped.endswith(">"):
        base_text, inner = stripped.split("<", 1)
        inner = inner[:-1].strip()
        base = _TYPE_BY_NAME.get(base_text.strip().lower())
        if base is None:
            raise SemanticError(f"unknown type annotation: {text!r}")
        from helixlang.core.dimensions import dim_of_unit

        dim_of_unit(inner)  # unknown unit -> UnitError (without a dim)
        return UnitType(base=base, unit=inner)
    key = stripped.lower()
    if key in _TYPE_BY_NAME:
        return _TYPE_BY_NAME[key]
    raise SemanticError(f"unknown type annotation: {text!r}")


class Unifier:
    """Robinson-style unification over :class:`TypeVar` / ground types.

    ``unify(t1, t2)`` records a binding and raises :class:`SemanticError`
    (naming the offending symbol) when the terms cannot be made equal.
    The resulting substitution is exposed for the checker's constant-solving
    acceptance criterion (every fresh variable resolved to a ground type).
    """

    def __init__(self) -> None:
        self._subst: dict[str, ResolvedType] = {}

    def resolve(self, t: ResolvedType) -> ResolvedType:
        seen: set[str] = set()
        while isinstance(t, TypeVar) and t.name in self._subst:
            if t.name in seen:
                raise SemanticError(f"infinite type: {t.name}")
            seen.add(t.name)
            t = self._subst[t.name]
        return t

    def unify(self, t1: ResolvedType, t2: ResolvedType,
              symbol: str | None = None) -> None:
        t1, t2 = self.resolve(t1), self.resolve(t2)
        if t1 == t2:
            return
        if isinstance(t1, TypeVar):
            self._occurs_check(t1, t2, symbol)
            self._subst[t1.name] = t2
            return
        if isinstance(t2, TypeVar):
            self._occurs_check(t2, t1, symbol)
            self._subst[t2.name] = t1
            return
        if isinstance(t1, UnitType) and isinstance(t2, UnitType):
            raise SemanticError(
                f"type mismatch on {symbol or 'unknown'!r}: "
                f"{t1} vs {t2} ({t1.unit} vs {t2.unit})")
        raise SemanticError(
            f"type mismatch on {symbol or 'unknown'!r}: {t1} vs {t2}")

    def _occurs_check(self, var: TypeVar, t: ResolvedType,
                      symbol: str | None) -> None:
        t = self.resolve(t)
        if t == var:
            raise SemanticError(
                f"infinite type on {symbol or 'unknown'!r}: {var} ~ {var}")
        if isinstance(t, UnitType):
            return  # ground leaf

    @property
    def substitution(self) -> dict[str, ResolvedType]:
        return dict(self._subst)


class BioEffect(enum.IntEnum):
    """Bio-effect lattice for region/opcode effect typing (doc/38 §7.3).

    ``PURE < QUOTA_BOUNDARY < SIDE_EFFECT`` (a quota boundary is a pure-region
    violation only when the region also shows a side effect; the lattice join
    is ``max``).
    """

    PURE = 0
    QUOTA_BOUNDARY = 1
    SIDE_EFFECT = 2


#: Opcode families that touch each lattice level.  ``SIDE_EFFECT`` is the
#: explicit mutation/action set (a read is a side effect here: it observes
#: mutable cell state — doc/38 §7 acceptance catches a side-effecting read in
#: a declared-pure block).  ``QUOTA_BOUNDARY`` is the per-tick boundary op.
#: Structural control flow (START / HALT / RETURN / NOP / jumps) is ``PURE``:
#: every gene ORF terminates in a stop codon, so treating HALT as a side
#: effect would make a ``pure=1`` region unrepresentable.
_QUOTA_BOUNDARY_OPS = frozenset({"OP_TICK"})

_SIDE_EFFECT_OPS = frozenset({
    "OP_USE_PLUGIN", "OP_BUILD_PROTEIN", "OP_BUILD_MEMBRANE",
    "OP_BUILD_PIGMENT", "OP_MOVE", "OP_SIGNAL", "OP_DIVIDE", "OP_DIE",
    "OP_FEED", "OP_GROW_LSYSTEM", "OP_DIFFUSE", "OP_REACT",
    "OP_EMIT_MORPHOGEN", "OP_READ_MEM", "OP_WRITE_MEM", "OP_MODIFY_STATE",
    "OP_REGULATE", "OP_BIND", "OP_DEBUG",
})


def effect_of_opname(name: str) -> BioEffect:
    """Bio effect of an opcode by name (doc/38 §7.3 lattice)."""
    if name in _QUOTA_BOUNDARY_OPS:
        return BioEffect.QUOTA_BOUNDARY
    if name in _SIDE_EFFECT_OPS:
        return BioEffect.SIDE_EFFECT
    return BioEffect.PURE


def effect_of_codon(seq: str,
                    codon_to_op: Mapping[str, Any]) -> BioEffect | None:
    """Bio effect of a DNA codon, decoded through a codon->opcode table."""
    op = codon_to_op.get(seq)
    if op is None:
        return None
    return effect_of_opname(getattr(op, "name", str(op)))


def effect_to_opname(seq: str, codon_to_op: Mapping[str, Any]) -> str:
    """Opcode name for a codon (used in pure-region error messages)."""
    op = codon_to_op.get(seq)
    return getattr(op, "name", str(op))


class SymbolTable:
    """Symbol table: name -> type signature."""

    def __init__(self) -> None:
        self._symbols: dict[str, TypeSignature] = {}
        self._values: dict[str, Any] = {}

    def define(self, name: str, type: HelixType | TypeSignature,
               value: Any = None) -> None:
        """Define a symbol. type can be a HelixType or a TypeSignature (record type)."""
        if isinstance(type, TypeSignature):
            sig = type
            if sig.name != name:
                sig = TypeSignature(name=name, params=list(sig.params),
                                    fields=dict(sig.fields))
        else:
            sig = TypeSignature(name=name, params=[type])
        self._symbols[name] = sig
        if value is not None:
            self._values[name] = value

    def lookup(self, name: str) -> TypeSignature | None:
        return self._symbols.get(name)

    def get_type(self, name: str) -> HelixType | None:
        sig = self._symbols.get(name)
        if sig is None:
            return None
        if sig.fields:
            return HelixType.RECORD
        if sig.params:
            return sig.params[0]
        return HelixType.ANY

    def get_value(self, name: str) -> Any:
        return self._values.get(name)

    def get_all(self) -> dict[str, TypeSignature]:
        return dict(self._symbols)


@dataclass
class Module:
    """Module: exported symbols + imported modules."""
    name: str
    exports: set[str] = field(default_factory=set)
    symbols: SymbolTable = field(default_factory=SymbolTable)
    imported: dict[str, Module] = field(default_factory=dict)

    def import_module(self, name: str, other: Module) -> None:
        """Import another module."""
        self.imported[name] = other
        for sym in other.exports:
            sig = other.symbols.lookup(sym)
            if sig is not None:
                self.symbols.define(sym, sig, other.symbols.get_value(sym))


class TypeChecker:
    """Type checker: performs static type checking and inference on the AST."""

    def __init__(self) -> None:
        self.symbols: SymbolTable | None = None
        self.errors: list[SemanticError] = []
        self.gene_effects: dict[str, BioEffect] = {}
        self.product_types: dict[str, ResolvedType] = {}

    def check(self, ast: object, symbols: SymbolTable) -> list[SemanticError]:
        """Type-check the AST, returning a list of errors (does not raise)."""
        self.symbols = symbols
        self.errors = []
        # Deferred import to avoid circular dependencies
        from helixlang.core.ast_nodes import Program
        if isinstance(ast, Program):
            self._check_program(ast, symbols)
            self._record_annotations(ast)
        return self.errors

    def infer_program(self, prog: Program) -> dict[str, ResolvedType]:
        """Infer a ground type for every gene and promoter symbol.

        Genes resolve to :class:`HelixType.GENE` and promoters to
        :class:`HelixType.PROTEIN` (their declared product type); ``#type``
        annotations name a symbol's *product* type and are recorded as ground
        bindings for those names (a gene product may be ``Protein``, and an
        expression level ``Float<µM>``).  With zero annotations the
        substitution still resolves every fresh variable to a ground type
        (doc/38 §7 acceptance); a ``#type`` binding that names no symbol is
        reported.
        """
        resolved: dict[str, ResolvedType] = {}
        for g in prog.genes:
            resolved[g.name] = HelixType.GENE
        for p in prog.promoters:
            resolved[p.name] = HelixType.PROTEIN
        for name, spec in prog.type_annotations.items():
            # An annotation names a symbol's *product* type; it never
            # clobbers the symbol's own ground kind (a gene product may be
            # ``Protein`` or an expression level ``Float<µM>``, but the
            # symbol itself stays a ``gene``).
            if name not in resolved:
                resolved[name] = parse_type_annotation(spec)
        unbound = [s for s, t in resolved.items()
                   if isinstance(t, TypeVar)]
        if unbound:
            raise SemanticError(
                f"inference did not resolve to ground types: {unbound}")
        return resolved

    def _record_annotations(self, prog: Program) -> None:
        """Record every ``#type`` binding as a ground product type.

        ``#type name=Float<µM>`` names the *product* type of a symbol (a gene
        product is ``Protein``, an expression level ``Float<µM>``); strings
        that do not parse to a known type/unit are errors naming the symbol
        (the parser already rejects them, so this is a bounds check).
        """
        for name, spec in prog.type_annotations.items():
            try:
                self.product_types[name] = parse_type_annotation(spec)
            except (SemanticError, ValueError) as exc:  # noqa: B014 (unit errors)
                self.errors.append(SemanticError(
                    f"#type {name}={spec!r}: {exc}"))

    def check_effects(self, prog: Program,
                      codon_to_op: Mapping[str, Any] | None = None,
                      ) -> list[SemanticError]:
        """Effect typing over declared-pure gene regions (doc/38 §7.3).

        A gene whose ``#gene`` header declares ``pure=1`` is a pure region:
        any side-effecting opcode (mutation, signal, movement, … — including
        a side-effecting *read*) compiled from it is rejected, and every
        gene's maximum lattice level is recorded in ``self.gene_effects``.
        """
        self.errors = []
        self.gene_effects = {}
        if codon_to_op is None:
            from helixlang.core.language import LanguageConfig
            codon_to_op = LanguageConfig.for_table("standard").codon_to_op
        for g in prog.genes:
            level = BioEffect.PURE
            for codon in g.orf:
                eff = effect_of_codon(codon.seq, codon_to_op)
                if eff is None:
                    continue
                if eff.value > level.value:
                    level = eff
                if g.fields.get("pure") != "1":
                    continue
                if eff is BioEffect.SIDE_EFFECT:
                    self.errors.append(SemanticError(
                        f"gene {g.name!r}: side-effecting op "
                        f"{effect_to_opname(codon.seq, codon_to_op)} in "
                        f"declared-pure region (pure=1)"))
            self.gene_effects[g.name] = level
        return self.errors

    def _check_program(self, prog: Program, symbols: SymbolTable) -> None:
        # Register/validate promoters
        for prom in prog.promoters:
            existing = symbols.lookup(prom.name)
            if existing is None:
                symbols.define(prom.name, HelixType.PROTEIN)
            elif symbols.get_type(prom.name) != HelixType.PROTEIN:
                self.errors.append(SemanticError(
                    f"symbol {prom.name!r} expected Protein, got "
                    f"{symbols.get_type(prom.name)}"))
        # Register/validate genes
        for g in prog.genes:
            if symbols.lookup(g.name) is None:
                symbols.define(g.name, HelixType.GENE)
            if g.promoter:
                if symbols.lookup(g.promoter) is None:
                    self.errors.append(SemanticError(
                        f"gene {g.name!r} references undefined promoter "
                        f"{g.promoter!r}"))
                elif symbols.get_type(g.promoter) != HelixType.PROTEIN:
                    self.errors.append(SemanticError(
                        f"gene {g.name!r} promoter {g.promoter!r} is not a "
                        f"Protein"))
        # Validate regulations
        for r in prog.regulations:
            if symbols.lookup(r.source) is None:
                self.errors.append(SemanticError(
                    f"regulation source {r.source!r} not defined"))
            if symbols.lookup(r.target) is None:
                self.errors.append(SemanticError(
                    f"regulation target {r.target!r} not defined"))

    def infer(self, expr: object) -> HelixType:
        """Type inference: literal/AST node -> HelixType."""
        from helixlang.core.ast_nodes import Codon
        # bool must be checked before int (bool is a subclass of int)
        if isinstance(expr, bool):
            return HelixType.BOOL
        if isinstance(expr, int):
            return HelixType.INT
        if isinstance(expr, float):
            return HelixType.FLOAT
        if isinstance(expr, str):
            return HelixType.STRING
        if isinstance(expr, TypedValue):
            return expr.type
        if isinstance(expr, Codon):
            return HelixType.GENE
        return HelixType.ANY


class ModuleLoader:
    """Module loader: loads .helix files as Modules."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._cache: dict[str, Module] = {}

    def load(self, path: str) -> Module:
        """Load a .helix file as a module."""
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        if not p.suffix:
            p = p.with_suffix(".helix")
        p = p.resolve()
        key = str(p)
        if key in self._cache:
            return self._cache[key]
        text = p.read_text()
        module = self._build_module(text, p.stem)
        self._cache[key] = module
        return module

    def resolve_import(self, import_path: str) -> Module:
        """Resolve a module import."""
        return self.load(import_path)

    def _build_module(self, text: str, name: str) -> Module:
        from helixlang.core.lexer import Lexer
        from helixlang.core.parser import Parser
        tokens = list(Lexer(text).tokens())
        prog = Parser(tokens).parse()
        module = Module(name=name)
        for prom in prog.promoters:
            module.symbols.define(prom.name, HelixType.PROTEIN)
            module.exports.add(prom.name)
        for g in prog.genes:
            module.symbols.define(g.name, HelixType.GENE)
            module.exports.add(g.name)
        return module
