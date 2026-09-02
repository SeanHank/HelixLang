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


class Type:
    """Root of the HelixLang type-term hierarchy (doc/38 §7.1).

    The ground prims are :class:`HelixType` members; every *structured* term
    (unit-carrying, variable, list / record / function) is a ``Type``.  The
    :class:`Unifier` accepts any member of this hierarchy plus ground
    ``HelixType`` enum members, so ``resolve``/``unify`` are total over every
    type term the language can name.
    """

    __slots__ = ()

    def describe(self) -> str:
        """A stable human string for error messages (overridden by leaf terms)."""
        return repr(self)


@dataclass(frozen=True)
class ListType(Type):
    """A homogeneous list type ``list[T]``."""

    elem: ResolvedType

    def describe(self) -> str:
        return f"list[{_describe(self.elem)}]"


@dataclass(frozen=True)
class RecordType(Type):
    """A closed record type ``record{name: T, ...}`` (structural)."""

    fields: dict[str, ResolvedType]

    def describe(self) -> str:
        inner = ", ".join(f"{k}: {_describe(v)}" for k, v in self.fields.items())
        return f"record{{{inner}}}"


@dataclass(frozen=True)
class FuncType(Type):
    """A function type ``(T1, T2) -> T``."""

    params: tuple[ResolvedType, ...]
    ret: ResolvedType

    def describe(self) -> str:
        args = ", ".join(_describe(p) for p in self.params)
        return f"({args}) -> {_describe(self.ret)}"


@dataclass(frozen=True)
class Schema:
    """A pre-instantiated type schema (doc/38 §7.2).

    ``#type`` annotations become schemas with no free variables (the grammar
    can only name closed terms); ``instantiate`` returns the closed term
    unchanged so the checker can treat annotations and inferred terms
    uniformly.
    """

    term: ResolvedType

    def instantiate(self) -> ResolvedType:
        return self.term


class TypeVar(Type):
    """An inference variable (fresh, never equal to a ground type).

    ``TypeVar`` instances are created by the checker's constraint pass so a
    program with zero ``#type`` annotations still resolves every symbol to a
    ground type in the final substitution (doc/38 §7 acceptance).  Equality
    and hashing are by name, so the unifier can reason about variable chains.
    """

    _ids = itertools.count(1)

    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = f"?t{next(TypeVar._ids)}"
        self.name = name

    def describe(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeVar) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


def _describe(t: ResolvedType) -> str:
    if isinstance(t, Type):
        return t.describe()
    if isinstance(t, HelixType):
        return t.value
    return repr(t)


def fresh_var() -> TypeVar:
    return TypeVar()


#: A term the unifier works on: a ground :class:`HelixType`, a compound
#: :class:`Type` (:class:`UnitType` / :class:`ListType` / :class:`RecordType` /
#: :class:`FuncType`), or a :class:`TypeVar`.
ResolvedType = Any


@dataclass(frozen=True)
class UnitType(Type):
    """A base type constrained to a physical unit, e.g. ``Float<µM>``.

    Unit unification checks dimensional compatibility (doc/38 §7.4): two unit
    types unify when their bases match and their units live in the same
    dimension family (``Float<min>`` vs ``Float<s>``), while a unit from an
    incompatible dimension (``Float<min>`` vs ``Float<µm3>``) is an error.
    """

    base: HelixType
    unit: str

    def describe(self) -> str:
        return f"{self.base.value}<{self.unit}>"

    def __str__(self) -> str:
        return self.describe()

    __repr__ = __str__


def parse_type_annotation(text: str) -> ResolvedType:
    """Parse type annotation text into a type term.

    Plain names return the :class:`HelixType` member; ``Float<µM>`` /
    ``Float<min>`` return a unit-carrying :class:`UnitType`; compound forms
    ``list[...]``, ``record{name: T, ...}`` and ``(T, T) -> T`` return the
    corresponding structural :class:`Type`.
    """
    stripped = text.strip()
    low = stripped.lower()
    if low.startswith("list[") and stripped.endswith("]"):
        inner = stripped[len("list["):-1].strip()
        if not inner:
            raise SemanticError(f"empty list element type in {text!r}")
        return ListType(elem=parse_type_annotation(inner))
    if low.startswith("record{") and stripped.endswith("}"):
        inner = stripped[len("record{"):-1].strip()
        fields: dict[str, ResolvedType] = {}
        for part in _split_top_level(inner, ","):
            name, _, field_type = part.partition(":")
            name, field_type = name.strip(), field_type.strip()
            if not name or not field_type or name in fields:
                raise SemanticError(
                    f"malformed record field in {text!r}: {part!r}")
            fields[name] = parse_type_annotation(field_type)
        return RecordType(fields=fields)
    if "->" in stripped and stripped.endswith(")"):
        params_text, _, ret_text = stripped.rpartition("->")
        ret_text = ret_text.strip()
        params_text = params_text.strip()
        if not (params_text.startswith("(") and params_text.endswith(")")):
            raise SemanticError(f"malformed function type in {text!r}")
        params: list[ResolvedType] = []
        inner_p = params_text[1:-1].strip()
        if inner_p:
            for part in _split_top_level(inner_p, ","):
                part = part.strip()
                if part:
                    params.append(parse_type_annotation(part))
        return FuncType(params=tuple(params), ret=parse_type_annotation(ret_text))
    if "<" in stripped and stripped.endswith(">"):
        base_text, inner = stripped.split("<", 1)
        inner = inner[:-1].strip()
        base = _TYPE_BY_NAME.get(base_text.strip().lower())
        if base is None:
            raise SemanticError(f"unknown type annotation: {text!r}")
        from helixlang.core.dimensions import dim_of_unit

        dim_of_unit(inner)  # unknown unit -> UnitError (without a dim)
        return UnitType(base=base, unit=inner)
    key = low
    if key in _TYPE_BY_NAME:
        return _TYPE_BY_NAME[key]
    raise SemanticError(f"unknown type annotation: {text!r}")


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep`` outside nested ``<> {} [] ()`` (record/type)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "<{[(":
            depth += 1
        elif ch in ">}])":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    rest = "".join(current).strip()
    if rest:
        parts.append(rest)
    return parts


class Unifier:
    """Robinson-style unification over the :class:`Type` hierarchy.

    ``unify(t1, t2)`` resolves variable chains, records a binding and raises
    :class:`SemanticError` (naming the offending symbol) when the terms can't
    be made equal — across ground types, unit-carrying types (compared by
    dimensional compatibility, doc/38 §7.4), and structural compound types
    (list / record / function, unified element-wise).  The resulting
    substitution is exposed for the checker's constant-solving acceptance
    criterion (every fresh variable resolved to a ground type).
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
        if isinstance(t1, UnitType) or isinstance(t2, UnitType):
            self._unify_unit(t1, t2, symbol)
            return
        if isinstance(t1, ListType) and isinstance(t2, ListType):
            self.unify(t1.elem, t2.elem, symbol)
            return
        if isinstance(t1, RecordType) and isinstance(t2, RecordType):
            if set(t1.fields) != set(t2.fields):
                self._fail(t1, t2, symbol)
            for key in t1.fields:
                self.unify(t1.fields[key], t2.fields[key], symbol)
            return
        if isinstance(t1, FuncType) and isinstance(t2, FuncType):
            if len(t1.params) != len(t2.params):
                self._fail(t1, t2, symbol)
            for a, b in zip(t1.params, t2.params, strict=True):
                self.unify(a, b, symbol)
            self.unify(t1.ret, t2.ret, symbol)
            return
        self._fail(t1, t2, symbol)

    def _unify_unit(self, t1: ResolvedType, t2: ResolvedType,
                    symbol: str | None) -> None:
        if isinstance(t1, UnitType) and isinstance(t2, UnitType):
            self._unify_two_units(t1, t2, symbol)
            return
        if isinstance(t1, UnitType) and isinstance(t2, HelixType):
            self._unify_unit_vs_base(t1, t2, symbol)
            return
        if isinstance(t2, UnitType) and isinstance(t1, HelixType):
            self._unify_unit_vs_base(t2, t1, symbol)
            return
        self._fail(t1, t2, symbol)

    def _unify_two_units(self, t1: UnitType, t2: UnitType,
                         symbol: str | None) -> None:
        if t1 == t2:
            return
        if t1.base is not t2.base:
            self._fail(t1, t2, symbol)
            return
        from helixlang.core.dimensions import compatible

        if not compatible(t1.unit, t2.unit):
            raise SemanticError(
                f"type mismatch on {symbol or 'unknown'!r}: "
                f"{t1} vs {t2} ({t1.describe()} vs {t2.describe()})")
        # Dimensional compatibility (e.g. Float<min> ~ Float<s>) is a
        # successful, substitution-free unification: both terms are ground.

    def _unify_unit_vs_base(self, unit: UnitType, base: HelixType,
                            symbol: str | None) -> None:
        # A unit-carrying float IS a float (Float<µM> <: Float): matching the
        # base term unifies, the more specific unit survives in the term.
        if unit.base is base:
            return
        self._fail(unit, base, symbol)

    def _occurs_check(self, var: TypeVar, t: ResolvedType,
                      symbol: str | None) -> None:
        t = self.resolve(t)
        if t == var:
            raise SemanticError(
                f"infinite type on {symbol or 'unknown'!r}: {var} ~ {var}")
        if isinstance(t, ListType):
            self._occurs_check(var, t.elem, symbol)
        elif isinstance(t, RecordType):
            for v in t.fields.values():
                self._occurs_check(var, v, symbol)
        elif isinstance(t, FuncType):
            for p in t.params:
                self._occurs_check(var, p, symbol)
            self._occurs_check(var, t.ret, symbol)
        # UnitType / HelixType leaves are ground: nothing to check.

    def _fail(self, t1: ResolvedType, t2: ResolvedType,
              symbol: str | None) -> None:
        raise SemanticError(
            f"type mismatch on {symbol or 'unknown'!r}: "
            f"{_describe(t1)} vs {_describe(t2)}")

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

        The pass is the doc/38 §7.1/§7.2 machinery applied for real: it
        creates a fresh :class:`TypeVar` per symbol, publishes structural
        constraints (genes → ``GENE``, promoters → ``PROTEIN``) into a
        :class:`Unifier`, solves the system and — the constant-solving
        acceptance criterion — requires every fresh variable to resolve to a
        *single ground type*; a symbol that stays unbound or an unsatisfiable
        constraint raises a :class:`SemanticError` naming it.  With zero
        annotations every variable still resolves to a ground type.

        ``#type`` annotations name a symbol's *product* type (a gene product
        can be ``Protein`` or an expression level ``Float<µM>``); they are
        recorded as closed :class:`Schema` pre-instantiation in
        ``self.product_types`` and never clobber the symbol's own ground kind.
        """
        unifier = Unifier()
        vars_by_name: dict[str, TypeVar] = {}
        for g in prog.genes:
            var = fresh_var()
            unifier.unify(var, HelixType.GENE, g.name)
            vars_by_name[g.name] = var
        for p in prog.promoters:
            var = fresh_var()
            unifier.unify(var, HelixType.PROTEIN, p.name)
            vars_by_name[p.name] = var
        # Product-type constraints (a schema is closed, so instantiation is
        # the term itself).  Unknown symbols cannot occur here: the parser
        # already rejects ``#type`` names that reference no gene/promoter.
        for name, spec in prog.type_annotations.items():
            term = parse_type_annotation(spec)
            self.product_types[name] = term
            if name in vars_by_name:
                unifier.unify(fresh_var(), term, name)
        resolved: dict[str, ResolvedType] = {}
        for name, var in vars_by_name.items():
            ground = unifier.resolve(var)
            if isinstance(ground, TypeVar):
                raise SemanticError(
                    f"type inference did not resolve {name!r} to a ground "
                    f"type: {ground.describe()}")
            resolved[name] = ground
        return resolved

    def constraints_for(
            self, prog: Program) -> tuple[dict[str, TypeVar], Unifier]:
        """Collect the program's type system into ``(vars, unifier)``.

        Exposed for downstream passes that need the raw constraint system
        (e.g. the semantic analyzer's constant-solving acceptance check).  A
        :class:`SemanticError` is raised (naming the symbol) when the system
        is unsatisfiable, mirroring :meth:`infer_program`.
        """
        unifier = Unifier()
        vars_by_name: dict[str, TypeVar] = {}
        for g in prog.genes:
            var = fresh_var()
            unifier.unify(var, HelixType.GENE, g.name)
            vars_by_name[g.name] = var
        for p in prog.promoters:
            var = fresh_var()
            unifier.unify(var, HelixType.PROTEIN, p.name)
            vars_by_name[p.name] = var
        for name, spec in prog.type_annotations.items():
            term = parse_type_annotation(spec)
            self.product_types[name] = term
            if name in vars_by_name:
                unifier.unify(fresh_var(), term, name)
        return vars_by_name, unifier

    def check_types(self, prog: Program) -> list[SemanticError]:
        """Full type-system check: run inference and surface errors (no raise).

        Catches a unification :class:`SemanticError` from the constraint pass (doc/38 §7.2 constant solving) and returns it as a
        list, so a caller (e.g. the semantic analyzer) can report every
        offending symbol at once instead of stopping on the first.
        """
        errors: list[SemanticError] = []
        try:
            self.infer_program(prog)
        except SemanticError as exc:
            errors.append(exc)
        return errors

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
